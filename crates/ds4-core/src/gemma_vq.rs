//! `gemma_vq` — the EmbeddingGemma-2 VQ artifact contract: the third ds4 VQ
//! family, after `glm_vq` and `ling31_vq`.
//!
//! The contract is `docs/ds4-gemma-payload-contract.md` (phase 0, row X.8),
//! subordinate to `embeddinggemma2-handoff.md`. The artifact is a GGUF whose
//! architecture is `gemma_vq` and whose 461 matrix tensors are ds4's own type
//! 42 (`vqblob`, one byte per element) carrying a DQVQ payload exactly as the
//! reference writes it: `[u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]`
//! then `nc*dim` f16 codebook values, then `rows` f16 row gains, then the index
//! stream, with one value's rule `codebook[idx][d] * g_r[row]`
//! (`ds41_vq_fmt.h:4-12`, `:73-105`).
//!
//! Host/device split (contract section 6, the same rule the two shipping
//! families state in the first line of their bind include): Rust validates the
//! complete artifact before the C adapter sees a pointer, and C never parses
//! the payload. This module is the whole Rust side — the family gate, the
//! per-payload validation, the file-wide structural rules, and the layout the
//! bind resolves through.
//!
//! What the checks are for, in the contract's terms: the artifact declares a
//! VQ tensor as `ne = {payload_bytes}`, because type 42 is one byte per element
//! and there is no size field inside a DQVQ payload — its length IS the byte
//! account. So the payload's own 16-byte header is the only place its geometry
//! is written, and the loader's job is to make that header agree with the byte
//! account, with the length the file declares, and with the family layout
//! (`crate::gemma_vq::layout`, which computes the same account from the tower's
//! architecture). A mistyped `dim`, `nc`, `rows` or `cols` cannot then slip
//! through as garbage.

mod layout;

use std::collections::HashSet;

use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo};

/// The family's architecture string, and the key that makes the file
/// self-identifying (contract section 0, section 4).
pub(super) const ARCH: &[u8] = b"gemma_vq";
const FORMAT_VERSION: u32 = 1;
const FORMAT_VERSION_KEY: &str = "gemma_vq.format_version";
/// The writer's default (`/data/YoungAi/src/common/ds4_gguf_write.h:37`) and
/// the VQ precedent's choice (`v41_to_gguf.c:361`).
const ALIGN: u64 = 32;
/// The four full-attention layers, from the checkpoint's `layer_types`
/// (`.models/EmbeddingGemma-2-914f7f8/config.json`): a full layer every sixth
/// one, and a full layer widens the head to 512 with a single KV head. The
/// other twenty slide.
pub(super) const FULL_ATTN_LAYERS: [u32; 4] = [5, 11, 17, 23];
/// The full-attention head width, the artifact's own
/// `gemma_vq.attention.key_length`/`.value_length` (`out/artifact-kv.txt`).
const FULL_HEAD: u64 = 512;
/// The per-layer-input width (`hidden_size_per_layer_input`).
const PLE_DIM: u64 = 512;
/// The embedding width this model exists to produce (`1_Pooling` and the
/// checkpoint's `embedding_dim`).
const OUT_DIM: u64 = 768;
/// ds4's private GGUF type 42, `vqblob`: opaque bytes, one per element
/// (`/data/YoungAi/src/core/core_gguf.c:94-97`, mirrored at
/// `crates/ds4-core/src/tensors.rs:106-112`).
const VQBLOB: u32 = 42;
const Q8_0: u32 = 8;
const BF16: u32 = 30;
/// `ds4_quantfmt.c:17`: 32 values in 34 bytes.
const Q8_0_BLOCK: u64 = 32;
const Q8_0_BYTE: u64 = 34;
/// `DS4VQ_MAT_MAGIC` = `'DQVQ'` (`ds41_vq_fmt.h:32`). The v3 form carries a
/// different magic precisely so an older reader fails to recognize it instead
/// of mis-decoding it (`vq_fmt.h:16-18`); this family writes v2 payloads only,
/// which is what the reference decoder's whitelist accepts (`:75`).
const DQVQ: u32 = 0x5156_5144;
const VQ_HEADER: u64 = 16;
/// An f16 word is infinite or NaN when every exponent bit is set.
const F16_INF_NAN: u16 = 0x7c00;
/// The maximum rank a GGUF tensor directory may declare
/// (`crates/ds4-core/src/tensors.rs:266`).
const MAX_RANK: u32 = 4;

/// Tokens in one item: the tower's ceiling. It is the device attention
/// kernel's key plane (`cuda/eg2vq_attn.cuh:49`, `EG2VQ_ATTN_KEYS_MAX`), which
/// the C entry point's own cap mirrors (`ds4_eg2vq.h:140`,
/// `DS4_EG2VQ_MAX_TOKENS`, 2048) and which the kernel adapter static-asserts
/// equal (`ds4_eg2vq_gpu.cuh:45-46`). It is deliberately NOT the checkpoint's
/// `max_position_embeddings` (262144, the artifact's `gemma_vq.context_length`,
/// `out/artifact-kv.txt`): the attention is windowed to the symmetric +-512
/// (`ds4_eg2vq.h:49-52`, `src/llama-hparams.h:524-533`), and beyond the plane
/// the kernels cannot hold the item at all. A longer item is refused by name,
/// never truncated -- a plane that silently covered half an item would be a
/// wrong answer rather than a slow one.
///
/// The three copies of this number (here, the C macro, the CUDA enum) are held
/// together by `crates/ds4-core/tests/eg2vq_serving_caps.rs`, which reads both
/// C sources and fails if either disagrees with this constant.
pub const ITEM_TOKENS_MAX: u32 = 2048;

/// Total tokens over one batch's items: the row budget a batched embedding call
/// may occupy in the tower's single workspace, and therefore the device memory
/// a batch may ask for. It is the thing unit 27 made reachable: before it, an
/// array request paid one tower pass per item, and this number bound the sum
/// without anything being able to use it.
///
/// The layout is exactly linear in rows at 116,748 bytes each (114.0 KiB: the
/// tower's 27,648 f32 planes per row, the 4-byte id, the batch's 4 + 4 bytes of
/// position/length metadata, and the projection's own 768-float output plane plus
/// the pooled 768-float result plane the batch pools per item,
/// `ds4_eg2vq_gpu.cuh`'s `eg2vq_gpu_layout_of`), plus the (rows + 1) item spans.
/// So 8192 rows is four times the single-item ceiling and 912.13 MiB of
/// workspace, checkable against the measured 228.03 MiB at 2048 rows
/// (`tools/eg2_embed_batch_gate27.sh` prints both from the device oracle).
///
/// The same numbers were 108,548 bytes and 848.0 MiB before unit 27, and that
/// arithmetic was already one plane stale: unit 25's image splice added the
/// 512-float `in_soft` plane (2,048 bytes a row), which this comment and
/// crates/ds4-core/tests/eg2vq_serving_caps.rs kept quoting at the pre-splice
/// cost while the layout charged the new one. Unit 27's gate measures the
/// workspace it actually allocates, which is why the correction is stated here
/// rather than silently folded in.
///
/// This is a POLICY of this port, not a file fact: nothing in the artifact
/// states a batch width, so the number is declared here, stated in the serving
/// caps, mirrored by the C entry point's `DS4_EG2VQ_MAX_ROWS` (held equal by
/// crates/ds4-core/tests/eg2vq_serving_caps.rs, which reads the header as text)
/// and enforced by name where a request exceeds it.
pub const BATCH_ROWS_MAX: u32 = 8192;

/// The id the artifact's own vocabulary gives the image placeholder, and the
/// position the checkpoint's model card marks an image at
/// (`.models/EmbeddingGemma-2-914f7f8/README.md` section 5: "`<|image|>` marks
/// the position of image input"). It is the checkpoint's `image_token_id`
/// (`config.json:51`) and it is ALSO the id the vision tower's soft rows fill:
/// the reference processor expands the one placeholder into the model's begin
/// marker, one placeholder per soft row and its end marker, which is the shape
/// the vision anchor's own 92-position sequence carries
/// (tests/parity/eg2vq_vision_anchor.cpp:106-113, and the C header's
/// `DS4_EG2VQ_IMAGE_TOKEN_ID`, ds4_eg2vq.h). Measured against the artifact:
///
///   cargo run -p ds4-core --release --example gemma_vq_vocab -- <artifact.gguf> 255999,258880,258882
///   ->  255999  <|image>   258880  <|image|>   258882  <image|>
///
/// and the tokenizer's own reading of the strings, which is how the wire
/// builds a sequence:
///
///   cargo run -p ds4-core --release --example gemma_vq_tok -- <artifact.gguf> <items.tsv> <prompts.json>
///   ->  <|image|> = 258880,  <|image> = 255999,  <image|> = 258882
pub const IMAGE_TOKEN_ID: i32 = 258_880;
/// The begin and end markers the reference wraps the expanded span with
/// (`tools/mtmd/mtmd.cpp:955-958` at the pin; `boi_token_id` /
/// `eoi_token_id`, `.models/EmbeddingGemma-2-914f7f8/config.json:47,50`).
/// They are carried as STRINGS because that is how they enter a sequence: the
/// family's tokenizer encodes them from the same vocabulary the reference reads
/// them out of, so no id here is invented.
pub const IMAGE_BEGIN_MARKER: &str = "<|image>";
pub const IMAGE_END_MARKER: &str = "<image|>";
/// The placeholder as the text carries it, the literal the wire looks for.
pub const IMAGE_PLACEHOLDER: &str = "<|image|>";

/// The vision tower's soft-token budget for one image: the model card's 70 to
/// 1120 (README.md section 5, "Vision Token Budget"), which is the checkpoint's
/// own `hparams.set_limit_image_tokens(70, 1120)` at the pin
/// (`tools/mtmd/clip.cpp:1734`) over `patch_area` 2304
/// (`tools/mtmd/clip-model.h:207-212`). The size solver keeps every image inside
/// this window -- a small one is upscaled, a large one downscaled -- so these
/// are the engine's own representation bounds, NOT refusal boundaries: the
/// refusals are the item ceiling above (which counts the soft rows) and the
/// solver's own "cannot place" stop.
pub const IMAGE_SOFT_TOKENS_MIN: u32 = 70;
pub const IMAGE_SOFT_TOKENS_MAX: u32 = 1120;
/// The pixel window the same budget maps to (`patch_area` x the soft-token
/// limits), and the largest content the preprocessing will produce.
pub const IMAGE_PIXELS_MIN: u32 = 161_280;
pub const IMAGE_PIXELS_MAX: u32 = 2_580_480;

/// Images one item may carry. The splice entry fills ONE contiguous run of
/// placeholders (`ds4_eg2vq_graph.inc`, `ds4_eg2vq.h`'s
/// `ds4_eg2vq_ref_embed_image`), so one is what this port accepts; the
/// checkpoint itself and the reference allow several interleaved media items
/// (README.md section 5), which would need one span per image. Declared here so
/// the route can refuse the second one by name rather than mis-splice it.
pub const IMAGE_PER_ITEM_MAX: u32 = 1;

/// The vision tower's row width: the 768-to-512 projection's output
/// (`ds4_eg2vq.h`'s `DS4_EG2VQ_N_EMBD`, which is also the mmproj's
/// `mm.input_projection.weight` row length). The wire needs it to size the row
/// buffer it hands the C side; `crates/ds4-core/tests/eg2vq_serving_caps.rs`
/// pins it against the C header so the two cannot drift.
pub const IMAGE_EMBD: u32 = 512;

type Result<T> = std::result::Result<T, TensorError>;

fn fail(reason: &'static str) -> TensorError {
    TensorError::GemmaVq(reason)
}

/// Whether this file claims to be a `gemma_vq` artifact. The architecture
/// string is the family gate; `gemma_vq.*` is the fallback for a file whose
/// architecture key was stripped, mirroring `glm_vq.rs:419-437`.
///
/// Private type 42 deliberately does NOT trigger this: V4.1 also uses
/// `vqblob` for its per-layer expert blobs, whose contents are DQVL
/// containers, not DQVQ payloads.
pub(super) fn requested(g: &GgufFile) -> bool {
    g.get_string("general.architecture") == Some(ARCH)
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(b"gemma_vq."))
}

pub(super) fn layouts() -> Vec<crate::LayoutSpec> {
    layout::layouts()
}

/// The artifact's metadata checked against the family's declared shape — the
/// role `ds4.c`'s `config_validate_model` per-arch arm plays for the families
/// it validates, and the role `validate_qwen35`/`validate_glm53` play here.
/// The payload and directory checks already ran, in
/// `crate::tensors::parse_one`; this is the layer above them.
pub(super) fn validate_metadata(
    g: &GgufFile,
    shape: &crate::Shape,
) -> std::result::Result<(), crate::ValidateError> {
    expect_u32(g, "gemma_vq.block_count", shape.n_layer)?;
    expect_u32(g, "gemma_vq.embedding_length", shape.n_embd)?;
    expect_u32(g, "gemma_vq.vocab_size", shape.n_vocab)?;
    expect_u32(g, "gemma_vq.attention.head_count", shape.n_head)?;
    expect_u32(g, "gemma_vq.feed_forward_length", shape.n_ff_dense)?;
    expect_u32(g, "gemma_vq.context_length", shape.rope_orig_ctx as u32)?;
    expect_u32(g, "gemma_vq.embedding_length_per_layer_input", PLE_DIM as u32)?;
    expect_u32(g, "gemma_vq.embedding_length_out", OUT_DIM as u32)?;
    expect_u32(g, "gemma_vq.attention.key_length", FULL_HEAD as u32)?;
    expect_u32(g, "gemma_vq.attention.value_length", FULL_HEAD as u32)?;
    expect_u32(g, "gemma_vq.attention.key_length_swa", shape.n_head_dim)?;
    expect_u32(g, "gemma_vq.attention.value_length_swa", shape.n_value_dim)?;
    expect_u32(g, "gemma_vq.rope.dimension_count", FULL_HEAD as u32)?;
    expect_u32(g, "gemma_vq.rope.dimension_count_swa", shape.n_rot)?;
    expect_u32(g, "gemma_vq.attention.sliding_window", shape.n_swa)?;
    // A mean-pooled embeddings family: the checkpoint's `1_Pooling/config.json`
    // is mean pooling, and the converter writes the same value
    // (conversion/gemma.py:829).
    expect_u32(g, "gemma_vq.pooling_type", 1)?;
    expect_f32(
        g,
        "gemma_vq.attention.layer_norm_rms_epsilon",
        shape.rms_eps,
    )?;
    expect_f32(g, "gemma_vq.rope.freq_base", shape.rope_freq_base)?;
    expect_f32(g, "gemma_vq.rope.freq_base_swa", shape.rope_freq_base_swa)?;
    if g.get_bool("gemma_vq.attention.causal") != Some(false) {
        return Err(crate::ValidateError::TokenKey(
            "gemma-vq metadata",
            "gemma_vq.attention.causal".into(),
        ));
    }
    // The per-layer KV head count: two heads while sliding, one on the four
    // full-attention layers, which is the array the checkpoint's
    // `per_layer_config` produces (conversion/gemma.py:714-726).
    let kv = array(g, "gemma_vq.attention.head_count_kv")?;
    if kv.len() != shape.n_layer as usize {
        return Err(crate::ValidateError::TokenKey(
            "gemma-vq metadata",
            "gemma_vq.attention.head_count_kv".into(),
        ));
    }
    for (il, heads) in kv.iter().enumerate() {
        let want = if is_full_attn(il as u32) {
            1
        } else {
            shape.n_head_kv
        };
        if *heads != want {
            return Err(crate::ValidateError::TokenKey(
                "gemma-vq metadata",
                "gemma_vq.attention.head_count_kv".into(),
            ));
        }
    }
    // The sliding-window pattern: 24 entries that split 20/4 exactly on the
    // full-attention layers. The polarity is phase 4's business — it has no
    // meaning until the mask exists — so this checks the split and the split
    // points, and neither assumes nor forbids either polarity.
    let pattern = bools(g, "gemma_vq.attention.sliding_window_pattern")?;
    if pattern.len() != shape.n_layer as usize {
        return Err(crate::ValidateError::TokenKey(
            "gemma-vq metadata",
            "gemma_vq.attention.sliding_window_pattern".into(),
        ));
    }
    let full_value = pattern[FULL_ATTN_LAYERS[0] as usize];
    let slide_value = pattern[if is_full_attn(0) { 1 } else { 0 }];
    for il in 0..shape.n_layer {
        let want = if is_full_attn(il) {
            full_value
        } else {
            slide_value
        };
        if pattern[il as usize] != want || full_value == slide_value {
            return Err(crate::ValidateError::TokenKey(
                "gemma-vq metadata",
                "gemma_vq.attention.sliding_window_pattern".into(),
            ));
        }
    }
    Ok(())
}

fn is_full_attn(il: u32) -> bool {
    FULL_ATTN_LAYERS.contains(&il)
}

fn expect_u32(
    g: &GgufFile,
    key: &'static str,
    want: u32,
) -> std::result::Result<(), crate::ValidateError> {
    if g.get_u32(key) == Some(want) {
        return Ok(());
    }
    Err(crate::ValidateError::TokenKey("gemma-vq metadata", key.into()))
}

fn expect_f32(
    g: &GgufFile,
    key: &'static str,
    want: f32,
) -> std::result::Result<(), crate::ValidateError> {
    if g.get_f32_compat(key) == Some(want) {
        return Ok(());
    }
    Err(crate::ValidateError::TokenKey("gemma-vq metadata", key.into()))
}

fn array(g: &GgufFile, key: &'static str) -> std::result::Result<Vec<u32>, crate::ValidateError> {
    let arr = g.get_array(key).ok_or_else(|| {
        crate::ValidateError::TokenKey("gemma-vq metadata", key.into())
    })?;
    g.array_u32s(&arr)
        .map_err(|_| crate::ValidateError::TokenKey("gemma-vq metadata", key.into()))
}

fn bools(g: &GgufFile, key: &'static str) -> std::result::Result<Vec<bool>, crate::ValidateError> {
    let arr = g.get_array(key).ok_or_else(|| {
        crate::ValidateError::TokenKey("gemma-vq metadata", key.into())
    })?;
    g.array_bools(&arr)
        .map_err(|_| crate::ValidateError::TokenKey("gemma-vq metadata", key.into()))
}

/// The family's own validator, run by `crate::tensors::parse_one` on a file
/// that requested it. Everything here is the contract's section 6 list plus the
/// file-wide structural rules the two shipping families share
/// (`glm_vq.rs:437-531`).
pub(super) fn validate_part(
    g: &GgufFile,
    tensors: &mut [TensorInfo],
    dir_end: usize,
    data_pos: u64,
) -> Result<()> {
    if !requested(g) {
        return Ok(());
    }
    // A file that names the family must be the family: the version gate is
    // what stops a future container from being read as this one.
    if g.get_string("general.architecture") != Some(ARCH)
        || g.get_u32(FORMAT_VERSION_KEY) != Some(FORMAT_VERSION)
        || g.alignment != ALIGN
        || g.split_count() > 1
    {
        return Err(fail("unsupported format/version/alignment"));
    }
    let mut keys = HashSet::new();
    for entry in g.kv_entries() {
        if !keys.insert(g.key_bytes(entry)) {
            return Err(fail("duplicate metadata"));
        }
    }
    let mut names = HashSet::new();
    for t in tensors.iter_mut() {
        if !names.insert(t.name.clone())
            || t.ndim == 0
            || t.ndim > MAX_RANK
            || t.dim[..t.ndim as usize].contains(&0)
            || t.rel_offset % ALIGN != 0
        {
            return Err(fail("invalid tensor name/shape/alignment"));
        }
        match t.typ {
            BF16 => {
                if t.bytes != t.elements.checked_mul(2).ok_or(TensorError::Overflow)? {
                    return Err(fail("bf16 byte account"));
                }
            }
            Q8_0 => {
                // The token table: rank 2, every row a whole number of blocks.
                if t.ndim != 2
                    || t.dim[0] % Q8_0_BLOCK != 0
                    || t.bytes
                        != t.elements
                            .checked_div(Q8_0_BLOCK)
                            .and_then(|n| n.checked_mul(Q8_0_BYTE))
                            .ok_or(TensorError::Overflow)?
                {
                    return Err(fail("q8_0 byte account"));
                }
            }
            VQBLOB => payload(g, t)?,
            _ => return Err(fail("unsupported tensor type")),
        }
    }
    // File-wide: the directory is zero-padded up to the data position, every
    // tensor starts at the aligned end of the one before it, and the last one
    // ends exactly at the file's end.
    zeros(g.as_bytes(), dir_end as u64, data_pos)?;
    let mut order: Vec<_> = tensors.iter().collect();
    order.sort_unstable_by_key(|t| t.abs_offset);
    let mut end = data_pos;
    for tensor in order {
        if tensor.abs_offset != crate::tensors::align_up(end, ALIGN) {
            return Err(fail("gap/overlap"));
        }
        zeros(g.as_bytes(), end, tensor.abs_offset)?;
        end = tensor
            .abs_offset
            .checked_add(tensor.bytes)
            .ok_or(TensorError::Overflow)?;
    }
    if end != g.as_bytes().len() as u64 {
        return Err(fail("trailing/missing bytes"));
    }
    Ok(())
}

/// One DQVQ payload, validated against its own header, its declared length and
/// the reference's byte account.
///
/// The index-stream bound needs no separate pass. The account already reserves
/// `ceil(rows*cols/dim * bits / 8)` bytes plus the decoder's guard byte, and at
/// this tier the width is exactly 12 bits for `nc` = 4096, so every encodable
/// index is in `0..nc`: an out-of-range index cannot be expressed, and the
/// reachable tamper — a header whose geometry no longer reproduces the declared
/// length — is what the account refuses.
fn payload(g: &GgufFile, t: &TensorInfo) -> Result<()> {
    if t.ndim != 1 {
        return Err(fail("vq payload rank"));
    }
    let bytes = span(g.as_bytes(), t.abs_offset, t.bytes)?;
    if u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != DQVQ {
        return Err(fail("vq payload magic"));
    }
    let dim = u64::from(u16::from_le_bytes([bytes[4], bytes[5]]));
    let nc = u64::from(u16::from_le_bytes([bytes[6], bytes[7]]));
    let rows = u64::from(u32::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11],
    ]));
    let cols = u64::from(u32::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15],
    ]));
    if dim == 0 || nc == 0 || cols % dim != 0 || rows == 0 || cols == 0 {
        return Err(fail("vq payload geometry"));
    }
    // The single copy of the account the writer, the size allocator and this
    // loader must agree on (`vq_qc_bytes.h:15`, `:20`). `t.bytes` is the length
    // the file declares, so a disagreement is a bad payload, not a bad file.
    if vq_payload_bytes(rows, cols, dim, nc) != t.bytes {
        return Err(fail("vq payload byte account"));
    }
    let codebook = span(bytes, VQ_HEADER, nc * dim * 2)?;
    for word in codebook.chunks_exact(2) {
        if u16::from_le_bytes([word[0], word[1]]) & F16_INF_NAN == F16_INF_NAN {
            return Err(fail("nonfinite codebook"));
        }
    }
    let gains = span(bytes, VQ_HEADER + nc * dim * 2, rows * 2)?;
    for word in gains.chunks_exact(2) {
        let bits = u16::from_le_bytes([word[0], word[1]]);
        // Zero, +0 and -0 alike, is a row the payload would decode to nothing.
        if bits & 0x7fff == 0 {
            return Err(fail("zero row gain"));
        }
        if bits & F16_INF_NAN == F16_INF_NAN {
            return Err(fail("nonfinite row gain"));
        }
    }
    Ok(())
}

/// The index width the reference derives from the codebook size
/// (`vq_qc_bytes.h:15`), and the byte account built on it
/// (`/data/YoungAi/gguf-tools/quantize/vq_qc_bytes.h:20`; the same formula the
/// family layout computes from the tower's architecture).
fn vq_payload_bytes(rows: u64, cols: u64, dim: u64, nc: u64) -> u64 {
    let nidx = rows * cols / dim;
    let mut bits = 1u64;
    while (1u64 << bits) < nc {
        bits += 1;
    }
    let idx_bytes = if bits == 8 {
        nidx
    } else {
        (nidx * bits + 7) / 8 + 1
    };
    16 + nc * dim * 2 + rows * 2 + idx_bytes
}

fn span(data: &[u8], start: u64, len: u64) -> Result<&[u8]> {
    let end = start.checked_add(len).ok_or(TensorError::Overflow)?;
    let start = usize::try_from(start).map_err(|_| TensorError::Overflow)?;
    let end = usize::try_from(end).map_err(|_| TensorError::Overflow)?;
    data.get(start..end).ok_or(TensorError::OutsideFile)
}

fn zeros(data: &[u8], start: u64, end: u64) -> Result<()> {
    if end < start || span(data, start, end - start)?.iter().any(|&x| x != 0) {
        return Err(fail("nonzero padding"));
    }
    Ok(())
}
