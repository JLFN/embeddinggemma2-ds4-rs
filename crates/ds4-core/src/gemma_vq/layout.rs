//! The `gemma_vq` family layout: every tensor the artifact carries, with the
//! rank and dimensions the FILE declares, in GGUF (inner-dimension-first)
//! order.
//!
//! This table is the family's shape authority (`docs/ds4-gemma-payload-contract.md`
//! section 5): `validate_layouts` checks the artifact against it and the bind
//! resolves through it. It is also where a VQ payload's byte length becomes a
//! contract value — the spec's single dimension is
//! `vq_payload_bytes(rows, cols, 4, 4096)` computed here from the tower's own
//! architecture, while `crate::gemma_vq::validate_part` derives the same number
//! from the payload's own 16-byte header. A mistyped geometry on either side is
//! a load-time failure rather than a mis-decoded matrix.
//!
//! Sources, all read in this session or committed in this repo:
//! `out/artifact-manifest.tsv` (the writer's own record, one row per tensor,
//! rank census 1354 rank 1 / 7 rank 2 / 13 rank 3 / 2 rank 4) and
//! `docs/encode-plan.tsv` (the encoder's per-tensor plan); the checkpoint's
//! shapes are `.models/EmbeddingGemma-2-914f7f8/config.json`
//! (`text_config`: hidden 512, 24 layers, 4 heads, head_dim 256, 2 kv heads,
//! intermediate 2048, vocab 262144, embedding_dim 768, `hidden_size_per_layer_input`
//! 512, `layer_types` with full attention at 5/11/17/23 and `sliding_window` 512).
//!
//! Only `embed_tokens` is quantized outside the VQ payloads, and it is `q8_0`:
//! the artifact's recipe string says `token=q8_0` and the byte account is the
//! `ds4_quantfmt.c:17` one (32 values in 34 bytes), which the loader re-checks.

use crate::{LayoutSpec, TypeClass};

use super::FULL_ATTN_LAYERS;

const T_BF16: u32 = 30;
const T_Q8_0: u32 = 8;
const T_VQBLOB: u32 = 42;

/// The artifact's VQ tier. `docs/ladder-embeddinggemma2.txt` measured this tier
/// against the alternatives; the recipe string records it as `towers=vq4x4096`.
const VQ_DIM: u64 = 4;
const VQ_NC: u64 = 4096;

const TEXT_LAYERS: u32 = 24;
const TEXT_EMBED: u64 = 512;
const TEXT_FFN: u64 = 2048;
const TEXT_OUT: u64 = 768;
const TEXT_HEADS: u64 = 4;
const TEXT_HEAD_DIM: u64 = 256;
const TEXT_KV_HEADS: u64 = 2;
const PLE_DIM: u64 = 512;
const VOCAB: u64 = 262_144;
/// The checkpoint's full-attention head width (its `per_layer_config`);
/// `super::FULL_ATTN_LAYERS` names the layers that take it.
const FULL_HEAD_DIM: u64 = 512;

const VISION_LAYERS: u32 = 16;
const VISION_WIDTH: u64 = 768;
const VISION_FFN: u64 = 3072;
const VISION_HEAD_DIM: u64 = 64;
/// The vision position table's row count, the checkpoint's own.
const VISION_POS: u64 = 10_240;

const AUDIO_LAYERS: u32 = 12;
const AUDIO_WIDTH: u64 = 1024;
const AUDIO_FFN: u64 = 4096;
/// `lconv1d.linear_start` widens 1024 to 2048; `linear_end` narrows it back.
const AUDIO_LINEAR_START: u64 = 2048;
/// `self_attn.per_dim_scale`, the checkpoint's own width.
const AUDIO_PER_DIM: u64 = 128;
/// The audio output projection's width, the checkpoint's own.
const AUDIO_OUT: u64 = 1536;
/// The subsample convolutions' channel counts.
const AUDIO_CONV0: u64 = 128;
const AUDIO_CONV1: u64 = 32;
/// The depthwise convolution's kernel extent (`lconv1d`).
const AUDIO_CONV_K: u64 = 5;

fn tensor(out: &mut Vec<LayoutSpec>, name: String, typ: u32, dim: &[u64]) {
    let mut d = [0u64; 8];
    d[..dim.len()].copy_from_slice(dim);
    out.push(LayoutSpec {
        name,
        class: TypeClass::Exact(typ),
        ndim: dim.len() as u32,
        dim: d,
    });
}

/// The index width the reference derives from the codebook size
/// (`vq_qc_bytes.h:15`: `bits = 1; while ((1 << bits) < nc) bits++`).
fn index_bits(nc: u64) -> u64 {
    let mut bits = 1u64;
    while (1u64 << bits) < nc {
        bits += 1;
    }
    bits
}

/// The reference's payload byte account, the single copy the writer and the
/// size allocator share (`/data/YoungAi/gguf-tools/quantize/vq_qc_bytes.h:15`
/// and `:20`):
///
///     16 + nc*dim*2 + rows*2 + idx_bytes
///     idx_bytes = nidx when the index width is exactly 8 bits,
///                 (nidx*bits + 7)/8 + 1 otherwise
///
/// The `+ 1` is the guard byte the decoder's 3-byte window read needs at the
/// tail (`ds41_vq_fmt.h:97`).
fn vq_payload_bytes(rows: u64, cols: u64, dim: u64, nc: u64) -> u64 {
    let nidx = rows * cols / dim;
    let bits = index_bits(nc);
    let idx_bytes = if bits == 8 {
        nidx
    } else {
        (nidx * bits + 7) / 8 + 1
    };
    16 + nc * dim * 2 + rows * 2 + idx_bytes
}

/// A VQ matrix: the file declares one dimension, the payload's own byte length
/// (`docs/ds4-gemma-payload-contract.md` section 4 — type 42 is one byte per
/// element, so a `[cols, rows]` declaration would make a generic reader compute
/// `cols*rows` bytes).
fn vq(out: &mut Vec<LayoutSpec>, name: String, rows: u64, cols: u64) {
    tensor(
        out,
        name,
        T_VQBLOB,
        &[vq_payload_bytes(rows, cols, VQ_DIM, VQ_NC)],
    );
}

/// The min/max each audio block records beside its linear (`input_max`,
/// `input_min`, `output_max`, `output_min`): rank-0 scalars in the checkpoint,
/// which the file carries as one-element vectors.
fn audio_range(out: &mut Vec<LayoutSpec>, prefix: &str) {
    for side in ["input", "output"] {
        for bound in ["max", "min"] {
            tensor(out, format!("{prefix}.{side}_{bound}"), T_BF16, &[1]);
        }
    }
}

fn text_layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("language_model.layers.{il}.");
    let full = FULL_ATTN_LAYERS.contains(&il);
    let head_dim = if full { FULL_HEAD_DIM } else { TEXT_HEAD_DIM };
    let kv_out = if full {
        head_dim
    } else {
        TEXT_KV_HEADS * TEXT_HEAD_DIM
    };
    tensor(
        out,
        format!("{p}input_layernorm.weight"),
        T_BF16,
        &[TEXT_EMBED],
    );
    tensor(
        out,
        format!("{p}post_attention_layernorm.weight"),
        T_BF16,
        &[TEXT_EMBED],
    );
    tensor(
        out,
        format!("{p}pre_feedforward_layernorm.weight"),
        T_BF16,
        &[TEXT_EMBED],
    );
    tensor(
        out,
        format!("{p}post_feedforward_layernorm.weight"),
        T_BF16,
        &[TEXT_EMBED],
    );
    tensor(out, format!("{p}layer_scalar"), T_BF16, &[1]);
    vq(
        out,
        format!("{p}self_attn.q_proj.weight"),
        TEXT_HEADS * head_dim,
        TEXT_EMBED,
    );
    vq(
        out,
        format!("{p}self_attn.k_proj.weight"),
        kv_out,
        TEXT_EMBED,
    );
    vq(
        out,
        format!("{p}self_attn.v_proj.weight"),
        kv_out,
        TEXT_EMBED,
    );
    vq(
        out,
        format!("{p}self_attn.o_proj.weight"),
        TEXT_EMBED,
        TEXT_HEADS * head_dim,
    );
    tensor(
        out,
        format!("{p}self_attn.q_norm.weight"),
        T_BF16,
        &[head_dim],
    );
    tensor(
        out,
        format!("{p}self_attn.k_norm.weight"),
        T_BF16,
        &[head_dim],
    );
    vq(
        out,
        format!("{p}mlp.gate_proj.weight"),
        TEXT_FFN,
        TEXT_EMBED,
    );
    vq(out, format!("{p}mlp.up_proj.weight"), TEXT_FFN, TEXT_EMBED);
    vq(
        out,
        format!("{p}mlp.down_proj.weight"),
        TEXT_EMBED,
        TEXT_FFN,
    );
    vq(
        out,
        format!("{p}ple_block.per_layer_input_gate.weight"),
        PLE_DIM,
        TEXT_EMBED,
    );
    vq(
        out,
        format!("{p}ple_block.per_layer_projection.weight"),
        PLE_DIM,
        PLE_DIM,
    );
    tensor(
        out,
        format!("{p}ple_block.post_per_layer_input_norm.weight"),
        T_BF16,
        &[TEXT_EMBED],
    );
}

fn vision_layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("vision_tower.encoder.layers.{il}.");
    for norm in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ] {
        tensor(out, format!("{p}{norm}.weight"), T_BF16, &[VISION_WIDTH]);
    }
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        vq(
            out,
            format!("{p}self_attn.{proj}.linear.weight"),
            VISION_WIDTH,
            VISION_WIDTH,
        );
    }
    for norm in ["q_norm", "k_norm"] {
        tensor(
            out,
            format!("{p}self_attn.{norm}.weight"),
            T_BF16,
            &[VISION_HEAD_DIM],
        );
    }
    vq(
        out,
        format!("{p}mlp.gate_proj.linear.weight"),
        VISION_FFN,
        VISION_WIDTH,
    );
    vq(
        out,
        format!("{p}mlp.up_proj.linear.weight"),
        VISION_FFN,
        VISION_WIDTH,
    );
    vq(
        out,
        format!("{p}mlp.down_proj.linear.weight"),
        VISION_WIDTH,
        VISION_FFN,
    );
}

fn audio_layer(out: &mut Vec<LayoutSpec>, il: u32) {
    let p = format!("audio_tower.layers.{il}.");
    for block in ["feed_forward1", "feed_forward2"] {
        vq(
            out,
            format!("{p}{block}.ffw_layer_1.linear.weight"),
            AUDIO_FFN,
            AUDIO_WIDTH,
        );
        vq(
            out,
            format!("{p}{block}.ffw_layer_2.linear.weight"),
            AUDIO_WIDTH,
            AUDIO_FFN,
        );
        for ffw in ["ffw_layer_1", "ffw_layer_2"] {
            audio_range(out, &format!("{p}{block}.{ffw}"));
        }
        tensor(
            out,
            format!("{p}{block}.pre_layer_norm.weight"),
            T_BF16,
            &[AUDIO_WIDTH],
        );
        tensor(
            out,
            format!("{p}{block}.post_layer_norm.weight"),
            T_BF16,
            &[AUDIO_WIDTH],
        );
    }
    vq(
        out,
        format!("{p}lconv1d.linear_start.linear.weight"),
        AUDIO_LINEAR_START,
        AUDIO_WIDTH,
    );
    vq(
        out,
        format!("{p}lconv1d.linear_end.linear.weight"),
        AUDIO_WIDTH,
        AUDIO_WIDTH,
    );
    for linear in ["linear_start", "linear_end"] {
        audio_range(out, &format!("{p}lconv1d.{linear}"));
    }
    tensor(
        out,
        format!("{p}lconv1d.pre_layer_norm.weight"),
        T_BF16,
        &[AUDIO_WIDTH],
    );
    tensor(
        out,
        format!("{p}lconv1d.conv_norm.weight"),
        T_BF16,
        &[AUDIO_WIDTH],
    );
    tensor(
        out,
        format!("{p}lconv1d.depthwise_conv1d.weight"),
        T_BF16,
        &[AUDIO_CONV_K, 1, AUDIO_WIDTH],
    );
    for proj in ["q_proj", "k_proj", "v_proj"] {
        vq(
            out,
            format!("{p}self_attn.{proj}.linear.weight"),
            AUDIO_WIDTH,
            AUDIO_WIDTH,
        );
        audio_range(out, &format!("{p}self_attn.{proj}"));
    }
    vq(
        out,
        format!("{p}self_attn.post.linear.weight"),
        AUDIO_WIDTH,
        AUDIO_WIDTH,
    );
    audio_range(out, &format!("{p}self_attn.post"));
    vq(
        out,
        format!("{p}self_attn.relative_k_proj.weight"),
        AUDIO_WIDTH,
        AUDIO_WIDTH,
    );
    tensor(
        out,
        format!("{p}self_attn.per_dim_scale"),
        T_BF16,
        &[AUDIO_PER_DIM],
    );
    for norm in ["norm_pre_attn", "norm_post_attn", "norm_out"] {
        tensor(
            out,
            format!("{p}{norm}.weight"),
            T_BF16,
            &[AUDIO_WIDTH],
        );
    }
}

/// Every tensor of the artifact, in the order the writer registered them
/// (ascending name order, which is also ascending file offset).
pub(super) fn layouts() -> Vec<LayoutSpec> {
    let mut out = Vec::with_capacity(1376);
    // Text tower: 5 top-level tensors plus 24 layers of 17.
    tensor(
        &mut out,
        "language_model.embed_tokens.weight".into(),
        T_Q8_0,
        &[TEXT_EMBED, VOCAB],
    );
    tensor(
        &mut out,
        "language_model.embedding_projection.weight".into(),
        T_BF16,
        &[TEXT_EMBED, TEXT_OUT],
    );
    tensor(
        &mut out,
        "language_model.norm.weight".into(),
        T_BF16,
        &[TEXT_EMBED],
    );
    vq(
        &mut out,
        "language_model.ple.per_layer_model_projection.weight".into(),
        u64::from(TEXT_LAYERS) * PLE_DIM,
        TEXT_EMBED,
    );
    tensor(
        &mut out,
        "language_model.ple.per_layer_projection_norm.weight".into(),
        T_BF16,
        &[TEXT_EMBED],
    );
    for il in 0..TEXT_LAYERS {
        text_layer(&mut out, il);
    }
    // Vision tower: 2 patch-embedder tensors plus 16 layers of 13, then the
    // projection that ties it to the text width.
    tensor(
        &mut out,
        "vision_tower.patch_embedder.input_proj.weight".into(),
        T_BF16,
        &[VISION_WIDTH, VISION_WIDTH],
    );
    tensor(
        &mut out,
        "vision_tower.patch_embedder.position_embedding_table".into(),
        T_BF16,
        &[VISION_WIDTH, VISION_POS, 2],
    );
    for il in 0..VISION_LAYERS {
        vision_layer(&mut out, il);
    }
    tensor(
        &mut out,
        "embed_vision.embedding_projection.weight".into(),
        T_BF16,
        &[VISION_WIDTH, TEXT_EMBED],
    );
    // Audio tower: 7 top-level tensors plus 12 layers of 62, then the
    // projection that ties it to the text width.
    tensor(
        &mut out,
        "audio_tower.subsample_conv_projection.input_proj_linear.weight".into(),
        T_BF16,
        &[AUDIO_WIDTH, AUDIO_WIDTH],
    );
    tensor(
        &mut out,
        "audio_tower.subsample_conv_projection.layer0.conv.weight".into(),
        T_BF16,
        &[3, 3, 1, AUDIO_CONV0],
    );
    tensor(
        &mut out,
        "audio_tower.subsample_conv_projection.layer0.norm.weight".into(),
        T_BF16,
        &[AUDIO_CONV0],
    );
    tensor(
        &mut out,
        "audio_tower.subsample_conv_projection.layer1.conv.weight".into(),
        T_BF16,
        &[3, 3, AUDIO_CONV0, AUDIO_CONV1],
    );
    tensor(
        &mut out,
        "audio_tower.subsample_conv_projection.layer1.norm.weight".into(),
        T_BF16,
        &[AUDIO_CONV1],
    );
    tensor(
        &mut out,
        "audio_tower.output_proj.weight".into(),
        T_BF16,
        &[AUDIO_WIDTH, AUDIO_OUT],
    );
    tensor(
        &mut out,
        "audio_tower.output_proj.bias".into(),
        T_BF16,
        &[AUDIO_OUT],
    );
    for il in 0..AUDIO_LAYERS {
        audio_layer(&mut out, il);
    }
    tensor(
        &mut out,
        "embed_audio.embedding_projection.weight".into(),
        T_BF16,
        &[AUDIO_OUT, TEXT_EMBED],
    );
    out
}
