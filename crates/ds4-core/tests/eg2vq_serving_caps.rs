//! Phase 6, row P2.3: the gemma_vq family's embedding-shaped serving caps.
//!
//! Two numbers describe what an embedding caller may ask this family for
//! (`crate::gemma_vq`, `ds4_core::embed_caps`):
//!
//!   ITEM_TOKENS_MAX    tokens in one item. The tower's ceiling, and it exists
//!                      in three places: the Rust constant, the C entry point's
//!                      `DS4_EG2VQ_MAX_TOKENS` macro (`ds4_eg2vq.h`) and the
//!                      CUDA attention kernel's `EG2VQ_ATTN_KEYS_MAX`
//!                      (`cuda/eg2vq_attn.cuh`). The adapter static-asserts the
//!                      last two equal at compile time; nothing asserts that
//!                      Rust agrees with either, which is what this test adds:
//!                      a divergence would let the Rust gate admit an item the
//!                      kernel then refuses, or refuse one it could encode.
//!
//!   BATCH_ROWS_MAX     tokens over one call's items, the row budget of the
//!                      tower's single device workspace. A policy of this port,
//!                      so the test only pins it to the family's declared value
//!                      and to the layout's linear cost per row.
//!
//! The C sources are read, not linked: this crate must build without the C tree,
//! and the numbers are text in two headers the tree already ships. A header that
//! moves or changes spelling fails the test loudly rather than silently passing.
use ds4_core::embed_caps;

/// The value of a `#define NAME VALUE[u]` line, or a panic naming the file.
fn define_in(source: &str, path: &str, name: &str) -> u64 {
    let needle = format!("#define {name}");
    let line = source
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{path} has no {needle}"));
    let value = line[line.find(&needle).unwrap() + needle.len()..].trim();
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    assert!(
        !digits.is_empty(),
        "{path}: {name} has no numeric value (line: {line:?})"
    );
    digits.parse().expect("digits")
}

/// The value of an `NAME = VALUE,` enumerator, or a panic naming the file.  The
/// `=` may be padded for column alignment (`NAME      = VALUE,`), which the
/// headers do; what is matched is the name as the line's first token, then the
/// first run of digits after its `=`.
fn enumerator_in(source: &str, path: &str, name: &str) -> u64 {
    let line = source
        .lines()
        .find(|l| {
            l.trim_start()
                .strip_prefix(name)
                .is_some_and(|rest| rest.trim_start().starts_with('='))
        })
        .unwrap_or_else(|| panic!("{path} has no enumerator {name}"));
    let value = line[line.find('=').unwrap() + 1..].trim();
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    assert!(
        !digits.is_empty(),
        "{path}: {name} has no numeric value (line: {line:?})"
    );
    digits.parse().expect("digits")
}

#[test]
fn item_ceiling_is_one_number_in_rust_c_and_cuda() {
    let c_header = include_str!("../../../ds4_eg2vq.h");
    let cuda_kernel = include_str!("../../../cuda/eg2vq_attn.cuh");
    let rust = u64::from(ds4_core::GEMMA_VQ_ITEM_TOKENS_MAX);
    let c = define_in(c_header, "ds4_eg2vq.h", "DS4_EG2VQ_MAX_TOKENS");
    let cuda = enumerator_in(cuda_kernel, "cuda/eg2vq_attn.cuh", "EG2VQ_ATTN_KEYS_MAX");
    assert_eq!(
        (rust, rust),
        (c, cuda),
        "the item ceiling diverged: Rust ITEM_TOKENS_MAX={rust}, \
         C DS4_EG2VQ_MAX_TOKENS={c}, CUDA EG2VQ_ATTN_KEYS_MAX={cuda}"
    );
}

#[test]
fn batch_row_budget_costs_what_the_layout_charges() {
    // The workspace layout (`ds4_eg2vq_gpu.cuh`'s `eg2vq_gpu_layout_of`) is
    // linear in the row budget.  Per row it carries:
    //
    //    27,648 f32 planes   110,592 B   h, x, f, a (512 each), q_swa (4 x 256),
    //                                    q_full (4 x 512), k + v (2 x 2 x 256),
    //                                    attn_swa, attn_full, fg, fu (2,048
    //                                    each), fd (512), pg, pp (512 each),
    //                                    ple (512 x 24) and the image splice's
    //                                    in_soft (512)
    //    ids                       4 B   int32, the row's token id
    //    pos, len                  8 B   the batch's per-row metadata (unit 27)
    //    out, acc              6,144 B   the projection's per-row plane and the
    //                                    pooled per-item result, 768 f32 each
    //                        ---------
    //                         116,748 B
    //
    // plus the (rows + 1) uint32 item spans, rounded up to the layout's 16-byte
    // alignment.  The measured cross-check is the device oracle's own printout:
    // 228.03 MiB at 2,048 rows and 912.13 MiB at the 8,192-row budget
    // (tools/eg2_embed_batch_gate27.sh prints both; before unit 27 the same
    // oracle read 216.01 MiB at 2,048 rows).
    //
    // What this test pinned before unit 27 was 108,548 B (4 + 27,136 x 4) and
    // 848.0 MiB, and that arithmetic had already gone stale one unit earlier:
    // unit 25 added the image splice's 512-float `in_soft` plane, so the layout
    // had been charging 110,592 B a row (2,048 more) while this test, the
    // constant's comment and the 848.0 MiB figure still said 108,548 -- which is
    // what the gate-7 measurement of 212.01 MiB at max_tokens 2,048 reflects.
    // The numbers below are the layout's own, plane by plane, and they are now
    // cross-checked against a measurement rather than against themselves.
    const PLANES_PER_ROW: u64 = 27_648;
    const PLANE_BYTES: u64 = PLANES_PER_ROW * 4;
    assert_eq!(PLANE_BYTES, 110_592, "the tower's per-row planes changed");
    const PER_ROW: u64 = PLANE_BYTES + 4 + 4 + 4 + 768 * 4 + 768 * 4;
    assert_eq!(PER_ROW, 116_748, "the per-row workspace charge changed");
    let rows = u64::from(ds4_core::GEMMA_VQ_BATCH_ROWS_MAX);
    assert_eq!(rows, 8192, "the row budget is the port's declared policy");
    // The C entry point's own row ceiling is the same number: the caps the
    // server advertises must be the bound the engine enforces.
    let c_header = include_str!("../../../ds4_eg2vq.h");
    assert_eq!(
        define_in(c_header, "ds4_eg2vq.h", "DS4_EG2VQ_MAX_ROWS"),
        rows,
        "the C row budget and crate::gemma_vq::BATCH_ROWS_MAX diverged"
    );
    let spans = ((rows + 1) * 4 + 15) & !15;
    let mib = (rows * PER_ROW + spans) as f64 / (1024.0 * 1024.0);
    assert!(
        (mib - 912.13).abs() < 0.05,
        "BATCH_ROWS_MAX {rows} now costs {mib:.2} MiB of workspace, not the 912.13 MiB \
         the constant's comment and the unit-27 evidence state"
    );
    // and the single-item ceiling's own workspace, which the device gate measures
    let spans_2048 = ((2048 + 1) * 4 + 15) & !15;
    let at_2048 = (2048 * PER_ROW + spans_2048) as f64 / (1024.0 * 1024.0);
    assert!(
        (at_2048 - 228.03).abs() < 0.05,
        "2,048 rows now costs {at_2048:.2} MiB, not the 228.03 MiB gate 27 measures"
    );
}

#[test]
fn only_the_embedding_family_states_embedding_caps() {
    let caps = embed_caps(ds4_core::ModelFamily::GemmaVq).expect("gemma_vq is an embedding family");
    assert_eq!(
        caps.item_tokens_max,
        ds4_core::GEMMA_VQ_ITEM_TOKENS_MAX
    );
    assert_eq!(caps.batch_rows_max, ds4_core::GEMMA_VQ_BATCH_ROWS_MAX);
    // A decode family has a context, banks and speculation instead, and must not
    // report embedding caps it does not have.
    for family in [
        ds4_core::ModelFamily::Glm53Vq,
        ds4_core::ModelFamily::Ling31Vq,
        ds4_core::ModelFamily::DeepSeek4,
    ] {
        assert_eq!(embed_caps(family), None, "{family:?} is not an embedding family");
    }
}

/// Phase 8, unit 26: the image half of the caps is the C header's own numbers,
/// read as text (this crate builds without the C tree), so a header that moves or
/// changes spelling fails loudly.  The comparison is the same one the item
/// ceiling gets above, for the same reason: the Rust entry (which sizes the row
/// buffer the C side fills) and the C tower (which fills it) must agree on every
/// number, and neither asserts the other's.
#[test]
fn image_caps_are_the_c_headers_own_numbers() {
    let c_header = include_str!("../../../ds4_eg2vq.h");
    let vision_header = include_str!("../../../ds4_eg2vq_vision.h");
    let caps = embed_caps(ds4_core::ModelFamily::GemmaVq).expect("gemma_vq has embedding caps");
    let image = caps.image.expect("gemma_vq's embedding path takes an image");
    let pairs = [
        (
            vision_header,
            "ds4_eg2vq_vision.h",
            "DS4_EG2VQ_VISION_SOFT_MAX",
            u64::from(image.soft_tokens_max),
            u64::from(ds4_core::GEMMA_VQ_IMAGE_SOFT_TOKENS_MAX),
        ),
        (
            vision_header,
            "ds4_eg2vq_vision.h",
            "DS4_EG2VQ_VISION_MIN_TOKENS",
            u64::from(image.soft_tokens_min),
            u64::from(ds4_core::GEMMA_VQ_IMAGE_SOFT_TOKENS_MIN),
        ),
        (
            vision_header,
            "ds4_eg2vq_vision.h",
            "DS4_EG2VQ_VISION_MAX_PIXELS",
            u64::from(image.pixels_max),
            u64::from(ds4_core::GEMMA_VQ_IMAGE_PIXELS_MAX),
        ),
        (
            vision_header,
            "ds4_eg2vq_vision.h",
            "DS4_EG2VQ_VISION_MIN_PIXELS",
            u64::from(image.pixels_min),
            u64::from(ds4_core::GEMMA_VQ_IMAGE_PIXELS_MIN),
        ),
        // One soft row is as wide as the text tower's hidden state: the
        // projection's output IS the row the splice feeds the tower
        // (ds4_eg2vq.h:54 N_EMBD 512, the same 512 as the mmproj's
        // mm.input_projection.weight row length).
        (
            c_header,
            "ds4_eg2vq.h",
            "DS4_EG2VQ_N_EMBD",
            u64::from(image.embd),
            u64::from(ds4_core::GEMMA_VQ_IMAGE_EMBD),
        ),
    ];
    for (source, path, name, capped, rust) in pairs {
        assert_eq!(capped, rust, "the caps and the Rust constant disagree on {name}");
        assert_eq!(
            enumerator_in(source, path, name),
            capped,
            "crates/ds4-core's {name} ({capped}) is not the C header's own value"
        );
    }
    assert_eq!(
        define_in(c_header, "ds4_eg2vq.h", "DS4_EG2VQ_IMAGE_TOKEN_ID"),
        ds4_core::GEMMA_VQ_IMAGE_TOKEN_ID as u64,
        "the Rust image token id is not the C header's own placeholder id"
    );
    // The one-image rule is a POLICY of this port (the splice fills one
    // contiguous run), so it is pinned to its own declaration rather than to C.
    assert_eq!(image.per_item_max, ds4_core::GEMMA_VQ_IMAGE_PER_ITEM_MAX);
    assert_eq!(image.per_item_max, 1, "the splice fills exactly one span");
    // The image's soft rows are counted by the item ceiling like any other
    // token, so a full-budget image plus its begin/end markers must still fit an
    // item: the two numbers have to leave room for each other.
    assert!(
        image.soft_tokens_max + 3 <= caps.item_tokens_max,
        "a maximum image plus its begin/end markers must fit the item ceiling"
    );
    assert!(image.serial, "one device vision workspace per engine is serial");
}
