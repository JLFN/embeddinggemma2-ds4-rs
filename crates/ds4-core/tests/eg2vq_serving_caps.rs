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

/// The value of an `NAME = VALUE,` enumerator, or a panic naming the file.
fn enumerator_in(source: &str, path: &str, name: &str) -> u64 {
    let needle = format!("{name} =");
    let line = source
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{path} has no enumerator {name}"));
    let value = line[line.find(&needle).unwrap() + needle.len()..].trim();
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
    // The workspace layout (`ds4_eg2vq_gpu.cuh:72-99`) is linear in rows: 27,136
    // f32 planes per row plus that row's 4-byte id. The measured cross-check is
    // docs/eg2vq-device-gate7.txt:407-411, where max_tokens 2048 reads 212.01 MiB
    // (2048 x 108,548 B = 212.0086 MiB). The constant is a policy, so what is
    // asserted here is the arithmetic a reader relies on, not the policy.
    const PER_ROW: u64 = 4 + 27_136 * 4;
    assert_eq!(PER_ROW, 108_548, "the per-row workspace charge changed");
    let rows = u64::from(ds4_core::GEMMA_VQ_BATCH_ROWS_MAX);
    let mib = (rows * PER_ROW) as f64 / (1024.0 * 1024.0);
    assert!(
        (mib - 848.0).abs() < 1.0,
        "BATCH_ROWS_MAX {rows} now costs {mib:.1} MiB of workspace, not the ~848 MiB \
         the constant's comment and the phase evidence state"
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
