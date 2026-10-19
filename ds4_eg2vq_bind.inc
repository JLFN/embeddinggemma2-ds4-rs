/* Rust validates the complete gemma_vq artifact before this pointer adapter
 * (crates/ds4-core/src/gemma_vq.rs: the family gate, every payload's magic,
 * geometry and byte account, the codebook, the row gains and the file-wide
 * layout), and C never parses a DQVQ payload (docs/ds4-gemma-payload-contract.md
 * section 6). What this adapter fills — the weight struct per layer and the
 * record table the device kernels read — arrives with the text tower, phase 4 of
 * embeddinggemma2-handoff.md (ds4_eg2vq_graph.inc), and the kernels in phase 7.
 *
 * Until then the bind stops by name. That is deliberate rather than a stub:
 * weights_bind falls through to the DeepSeek binding when a family has no arm,
 * so without this the artifact would be bound as another model's architecture
 * and load a shape of garbage instead of failing. */
static void eg2vq_bind(ds4_weights *w, const ds4_model *m) {
    (void)w;
    (void)m;
    ds4_die("gemma_vq: the text tower is not ported yet (embeddinggemma2-handoff.md PHASE 4)");
}
