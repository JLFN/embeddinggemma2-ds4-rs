/* ds4_eg2vq.h — the C side of the gemma_vq family (EmbeddingGemma-2's VQ
 * artifact). The small header on purpose: like the two shipping VQ families,
 * Rust validates the complete artifact before C sees a pointer, and C never
 * parses a DQVQ payload (docs/ds4-gemma-payload-contract.md section 6,
 * crates/ds4-core/src/gemma_vq.rs). What crosses the boundary is a flat record
 * per payload tensor, carrying the geometry Rust read out of the payload's own
 * header and checked against the family layout. */
#ifndef DS4_EG2VQ_H
#define DS4_EG2VQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One VQ matrix, as the host validated it. `type` is ds4's own 42 (`vqblob`),
 * `offset`/`bytes` are the tensor's absolute file span, and `rows`/`cols` are
 * the payload's declared geometry (`rows` = output features, `cols` = input).
 * The graph that consumes these records, and the weight struct it fills, are
 * phase 4 of embeddinggemma2-handoff.md (ds4_eg2vq_graph.inc); the device
 * kernels are phase 7. Nothing in this unit reads a record yet. */
typedef struct {
    uint64_t offset, bytes;
    uint32_t type, rows, cols;
} ds4_eg2vq_record;

/* The tier the artifact's recipe pins (`towers=vq4x4096`): a codebook vector
 * of four f16 values over 4096 codewords, i.e. a 12-bit index. The payload
 * header Rust validated is [u32 magic][u16 dim][u16 nc][u32 rows][u32 cols].
 * These are the numbers crates/ds4-core/src/gemma_vq.rs validates against. */
enum {
    DS4_EG2VQ_VQBLOB = 42,
    DS4_EG2VQ_DIM = 4,
    DS4_EG2VQ_NC = 4096,
    DS4_EG2VQ_HEADER = 16,
};

#ifdef __cplusplus
}
#endif
#endif
