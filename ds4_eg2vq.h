/* ds4_eg2vq.h — the C side of the gemma_vq family (EmbeddingGemma-2's VQ
 * artifact). The small header on purpose: like the two shipping VQ families,
 * Rust validates the complete artifact before C sees a pointer — the family
 * gate, every payload's magic, geometry, codebook, row gains and byte account,
 * and the file-wide layout (crates/ds4-core/src/gemma_vq.rs). C binds the
 * tensors by name (ds4_eg2vq_bind.inc), re-reads each payload's five header
 * fields and decodes it with the vendored dim-generic decoder
 * (ds41_vq_fmt.h:73-105); it never re-derives the family's validation. */
#ifndef DS4_EG2VQ_H
#define DS4_EG2VQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One VQ matrix, as the host validated it. `type` is ds4's own 42 (`vqblob`),
 * `offset`/`bytes` are the tensor's absolute file span, and `rows`/`cols` are
 * the payload's declared geometry (`rows` = output features, `cols` = input).
 * The CPU reference forward (ds4_eg2vq_graph.inc, phase 4 of
 * embeddinggemma2-handoff.md) reads the tensors by name rather than this
 * record; the device kernels are phase 7. */
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

/* ------------------------------------------------------------------------- *
 * The text tower (embeddinggemma2-handoff.md PHASE 4).
 *
 * Shape facts, all from the artifact's own `gemma_vq.*` metadata
 * (out/artifact-kv.txt): 24 layers, hidden 512, 4 query heads, head_dim 256
 * on the sliding layers and 512 on the four full ones, 2 KV heads while
 * sliding and 1 on the full ones, ffn 2048, rms eps 1e-6, vocab 262144,
 * per-layer input 512, output 768.  `attention.sliding_window` is 1024 and
 * the mask type is SYMMETRIC, so a token attends within |p1 - p0| <= 512
 * (src/llama-hparams.h:524-533); the doubling from the checkpoint's
 * one-sided 512 is the converter's (conversion/gemma.py:824-825).
 * ------------------------------------------------------------------------- */
enum {
    DS4_EG2VQ_N_LAYER       = 24,
    DS4_EG2VQ_N_EMBD        = 512,
    DS4_EG2VQ_N_HEAD        = 4,
    DS4_EG2VQ_HEAD_DIM_SWA  = 256,
    DS4_EG2VQ_HEAD_DIM_FULL = 512,
    DS4_EG2VQ_HEAD_KV_SWA   = 2,
    DS4_EG2VQ_HEAD_KV_FULL  = 1,
    DS4_EG2VQ_N_FF          = 2048,
    DS4_EG2VQ_PER_LAYER     = 512,
    DS4_EG2VQ_N_OUT         = 768,
    DS4_EG2VQ_VOCAB_SIZE    = 262144,
    DS4_EG2VQ_SWA           = 1024,    /* file sliding_window; symmetric half is 512 */
    DS4_EG2VQ_SWA_PERIOD    = 6,
    DS4_EG2VQ_ROPE_BASE     = 1000000, /* rope.freq_base: the full layers */
    DS4_EG2VQ_ROPE_BASE_SWA = 10000,   /* rope.freq_base_swa: the sliding ones */
};

/* attention.layer_norm_rms_epsilon = 1e-6 (out/artifact-kv.txt); every RMS
 * norm in the tower uses it inside the square root. */
#define DS4_EG2VQ_RMS_EPS 1e-6f

/* llama_hparams::set_swa_pattern(6, dense_first=false) => is_swa(il) is
 * il % 6 < 5 (src/llama-hparams.cpp:8-20), so the full-attention layers are
 * 5, 11, 17 and 23 (src/llama-model.cpp:2748-2750 picks the rope base by it). */
static inline int ds4_eg2vq_layer_is_swa(uint32_t il) {
    return (int)(il % (uint32_t)DS4_EG2VQ_SWA_PERIOD) < (DS4_EG2VQ_SWA_PERIOD - 1);
}

static inline uint32_t ds4_eg2vq_head_dim(uint32_t il) {
    return ds4_eg2vq_layer_is_swa(il) ? (uint32_t)DS4_EG2VQ_HEAD_DIM_SWA
                                      : (uint32_t)DS4_EG2VQ_HEAD_DIM_FULL;
}

static inline uint32_t ds4_eg2vq_head_kv(uint32_t il) {
    return ds4_eg2vq_layer_is_swa(il) ? (uint32_t)DS4_EG2VQ_HEAD_KV_SWA
                                      : (uint32_t)DS4_EG2VQ_HEAD_KV_FULL;
}

struct ds4_tensor;
struct ds4_model;
struct ds4_weights;

/* One VQ matrix: the file record plus, on the CPU reference path, its decoded
 * dense values.  `dense` is row-major [rows][cols] (HF [out][in], the same
 * memory order as ggml's ne0=in/ne1=out), so a matvec is
 * out[o] = sum_i dense[o*cols + i] * x[i]. */
typedef struct {
    struct ds4_tensor *tensor;
    float             *dense;
    uint32_t           rows, cols;
} ds4_eg2vq_matrix;

typedef struct {
    struct ds4_tensor *input_norm;      /* input_layernorm.weight */
    struct ds4_tensor *q_norm;          /* self_attn.q_norm.weight */
    struct ds4_tensor *k_norm;          /* self_attn.k_norm.weight */
    struct ds4_tensor *attn_post_norm;  /* post_attention_layernorm.weight */
    struct ds4_tensor *ffn_norm;        /* pre_feedforward_layernorm.weight */
    struct ds4_tensor *ffn_post_norm;   /* post_feedforward_layernorm.weight */
    struct ds4_tensor *ple_post_norm;   /* ple_block.post_per_layer_input_norm.weight */
    struct ds4_tensor *layer_scalar;    /* one learned scalar */
    ds4_eg2vq_matrix   q, k, v, o;
    ds4_eg2vq_matrix   ffn_gate, ffn_up, ffn_down;
    ds4_eg2vq_matrix   ple_gate, ple_proj;
} ds4_eg2vq_layer;

typedef struct {
    struct ds4_tensor *token_embd;          /* q8_0 [512, 262144] */
    struct ds4_tensor *output_norm;         /* bf16 512 */
    struct ds4_tensor *output;              /* embedding_projection, bf16 [512, 768] */
    struct ds4_tensor *per_layer_proj_norm; /* bf16 512 */
    ds4_eg2vq_matrix   per_layer_model_proj;/* [12288, 512] */
    ds4_eg2vq_layer    layer[DS4_EG2VQ_N_LAYER];
    float             *dense;               /* the one decode cache, all matrices */
    uint64_t           dense_bytes;
} ds4_eg2vq_weights;

/* The CPU reference forward: one item's token ids (BOS..EOS, as the host's
 * tokenizer emits them) -> the mean-pooled, L2-normalized 768-dim vector. */
void ds4_eg2vq_ref_embed(const struct ds4_model *m, const struct ds4_weights *w,
                         const int32_t *ids, uint32_t n_ids, float *out768);

#ifdef __cplusplus
}
#endif
#endif
