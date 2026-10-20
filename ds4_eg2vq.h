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

/* The CPU reference's BATCH entry (unit 27): the same forward, `n_items` items
 * packed into `n_ids` contiguous rows with `item_off` the n_items + 1 spans
 * (off[0] = 0, off[n_items] = n_ids), and `out768` taking
 * n_items * DS4_EG2VQ_N_OUT floats, item i's vector at i * DS4_EG2VQ_N_OUT.
 *
 * It LOOPS the one-item reference above, item by item, deliberately: the CPU leg
 * is this port's oracle, not its speed path (the device leg is), and looping the
 * already-gated reference is what makes the two legs comparable at all -- the
 * batch's whole claim is that a batched item's vector EQUALS its single-item
 * vector.  The one-item entry is therefore bit-identical to the batch of one by
 * construction here, and the device leg is measured against it. */
void ds4_eg2vq_ref_embed_batch(const struct ds4_model *m, const struct ds4_weights *w,
                               const int32_t *ids, const uint32_t *item_off,
                               uint32_t n_items, uint32_t n_ids, float *out768);

/* The image span's placeholder id, from the checkpoint's own config.json
 * (image_token_id 258880).  Its vocabulary string is "<|image|>", while the two
 * MARKERS the reference wraps the span with are "<|image>" = 255999 and
 * "<image|>" = 258882 (config.json's boi/eoi_token_id); the port's own tokenizer
 * emits those two from the same strings, which is how the 92-position spliced
 * sequence is built.  The container carries no metadata key for the placeholder
 * (unlike the other families' vision GGUFs, which is why their adapters read one),
 * so the id is named here once, and this is how it is checked against the
 * artifact:
 *
 *   cargo run -p ds4-core --release --example gemma_vq_vocab -- <artifact.gguf> 255999,258880,258882
 *   ->  255999  <|image>   258880  <|image|>   258882  <image|>
 */
#define DS4_EG2VQ_IMAGE_TOKEN_ID 258880

/* THE IMAGE SPLICE: the same forward, with the image's soft tokens filling the
 * positions the ids mark.  `ids` carries the span as DS4_EG2VQ_IMAGE_TOKEN_ID
 * placeholders (one per soft row), and `soft` is n_soft rows of
 * DS4_EG2VQ_N_EMBD floats in span order.  The soft rows enter the tower's input
 * UNSCALED, while a token row is scaled by sqrt(n_embd): llama.cpp's
 * build_inp_embd takes a scale for the token lookup and leaves a raw embedding
 * entry alone — "important: do not normalize weights for raw embeddings input
 * (i.e. encoded image embeddings)", src/models/gemma-embedding2.cpp:85-86.  A
 * placeholder count that disagrees with n_soft, or a placeholder with no soft row
 * left, is a hard stop: a silently misaligned span would be a wrong answer. */
void ds4_eg2vq_ref_embed_image(const struct ds4_model *m, const struct ds4_weights *w,
                               const int32_t *ids, uint32_t n_ids,
                               const float *soft, uint32_t n_soft, float *out768);

/* The longest item the device path accepts: the attention kernel's key plane
 * (cuda/eg2vq_attn.cuh:49, EG2VQ_ATTN_KEYS_MAX; the equality is asserted in
 * ds4_eg2vq_gpu.cuh).  A longer item is REFUSED by the entry point rather than
 * truncated -- a plane that silently covered half an item would be a wrong
 * answer, not a slow one. */
#define DS4_EG2VQ_MAX_TOKENS 2048u

/* The row budget of the tower's ONE device workspace: how many rows all of one
 * call's items may add up to (unit 27, the batch).  This is a POLICY of the port,
 * not a file fact -- nothing in the artifact states a batch width -- and it is the
 * number the serving caps advertise as `embed.batch_rows_max`
 * (crates/ds4-core/src/gemma_vq.rs BATCH_ROWS_MAX, which this macro mirrors and
 * crates/ds4-core/tests/eg2vq_serving_caps.rs holds equal by reading this header
 * as text).  The workspace is linear in rows at 114,700 bytes each
 * (ds4_eg2vq_gpu.cuh's layout: the tower's per-row planes, the batch metadata and
 * the projection's own per-row planes), so the cap is 896.2 MiB of device
 * workspace.  A call past it is refused by name, never truncated. */
#define DS4_EG2VQ_MAX_ROWS 8192u

/* ------------------------------------------------------------------------- *
 * The device path (embeddinggemma2-handoff.md PHASE 7, phase-doc rows
 * P3.1-P3.3).
 *
 * The same tower, run on the GPU by the kernels of ds4-rs/cuda/eg2vq_{vq,attn,
 * ops}.cuh composed in the CPU reference forward's own op order.  The payloads
 * are consumed in the kernel (never expanded to the dense f32 decode the CPU
 * path re-reads once per token), so the device plan holds the tower's 217
 * payloads -- 55.9 MiB, out/artifact-manifest.tsv -- next to the q8_0 token
 * table, the norms and the RoPE tables.  The adapter is
 * ds4-rs/ds4_eg2vq_gpu.cuh (included once by ds4_cuda.cu, the
 * ds4_glmvq_gpu.cuh pattern); the host side of the contract is the table shape
 * below, which ds4_eg2vq_bind.inc fills from the bound weights.
 * ------------------------------------------------------------------------- */

/* The storage form of one plain (non-payload) vector, as the file carries it.
 * The numbers are ds4.c's own tensor type ids, kept here for the same reason
 * DS4_EG2VQ_VQBLOB is: the family header is the device path's whole contract. */
typedef enum {
    DS4_EG2VQ_ROW_F32  = 0,
    DS4_EG2VQ_ROW_F16  = 1,
    DS4_EG2VQ_ROW_Q8_0 = 8,
    DS4_EG2VQ_ROW_BF16 = 30,
} ds4_eg2vq_row;

/* One plain vector for the device plan.  `host` points into the artifact's
 * mapping (or into a caller-owned buffer) and must stay valid until
 * ds4_gpu_eg2vq_bind returns; the source is TYPE-tagged and the adapter widens
 * a bf16/f16 source where the slot's device form is f32. */
typedef struct {
    const void *host;
    uint64_t    bytes;
    uint32_t    row_type;   /* ds4_eg2vq_row */
    uint32_t    rows, cols;
} ds4_eg2vq_aux;

/* The aux slots, in the order ds4_eg2vq_device_tables fills them.  Each slot's
 * device form is fixed: OUTPUT_NORM, PLE_NORM and the eight per-layer slots
 * become f32 (the ops kernels take f32 weights; the artifact stores them bf16
 * and the baseline f32); OUTPUT keeps its bf16 words (the 768 projection kernel
 * reads bf16 rows exactly as the CPU binder does); TOKEN_EMBD keeps the q8_0
 * block stream the gather kernel reads. */
enum {
    DS4_EG2VQ_AUX_OUTPUT_NORM = 0,   /* f32 [512] */
    DS4_EG2VQ_AUX_PLE_NORM    = 1,   /* f32 [512] */
    DS4_EG2VQ_AUX_OUTPUT      = 2,   /* bf16 [768][512]  (embedding_projection) */
    DS4_EG2VQ_AUX_TOKEN_EMBD  = 3,   /* q8_0 [512][262144] */
    DS4_EG2VQ_AUX_LAYER_BASE  = 4,   /* + il * 8 */
    DS4_EG2VQ_AUX_LAYER_STRIDE = 8,
    DS4_EG2VQ_AUX_L_INPUT_NORM = 0,  /* f32 [512] */
    DS4_EG2VQ_AUX_L_ATTN_POST  = 1,  /* f32 [512] */
    DS4_EG2VQ_AUX_L_FFN_NORM   = 2,  /* f32 [512] */
    DS4_EG2VQ_AUX_L_FFN_POST   = 3,  /* f32 [512] */
    DS4_EG2VQ_AUX_L_PLE_POST   = 4,  /* f32 [512] */
    DS4_EG2VQ_AUX_L_Q_NORM     = 5,  /* f32 [head_dim] */
    DS4_EG2VQ_AUX_L_K_NORM     = 6,  /* f32 [head_dim] */
    DS4_EG2VQ_AUX_L_SCALAR     = 7,  /* f32 [1] */
    DS4_EG2VQ_AUX_N = 4 + DS4_EG2VQ_N_LAYER * 8,
};

/* The tower's KV width is 512 on every layer (2 x 256 sliding, 1 x 512 full),
 * which the device sequence relies on: k and v are one packed buffer of that
 * width for both layer kinds. */
typedef char ds4_eg2vq_kv_width_is_shared
    [(DS4_EG2VQ_HEAD_KV_SWA * DS4_EG2VQ_HEAD_DIM_SWA == DS4_EG2VQ_HEAD_KV_FULL * DS4_EG2VQ_HEAD_DIM_FULL) ? 1 : -1];

/* The payload records' order, which is the CPU binder's own matrix list
 * (ds4_eg2vq_bind.inc: the per-layer projection once, then nine matrices per
 * layer in this order).  The device sequence indexes the records by it. */
enum {
    DS4_EG2VQ_REC_PLE_PROJ = 0,
    DS4_EG2VQ_REC_LAYER_BASE = 1,       /* + il * 9 */
    DS4_EG2VQ_REC_LAYER_STRIDE = 9,
    DS4_EG2VQ_REC_L_Q = 0,
    DS4_EG2VQ_REC_L_K = 1,
    DS4_EG2VQ_REC_L_V = 2,
    DS4_EG2VQ_REC_L_O = 3,
    DS4_EG2VQ_REC_L_FFN_GATE = 4,
    DS4_EG2VQ_REC_L_FFN_UP = 5,
    DS4_EG2VQ_REC_L_FFN_DOWN = 6,
    DS4_EG2VQ_REC_L_PLE_GATE = 7,
    DS4_EG2VQ_REC_L_PLE_PROJ = 8,
    DS4_EG2VQ_REC_N = 1 + DS4_EG2VQ_N_LAYER * 9,
};

typedef struct ds4_eg2vq_gpu_model ds4_eg2vq_gpu_model;
typedef struct ds4_eg2vq_gpu_ctx ds4_eg2vq_gpu_ctx;

/* Bind a plan: `records` are the tower's VQ payloads (offset/bytes/type/rows/
 * cols as ds4_eg2vq_record, mapped into `map` of `size` bytes), `aux` is the
 * slot table above.  Every payload's own header geometry is re-checked in the
 * kernels and every payload span is checked against `size` here, so a plan that
 * disagrees with the artifact stops at bind by name.  NULL on any refusal. */
ds4_eg2vq_gpu_model *ds4_gpu_eg2vq_bind(const void *map, uint64_t size,
    const ds4_eg2vq_record *records, uint32_t count,
    const ds4_eg2vq_aux *aux, uint32_t aux_count);
void ds4_gpu_eg2vq_unbind(ds4_eg2vq_gpu_model *model);

/* Device bytes of the workspace for a call whose items add up to at most
 * `max_rows` rows (the handoff's PHASE 7 shape: one arena, reused by every call;
 * the layout is linear in rows, ds4_eg2vq_gpu.cuh). */
uint64_t ds4_gpu_eg2vq_workspace(uint32_t max_rows);
/* `max_rows` <= DS4_EG2VQ_MAX_ROWS (the row budget; the ITEM ceiling is
 * DS4_EG2VQ_MAX_TOKENS and is checked per item, at every call); 0 for a plan or a
 * size outside that. */
ds4_eg2vq_gpu_ctx *ds4_gpu_eg2vq_create(const ds4_eg2vq_gpu_model *model,
    uint64_t scratch, uint32_t max_rows);
void ds4_gpu_eg2vq_free(ds4_eg2vq_gpu_ctx *ctx);

/* One item on the device: `n_ids` token ids (BOS..EOS, as the host's tokenizer
 * emits them) -> the mean-pooled, L2-normalized 768-dim vector, the same
 * function ds4_eg2vq_ref_embed computes on the CPU.  Returns 0 (and writes
 * nothing) for an empty item, an item longer than DS4_EG2VQ_MAX_TOKENS, or a
 * token id the embedding table cannot address. */
int ds4_gpu_eg2vq_embed(ds4_eg2vq_gpu_ctx *ctx, const int32_t *ids, uint32_t n_ids, float *out768);

/* SEVERAL items in ONE call (unit 27): `n_items` items packed into one
 * contiguous row block of `n_ids` rows, `item_off` their n_items + 1 spans, and
 * `out768` taking n_items * DS4_EG2VQ_N_OUT floats (item i at i *
 * DS4_EG2VQ_N_OUT).  Positions and item bounds are DERIVED from the spans here,
 * so there is one statement about the block, not two.
 *
 * Returns 0 (and writes nothing) for the one-item refusals plus: n_items == 0,
 * n_ids past the context's row budget, a span table that does not cover n_ids
 * from 0, an empty item, or an item past DS4_EG2VQ_MAX_TOKENS.  Every item's own
 * arithmetic is independent of the call's composition -- that is the batch's
 * contract and what tools/eg2_embed_batch_gate27.sh measures. */
int ds4_gpu_eg2vq_embed_batch(ds4_eg2vq_gpu_ctx *ctx, const int32_t *ids,
                              const uint32_t *item_off, uint32_t n_items,
                              uint32_t n_ids, float *out768);

/* The image splice on the device: the same call, with the vision tower's
 * `n_soft` soft rows (DS4_EG2VQ_N_EMBD floats each) filling the positions the
 * ids mark as DS4_EG2VQ_IMAGE_TOKEN_ID.  Returns 0 for the same refusals as the
 * text call plus a placeholder count that disagrees with n_soft. */
int ds4_gpu_eg2vq_embed_image(ds4_eg2vq_gpu_ctx *ctx, const int32_t *ids, uint32_t n_ids,
                              const float *soft, uint32_t n_soft, float *out768);

#ifdef __cplusplus
}
#endif
#endif
