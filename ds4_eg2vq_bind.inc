/* ds4_eg2vq_bind.inc — the gemma_vq weight adapter and its dense decode.
 *
 * Rust validates the complete artifact before this pointer adapter
 * (crates/ds4-core/src/gemma_vq.rs: the family gate, every payload's magic,
 * geometry and byte account, the codebook, the row gains and the file-wide
 * layout).  The names below are the checkpoint's own, in the artifact's order
 * (out/artifact-manifest.tsv), and the geometry each VQ payload must carry is
 * the tower architecture's, so a payload whose header disagrees with the layer
 * it is bound to stops the load by name instead of decoding into the shape of
 * a different layer.
 *
 * The decoded matrices are the CPU reference path's own operand form: the DQVQ
 * value rule is value[row][col] = codebook[idx][col % dim] * gain[row]
 * (ds41_vq_fmt.h:73-105, ds4vq_dequant_f32), which is HF row-major [out][in]
 * and therefore the same memory order as this tree's ds4_ref_row convention.
 * Every matrix is decoded exactly once, into one cache, here at bind time.
 *
 * A second, equally strict leg below binds the phase-1a baseline GGUF, whose
 * tensor names are llama.cpp's and whose weights are plain BF16/F32.  The
 * handoff's C-vs-anchor gate is measured on that file: the anchor was produced
 * by llama.cpp on the baseline, so only the baseline leg shares its weights.
 * The VQ artifact's own gap to the anchor is the quantization, which is phase
 * 5's subject, not a port error. */

#include "ds41_vq_fmt.h"

/* Payload byte account, from the writer's own single source of truth
 * (tools/eg2_vq.c:16-23): 16 B header + nc*dim f16 codebook + rows f16 gains +
 * ceil(blocks*nbit/8) index bytes + one guard byte, with nbit = ceil(log2 nc)
 * and blocks = rows * (cols / dim).  Verified against every entry of
 * out/artifact-manifest.tsv. */
static uint32_t eg2vq_index_bits(void) {
    uint32_t nbit = 0;
    while ((1u << nbit) < (uint32_t)DS4_EG2VQ_NC) { nbit++; }
    return nbit ? nbit : 1u;
}

static uint64_t eg2vq_payload_bytes(uint32_t rows, uint32_t cols) {
    const uint64_t blocks = (uint64_t)rows * (uint64_t)(cols / (uint32_t)DS4_EG2VQ_DIM);
    const uint64_t idx_bytes = (blocks * eg2vq_index_bits() + 7u) / 8u + 1u;
    return (uint64_t)DS4_EG2VQ_HEADER
         + (uint64_t)DS4_EG2VQ_NC * (uint64_t)DS4_EG2VQ_DIM * 2u
         + (uint64_t)rows * 2u + idx_bytes;
}

/* Name the tensor and stop: the by-name refusal phase 3's gate measured. */
static void eg2vq_refuse(const ds4_tensor *t, const char *reason) {
    char msg[256];
    snprintf(msg, sizeof(msg), "gemma_vq: tensor %.*s %s",
             (int)t->name.len, t->name.ptr, reason);
    ds4_die(msg);
}

static void eg2vq_bind_matrix(const ds4_model *m, ds4_eg2vq_matrix *mx,
        const char *name, uint32_t rows, uint32_t cols) {
    ds4_tensor *t = required_tensor(m, name);
    if (t->type != DS4_TENSOR_VQBLOB) {
        eg2vq_refuse(t, "is not a vqblob payload");
    }
    if (t->ndim != 1u) {
        eg2vq_refuse(t, "is not a rank-1 VQ payload");
    }
    /* The payload's own header carries its geometry: [u32 magic][u16 dim]
     * [u16 nc][u32 rows][u32 cols] (ds41_vq_fmt.h:10-12, :73-77).  C reads
     * only these five fields here; the values themselves come from
     * ds4vq_dequant_f32. */
    const uint8_t *p = (const uint8_t *)tensor_data(m, t);
    uint32_t magic, hrows, hcols;
    uint16_t dim, nc;
    memcpy(&magic, p, 4);
    memcpy(&dim, p + 4, 2);
    memcpy(&nc, p + 6, 2);
    memcpy(&hrows, p + 8, 4);
    memcpy(&hcols, p + 12, 4);
    if (magic != (uint32_t)DS4VQ_MAT_MAGIC ||
        dim != (uint16_t)DS4_EG2VQ_DIM || nc != (uint16_t)DS4_EG2VQ_NC) {
        eg2vq_refuse(t, "is not a vq4x4096 payload");
    }
    if (hrows != rows || hcols != cols) {
        eg2vq_refuse(t, "has a payload geometry the tower does not expect");
    }
    if (t->dim[0] != eg2vq_payload_bytes(rows, cols)) {
        eg2vq_refuse(t, "has a payload byte length its geometry does not account for");
    }
    mx->tensor = t;
    mx->dense = NULL;
    mx->rows = rows;
    mx->cols = cols;
}

static void eg2vq_bind_artifact(ds4_weights *w, const ds4_model *m);
static void eg2vq_bind_baseline(ds4_weights *w, const ds4_model *m);

/* ---------------------------------------------------------------------- *
 * The reference-weight leg.  The phase-1a baseline GGUF
 * (out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf) is the same checkpoint
 * under llama.cpp's own tensor names, written so the reference's embedding
 * example and our C reference path load one file (docs/ds4-gemma-port-phases.md
 * X.5).  Its matrices are plain BF16 (the token table too) and its norms F32,
 * so no VQ decode applies: each matrix is widened once into the same dense
 * cache, row by row, exactly as ds4_ref_row reads a tensor.  This leg is what
 * makes the handoff's C-vs-anchor cosine gate evaluable at all — the anchor was
 * produced on this baseline, and the VQ artifact carries its own weight error.
 * ---------------------------------------------------------------------- */
static void eg2vq_plain_row(const ds4_model *m, const ds4_tensor *t, uint64_t row, float *out) {
    const uint64_t n = t->dim[0];
    const uint8_t *p = (const uint8_t *)tensor_data(m, t);
    if (t->type == DS4_TENSOR_F32) {
        memcpy(out, (const float *)p + row * n, (size_t)n * sizeof(float));
        return;
    }
    const uint16_t *h = (const uint16_t *)p + row * n;
    for (uint64_t i = 0; i < n; i++) {
        if (t->type == DS4_TENSOR_F16) {
            out[i] = f16_to_f32(h[i]);
        } else {
            const uint32_t u = (uint32_t)h[i] << 16;
            memcpy(&out[i], &u, sizeof(float));
        }
    }
}

static void eg2vq_bind_plain_matrix(const ds4_model *m, ds4_eg2vq_matrix *mx,
        const char *name, uint32_t rows, uint32_t cols) {
    ds4_tensor *t = required_tensor(m, name);
    if (t->type != DS4_TENSOR_F32 && t->type != DS4_TENSOR_BF16 && t->type != DS4_TENSOR_F16) {
        eg2vq_refuse(t, "is not a plain F16/BF16/F32 weight matrix");
    }
    if (t->ndim != 2u || t->dim[0] != cols || t->dim[1] != rows) {
        eg2vq_refuse(t, "has a shape the tower does not expect");
    }
    mx->tensor = t;
    mx->dense = NULL;
    mx->rows = rows;
    mx->cols = cols;
}

#define EG2VQ_PLAIN(mx_) do { \
        for (uint32_t o = 0; o < (mx_)->rows; o++) { \
            eg2vq_plain_row(m, (mx_)->tensor, o, (mx_)->dense + (uint64_t)o * (mx_)->cols); \
        } \
    } while (0)

static void eg2vq_bind_baseline(ds4_weights *w, const ds4_model *m) {
    ds4_eg2vq_weights *g = xcalloc(1, sizeof(*g));
    w->eg2vq = g;

    g->token_embd = required_tensor(m, "token_embd.weight");
    tensor_expect_layout(g->token_embd, DS4_TENSOR_BF16, 2,
        DS4_EG2VQ_N_EMBD, DS4_EG2VQ_VOCAB_SIZE, 0);
    g->output_norm = required_tensor(m, "output_norm.weight");
    tensor_expect_layout(g->output_norm, DS4_TENSOR_F32, 1, DS4_EG2VQ_N_EMBD, 0, 0);
    g->output = required_tensor(m, "output.weight");
    tensor_expect_layout(g->output, DS4_TENSOR_BF16, 2, DS4_EG2VQ_N_EMBD, DS4_EG2VQ_N_OUT, 0);
    g->per_layer_proj_norm = required_tensor(m, "per_layer_proj_norm.weight");
    tensor_expect_layout(g->per_layer_proj_norm, DS4_TENSOR_F32, 1, DS4_EG2VQ_PER_LAYER, 0, 0);

    ds4_eg2vq_matrix *mats[1 + DS4_EG2VQ_N_LAYER * 9];
    uint32_t n_mats = 0;

    eg2vq_bind_plain_matrix(m, &g->per_layer_model_proj, "per_layer_model_proj.weight",
        DS4_EG2VQ_PER_LAYER * DS4_EG2VQ_N_LAYER, DS4_EG2VQ_N_EMBD);
    mats[n_mats++] = &g->per_layer_model_proj;

    char name[160];
#define EG2VQ_BNORM(field, suffix, d0) do { \
        snprintf(name, sizeof(name), "blk.%u." suffix, il); \
        g->layer[il].field = required_tensor(m, name); \
        tensor_expect_layout(g->layer[il].field, DS4_TENSOR_F32, 1, d0, 0, 0); \
    } while (0)
#define EG2VQ_BMATRIX(field, suffix, rows_, cols_) do { \
        snprintf(name, sizeof(name), "blk.%u." suffix, il); \
        eg2vq_bind_plain_matrix(m, &g->layer[il].field, name, rows_, cols_); \
        mats[n_mats++] = &g->layer[il].field; \
    } while (0)

    for (uint32_t il = 0; il < DS4_EG2VQ_N_LAYER; il++) {
        const uint32_t hd = ds4_eg2vq_head_dim(il);
        const uint32_t nkv = ds4_eg2vq_head_kv(il);
        const uint32_t qw = (uint32_t)DS4_EG2VQ_N_HEAD * hd;
        const uint32_t kw = nkv * hd;
        EG2VQ_BNORM(input_norm,     "attn_norm.weight",          DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(q,            "attn_q.weight",             qw, DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(k,            "attn_k.weight",             kw, DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(v,            "attn_v.weight",             kw, DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(o,            "attn_output.weight",        DS4_EG2VQ_N_EMBD, qw);
        EG2VQ_BNORM(q_norm,         "attn_q_norm.weight",        hd);
        EG2VQ_BNORM(k_norm,         "attn_k_norm.weight",        hd);
        EG2VQ_BNORM(attn_post_norm, "post_attention_norm.weight", DS4_EG2VQ_N_EMBD);
        EG2VQ_BNORM(ffn_norm,       "ffn_norm.weight",           DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(ffn_gate,     "ffn_gate.weight",           DS4_EG2VQ_N_FF, DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(ffn_up,       "ffn_up.weight",             DS4_EG2VQ_N_FF, DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(ffn_down,     "ffn_down.weight",           DS4_EG2VQ_N_EMBD, DS4_EG2VQ_N_FF);
        EG2VQ_BNORM(ffn_post_norm,  "post_ffw_norm.weight",      DS4_EG2VQ_N_EMBD);
        EG2VQ_BMATRIX(ple_gate,     "inp_gate.weight",           DS4_EG2VQ_PER_LAYER, DS4_EG2VQ_PER_LAYER);
        EG2VQ_BMATRIX(ple_proj,     "proj.weight",               DS4_EG2VQ_PER_LAYER, DS4_EG2VQ_PER_LAYER);
        EG2VQ_BNORM(ple_post_norm,  "post_norm.weight",          DS4_EG2VQ_N_EMBD);
        EG2VQ_BNORM(layer_scalar,   "layer_output_scale.weight", 1);
    }
#undef EG2VQ_BNORM
#undef EG2VQ_BMATRIX

    uint64_t total = 0;
    for (uint32_t i = 0; i < n_mats; i++) {
        total += (uint64_t)mats[i]->rows * (uint64_t)mats[i]->cols;
    }
    g->dense = xmalloc((size_t)total * sizeof(float));
    g->dense_bytes = total * sizeof(float);

    float *cursor = g->dense;
    for (uint32_t i = 0; i < n_mats; i++) {
        mats[i]->dense = cursor;
        cursor += (uint64_t)mats[i]->rows * (uint64_t)mats[i]->cols;
    }
    for (uint32_t i = 0; i < n_mats; i++) {
        EG2VQ_PLAIN(mats[i]);
    }
}

#undef EG2VQ_PLAIN

static void eg2vq_bind(ds4_weights *w, const ds4_model *m) {
    /* One shape, two weight sources: the VQ artifact carries the checkpoint's
     * Hugging Face names, the phase-1a baseline carries llama.cpp's.  The file
     * decides which leg runs; a file with neither name set stops by name. */
    if (model_find_tensor(m, "language_model.embed_tokens.weight")) {
        eg2vq_bind_artifact(w, m);
    } else {
        eg2vq_bind_baseline(w, m);
    }
}

static void eg2vq_bind_artifact(ds4_weights *w, const ds4_model *m) {
    ds4_eg2vq_weights *g = xcalloc(1, sizeof(*g));
    w->eg2vq = g;

    g->token_embd = required_tensor(m, "language_model.embed_tokens.weight");
    tensor_expect_layout(g->token_embd, DS4_TENSOR_Q8_0, 2,
        DS4_EG2VQ_N_EMBD, DS4_EG2VQ_VOCAB_SIZE, 0);
    g->output_norm = required_tensor(m, "language_model.norm.weight");
    tensor_expect_layout(g->output_norm, DS4_TENSOR_BF16, 1, DS4_EG2VQ_N_EMBD, 0, 0);
    g->output = required_tensor(m, "language_model.embedding_projection.weight");
    tensor_expect_layout(g->output, DS4_TENSOR_BF16, 2, DS4_EG2VQ_N_EMBD, DS4_EG2VQ_N_OUT, 0);
    g->per_layer_proj_norm = required_tensor(m, "language_model.ple.per_layer_projection_norm.weight");
    tensor_expect_layout(g->per_layer_proj_norm, DS4_TENSOR_BF16, 1, DS4_EG2VQ_PER_LAYER, 0, 0);

    ds4_eg2vq_matrix *mats[1 + DS4_EG2VQ_N_LAYER * 9];
    uint32_t n_mats = 0;

    eg2vq_bind_matrix(m, &g->per_layer_model_proj,
        "language_model.ple.per_layer_model_projection.weight",
        DS4_EG2VQ_PER_LAYER * DS4_EG2VQ_N_LAYER, DS4_EG2VQ_N_EMBD);
    mats[n_mats++] = &g->per_layer_model_proj;

    char name[160];
#define EG2VQ_NORM(field, suffix, d0) do { \
        snprintf(name, sizeof(name), "language_model.layers.%u." suffix, il); \
        g->layer[il].field = required_tensor(m, name); \
        tensor_expect_layout(g->layer[il].field, DS4_TENSOR_BF16, 1, d0, 0, 0); \
    } while (0)
#define EG2VQ_MATRIX(field, suffix, rows_, cols_) do { \
        snprintf(name, sizeof(name), "language_model.layers.%u." suffix, il); \
        eg2vq_bind_matrix(m, &g->layer[il].field, name, rows_, cols_); \
        mats[n_mats++] = &g->layer[il].field; \
    } while (0)

    for (uint32_t il = 0; il < DS4_EG2VQ_N_LAYER; il++) {
        const uint32_t hd = ds4_eg2vq_head_dim(il);
        const uint32_t nkv = ds4_eg2vq_head_kv(il);
        const uint32_t qw = (uint32_t)DS4_EG2VQ_N_HEAD * hd;
        const uint32_t kw = nkv * hd;
        EG2VQ_NORM(input_norm,     "input_layernorm.weight",           DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(q,            "self_attn.q_proj.weight",          qw, DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(k,            "self_attn.k_proj.weight",          kw, DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(v,            "self_attn.v_proj.weight",          kw, DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(o,            "self_attn.o_proj.weight",          DS4_EG2VQ_N_EMBD, qw);
        EG2VQ_NORM(q_norm,         "self_attn.q_norm.weight",          hd);
        EG2VQ_NORM(k_norm,         "self_attn.k_norm.weight",          hd);
        EG2VQ_NORM(attn_post_norm, "post_attention_layernorm.weight",  DS4_EG2VQ_N_EMBD);
        EG2VQ_NORM(ffn_norm,       "pre_feedforward_layernorm.weight", DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(ffn_gate,     "mlp.gate_proj.weight",             DS4_EG2VQ_N_FF, DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(ffn_up,       "mlp.up_proj.weight",               DS4_EG2VQ_N_FF, DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(ffn_down,     "mlp.down_proj.weight",             DS4_EG2VQ_N_EMBD, DS4_EG2VQ_N_FF);
        EG2VQ_NORM(ffn_post_norm,  "post_feedforward_layernorm.weight", DS4_EG2VQ_N_EMBD);
        EG2VQ_MATRIX(ple_gate,     "ple_block.per_layer_input_gate.weight", DS4_EG2VQ_PER_LAYER, DS4_EG2VQ_PER_LAYER);
        EG2VQ_MATRIX(ple_proj,     "ple_block.per_layer_projection.weight", DS4_EG2VQ_PER_LAYER, DS4_EG2VQ_PER_LAYER);
        EG2VQ_NORM(ple_post_norm,  "ple_block.post_per_layer_input_norm.weight", DS4_EG2VQ_N_EMBD);
        EG2VQ_NORM(layer_scalar,   "layer_scalar",                     1);
    }
#undef EG2VQ_NORM
#undef EG2VQ_MATRIX

    /* One cache for every decoded matrix.  The decoded geometry comes from the
     * payload headers Rust already validated (checked above against the
     * architecture), so the size is exact. */
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_mats; i++) {
        total += (uint64_t)mats[i]->rows * (uint64_t)mats[i]->cols;
    }
    g->dense = xmalloc((size_t)total * sizeof(float));
    g->dense_bytes = total * sizeof(float);

    float *cursor = g->dense;
    for (uint32_t i = 0; i < n_mats; i++) {
        ds4_eg2vq_matrix *mx = mats[i];
        if (ds4vq_dequant_f32((const uint8_t *)tensor_data(m, mx->tensor),
                              cursor, (int)mx->rows, (int)mx->cols) != 0) {
            eg2vq_refuse(mx->tensor, "does not decode as a DQVQ payload");
        }
        mx->dense = cursor;
        cursor += (uint64_t)mx->rows * (uint64_t)mx->cols;
    }
}
