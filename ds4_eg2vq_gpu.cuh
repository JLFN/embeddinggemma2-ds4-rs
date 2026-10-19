/* ds4_eg2vq_gpu.cuh -- the gemma_vq (EmbeddingGemma-2) text tower's device
 * adapter; included once by ds4_cuda.cu beside ds4_glmvq_gpu.cuh.
 *
 * Authority, read first-hand (global rule 0.2):
 *   - the tower: ds4-rs/ds4_eg2vq_graph.inc, this port's own CPU reference
 *     forward, the gated oracle of phase 4 (768 mean cosine 0.999996809 against
 *     the frozen anchor vectors, docs/ds4-gemma-port-phases.md row X.6).  The
 *     device sequence below is that function's op order, op for op, with the same
 *     buffers and the same constants (ds4_eg2vq.h pins them).
 *   - the kernels: cuda/eg2vq_vq.cuh (the dim-4 payload GEMM), cuda/eg2vq_attn.cuh
 *     (the symmetric-mask attention), cuda/eg2vq_ops.cuh (embed, RMS, RoPE, the
 *     768 projection, pool/L2, scale, add, gelu-mul).
 *   - the adapter shape: ds4_glmvq_gpu.cuh (the shipping family device adapter) --
 *     a plan bound once per process, a per-session workspace, device tensors
 *     allocated through ds4_gpu_tensor_alloc and filled with ds4_gpu_tensor_write.
 *
 * What this adapter deliberately does NOT do: it never decodes a payload to
 * dense values.  The CPU path decodes all 217 tower payloads into one f32 cache
 * (ds4_eg2vq_bind.inc, ds4vq_dequant_f32) and re-reads that cache once per token
 * -- the memory bound the device path exists to remove -- so here the kernels
 * read the payload's own bytes and the plan holds 55.9 MiB
 * (out/artifact-manifest.tsv), not the ~536 MB dense decode.
 *
 * Per-item host traffic is the ids up and the 768-vector down; every weight is
 * device-resident for the life of the plan.
 */
#include "ds4_eg2vq.h"
#include "cuda/eg2vq_vq.cuh"
#include "cuda/eg2vq_attn.cuh"
#include "cuda/eg2vq_ops.cuh"

#include <cstdio>
#include <cstring>
#include <vector>

/* 16-byte alignment for every aux slot and every workspace buffer: the widest
 * access in the kernels is float4 (the GEMM's x loads) and uint4 (the codebook).
 */
enum { EG2VQ_GPU_ALIGN = 16 };

/* The public entry point's item limit and the attention kernel's key plane are
 * one number (ds4_eg2vq.h, DS4_EG2VQ_MAX_TOKENS): if they diverged, the entry
 * point would admit an item the kernel then refuses, which is exactly the
 * silent-truncation class of bug the launcher's refusal exists to prevent. */
static_assert((uint32_t)DS4_EG2VQ_MAX_TOKENS == (uint32_t)EG2VQ_ATTN_KEYS_MAX,
              "DS4_EG2VQ_MAX_TOKENS must equal EG2VQ_ATTN_KEYS_MAX");

static inline uint64_t eg2vq_gpu_round16(uint64_t bytes) {
    return (bytes + (uint64_t)(EG2VQ_GPU_ALIGN - 1)) & ~(uint64_t)(EG2VQ_GPU_ALIGN - 1);
}

static inline uint64_t eg2vq_gpu_elems(const ds4_eg2vq_aux *a) {
    return (uint64_t)a->rows * (uint64_t)a->cols;
}

/* The per-item workspace layout, computed in one place so ds4_gpu_eg2vq_workspace
 * and ds4_gpu_eg2vq_create can never disagree.  Offsets are bytes. */
typedef struct {
    uint64_t ids, out, h, x, f, a;
    uint64_t q_swa, q_full, kv, attn_swa, attn_full;
    uint64_t fg, fu, fd, pg, pp, ple;
    uint64_t total;
} eg2vq_gpu_layout;

/* k and v are one packed buffer of the tower's KV width, which is 512 on every
 * layer (2 x 256 sliding, 1 x 512 full; ds4_eg2vq.h's static assertion). */
static inline uint64_t eg2vq_gpu_kv_bytes(uint32_t max_tokens) {
    return (uint64_t)max_tokens * (uint64_t)DS4_EG2VQ_HEAD_KV_SWA *
           (uint64_t)DS4_EG2VQ_HEAD_DIM_SWA * sizeof(float);
}

static eg2vq_gpu_layout eg2vq_gpu_layout_of(uint32_t max_tokens) {
    const uint64_t t = max_tokens;
    const uint64_t E = (uint64_t)DS4_EG2VQ_N_EMBD;
    const uint64_t PL = (uint64_t)DS4_EG2VQ_PER_LAYER;
    const uint64_t NF = (uint64_t)DS4_EG2VQ_N_FF;
    const uint64_t q_swa = t * (uint64_t)DS4_EG2VQ_N_HEAD * (uint64_t)DS4_EG2VQ_HEAD_DIM_SWA;
    const uint64_t q_full = t * (uint64_t)DS4_EG2VQ_N_HEAD * (uint64_t)DS4_EG2VQ_HEAD_DIM_FULL;
    const uint64_t kv_terms = eg2vq_gpu_kv_bytes(max_tokens);
    eg2vq_gpu_layout L = {};
    uint64_t c = 0;
    L.ids = c;        c += eg2vq_gpu_round16(t * sizeof(int32_t));
    L.out = c;        c += eg2vq_gpu_round16((uint64_t)DS4_EG2VQ_N_OUT * sizeof(float));
    L.h = c;          c += eg2vq_gpu_round16(t * E * sizeof(float));
    L.x = c;          c += eg2vq_gpu_round16(t * E * sizeof(float));
    L.f = c;          c += eg2vq_gpu_round16(t * E * sizeof(float));
    L.a = c;          c += eg2vq_gpu_round16(t * E * sizeof(float));
    L.q_swa = c;      c += eg2vq_gpu_round16(q_swa * sizeof(float));
    L.q_full = c;     c += eg2vq_gpu_round16(q_full * sizeof(float));
    L.kv = c;         c += 2u * eg2vq_gpu_round16(kv_terms);                /* k then v */
    L.attn_swa = c;   c += eg2vq_gpu_round16(q_swa * sizeof(float));
    L.attn_full = c;  c += eg2vq_gpu_round16(q_full * sizeof(float));
    L.fg = c;         c += eg2vq_gpu_round16(t * NF * sizeof(float));
    L.fu = c;         c += eg2vq_gpu_round16(t * NF * sizeof(float));
    L.fd = c;         c += eg2vq_gpu_round16(t * E * sizeof(float));
    L.pg = c;         c += eg2vq_gpu_round16(t * PL * sizeof(float));
    L.pp = c;         c += eg2vq_gpu_round16(t * PL * sizeof(float));
    L.ple = c;        c += eg2vq_gpu_round16(t * PL * (uint64_t)DS4_EG2VQ_N_LAYER * sizeof(float));
    L.total = c;
    return L;
}

/* The plan: the tower's payloads in one device arena, the plain vectors in
 * another, and the per-record/per-slot offsets the item sequence indexes by. */
struct ds4_eg2vq_gpu_model {
    std::vector<ds4_eg2vq_record> records;
    std::vector<uint64_t> payload_off;      /* device offset of each record */
    std::vector<uint64_t> aux_off;          /* device offset of each aux slot */
    ds4_gpu_tensor *payload = nullptr;
    ds4_gpu_tensor *aux = nullptr;
    uint64_t payload_bytes = 0, aux_bytes = 0;
    uint32_t aux_count = 0;
    uint64_t n_vocab = 0;
};

struct ds4_eg2vq_gpu_ctx {
    const ds4_eg2vq_gpu_model *model = nullptr;
    ds4_gpu_tensor *scratch = nullptr;      /* ids, the 768 accumulator, the tower's rows */
    ds4_gpu_tensor *rope_tab = nullptr;     /* cos|sin for both rope bases */
    eg2vq_gpu_layout L = {};
    uint32_t max_tokens = 0;
    float layer_scalar[DS4_EG2VQ_N_LAYER] = {};
    uint64_t calls = 0, tokens = 0;
};

static inline const uint8_t *eg2vq_gpu_pay(const ds4_eg2vq_gpu_model *g, uint32_t rec) {
    return (const uint8_t *)g->payload->ptr + g->payload_off[rec];
}

static inline char *eg2vq_gpu_buf(const ds4_eg2vq_gpu_ctx *c, uint64_t off) {
    return (char *)c->scratch->ptr + off;
}

static inline float *eg2vq_gpu_frow(const ds4_eg2vq_gpu_ctx *c, uint64_t off) {
    return (float *)eg2vq_gpu_buf(c, off);
}

static inline const float *eg2vq_gpu_w(const ds4_eg2vq_gpu_model *g, uint32_t slot) {
    return (const float *)((const char *)g->aux->ptr + g->aux_off[slot]);
}

static inline uint32_t eg2vq_gpu_layer_slot(uint32_t il, uint32_t sub) {
    return (uint32_t)DS4_EG2VQ_AUX_LAYER_BASE + il * (uint32_t)DS4_EG2VQ_AUX_LAYER_STRIDE + sub;
}

/* Every refusal names what disagreed, the way the CPU binder does: a plan that
 * does not match the artifact stops at bind rather than computing with the
 * wrong bytes. */
static ds4_eg2vq_gpu_model *eg2vq_gpu_fail(ds4_eg2vq_gpu_model *plan, const char *what) {
    fprintf(stderr, "ds4: gemma_vq device plan: %s\n", what);
    ds4_gpu_eg2vq_unbind(plan);
    return nullptr;
}

extern "C" void ds4_gpu_eg2vq_unbind(ds4_eg2vq_gpu_model *model) {
    if (!model) { return; }
    ds4_gpu_tensor_free(model->aux);
    ds4_gpu_tensor_free(model->payload);
    delete model;
}

/* The geometry a slot must carry, or 0 for "outside the table".  The column
 * count is the in-features / vector width: 512 for every 512-wide vector, the
 * layer's head_dim for q_norm and k_norm, 1 for the scalar, and the 768
 * projection's 512 inputs (its 768 output rows are checked with the slots). */
static uint32_t eg2vq_gpu_slot_cols(uint32_t slot) {
    if (slot == (uint32_t)DS4_EG2VQ_AUX_OUTPUT || slot == (uint32_t)DS4_EG2VQ_AUX_TOKEN_EMBD) {
        return DS4_EG2VQ_N_EMBD;
    }
    if (slot < (uint32_t)DS4_EG2VQ_AUX_LAYER_BASE) { return DS4_EG2VQ_N_EMBD; }
    const uint32_t il = (slot - (uint32_t)DS4_EG2VQ_AUX_LAYER_BASE) / (uint32_t)DS4_EG2VQ_AUX_LAYER_STRIDE;
    const uint32_t sub = (slot - (uint32_t)DS4_EG2VQ_AUX_LAYER_BASE) % (uint32_t)DS4_EG2VQ_AUX_LAYER_STRIDE;
    if (il >= (uint32_t)DS4_EG2VQ_N_LAYER) { return 0; }
    switch (sub) {
        case DS4_EG2VQ_AUX_L_Q_NORM:
        case DS4_EG2VQ_AUX_L_K_NORM:   return ds4_eg2vq_head_dim(il);
        case DS4_EG2VQ_AUX_L_SCALAR:   return 1u;
        default:                       return DS4_EG2VQ_N_EMBD;
    }
}

extern "C" ds4_eg2vq_gpu_model *ds4_gpu_eg2vq_bind(const void *map, uint64_t size,
        const ds4_eg2vq_record *records, uint32_t count,
        const ds4_eg2vq_aux *aux, uint32_t aux_count) {
    if (!map || !records || !count || !aux || !aux_count) { return nullptr; }
    ds4_eg2vq_gpu_model *plan = new (std::nothrow) ds4_eg2vq_gpu_model;
    if (!plan) { return nullptr; }
    plan->records.assign(records, records + count);

    /* ---- the payload arena: one span per record, each checked against the
     * mapping and each re-checked for this tier's own header fields (the same
     * five the CPU binder re-reads).  The geometry the record declares is the
     * geometry the kernels are launched with, and the kernel re-checks it again
     * at every launch, so a payload bound to the wrong layer cannot decode. */
    plan->payload_off.assign(count, 0);
    uint64_t payload_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        const ds4_eg2vq_record &r = records[i];
        if (r.type != (uint32_t)DS4_EG2VQ_VQBLOB) { return eg2vq_gpu_fail(plan, "a record is not a vqblob payload"); }
        if (!r.rows || !r.cols || (r.cols % 256u) || (r.rows % 64u)) {
            return eg2vq_gpu_fail(plan, "a payload geometry the device tiles do not cover");
        }
        if (r.offset > size || r.bytes > size - r.offset) {
            return eg2vq_gpu_fail(plan, "a payload span outside the mapped artifact");
        }
        const uint8_t *p = (const uint8_t *)map + r.offset;
        uint32_t magic, hrows, hcols;
        uint16_t dim, nc;
        memcpy(&magic, p, 4);
        memcpy(&dim, p + 4, 2);
        memcpy(&nc, p + 6, 2);
        memcpy(&hrows, p + 8, 4);
        memcpy(&hcols, p + 12, 4);
        if (magic != (uint32_t)DS4VQ_MAT_MAGIC || dim != (uint16_t)DS4_EG2VQ_DIM ||
            nc != (uint16_t)DS4_EG2VQ_NC || hrows != r.rows || hcols != r.cols) {
            return eg2vq_gpu_fail(plan, "a payload header that is not this tier's geometry");
        }
        plan->payload_off[i] = eg2vq_gpu_round16(payload_bytes);
        payload_bytes = plan->payload_off[i] + r.bytes;
    }
    plan->payload_bytes = eg2vq_gpu_round16(payload_bytes);
    plan->payload = ds4_gpu_tensor_alloc(plan->payload_bytes);
    if (!plan->payload) { return eg2vq_gpu_fail(plan, "no device memory for the payload arena"); }
    for (uint32_t i = 0; i < count; i++) {
        if (!ds4_gpu_tensor_write(plan->payload, plan->payload_off[i],
                                  (const uint8_t *)map + records[i].offset, records[i].bytes)) {
            return eg2vq_gpu_fail(plan, "a payload did not reach the device");
        }
    }

    /* ---- the aux arena: the slot table ds4_eg2vq.h pins.  A slot whose device
     * form is f32 is widened on the host from the source's own width (bf16 in
     * the artifact, f32 in the baseline), because the ops kernels take f32
     * weights; the reading is the file's IEEE one, not the vendored decoder's
     * halved subnormals (the note in cuda/eg2vq_ops.cuh, measured in
     * docs/eg2vq-device-gate7.txt section 2a). */
    plan->aux_off.assign(aux_count, 0);
    std::vector<float> widen;
    uint64_t aux_bytes = 0;
    for (uint32_t i = 0; i < aux_count; i++) {
        const ds4_eg2vq_aux &a = aux[i];
        if (!a.host || !a.rows || !a.cols || !a.bytes) { return eg2vq_gpu_fail(plan, "an aux slot is empty"); }
        const uint32_t want_cols = eg2vq_gpu_slot_cols(i);
        if (!want_cols) { return eg2vq_gpu_fail(plan, "an aux slot is outside the table"); }
        if (a.cols != want_cols) { return eg2vq_gpu_fail(plan, "an aux slot has the wrong width"); }
        const uint64_t n = eg2vq_gpu_elems(&a);
        uint64_t bytes = 0;
        if (i == (uint32_t)DS4_EG2VQ_AUX_TOKEN_EMBD) {
            if (a.row_type != (uint32_t)DS4_EG2VQ_ROW_Q8_0 ||
                a.bytes != (uint64_t)a.rows * (a.cols / 32u) * 34u) {
                return eg2vq_gpu_fail(plan, "the token table is not a q8_0 table of this geometry");
            }
            plan->n_vocab = a.rows;
            bytes = a.bytes;
        } else if (i == (uint32_t)DS4_EG2VQ_AUX_OUTPUT) {
            if (a.rows != (uint32_t)DS4_EG2VQ_N_OUT || a.row_type != (uint32_t)DS4_EG2VQ_ROW_BF16 ||
                a.bytes != n * 2u) {
                return eg2vq_gpu_fail(plan, "the 768 projection is not a bf16 [768][512] matrix");
            }
            bytes = n * 2u;
        } else {
            if (a.row_type != (uint32_t)DS4_EG2VQ_ROW_F32 && a.row_type != (uint32_t)DS4_EG2VQ_ROW_BF16 &&
                a.row_type != (uint32_t)DS4_EG2VQ_ROW_F16) {
                return eg2vq_gpu_fail(plan, "a weight vector is not f32, f16 or bf16");
            }
            if (a.bytes != n * (a.row_type == (uint32_t)DS4_EG2VQ_ROW_F32 ? 4u : 2u)) {
                return eg2vq_gpu_fail(plan, "a weight vector's byte count disagrees with its geometry");
            }
            bytes = n * 4u;
        }
        plan->aux_off[i] = eg2vq_gpu_round16(aux_bytes);
        aux_bytes = plan->aux_off[i] + bytes;
    }
    plan->aux_bytes = eg2vq_gpu_round16(aux_bytes);
    plan->aux = ds4_gpu_tensor_alloc(plan->aux_bytes);
    if (!plan->aux) { return eg2vq_gpu_fail(plan, "no device memory for the weight-vector arena"); }
    for (uint32_t i = 0; i < aux_count; i++) {
        const ds4_eg2vq_aux &a = aux[i];
        const uint64_t n = eg2vq_gpu_elems(&a);
        if (i == (uint32_t)DS4_EG2VQ_AUX_TOKEN_EMBD || i == (uint32_t)DS4_EG2VQ_AUX_OUTPUT ||
            a.row_type == (uint32_t)DS4_EG2VQ_ROW_F32) {
            if (!ds4_gpu_tensor_write(plan->aux, plan->aux_off[i], a.host, a.bytes)) {
                return eg2vq_gpu_fail(plan, "an aux slot did not reach the device");
            }
            continue;
        }
        widen.resize((size_t)n);
        for (uint64_t k = 0; k < n; k++) {
            uint16_t h;
            memcpy(&h, (const uint8_t *)a.host + 2u * k, 2);
            if (a.row_type == (uint32_t)DS4_EG2VQ_ROW_BF16) {
                const uint32_t u = (uint32_t)h << 16;      /* bf16 -> f32 is the top half */
                memcpy(&widen[(size_t)k], &u, 4);
            } else {
                widen[(size_t)k] = eg2vq_ops_half_to_f32(h);
            }
        }
        if (!ds4_gpu_tensor_write(plan->aux, plan->aux_off[i], widen.data(), n * 4u)) {
            return eg2vq_gpu_fail(plan, "a widened weight vector did not reach the device");
        }
    }
    plan->aux_count = aux_count;
    return plan;
}

extern "C" uint64_t ds4_gpu_eg2vq_workspace(uint32_t max_tokens) {
    if (!max_tokens) { return 0; }
    return eg2vq_gpu_layout_of(max_tokens).total;
}

extern "C" void ds4_gpu_eg2vq_free(ds4_eg2vq_gpu_ctx *ctx) {
    if (!ctx) { return; }
    if (ctx->calls) {
        fprintf(stderr, "ds4: gemma_vq device path calls=%llu tokens=%llu "
                        "payload=%.2f MiB weights=%.2f MiB workspace=%.2f MiB\n",
                (unsigned long long)ctx->calls, (unsigned long long)ctx->tokens,
                (double)ctx->model->payload_bytes / 1048576.0,
                (double)ctx->model->aux_bytes / 1048576.0,
                (double)ctx->L.total / 1048576.0);
    }
    ds4_gpu_tensor_free(ctx->rope_tab);
    ds4_gpu_tensor_free(ctx->scratch);
    delete ctx;
}

extern "C" ds4_eg2vq_gpu_ctx *ds4_gpu_eg2vq_create(const ds4_eg2vq_gpu_model *model,
        uint64_t scratch, uint32_t max_tokens) {
    if (!model || !model->payload || !model->aux || !max_tokens) { return nullptr; }
    if (max_tokens > (uint32_t)EG2VQ_ATTN_KEYS_MAX) {
        fprintf(stderr, "ds4: gemma_vq device path takes at most %d tokens an item\n", (int)EG2VQ_ATTN_KEYS_MAX);
        return nullptr;
    }
    const eg2vq_gpu_layout L = eg2vq_gpu_layout_of(max_tokens);
    if (scratch < L.total) {
        fprintf(stderr, "ds4: gemma_vq device path needs %llu bytes of workspace, given %llu\n",
                (unsigned long long)L.total, (unsigned long long)scratch);
        return nullptr;
    }
    ds4_eg2vq_gpu_ctx *ctx = new (std::nothrow) ds4_eg2vq_gpu_ctx;
    if (!ctx) { return nullptr; }
    ctx->model = model;
    ctx->L = L;
    ctx->max_tokens = max_tokens;
    ctx->scratch = ds4_gpu_tensor_alloc(L.total);
    /* The RoPE tables: one [max_tokens][hd/2] cos and one sin per rope base, the
     * sliding layers' 1e4 over head_dim 256 and the full layers' 1e6 over 512
     * (ds4_eg2vq.h:65-68).  The tables exist because this build compiles with
     * --use_fast_math, whose sinf/cosf lose ~8e-05 rad at position 1414; the
     * angle is accumulated exactly as the graph accumulates it (theta *=
     * theta_scale in f32) and only the trig is computed in double
     * (cuda/eg2vq_ops.cuh's note; measured bit-identical in the ops gate). */
    const uint64_t swa_half = (uint64_t)DS4_EG2VQ_HEAD_DIM_SWA / 2u;
    const uint64_t full_half = (uint64_t)DS4_EG2VQ_HEAD_DIM_FULL / 2u;
    const uint64_t rope_bytes = eg2vq_gpu_round16(2u * max_tokens * swa_half * sizeof(float))
                              + eg2vq_gpu_round16(2u * max_tokens * full_half * sizeof(float));
    ctx->rope_tab = ds4_gpu_tensor_alloc(rope_bytes);
    if (!ctx->scratch || !ctx->rope_tab) { ds4_gpu_eg2vq_free(ctx); return nullptr; }
    {
        std::vector<float> tab((size_t)(2u * max_tokens * swa_half + 2u * max_tokens * full_half), 0.0f);
        float *swa = tab.data();
        float *full = swa + 2u * max_tokens * swa_half;
        const float base_swa = (float)DS4_EG2VQ_ROPE_BASE_SWA;
        const float base_full = (float)DS4_EG2VQ_ROPE_BASE;
        const float ts_swa = powf(base_swa, -2.0f / (float)DS4_EG2VQ_HEAD_DIM_SWA);
        const float ts_full = powf(base_full, -2.0f / (float)DS4_EG2VQ_HEAD_DIM_FULL);
        for (uint32_t p = 0; p < max_tokens; p++) {
            float theta = (float)p;
            for (uint64_t i = 0; i < swa_half; i++) {
                swa[p * swa_half + i] = (float)cos((double)theta);
                swa[max_tokens * swa_half + p * swa_half + i] = (float)sin((double)theta);
                theta *= ts_swa;
            }
            theta = (float)p;
            for (uint64_t i = 0; i < full_half; i++) {
                full[p * full_half + i] = (float)cos((double)theta);
                full[max_tokens * full_half + p * full_half + i] = (float)sin((double)theta);
                theta *= ts_full;
            }
        }
        if (!ds4_gpu_tensor_write(ctx->rope_tab, 0, tab.data(), tab.size() * sizeof(float))) {
            ds4_gpu_eg2vq_free(ctx);
            return nullptr;
        }
    }
    /* The 24 layer scalars are read back once: each is a single kernel argument
     * (the graph keeps them in a local too, :240-249), and the device form in the
     * aux arena is already f32. */
    for (uint32_t il = 0; il < (uint32_t)DS4_EG2VQ_N_LAYER; il++) {
        const uint32_t slot = eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_SCALAR);
        if (!ds4_gpu_tensor_read(model->aux, model->aux_off[slot], &ctx->layer_scalar[il], sizeof(float))) {
            ds4_gpu_eg2vq_free(ctx);
            return nullptr;
        }
    }
    ctx->calls = 0;
    ctx->tokens = 0;
    return ctx;
}

/* One item, the CPU reference forward's op order (ds4_eg2vq_graph.inc:169-370).
 * The numeric comments cite that file; every constant is ds4_eg2vq.h's. */
extern "C" int ds4_gpu_eg2vq_embed(ds4_eg2vq_gpu_ctx *ctx, const int32_t *ids,
        uint32_t n_ids, float *out768) {
    if (!ctx || !ids || !out768) { return 0; }
    if (!n_ids || n_ids > ctx->max_tokens) { return 0; }
    const ds4_eg2vq_gpu_model *g = ctx->model;
    if (!g->n_vocab) { return 0; }
    for (uint32_t t = 0; t < n_ids; t++) {                       /* the graph dies here; the API refuses */
        if (ids[t] < 0 || (uint64_t)(uint32_t)ids[t] >= g->n_vocab) { return 0; }
    }

    const uint32_t E = (uint32_t)DS4_EG2VQ_N_EMBD;
    const uint32_t PL = (uint32_t)DS4_EG2VQ_PER_LAYER;
    const uint32_t NF = (uint32_t)DS4_EG2VQ_N_FF;
    const float eps = DS4_EG2VQ_RMS_EPS;
    const float *swa_base = (const float *)ctx->rope_tab->ptr;
    const uint64_t swa_floats = (uint64_t)ctx->max_tokens * DS4_EG2VQ_HEAD_DIM_SWA / 2u;
    const float *full_base = (const float *)((const char *)ctx->rope_tab->ptr +
                                             eg2vq_gpu_round16(2u * swa_floats * sizeof(float)));
    const uint64_t full_floats = (uint64_t)ctx->max_tokens * DS4_EG2VQ_HEAD_DIM_FULL / 2u;

    float *h = eg2vq_gpu_frow(ctx, ctx->L.h);
    float *x = eg2vq_gpu_frow(ctx, ctx->L.x);
    float *f = eg2vq_gpu_frow(ctx, ctx->L.f);
    float *a = eg2vq_gpu_frow(ctx, ctx->L.a);
    float *k = eg2vq_gpu_frow(ctx, ctx->L.kv);
    float *v = eg2vq_gpu_frow(ctx, ctx->L.kv + eg2vq_gpu_kv_bytes(ctx->max_tokens));
    float *fg = eg2vq_gpu_frow(ctx, ctx->L.fg);
    float *fu = eg2vq_gpu_frow(ctx, ctx->L.fu);
    float *fd = eg2vq_gpu_frow(ctx, ctx->L.fd);
    float *pg = eg2vq_gpu_frow(ctx, ctx->L.pg);
    float *pp = eg2vq_gpu_frow(ctx, ctx->L.pp);
    float *ple = eg2vq_gpu_frow(ctx, ctx->L.ple);
    float *out = eg2vq_gpu_frow(ctx, ctx->L.out);

    if (!ds4_gpu_tensor_write(ctx->scratch, ctx->L.ids, ids, (size_t)n_ids * sizeof(int32_t))) { return 0; }

    /* 1. token embeddings, scaled by sqrt(n_embd) (graph:196-208). */
    if (!eg2vq_ops_embed(h, (const int32_t *)eg2vq_gpu_buf(ctx, ctx->L.ids),
                         (const uint8_t *)g->aux->ptr + g->aux_off[DS4_EG2VQ_AUX_TOKEN_EMBD],
                         n_ids, E, g->n_vocab)) { return 0; }

    /* 2. the per-layer input, once from the embedding (graph:210-227): the
     * projection, its 1/sqrt(n_embd), then the RMS of each layer's 512-slice. */
    if (!eg2vq_vq_gemm(ple, h, E, eg2vq_gpu_pay(g, DS4_EG2VQ_REC_PLE_PROJ),
                       PL * (uint32_t)DS4_EG2VQ_N_LAYER, E, n_ids)) { return 0; }
    eg2vq_ops_scale(ple, 1.0f / sqrtf((float)E), (uint64_t)n_ids * PL * DS4_EG2VQ_N_LAYER);
    if (!eg2vq_ops_rms(ple, ple, eg2vq_gpu_w(g, DS4_EG2VQ_AUX_PLE_NORM),
                       n_ids * (uint32_t)DS4_EG2VQ_N_LAYER, PL, eps)) { return 0; }

    for (uint32_t il = 0; il < (uint32_t)DS4_EG2VQ_N_LAYER; il++) {
        const uint32_t hd = ds4_eg2vq_head_dim(il);
        const uint32_t nkv = ds4_eg2vq_head_kv(il);
        const uint32_t qw = (uint32_t)DS4_EG2VQ_N_HEAD * hd;
        const int sliding = ds4_eg2vq_layer_is_swa(il);
        const int32_t half_swa = sliding ? (int32_t)((uint32_t)DS4_EG2VQ_SWA / 2u) : -1;
        const uint32_t rec = (uint32_t)DS4_EG2VQ_REC_LAYER_BASE + il * (uint32_t)DS4_EG2VQ_REC_LAYER_STRIDE;
        float *q = eg2vq_gpu_frow(ctx, sliding ? ctx->L.q_swa : ctx->L.q_full);
        float *attn = eg2vq_gpu_frow(ctx, sliding ? ctx->L.attn_swa : ctx->L.attn_full);
        const float *ln_in = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_INPUT_NORM));
        const float *ln_ap = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_ATTN_POST));
        const float *ln_fn = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_FFN_NORM));
        const float *ln_fp = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_FFN_POST));
        const float *ln_pp = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_PLE_POST));
        const float *qn = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_Q_NORM));
        const float *kn = eg2vq_gpu_w(g, eg2vq_gpu_layer_slot(il, DS4_EG2VQ_AUX_L_K_NORM));
        const float *rc = sliding ? swa_base : full_base;
        const float *rs = rc + (sliding ? swa_floats : full_floats);

        /* a. attn_norm (graph:250-253) */
        if (!eg2vq_ops_rms(x, h, ln_in, n_ids, E, eps)) { return 0; }
        /* b. q/k/v projections (graph:256-258) */
        if (!eg2vq_vq_gemm(q, x, E, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_Q), qw, E, n_ids)) { return 0; }
        if (!eg2vq_vq_gemm(k, x, E, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_K), nkv * hd, E, n_ids)) { return 0; }
        if (!eg2vq_vq_gemm(v, x, E, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_V), nkv * hd, E, n_ids)) { return 0; }
        /* c/d. per-head RMS (weighted on q and k, weightless on v) and RoPE over
         * the whole head (graph:260-277) */
        if (!eg2vq_ops_rms_head(q, qn, n_ids, DS4_EG2VQ_N_HEAD, hd, eps)) { return 0; }
        if (!eg2vq_ops_rope(q, rc, rs, n_ids, DS4_EG2VQ_N_HEAD, hd)) { return 0; }
        if (!eg2vq_ops_rms_head(k, kn, n_ids, nkv, hd, eps)) { return 0; }
        if (!eg2vq_ops_rope(k, rc, rs, n_ids, nkv, hd)) { return 0; }
        if (!eg2vq_ops_rms_head(v, NULL, n_ids, nkv, hd, eps)) { return 0; }
        /* e. attention, scale 1.0, non-causal, the symmetric +-512 window
         * (graph:279-288) */
        if (!eg2vq_attn(attn, q, k, v, n_ids, DS4_EG2VQ_N_HEAD, nkv, hd, half_swa)) { return 0; }
        /* f. o_proj (graph:290-291) */
        if (!eg2vq_vq_gemm(a, attn, qw, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_O), E, qw, n_ids)) { return 0; }
        /* g. h += attn_post_norm(o) (graph:293-300) */
        if (!eg2vq_ops_rms(a, a, ln_ap, n_ids, E, eps)) { return 0; }
        eg2vq_ops_add(h, a, (uint64_t)n_ids * E);
        /* h. the FFN (graph:302-310) */
        if (!eg2vq_ops_rms(f, h, ln_fn, n_ids, E, eps)) { return 0; }
        if (!eg2vq_vq_gemm(fg, f, E, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_FFN_GATE), NF, E, n_ids)) { return 0; }
        if (!eg2vq_vq_gemm(fu, f, E, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_FFN_UP), NF, E, n_ids)) { return 0; }
        eg2vq_ops_gelu_mul(fg, fu, (uint64_t)n_ids * NF);
        if (!eg2vq_vq_gemm(fd, fg, NF, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_FFN_DOWN), E, NF, n_ids)) { return 0; }
        /* i. h += ffn_post_norm(ffn) (graph:312-318) */
        if (!eg2vq_ops_rms(fd, fd, ln_fp, n_ids, E, eps)) { return 0; }
        eg2vq_ops_add(h, fd, (uint64_t)n_ids * E);
        /* j. the per-layer input block (graph:320-333) */
        if (!eg2vq_vq_gemm(pg, h, E, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_PLE_GATE), PL, E, n_ids)) { return 0; }
        if (!eg2vq_ops_gelu_mul_strided(pg, ple, n_ids, PL,
                (uint64_t)PL * DS4_EG2VQ_N_LAYER, (uint64_t)il * PL)) { return 0; }
        if (!eg2vq_vq_gemm(pp, pg, PL, eg2vq_gpu_pay(g, rec + DS4_EG2VQ_REC_L_PLE_PROJ), PL, PL, n_ids)) { return 0; }
        if (!eg2vq_ops_rms(pp, pp, ln_pp, n_ids, E, eps)) { return 0; }
        eg2vq_ops_add(h, pp, (uint64_t)n_ids * E);
        /* k. layer_scalar (graph:335-336) */
        eg2vq_ops_scale(h, ctx->layer_scalar[il], (uint64_t)n_ids * E);
    }

    /* 4. output_norm, the 768 projection, the mean pool and L2 (graph:339-367) */
    if (!eg2vq_ops_proj768_pool(out, h,
            (const uint16_t *)((const char *)g->aux->ptr + g->aux_off[DS4_EG2VQ_AUX_OUTPUT]),
            eg2vq_gpu_w(g, DS4_EG2VQ_AUX_OUTPUT_NORM), n_ids, E, DS4_EG2VQ_N_OUT, eps)) { return 0; }
    if (!cuda_ok(cudaGetLastError(), "gemma_vq device tower")) { return 0; }
    if (!ds4_gpu_tensor_read(ctx->scratch, ctx->L.out, out768,
                             (size_t)DS4_EG2VQ_N_OUT * sizeof(float))) { return 0; }
    ctx->calls++;
    ctx->tokens += n_ids;
    return 1;
}
