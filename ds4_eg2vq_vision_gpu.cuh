/* ds4_eg2vq_vision_gpu.cuh -- the gemma_vq (EmbeddingGemma-2) VISION tower's
 * device adapter; included once by ds4_cuda.cu beside ds4_eg2vq_gpu.cuh.
 *
 * Authority, read first-hand (global rule 0.2):
 *   - the tower: ds4-rs/ds4_eg2vq_vision_graph.inc, this port's own CPU
 *     reference forward and the gated oracle of unit 23 (0.82% / 0.53% relative
 *     RMS against the frozen dumps of /data/llama.cpp @ 31288c9c, gate 23 in the
 *     fast path).  The device sequence below is that function's op order, op for
 *     op, with the same buffers and the same constants.
 *   - the kernels: cuda/eg2vq_vis.cuh (this tower's own primitives) and
 *     cuda/eg2vq_ops.cuh (the row-wise and per-head RMS the text tower shares).
 *   - the adapter shape: ds4_eg2vq_gpu.cuh (the same family's text tower) --
 *     a plan bound once per process, a per-item workspace, device tensors
 *     allocated through ds4_gpu_tensor_alloc and filled with
 *     ds4_gpu_tensor_write.
 *
 * THE ONE SIMPLIFICATION THE DEVICE TAKES, and why it is not a deviation: the
 * matrices arrive from the mmproj as bf16 and STAY bf16 on the device -- there is
 * no widening step at all, where the CPU leg widens all 208 matrices once into a
 * 576.0 MiB f32 cache (ds4_eg2vq_vision_bind.inc:101-139).  The GEMM reads bf16
 * rows and widens them in the register, which is the arithmetic the CPU leg's
 * cache holds after its own widening: bf16 -> f32 is exact.  The weight bytes on
 * the device are therefore 288.0 MiB against the CPU leg's 576.0 MiB, and the
 * gate measures the two legs against each other, not against this reasoning.
 *
 * Per-call host traffic is the canvas and the two position vectors up, the soft
 * tokens down; every weight is device-resident for the life of the plan.
 */
#include "ds4_eg2vq_vision.h"
#include "cuda/eg2vq_ops.cuh"
#include "cuda/eg2vq_vis.cuh"

#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

/* 16-byte alignment for every arena slot and every workspace buffer: the widest
 * access in the kernels is the GEMM's shared-tile loads (float4-sized strides)
 * and the bf16 rows. */
enum { EG2VQ_VIS_GPU_ALIGN = 16 };

static inline uint64_t eg2vq_vis_gpu_round16(uint64_t bytes) {
    return (bytes + (uint64_t)(EG2VQ_VIS_GPU_ALIGN - 1)) & ~(uint64_t)(EG2VQ_VIS_GPU_ALIGN - 1);
}

static inline uint32_t eg2vq_vis_layer_slot(uint32_t il, uint32_t sub) {
    return (uint32_t)DS4_EG2VQ_VISION_SLOT_LAYER_BASE + il * (uint32_t)DS4_EG2VQ_VISION_SUB_N + sub;
}

/* ---- the plan ------------------------------------------------------------- */

struct ds4_eg2vq_vision_gpu_model {
    std::vector<uint64_t> w_off;      /* device offset of every bf16 matrix slot */
    std::vector<uint64_t> aux_off;    /* device offset of every f32 vector slot */
    ds4_gpu_tensor *w = nullptr;      /* every bf16 matrix, in slot order */
    ds4_gpu_tensor *aux = nullptr;    /* the f32 vectors, then the rope table */
    uint64_t w_bytes = 0, aux_bytes = 0;
    uint64_t rope_off = 0;
    uint32_t n_slots = 0;
};

static inline const uint16_t *eg2vq_vis_gpu_w(const ds4_eg2vq_vision_gpu_model *m, uint32_t slot) {
    return (const uint16_t *)((const char *)m->w->ptr + m->w_off[slot]);
}

static inline const float *eg2vq_vis_gpu_f(const ds4_eg2vq_vision_gpu_model *m, uint32_t slot) {
    return (const float *)((const char *)m->aux->ptr + m->aux_off[slot]);
}

static inline const float *eg2vq_vis_gpu_rope(const ds4_eg2vq_vision_gpu_model *m) {
    return (const float *)((const char *)m->aux->ptr + m->rope_off);
}

static ds4_eg2vq_vision_gpu_model *eg2vq_vis_gpu_fail(ds4_eg2vq_vision_gpu_model *plan,
                                                      const char *what) {
    fprintf(stderr, "ds4: gemma_vq vision device plan: %s\n", what);
    ds4_gpu_eg2vq_vision_unbind(plan);
    return nullptr;
}

extern "C" void ds4_gpu_eg2vq_vision_unbind(ds4_eg2vq_vision_gpu_model *model) {
    if (!model) { return; }
    ds4_gpu_tensor_free(model->aux);
    ds4_gpu_tensor_free(model->w);
    delete model;
}

/* What one slot must carry, from the architecture (ds4_eg2vq_vision.h pins every
 * constant).  A slot table that disagrees stops the bind, per slot: the same
 * refusal-by-name discipline ds4_gpu_eg2vq_bind applies to its aux slots. */
static int eg2vq_vis_gpu_expect(uint32_t slot, uint32_t *kind, uint32_t *rows, uint32_t *cols) {
    const uint32_t E = (uint32_t)DS4_EG2VQ_VISION_N_EMBD;
    if (slot == (uint32_t)DS4_EG2VQ_VISION_SLOT_PATCH_CONV) {
        *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX;
        *rows = E;                                        /* out features */
        *cols = (uint32_t)DS4_EG2VQ_VISION_PATCH * (uint32_t)DS4_EG2VQ_VISION_PATCH * 3u;
        return 1;
    }
    if (slot == (uint32_t)DS4_EG2VQ_VISION_SLOT_PROJ) {
        *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX;
        *rows = (uint32_t)DS4_EG2VQ_VISION_N_OUT;
        *cols = E;
        return 1;
    }
    if (slot == (uint32_t)DS4_EG2VQ_VISION_SLOT_POS_TABLE) {
        *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_F32_TABLE;
        *rows = 2u * (uint32_t)DS4_EG2VQ_VISION_POS_SIZE;
        *cols = E;
        return 1;
    }
    if (slot < (uint32_t)DS4_EG2VQ_VISION_SLOT_LAYER_BASE) { return 0; }
    const uint32_t rem = slot - (uint32_t)DS4_EG2VQ_VISION_SLOT_LAYER_BASE;
    const uint32_t il = rem / (uint32_t)DS4_EG2VQ_VISION_SUB_N;
    const uint32_t sub = rem % (uint32_t)DS4_EG2VQ_VISION_SUB_N;
    if (il >= (uint32_t)DS4_EG2VQ_VISION_N_LAYER) { return 0; }
    switch (sub) {
        case DS4_EG2VQ_VISION_SUB_LN1:
        case DS4_EG2VQ_VISION_SUB_ATTN_POST:
        case DS4_EG2VQ_VISION_SUB_LN2:
        case DS4_EG2VQ_VISION_SUB_FFN_POST:
            *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_F32_VECTOR; *rows = E; *cols = 0; return 1;
        case DS4_EG2VQ_VISION_SUB_Q_NORM:
        case DS4_EG2VQ_VISION_SUB_K_NORM:
            *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_F32_VECTOR;
            *rows = (uint32_t)DS4_EG2VQ_VISION_HEAD_DIM; *cols = 0; return 1;
        case DS4_EG2VQ_VISION_SUB_Q:
        case DS4_EG2VQ_VISION_SUB_K:
        case DS4_EG2VQ_VISION_SUB_V:
        case DS4_EG2VQ_VISION_SUB_O:
            *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX; *rows = E; *cols = E; return 1;
        case DS4_EG2VQ_VISION_SUB_FFN_GATE:
        case DS4_EG2VQ_VISION_SUB_FFN_UP:
            *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX;
            *rows = (uint32_t)DS4_EG2VQ_VISION_N_FF; *cols = E; return 1;
        case DS4_EG2VQ_VISION_SUB_FFN_DOWN:
            *kind = (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX;
            *rows = E; *cols = (uint32_t)DS4_EG2VQ_VISION_N_FF; return 1;
        default:
            return 0;
    }
}

extern "C" ds4_eg2vq_vision_gpu_model *ds4_gpu_eg2vq_vision_bind(
        const ds4_eg2vq_vision_slot *slots, uint32_t n_slots) {
    if (!slots || n_slots != (uint32_t)DS4_EG2VQ_VISION_SLOT_N) { return nullptr; }
    ds4_eg2vq_vision_gpu_model *plan = new (std::nothrow) ds4_eg2vq_vision_gpu_model;
    if (!plan) { return nullptr; }

    /* ---- pass 1: geometry, and the two arenas' offsets. */
    plan->w_off.assign(n_slots, 0);
    plan->aux_off.assign(n_slots, 0);
    uint64_t w_bytes = 0, aux_bytes = 0;
    for (uint32_t i = 0; i < n_slots; i++) {
        const ds4_eg2vq_vision_slot &s = slots[i];
        uint32_t kind = 0, rows = 0, cols = 0;
        if (!eg2vq_vis_gpu_expect(i, &kind, &rows, &cols)) {
            return eg2vq_vis_gpu_fail(plan, "a slot is outside the tower's slot table");
        }
        if (!s.host || s.kind != kind || s.rows != rows || s.cols != cols) {
            return eg2vq_vis_gpu_fail(plan, "a slot's geometry is not this tower's");
        }
        const uint64_t want = (kind == (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX)
                            ? (uint64_t)rows * cols * 2u
                            : (uint64_t)rows * (cols ? cols : 1u) * 4u;
        if (s.bytes != want) {
            return eg2vq_vis_gpu_fail(plan, "a slot's byte count disagrees with its geometry");
        }
        if (kind == (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX) {
            plan->w_off[i] = eg2vq_vis_gpu_round16(w_bytes);
            w_bytes = plan->w_off[i] + want;
        } else {
            plan->aux_off[i] = eg2vq_vis_gpu_round16(aux_bytes);
            aux_bytes = plan->aux_off[i] + want;
        }
    }
    /* the rope table: 4 sections (X cos, X sin, Y cos, Y sin), one float each,
     * the same order eg2vq_vis_rope_table writes. */
    const uint64_t rope_floats = (uint64_t)EG2VQ_VIS_ROPE_FLOATS;
    plan->rope_off = eg2vq_vis_gpu_round16(aux_bytes);
    aux_bytes = plan->rope_off + rope_floats * 4u;
    plan->w_bytes = eg2vq_vis_gpu_round16(w_bytes);
    plan->aux_bytes = eg2vq_vis_gpu_round16(aux_bytes);

    plan->w = ds4_gpu_tensor_alloc(plan->w_bytes);
    plan->aux = ds4_gpu_tensor_alloc(plan->aux_bytes);
    if (!plan->w || !plan->aux) { return eg2vq_vis_gpu_fail(plan, "no device memory for the tower's weights"); }

    /* ---- pass 2: the bytes.  The matrices go up exactly as the file carries
     * them (bf16, row-major, dim[0] fastest), which is the GEMM's own operand
     * order; the vectors and the position table verbatim. */
    for (uint32_t i = 0; i < n_slots; i++) {
        const ds4_eg2vq_vision_slot &s = slots[i];
        const uint64_t off = (s.kind == (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX)
                           ? plan->w_off[i] : plan->aux_off[i];
        ds4_gpu_tensor *dst = (s.kind == (uint32_t)DS4_EG2VQ_VISION_KIND_BF16_MATRIX)
                            ? plan->w : plan->aux;
        if (!ds4_gpu_tensor_write(dst, off, s.host, s.bytes)) {
            return eg2vq_vis_gpu_fail(plan, "a weight did not reach the device");
        }
    }
    {
        std::vector<float> tab((size_t)rope_floats, 0.0f);
        eg2vq_vis_rope_table(tab.data(), (uint32_t)DS4_EG2VQ_VISION_POS_SIZE);
        if (!ds4_gpu_tensor_write(plan->aux, plan->rope_off, tab.data(), rope_floats * 4u)) {
            return eg2vq_vis_gpu_fail(plan, "the rope table did not reach the device");
        }
    }
    plan->n_slots = n_slots;
    return plan;
}

/* ---- the per-item workspace ---------------------------------------------- */

/* Offsets in bytes, computed in one place so the workspace query and the context
 * creation can never disagree.  `n` patches and `s` soft tokens: the canvas is
 * exactly n*256 pixels * 3 channels = n*768 floats (one patch is 16x16 pixels
 * and the patch grid tiles the canvas). */
typedef struct {
    uint64_t canvas, pos_x, pos_y, im2col, h, z, q, k, v, attn, fg, fu, fo;
    uint64_t pooled, soft, stage_patch, stage_l0;
    uint64_t total;
} eg2vq_vis_gpu_layout;

static eg2vq_vis_gpu_layout eg2vq_vis_gpu_layout_of(uint32_t n, uint32_t s) {
    const uint64_t NP = n, NS = s;
    const uint64_t E = (uint64_t)DS4_EG2VQ_VISION_N_EMBD;
    const uint64_t NF = (uint64_t)DS4_EG2VQ_VISION_N_FF;
    const uint64_t NO = (uint64_t)DS4_EG2VQ_VISION_N_OUT;
    eg2vq_vis_gpu_layout L = {};
    uint64_t c = 0;
    L.canvas = c;      c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.pos_x = c;       c += eg2vq_vis_gpu_round16(NP * sizeof(uint32_t));
    L.pos_y = c;       c += eg2vq_vis_gpu_round16(NP * sizeof(uint32_t));
    L.im2col = c;      c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.h = c;           c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.z = c;           c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.q = c;           c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.k = c;           c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.v = c;           c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.attn = c;        c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.fg = c;          c += eg2vq_vis_gpu_round16(NP * NF * sizeof(float));
    L.fu = c;          c += eg2vq_vis_gpu_round16(NP * NF * sizeof(float));
    L.fo = c;          c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.pooled = c;      c += eg2vq_vis_gpu_round16(NS * E * sizeof(float));
    L.soft = c;        c += eg2vq_vis_gpu_round16(NS * NO * sizeof(float));
    L.stage_patch = c; c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.stage_l0 = c;    c += eg2vq_vis_gpu_round16(NP * E * sizeof(float));
    L.total = c;
    return L;
}

/* The gate's discrimination switches, read once per context (diagnostics, not
 * mechanisms: each one turns OFF something the reference does, which is how the
 * CPU leg's own table in docs/eg2vq-vision-tower-gate23.txt section 4 was
 * measured).  The defaults are the reference's behaviour. */
enum {
    EG2VQ_VIS_GPU_NO_ACT_BF16  = 1u << 0,   /* layer matmuls stop rounding the activation */
    EG2VQ_VIS_GPU_NO_CONV_BF16 = 1u << 1,   /* the conv starts rounding it               */
    EG2VQ_VIS_GPU_NO_GELU_F16  = 1u << 2,   /* gelu_quick loses its f16 in/out chain      */
};

struct ds4_eg2vq_vision_gpu_ctx {
    const ds4_eg2vq_vision_gpu_model *model = nullptr;
    ds4_gpu_tensor *scratch = nullptr;
    eg2vq_vis_gpu_layout L = {};
    uint32_t max_patches = 0, max_soft = 0;
    uint32_t flags = 0;
    /* the last forward, for the gate's stage read-back */
    uint32_t last_patches = 0, last_soft = 0, last_grid_w = 0;
    uint64_t calls = 0;
};

extern "C" uint64_t ds4_gpu_eg2vq_vision_workspace(uint32_t max_patches, uint32_t max_soft) {
    if (!max_patches || !max_soft) { return 0; }
    if (max_patches > (uint32_t)DS4_EG2VQ_VISION_MAX_PATCHES) { return 0; }
    if (max_soft > (uint32_t)DS4_EG2VQ_VISION_SOFT_MAX) { return 0; }
    return eg2vq_vis_gpu_layout_of(max_patches, max_soft).total;
}

extern "C" void ds4_gpu_eg2vq_vision_free(ds4_eg2vq_vision_gpu_ctx *ctx) {
    if (!ctx) { return; }
    if (ctx->calls) {
        fprintf(stderr, "ds4: gemma_vq vision device path calls=%llu weights=%.2f MiB "
                        "aux=%.2f MiB workspace=%.2f MiB (patches<=%u, soft<=%u)\n",
                (unsigned long long)ctx->calls,
                (double)ctx->model->w_bytes / 1048576.0,
                (double)ctx->model->aux_bytes / 1048576.0,
                (double)ctx->L.total / 1048576.0, ctx->max_patches, ctx->max_soft);
    }
    ds4_gpu_tensor_free(ctx->scratch);
    delete ctx;
}

extern "C" ds4_eg2vq_vision_gpu_ctx *ds4_gpu_eg2vq_vision_create(
        const ds4_eg2vq_vision_gpu_model *model, uint64_t scratch,
        uint32_t max_patches, uint32_t max_soft) {
    if (!model || !model->w || !model->aux || !max_patches || !max_soft) { return nullptr; }
    if (max_patches > (uint32_t)DS4_EG2VQ_VISION_MAX_PATCHES ||
        max_soft > (uint32_t)DS4_EG2VQ_VISION_SOFT_MAX) {
        fprintf(stderr, "ds4: gemma_vq vision device path takes at most %u patches and %u soft tokens\n",
                (unsigned)DS4_EG2VQ_VISION_MAX_PATCHES, (unsigned)DS4_EG2VQ_VISION_SOFT_MAX);
        return nullptr;
    }
    const eg2vq_vis_gpu_layout L = eg2vq_vis_gpu_layout_of(max_patches, max_soft);
    if (scratch < L.total) {
        fprintf(stderr, "ds4: gemma_vq vision device path needs %llu bytes of workspace, given %llu\n",
                (unsigned long long)L.total, (unsigned long long)scratch);
        return nullptr;
    }
    ds4_eg2vq_vision_gpu_ctx *ctx = new (std::nothrow) ds4_eg2vq_vision_gpu_ctx;
    if (!ctx) { return nullptr; }
    ctx->model = model;
    ctx->L = L;
    ctx->max_patches = max_patches;
    ctx->max_soft = max_soft;
    ctx->scratch = ds4_gpu_tensor_alloc(L.total);
    if (!ctx->scratch) { delete ctx; return nullptr; }
    if (getenv("DS4_EG2VQ_VIS_NO_ACT_BF16"))  { ctx->flags |= EG2VQ_VIS_GPU_NO_ACT_BF16; }
    if (getenv("DS4_EG2VQ_VIS_CONV_BF16"))    { ctx->flags |= EG2VQ_VIS_GPU_NO_CONV_BF16; }
    if (getenv("DS4_EG2VQ_VIS_NO_GELU_F16"))  { ctx->flags |= EG2VQ_VIS_GPU_NO_GELU_F16; }
    return ctx;
}

/* ---- the forward ---------------------------------------------------------- */

static inline char *eg2vq_vis_gpu_buf(const ds4_eg2vq_vision_gpu_ctx *c, uint64_t off) {
    return (char *)c->scratch->ptr + off;
}

static inline float *eg2vq_vis_gpu_frow(const ds4_eg2vq_vision_gpu_ctx *c, uint64_t off) {
    return (float *)eg2vq_vis_gpu_buf(c, off);
}

/* The gate's stage read-back copies device-to-device, on the same stream as the
 * kernels that produced the buffer. */
static inline int eg2vq_vis_gpu_copy(const ds4_eg2vq_vision_gpu_ctx *c, uint64_t dst,
                                     const void *src, uint64_t bytes) {
    return cudaMemcpyAsync(eg2vq_vis_gpu_buf(c, dst), src, (size_t)bytes,
                           cudaMemcpyDeviceToDevice, ds4_current_stream()) == cudaSuccess;
}

/* gelu_quick(gate) * up, with the gate's discrimination switch honored in one
 * place: the kernel carries the f16 chain (cuda/eg2vq_vis.cuh's
 * eg2vq_vis_gelu_quick), and the switch's form is the same activation without
 * it, which is the CPU leg's own "gelu_quick without the f16 table" row. */
__global__ void eg2vq_vis_no_f16_gelu_mul_kernel(float *__restrict__ gate,
                                                 const float *__restrict__ up, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const float xf = gate[i];
    const float y = xf * (1.0f / (1.0f + expf(-1.702f * xf)));
    gate[i] = y * up[i];
}

extern "C" int ds4_gpu_eg2vq_vision_forward(ds4_eg2vq_vision_gpu_ctx *ctx,
        const uint8_t *data, size_t len, const ds4_eg2vq_vision_opts *opts,
        float *soft, uint64_t soft_cap, ds4_eg2vq_vision_geometry *geo,
        char *err, size_t errlen) {
    if (err && errlen) { err[0] = '\0'; }
    if (!ctx || !ctx->model || !data || !len || !soft || !geo) {
        if (err && errlen) { snprintf(err, errlen, "invalid vision device arguments"); }
        return 0;
    }
    const int f16_kv = opts ? opts->f16_kv : 1;

    /* 1. the preprocessing: the SAME function the CPU leg calls (gate 23 pins
     * it).  The canvas and the two position vectors are the only per-call host
     * buffers. */
    uint32_t *pos_x = (uint32_t *)malloc((size_t)DS4_EG2VQ_VISION_MAX_PATCHES * sizeof(uint32_t));
    uint32_t *pos_y = (uint32_t *)malloc((size_t)DS4_EG2VQ_VISION_MAX_PATCHES * sizeof(uint32_t));
    float *canvas = (float *)malloc((size_t)DS4_EG2VQ_VISION_MAX_PIXELS * 3u * sizeof(float));
    if (!pos_x || !pos_y || !canvas) {
        free(canvas); free(pos_y); free(pos_x);
        if (err && errlen) { snprintf(err, errlen, "no host memory for the preprocessing"); }
        return 0;
    }
    ds4_eg2vq_vision_geometry g;
    if (!ds4_eg2vq_vision_prepare(data, len, &g, pos_x, pos_y, canvas,
                                  (uint64_t)DS4_EG2VQ_VISION_MAX_PIXELS * 3u, err, errlen)) {
        free(canvas); free(pos_y); free(pos_x);
        return 0;
    }
    if (g.n_patches > ctx->max_patches || g.n_soft > ctx->max_soft ||
        (uint64_t)g.n_soft * DS4_EG2VQ_VISION_N_OUT > soft_cap) {
        free(canvas); free(pos_y); free(pos_x);
        if (err && errlen) { snprintf(err, errlen, "the image does not fit the device workspace"); }
        return 0;
    }

    const uint32_t E  = (uint32_t)DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t NF = (uint32_t)DS4_EG2VQ_VISION_N_FF;
    const uint32_t NP = g.n_patches;
    const float eps = DS4_EG2VQ_VISION_RMS_EPS;
    const ds4_eg2vq_vision_gpu_model *M = ctx->model;

    float *h      = eg2vq_vis_gpu_frow(ctx, ctx->L.h);
    float *z      = eg2vq_vis_gpu_frow(ctx, ctx->L.z);
    float *q      = eg2vq_vis_gpu_frow(ctx, ctx->L.q);
    float *k      = eg2vq_vis_gpu_frow(ctx, ctx->L.k);
    float *vv     = eg2vq_vis_gpu_frow(ctx, ctx->L.v);
    float *attn   = eg2vq_vis_gpu_frow(ctx, ctx->L.attn);
    float *fg     = eg2vq_vis_gpu_frow(ctx, ctx->L.fg);
    float *fu     = eg2vq_vis_gpu_frow(ctx, ctx->L.fu);
    float *fo     = eg2vq_vis_gpu_frow(ctx, ctx->L.fo);
    float *pooled = eg2vq_vis_gpu_frow(ctx, ctx->L.pooled);
    float *sdev   = eg2vq_vis_gpu_frow(ctx, ctx->L.soft);
    const float *rope = eg2vq_vis_gpu_rope(M);

    /* The gate's amplification probe, and only that (see
     * tests/parity/eg2vq_vision_cuda_oracle.c): one canvas value moved by a
     * relative amount, so the tower's input-to-output gain is a measured number
     * rather than an explanation.  Unset in every shipping path. */
    if (const char *kick_s = getenv("DS4_EG2VQ_VIS_KICK")) {
        const double kick = atof(kick_s);
        if (kick > 0.0) { canvas[0] = (float)((double)canvas[0] * (1.0 + kick)); }
    }

    if (!ds4_gpu_tensor_write(ctx->scratch, ctx->L.canvas, canvas,
                              (size_t)g.canvas_w * g.canvas_h * 3u * sizeof(float)) ||
        !ds4_gpu_tensor_write(ctx->scratch, ctx->L.pos_x, pos_x, (size_t)NP * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_write(ctx->scratch, ctx->L.pos_y, pos_y, (size_t)NP * sizeof(uint32_t))) {
        free(canvas); free(pos_y); free(pos_x);
        if (err && errlen) { snprintf(err, errlen, "the canvas did not reach the device"); }
        return 0;
    }
    free(canvas); free(pos_y); free(pos_x);

    const uint32_t *d_px = (const uint32_t *)eg2vq_vis_gpu_buf(ctx, ctx->L.pos_x);
    const uint32_t *d_py = (const uint32_t *)eg2vq_vis_gpu_buf(ctx, ctx->L.pos_y);
    float *col = eg2vq_vis_gpu_frow(ctx, ctx->L.im2col);
    const int act_round = (ctx->flags & EG2VQ_VIS_GPU_NO_ACT_BF16) ? 0 : 1;
    const int conv_round = (ctx->flags & EG2VQ_VIS_GPU_NO_CONV_BF16) ? 1 : 0;

    /* 2. the patch convolution (im2col, then the GEMM whose activation is NOT
     * rounded: ggml_conv_2d passes the f32 im2col as src0, ggml.c:4785-4790),
     * then the two position tables. */
    if (!eg2vq_vis_im2col(col, eg2vq_vis_gpu_frow(ctx, ctx->L.canvas),
                          g.canvas_w, g.canvas_h, g.grid_w, NP) ||
        !eg2vq_vis_gemm(h, col, eg2vq_vis_gpu_w(M, (uint32_t)DS4_EG2VQ_VISION_SLOT_PATCH_CONV),
                        NP, E, E, conv_round) ||
        !eg2vq_vis_pos_add(h, eg2vq_vis_gpu_f(M, (uint32_t)DS4_EG2VQ_VISION_SLOT_POS_TABLE),
                           d_px, d_py, NP)) { goto fail; }
    if (!eg2vq_vis_gpu_copy(ctx, ctx->L.stage_patch, h, (uint64_t)NP * E * sizeof(float))) { goto fail; }

    for (uint32_t il = 0; il < (uint32_t)DS4_EG2VQ_VISION_N_LAYER; il++) {
        const uint32_t s_ln1  = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_LN1);
        const uint32_t s_lap  = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_ATTN_POST);
        const uint32_t s_ln2  = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_LN2);
        const uint32_t s_lfp  = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_FFN_POST);
        const uint32_t s_qn   = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_Q_NORM);
        const uint32_t s_kn   = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_K_NORM);
        const uint32_t s_q    = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_Q);
        const uint32_t s_k    = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_K);
        const uint32_t s_v    = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_V);
        const uint32_t s_o    = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_O);
        const uint32_t s_fg   = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_FFN_GATE);
        const uint32_t s_fu   = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_FFN_UP);
        const uint32_t s_fd   = eg2vq_vis_layer_slot(il, DS4_EG2VQ_VISION_SUB_FFN_DOWN);

        /* a. ln1, then the three projections (their activation IS rounded: the
         * reference's mul_mat quantizes src1 to the weight's bf16 vec_dot_type,
         * ggml-cpu/ggml-cpu.c:395-400, :1328-1340 -- the mechanism of unit 23). */
        if (!eg2vq_ops_rms(z, h, eg2vq_vis_gpu_f(M, s_ln1), NP, E, eps) ||
            !eg2vq_vis_gemm(q,  z, eg2vq_vis_gpu_w(M, s_q), NP, E, E, act_round) ||
            !eg2vq_vis_gemm(k,  z, eg2vq_vis_gpu_w(M, s_k), NP, E, E, act_round) ||
            !eg2vq_vis_gemm(vv, z, eg2vq_vis_gpu_w(M, s_v), NP, E, E, act_round)) { goto fail; }

        /* b. per-head RMS (weighted on q and k, weightless on v) and the axial
         * rope: graph:326-334, in the reference's order. */
        if (!eg2vq_ops_rms_head(q, eg2vq_vis_gpu_f(M, s_qn), NP,
                                (uint32_t)DS4_EG2VQ_VISION_N_HEAD,
                                (uint32_t)DS4_EG2VQ_VISION_HEAD_DIM, eps) ||
            !eg2vq_ops_rms_head(k, eg2vq_vis_gpu_f(M, s_kn), NP,
                                (uint32_t)DS4_EG2VQ_VISION_N_HEAD,
                                (uint32_t)DS4_EG2VQ_VISION_HEAD_DIM, eps) ||
            !eg2vq_vis_rope(q, k, rope, d_px, d_py, NP, 1) ||
            !eg2vq_ops_rms_head(vv, NULL, NP, (uint32_t)DS4_EG2VQ_VISION_N_HEAD,
                                (uint32_t)DS4_EG2VQ_VISION_HEAD_DIM, eps)) { goto fail; }

        /* c. the reference's executed attention rounds K and V to f16 (the flash
         * branch, clip.cpp:787-790; graph:546-549). */
        if (f16_kv && (!eg2vq_vis_f16_round(k, (uint64_t)NP * E) ||
                       !eg2vq_vis_f16_round(vv, (uint64_t)NP * E))) { goto fail; }

        /* d. attention (scale 1.0, no mask), then the output projection, its
         * norm and the residual. */
        if (!eg2vq_vis_attn(attn, q, k, vv, NP) ||
            !eg2vq_vis_gemm(fo, attn, eg2vq_vis_gpu_w(M, s_o), NP, E, E, act_round) ||
            !eg2vq_ops_rms(fo, fo, eg2vq_vis_gpu_f(M, s_lap), NP, E, eps) ||
            !eg2vq_ops_add(h, fo, (uint64_t)NP * E)) { goto fail; }

        /* e. ln2, the GEGLU FFN, its norm and the residual. */
        if (!eg2vq_ops_rms(z, h, eg2vq_vis_gpu_f(M, s_ln2), NP, E, eps) ||
            !eg2vq_vis_gemm(fg, z, eg2vq_vis_gpu_w(M, s_fg), NP, NF, E, act_round) ||
            !eg2vq_vis_gemm(fu, z, eg2vq_vis_gpu_w(M, s_fu), NP, NF, E, act_round)) { goto fail; }
        if (ctx->flags & EG2VQ_VIS_GPU_NO_GELU_F16) {
            const uint64_t n = (uint64_t)NP * NF;
            eg2vq_vis_no_f16_gelu_mul_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256>>>(fg, fu, n);
        } else if (!eg2vq_vis_gelu_mul(fg, fu, (uint64_t)NP * NF)) { goto fail; }
        if (!eg2vq_vis_gemm(fo, fg, eg2vq_vis_gpu_w(M, s_fd), NP, E, NF, act_round) ||
            !eg2vq_ops_rms(fo, fo, eg2vq_vis_gpu_f(M, s_lfp), NP, E, eps) ||
            !eg2vq_ops_add(h, fo, (uint64_t)NP * E)) { goto fail; }

        if (il == 0u &&
            !eg2vq_vis_gpu_copy(ctx, ctx->L.stage_l0, h, (uint64_t)NP * E * sizeof(float))) {
            goto fail;
        }
    }

    /* 3. the pooler, then the embedder: weightless RMS (eps 1e-6) and the
     * 768->512 projection (graph:586-604; gemma4v.cpp:83-114).  The embedder's
     * RMS writes into `z`, not in place, so the pooled rows the gate compares
     * against the CPU leg's stage stay what the pooler produced. */
    if (!eg2vq_vis_pool(h, pooled, g.grid_w, g.n_soft) ||
        !eg2vq_ops_rms(z, pooled, NULL, g.n_soft, E, eps) ||
        !eg2vq_vis_gemm(sdev, z, eg2vq_vis_gpu_w(M, (uint32_t)DS4_EG2VQ_VISION_SLOT_PROJ),
                        g.n_soft, (uint32_t)DS4_EG2VQ_VISION_N_OUT, E, act_round)) { goto fail; }

    if (!cuda_ok(cudaGetLastError(), "gemma_vq vision device tower")) { goto fail; }
    if (!ds4_gpu_tensor_read(ctx->scratch, ctx->L.soft, soft,
                             (size_t)g.n_soft * DS4_EG2VQ_VISION_N_OUT * sizeof(float))) { goto fail; }

    ctx->last_patches = NP;
    ctx->last_soft = g.n_soft;
    ctx->last_grid_w = g.grid_w;
    ctx->calls++;
    *geo = g;
    return 1;

fail:
    if (err && errlen) { snprintf(err, errlen, "a device kernel refused the tower's shape"); }
    return 0;
}

extern "C" uint64_t ds4_gpu_eg2vq_vision_stage(const ds4_eg2vq_vision_gpu_ctx *ctx,
                                               const char *stage, float *dst, uint64_t cap) {
    if (!ctx || !ctx->scratch || !stage || !dst || !ctx->last_patches) { return 0; }
    const uint64_t E = (uint64_t)DS4_EG2VQ_VISION_N_EMBD;
    uint64_t n = 0, off = 0;
    if (!strcmp(stage, "patch"))       { n = (uint64_t)ctx->last_patches * E; off = ctx->L.stage_patch; }
    else if (!strcmp(stage, "l0"))     { n = (uint64_t)ctx->last_patches * E; off = ctx->L.stage_l0; }
    else if (!strcmp(stage, "pooled")) { n = (uint64_t)ctx->last_soft * E;    off = ctx->L.pooled; }
    else if (!strcmp(stage, "soft"))   {
        n = (uint64_t)ctx->last_soft * (uint64_t)DS4_EG2VQ_VISION_N_OUT; off = ctx->L.soft;
    } else { return 0; }
    if (cap < n) { return 0; }
    if (!ds4_gpu_tensor_read(ctx->scratch, off, dst, n * sizeof(float))) { return 0; }
    return n;
}
