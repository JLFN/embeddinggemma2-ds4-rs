/* eg2vq_vis.cuh — the gemma_vq VISION tower's device primitives
 * (embeddinggemma2-handoff.md PHASE 8, row P4.2, unit 24).
 *
 * Authority, read first-hand (global rule 0.2): the CPU leg this unit is gated
 * against is ds4-rs/ds4_eg2vq_vision_graph.inc, this port's own reference
 * forward (unit 23, within 0.82%/0.53% relative RMS of the frozen dumps of
 * /data/llama.cpp @ 31288c9c).  Every kernel below is one of that file's
 * expressions, in that file's order, with that file's constants, and the
 * citations are to it and through it to the reference it transcribes:
 *
 *   eg2vq_vis_gemm        the dense bf16-weight matvec (graph:420-434), with the
 *                         ACTIVATION QUANTIZED TO BF16 exactly as the CPU leg
 *                         quantizes it (graph:429-431, eg2vq_vision_bf16 at
 *                         graph:94-103) — the mechanism the reference's kernels
 *                         impose (ggml-cpu/ggml-cpu.c:395-400, :1328-1340) and
 *                         the one the conv is exempt from
 *                         (ggml.c:4785-4790: ggml_conv_2d passes the f32 im2col
 *                         as src0, so the bf16 weight is merely widened).
 *                         `act_round` is that switch: 1 = the layer matmuls and
 *                         the tie-in projection, 0 = the patch conv, which is
 *                         the CPU leg's own two paths and not a device choice.
 *   eg2vq_vis_im2col      the im2col ggml_conv_2d performs before its mul_mat
 *                         (ggml.c:4785-4790): canvas -> [n_patch][768], with
 *                         j = ic*256 + kh*16 + kw, which is the convolution
 *                         weight's own memory order (graph:288-290).
 *   eg2vq_vis_pos_add     + the X table then + the Y table, in that order
 *                         (graph:499-505; gemma4v.cpp:26-43).
 *   eg2vq_vis_rope        the axial NeoX rotation: dims [0,32) against pos_x
 *                         and dims [32,64) against pos_y at the +32 offset, each
 *                         half rotating pairs (i, i+16) with theta starting at
 *                         the position and multiplying by
 *                         theta_scale = 100^(-2/32) per step (graph:69-77,
 *                         :326-330; ops.cpp:6455, :6375-6391).
 *   eg2vq_vis_attn        attention at kq_scale exactly 1.0, NO mask, 12 query
 *                         heads over 12 KV heads, head_dim 64 (graph:344-381;
 *                         gemma4v.cpp:75, clip.cpp:787-802 with kq_mask null).
 *   eg2vq_vis_f16_round   the reference's executed flash branch casting K and V
 *                         to f16 (clip.cpp:787-790; graph:546-549).
 *   eg2vq_vis_gelu_mul    gelu_quick(gate) * up through the same f16 in / f16
 *                         out chain the CPU leg uses (graph:82-89, :573-575;
 *                         vec.h:47, :1031, :1047-1057; ggml-cpu.c:3884-3891).
 *   eg2vq_vis_pool        the 3x3 stride-3 average pool with row q = x' +
 *                         (grid_w/3)*y', the /9 and the sqrt(n_embd) scale
 *                         (graph:394-412, :586-592; gemma4v.cpp:83-99).
 *
 * THE ROPE TABLE IS BUILT ON THE HOST, NOT IN THE KERNEL.  This tree compiles
 * its device code with --use_fast_math (Makefile:63), so sinf/cosf become the
 * SFU approximations; the CPU leg (and the reference) use the accurate ones.
 * The host table below accumulates the angle exactly as the CPU leg accumulates
 * it (float theta, multiplied by theta_scale each step) and computes only the
 * trig in double, which is the same device the text tower's rope table takes
 * (ds4_eg2vq_gpu.cuh:346-361) and for the same reason.  It also removes the
 * per-call cost: 16 pairs times 2 axes per (patch, head) is 3.9 M evaluations
 * for the largest image, and the table is 10.5 MiB for all 10240 positions.
 *
 * WHAT THE DEVICE ADDS, named rather than hidden (the gate measures both):
 *  - the accumulations that the CPU leg runs in double (the GEMM's dots, the
 *    attention scores, the pool's 9-term sum) accumulate in their own order
 *    here: f32 for the GEMM and the attention (eg2vq_attn.cuh:102-140, the
 *    scalar and vectorized four-partial-sum dots), double for the pool's nine
 *    terms.
 *  - the softmax is ONLINE (a running max and sum rescaled per KV tile) instead
 *    of the CPU leg's two-pass max-then-exp; the two are the same function up to
 *    rounding, and the online form is the one the reference's own flash kernel
 *    runs (clip.cpp:787-802).  The K/V cache the tile sizes need is the plane
 *    below; a longer image than the plane holds is refused by the adapter, never
 *    truncated.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>
#include <cstring>

#include "eg2vq_ops.cuh"   /* eg2vq_ops_rms, eg2vq_ops_scale, eg2vq_ops_add */

/* ---- the tower's own shape (ds4_eg2vq_vision.h pins every one of them) ---- */
enum {
    EG2VQ_VIS_N_LAYER   = 16,      /* DS4_EG2VQ_VISION_N_LAYER */
    EG2VQ_VIS_N_EMBD    = 768,     /* DS4_EG2VQ_VISION_N_EMBD */
    EG2VQ_VIS_N_HEAD    = 12,      /* DS4_EG2VQ_VISION_N_HEAD */
    EG2VQ_VIS_HEAD_DIM  = 64,      /* DS4_EG2VQ_VISION_HEAD_DIM */
    EG2VQ_VIS_N_FF      = 3072,    /* DS4_EG2VQ_VISION_N_FF */
    EG2VQ_VIS_PATCH     = 16,      /* DS4_EG2VQ_VISION_PATCH */
    EG2VQ_VIS_MERGE     = 3,       /* DS4_EG2VQ_VISION_MERGE */
    EG2VQ_VIS_POS_SIZE  = 10240,   /* DS4_EG2VQ_VISION_POS_SIZE */
    EG2VQ_VIS_N_OUT     = 512,     /* DS4_EG2VQ_VISION_N_OUT */

    /* The rope table: 16 pairs per axis-half (head_dim/2 = 32 dims, pairs
     * (i, i+16)), two axes, cos and sin each, for every position the position
     * table can address. */
    EG2VQ_VIS_ROPE_PAIRS = EG2VQ_VIS_HEAD_DIM / 4,
    EG2VQ_VIS_ROPE_FLOATS = 4u * EG2VQ_VIS_POS_SIZE * (EG2VQ_VIS_HEAD_DIM / 4),

    /* The GEMM's tile: 32 patches x 64 outputs x 16 in-features per step, with
     * one 16x16 block of threads and a 2 x 4 micro-tile per thread.  The
     * activation tile is rounded to bf16 AT LOAD when the caller asks for it,
     * so the rounding happens once per element per matmul, as the CPU leg's
     * pre-pass does it (graph:429-431). */
    EG2VQ_VIS_PT = 32,
    EG2VQ_VIS_OT = 64,
    EG2VQ_VIS_KT = 16,

    /* Attention: one warp per head (12 warps per block), key tiles of 512
     * cached in shared as probabilities.  12 * 512 * 4 B = 24 KiB. */
    EG2VQ_VIS_ATTN_WARPS = EG2VQ_VIS_N_HEAD,
    EG2VQ_VIS_ATTN_KV = 512,
    EG2VQ_VIS_ATTN_SHARED = EG2VQ_VIS_ATTN_WARPS * EG2VQ_VIS_ATTN_KV,
};

/* ---- the graph's own conversions, on the device --------------------------- */

/* Round-to-nearest-even f32 -> bf16 -> f32.  This is the CPU leg's
 * eg2vq_vision_bf16 bit for bit (graph:94-103; ggml-impl.h:623-635).  It is
 * written out rather than left to __float2bfloat16_rn so that the device's
 * rounding is the CPU leg's by construction, not by an assumption about a
 * library call. */
__host__ __device__ __forceinline__ static float eg2vq_vis_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return x; }
    bits = (bits + (0x7fffu + ((bits >> 16) & 1u))) >> 16;
    bits <<= 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* f32 -> f16 -> f32 round trip.  __float2half_rn is round-to-nearest-even, the
 * same conversion the CPU leg's f32_to_f16/f16_to_f32 pair performs (ds4.c's
 * pair, accurate on subnormals as this is). */
__host__ __device__ __forceinline__ static float eg2vq_vis_f16(float x) {
    return __half2float(__float2half_rn(x));
}

/* ggml's f32 gelu_quick, computed the way ggml computes it on a CPU whose
 * GGML_GELU_QUICK_FP16 is defined -- which is unconditional in this revision
 * (vec.h:47): the input is quantized to f16, the value gelu_quick_f32 is taken
 * of the widened input, and that result is quantized to f16 and widened back
 * (vec.h:1047-1057; the table it looks up is built as
 * FP32_TO_FP16(gelu_quick_f32(FP16_TO_FP32(i))), ggml-cpu.c:3884-3891).  The
 * CPU leg carries the same four conversions (graph:82-89); the expression
 * gelu_quick_f32 itself is vec.h:1031, x*(1/(1+expf(-1.702 x))).
 *
 * The expf here is --use_fast_math's, which is the one device-side difference
 * in this chain: the f16 quantizations on both sides of it are worth ~5e-4
 * relative and dominate it, and the gate measures what is left. */
__host__ __device__ __forceinline__ static float eg2vq_vis_gelu_quick(float x) {
    const float xf = eg2vq_vis_f16(x);
    const float y = xf * (1.0f / (1.0f + expf(-1.702f * xf)));
    return eg2vq_vis_f16(y);
}

/* ---- the patch convolution as im2col + GEMM ------------------------------ */

/* canvas -> [n_patches][768], j = ic*256 + kh*16 + kw: the weight's own order
 * (graph:288-296 walks wp[ic*P*P + kh*P + kw] against
 * canvas[x + canvas_w*(y + canvas_h*ic)], which is this element). */
__global__ void eg2vq_vis_im2col_kernel(
        float *__restrict__ xcol, const float *__restrict__ canvas,
        uint32_t canvas_w, uint32_t canvas_h, uint32_t grid_w, uint32_t n_patches) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_patches * (uint64_t)EG2VQ_VIS_N_EMBD) { return; }
    const uint32_t p = (uint32_t)(i / (uint64_t)EG2VQ_VIS_N_EMBD);
    const uint32_t j = (uint32_t)(i % (uint64_t)EG2VQ_VIS_N_EMBD);
    const uint32_t ic = j >> 8u;                 /* j / 256 */
    const uint32_t kh = (j >> 4u) & 15u;         /* (j % 256) / 16 */
    const uint32_t kw = j & 15u;                 /* j % 16 */
    const uint32_t px = p % grid_w;
    const uint32_t py = p / grid_w;
    const uint64_t plane = (uint64_t)canvas_w * (uint64_t)canvas_h;
    xcol[i] = canvas[(uint64_t)(px * (uint32_t)EG2VQ_VIS_PATCH + kw)
                   + (uint64_t)canvas_w * (uint64_t)(py * (uint32_t)EG2VQ_VIS_PATCH + kh)
                   + plane * (uint64_t)ic];
}

static inline int eg2vq_vis_im2col(float *xcol, const float *canvas, uint32_t canvas_w,
                                   uint32_t canvas_h, uint32_t grid_w, uint32_t n_patches) {
    if (!xcol || !canvas || !canvas_w || !canvas_h || !grid_w || !n_patches) { return 0; }
    const uint64_t n = (uint64_t)n_patches * (uint64_t)EG2VQ_VIS_N_EMBD;
    eg2vq_vis_im2col_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256>>>(
            xcol, canvas, canvas_w, canvas_h, grid_w, n_patches);
    return 1;
}

/* One blocked dense GEMM against a bf16 weight held row-major [rows][cols], the
 * memory order the mmproj carries and the order the CPU leg's widened cache has
 * (graph:100-107): out[p][o] = sum_k w[o][k] * round?(x[p][k]).  `act_round` is
 * the mechanism switch documented at the top of this file. */
__global__ void eg2vq_vis_gemm_kernel(
        float *__restrict__ out, const float *__restrict__ x,
        const uint16_t *__restrict__ w,
        uint32_t n, uint32_t rows, uint32_t cols, int act_round) {
    __shared__ float xs[EG2VQ_VIS_PT][EG2VQ_VIS_KT];
    __shared__ float ws[EG2VQ_VIS_OT][EG2VQ_VIS_KT];

    const uint32_t tx = threadIdx.x & 15u;
    const uint32_t ty = threadIdx.x >> 4u;
    const uint32_t pb = blockIdx.x * (uint32_t)EG2VQ_VIS_PT;
    const uint32_t ob = blockIdx.y * (uint32_t)EG2VQ_VIS_OT;

    float acc[2][4];
    #pragma unroll
    for (uint32_t a = 0; a < 2u; a++) {
        #pragma unroll
        for (uint32_t b = 0; b < 4u; b++) { acc[a][b] = 0.0f; }
    }

    for (uint32_t k0 = 0; k0 < cols; k0 += (uint32_t)EG2VQ_VIS_KT) {
        for (uint32_t i = threadIdx.x; i < (uint32_t)(EG2VQ_VIS_PT * EG2VQ_VIS_KT); i += blockDim.x) {
            const uint32_t p = i / (uint32_t)EG2VQ_VIS_KT;
            const uint32_t k = i % (uint32_t)EG2VQ_VIS_KT;
            float v = 0.0f;
            if (pb + p < n && k0 + k < cols) {
                v = x[(uint64_t)(pb + p) * cols + (k0 + k)];
                if (act_round) { v = eg2vq_vis_bf16(v); }
            }
            xs[p][k] = v;
        }
        for (uint32_t i = threadIdx.x; i < (uint32_t)(EG2VQ_VIS_OT * EG2VQ_VIS_KT); i += blockDim.x) {
            const uint32_t o = i / (uint32_t)EG2VQ_VIS_KT;
            const uint32_t k = i % (uint32_t)EG2VQ_VIS_KT;
            float v = 0.0f;
            if (ob + o < rows && k0 + k < cols) {
                const uint16_t h = w[(uint64_t)(ob + o) * cols + (k0 + k)];
                const uint32_t u = (uint32_t)h << 16;      /* bf16 -> f32 is the top half */
                memcpy(&v, &u, 4);
            }
            ws[o][k] = v;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t k = 0; k < (uint32_t)EG2VQ_VIS_KT; k++) {
            const float x0 = xs[ty * 2u][k];
            const float x1 = xs[ty * 2u + 1u][k];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const float wv = ws[tx * 4u + j][k];
                acc[0][j] += x0 * wv;
                acc[1][j] += x1 * wv;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
        const uint32_t o = ob + tx * 4u + j;
        if (o < rows) {
            if (pb + ty * 2u < n) { out[(uint64_t)(pb + ty * 2u) * rows + o] = acc[0][j]; }
            if (pb + ty * 2u + 1u < n) { out[(uint64_t)(pb + ty * 2u + 1u) * rows + o] = acc[1][j]; }
        }
    }
}

static inline int eg2vq_vis_gemm(float *out, const float *x, const uint16_t *w,
                                 uint32_t n, uint32_t rows, uint32_t cols, int act_round) {
    if (!out || !x || !w || !n || !rows || !cols) { return 0; }
    if (cols > UINT32_MAX - (uint32_t)EG2VQ_VIS_KT) { return 0; }
    const dim3 grid((n + (uint32_t)EG2VQ_VIS_PT - 1u) / (uint32_t)EG2VQ_VIS_PT,
                    (rows + (uint32_t)EG2VQ_VIS_OT - 1u) / (uint32_t)EG2VQ_VIS_OT, 1);
    eg2vq_vis_gemm_kernel<<<grid, 256>>>(out, x, w, n, rows, cols, act_round);
    return 1;
}

/* ---- the two axial position tables --------------------------------------- */

/* h[p][d] += X[pos_x[p]][d], then += Y[pos_y[p]][d]: the reference's order
 * (graph:499-505 adds emb_x first, then emb_y). */
__global__ void eg2vq_vis_pos_add_kernel(
        float *__restrict__ h, const float *__restrict__ tab,
        const uint32_t *__restrict__ pos_x, const uint32_t *__restrict__ pos_y,
        uint32_t n_patches) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_patches * (uint64_t)EG2VQ_VIS_N_EMBD) { return; }
    const uint32_t p = (uint32_t)(i / (uint64_t)EG2VQ_VIS_N_EMBD);
    const uint32_t d = (uint32_t)(i % (uint64_t)EG2VQ_VIS_N_EMBD);
    const uint64_t slice = (uint64_t)EG2VQ_VIS_POS_SIZE * (uint64_t)EG2VQ_VIS_N_EMBD;
    float v = h[i];
    v += tab[(uint64_t)pos_x[p] * (uint64_t)EG2VQ_VIS_N_EMBD + d];
    v += tab[slice + (uint64_t)pos_y[p] * (uint64_t)EG2VQ_VIS_N_EMBD + d];
    h[i] = v;
}

static inline int eg2vq_vis_pos_add(float *h, const float *tab, const uint32_t *pos_x,
                                    const uint32_t *pos_y, uint32_t n_patches) {
    if (!h || !tab || !pos_x || !pos_y || !n_patches) { return 0; }
    const uint64_t n = (uint64_t)n_patches * (uint64_t)EG2VQ_VIS_N_EMBD;
    eg2vq_vis_pos_add_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256>>>(
            h, tab, pos_x, pos_y, n_patches);
    return 1;
}

/* ---- the axial NeoX RoPE -------------------------------------------------- */

/* The table: [4][POS_SIZE][16] as (X cos, X sin, Y cos, Y sin).  Built on the
 * host, in the CPU leg's angle order (graph:69-77, :509-510): theta starts at
 * the position as a FLOAT and is multiplied by theta_scale once per pair, and
 * only cos/sin are computed in double. */
static inline void eg2vq_vis_rope_table(float *tab, uint32_t pos_size) {
    const uint32_t pairs = (uint32_t)EG2VQ_VIS_ROPE_PAIRS;
    const float theta_scale = powf(100.0f, -2.0f / (float)(EG2VQ_VIS_HEAD_DIM / 2u));
    const uint64_t section = (uint64_t)pos_size * pairs;
    for (uint32_t p = 0; p < pos_size; p++) {
        float theta = (float)p;
        for (uint32_t i = 0; i < pairs; i++) {
            const double c = cos((double)theta);
            const double s = sin((double)theta);
            tab[(uint64_t)p * pairs + i] = (float)c;
            tab[section + (uint64_t)p * pairs + i] = (float)s;
            tab[2u * section + (uint64_t)p * pairs + i] = (float)c;
            tab[3u * section + (uint64_t)p * pairs + i] = (float)s;
            theta *= theta_scale;
        }
    }
}

/* One thread per (patch, head, pair): the first half of the head rotates against
 * pos_x, the second against pos_y (graph:326-330: the second call is the +D/2
 * offset the reference sets with ggml_rope_set_offset, gemma4v.cpp:70).  Each
 * call rotates n_rot = D/2 = 32 dimensions, so its own pairs are (i, i + 16)
 * (graph:69-77: half = n_rot/2), which is the `half` below -- NOT D/2. */
__global__ void eg2vq_vis_rope_kernel(
        float *__restrict__ q, float *__restrict__ k, const float *__restrict__ tab,
        const uint32_t *__restrict__ pos_x, const uint32_t *__restrict__ pos_y,
        uint32_t n_patches, int do_k) {
    const uint32_t pairs = (uint32_t)EG2VQ_VIS_ROPE_PAIRS;
    const uint32_t half = (uint32_t)EG2VQ_VIS_HEAD_DIM / 4u;
    const uint32_t halfway = (uint32_t)EG2VQ_VIS_HEAD_DIM / 2u;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_patches * (uint64_t)EG2VQ_VIS_N_HEAD * pairs;
    if (i >= total) { return; }
    const uint32_t pair = (uint32_t)(i % pairs);
    const uint32_t head = (uint32_t)((i / pairs) % (uint64_t)EG2VQ_VIS_N_HEAD);
    const uint32_t p    = (uint32_t)(i / (pairs * (uint64_t)EG2VQ_VIS_N_HEAD));
    const uint64_t section = (uint64_t)EG2VQ_VIS_POS_SIZE * pairs;
    float *row = q + (uint64_t)p * (uint64_t)EG2VQ_VIS_N_EMBD
                   + (uint64_t)head * (uint64_t)EG2VQ_VIS_HEAD_DIM;
    const float *cx = tab + (uint64_t)pos_x[p] * pairs;
    const float *sx = tab + section + (uint64_t)pos_x[p] * pairs;
    const float *cy = tab + 2u * section + (uint64_t)pos_y[p] * pairs;
    const float *sy = tab + 3u * section + (uint64_t)pos_y[p] * pairs;
    for (int which = 0; which < (do_k ? 2 : 1); which++) {
        float *r = which ? (k + (uint64_t)p * (uint64_t)EG2VQ_VIS_N_EMBD
                              + (uint64_t)head * (uint64_t)EG2VQ_VIS_HEAD_DIM) : row;
        {
            const float x0 = r[pair], x1 = r[pair + half];
            r[pair]        = x0 * cx[pair] - x1 * sx[pair];
            r[pair + half] = x0 * sx[pair] + x1 * cx[pair];
        }
        {
            float *ry = r + halfway;
            const float x0 = ry[pair], x1 = ry[pair + half];
            ry[pair]        = x0 * cy[pair] - x1 * sy[pair];
            ry[pair + half] = x0 * sy[pair] + x1 * cy[pair];
        }
    }
}

static inline int eg2vq_vis_rope(float *q, float *k, const float *tab,
                                 const uint32_t *pos_x, const uint32_t *pos_y,
                                 uint32_t n_patches, int do_k) {
    if (!q || !tab || !pos_x || !pos_y || !n_patches) { return 0; }
    const uint64_t n = (uint64_t)n_patches * (uint64_t)EG2VQ_VIS_N_HEAD
                     * (uint64_t)EG2VQ_VIS_ROPE_PAIRS;
    eg2vq_vis_rope_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256>>>(
            q, k, tab, pos_x, pos_y, n_patches, do_k);
    return 1;
}

/* ---- the f16 K/V cast the reference's executed flash branch performs ------ */

__global__ void eg2vq_vis_f16_round_kernel(float *__restrict__ x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = eg2vq_vis_f16(x[i]); }
}

static inline int eg2vq_vis_f16_round(float *x, uint64_t n) {
    if (!x || !n) { return 0; }
    eg2vq_vis_f16_round_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256>>>(x, n);
    return 1;
}

/* ---- attention: 12 heads, no mask, scale 1.0 ----------------------------- */

/* One block per query patch, one warp per head.  The key range is the whole
 * item (no mask, no window: graph:344-381).  Per KV tile the warp computes the
 * scores, updates the running max, rescales the running denominator and output
 * by exp(m_old - m_new) and accumulates this tile's weighted V.  The score and
 * output accumulations are f32 with four partial sums (eg2vq_attn.cuh:102-140
 * for the same choice, measured there); the probabilities are cached in shared
 * and read back by the output pass, so each (query, key) dot is computed once. */
__device__ __forceinline__ static float eg2vq_vis_attn_wmax(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); }
    return v;
}

__device__ __forceinline__ static float eg2vq_vis_attn_wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
    return v;
}

__global__ void eg2vq_vis_attn_kernel(
        float *__restrict__ out, const float *__restrict__ q,
        const float *__restrict__ k, const float *__restrict__ v,
        uint32_t n_patches) {
    __shared__ float plane[EG2VQ_VIS_ATTN_SHARED];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (warp >= (uint32_t)EG2VQ_VIS_ATTN_WARPS) { return; }
    const uint32_t qp = blockIdx.x;
    const uint32_t D = (uint32_t)EG2VQ_VIS_HEAD_DIM;
    const uint64_t row_stride = (uint64_t)EG2VQ_VIS_N_EMBD;

    const float *qh = q + (uint64_t)qp * row_stride + (uint64_t)warp * D;
    const float *kh0 = k + (uint64_t)warp * D;
    const float *vh0 = v + (uint64_t)warp * D;
    float *oh = out + (uint64_t)qp * row_stride + (uint64_t)warp * D;
    float *scores = plane + (size_t)warp * (uint32_t)EG2VQ_VIS_ATTN_KV;

    float m = -FLT_MAX;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f;      /* dims lane and lane + 32 */

    for (uint32_t k0 = 0; k0 < n_patches; k0 += (uint32_t)EG2VQ_VIS_ATTN_KV) {
        const uint32_t nkey = (k0 + (uint32_t)EG2VQ_VIS_ATTN_KV <= n_patches)
                            ? (uint32_t)EG2VQ_VIS_ATTN_KV : (n_patches - k0);
        float tile_max = -FLT_MAX;
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float *kh = kh0 + (uint64_t)(k0 + kk) * row_stride;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            uint32_t d = 0;
            for (; d + 4u <= D; d += 4u) {
                a0 += qh[d] * kh[d];
                a1 += qh[d + 1u] * kh[d + 1u];
                a2 += qh[d + 2u] * kh[d + 2u];
                a3 += qh[d + 3u] * kh[d + 3u];
            }
            for (; d < D; d++) { a0 += qh[d] * kh[d]; }
            const float s = (a0 + a1) + (a2 + a3);   /* kq_scale is exactly 1.0 */
            scores[kk] = s;
            tile_max = fmaxf(tile_max, s);
        }
        const float M = eg2vq_vis_attn_wmax(tile_max);
        const float Mnew = fmaxf(m, M);
        float rescale = 0.0f;
        if (m > -FLT_MAX) { rescale = expf(m - Mnew); }
        l *= rescale;
        o0 *= rescale;
        o1 *= rescale;

        float psum = 0.0f;
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float e = expf(scores[kk] - Mnew);
            scores[kk] = e;
            psum += e;
        }
        l += eg2vq_vis_attn_wsum(psum);

        const uint32_t d0 = lane, d1 = lane + 32u;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint32_t kk = 0; kk < nkey; kk++) {
            const float p = scores[kk];
            const float *vr = vh0 + (uint64_t)(k0 + kk) * row_stride;
            acc0 += p * vr[d0];
            acc1 += p * vr[d1];
        }
        o0 += acc0;
        o1 += acc1;
        m = Mnew;
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;   /* graph:370 zero guard */
    oh[lane] = o0 * inv;
    oh[lane + 32u] = o1 * inv;
}

static inline int eg2vq_vis_attn(float *out, const float *q, const float *k, const float *v,
                                 uint32_t n_patches) {
    if (!out || !q || !k || !v || !n_patches) { return 0; }
    eg2vq_vis_attn_kernel<<<n_patches, EG2VQ_VIS_ATTN_WARPS * 32>>>(out, q, k, v, n_patches);
    return 1;
}

/* ---- the GEGLU FFN's activation ------------------------------------------ */

__global__ void eg2vq_vis_gelu_mul_kernel(float *__restrict__ gate,
                                          const float *__restrict__ up, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { gate[i] = eg2vq_vis_gelu_quick(gate[i]) * up[i]; }
}

static inline int eg2vq_vis_gelu_mul(float *gate, const float *up, uint64_t n) {
    if (!gate || !up || !n) { return 0; }
    eg2vq_vis_gelu_mul_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256>>>(gate, up, n);
    return 1;
}

/* ---- the pooler: 3x3 stride 3, then sqrt(n_embd) ------------------------- */

/* One block per pooled row; row q = x' + (grid_w/3)*y' (graph:394-412).  The
 * nine-term sum is accumulated in double, as the CPU leg accumulates it. */
__global__ void eg2vq_vis_pool_kernel(
        const float *__restrict__ patch, float *__restrict__ pooled,
        uint32_t grid_w, float scale) {
    const uint32_t q = blockIdx.x;
    const uint32_t out_x = grid_w / (uint32_t)EG2VQ_VIS_MERGE;
    const uint32_t x0 = (q % out_x) * (uint32_t)EG2VQ_VIS_MERGE;
    const uint32_t y0 = (q / out_x) * (uint32_t)EG2VQ_VIS_MERGE;
    for (uint32_t d = threadIdx.x; d < (uint32_t)EG2VQ_VIS_N_EMBD; d += blockDim.x) {
        double acc = 0.0;
        #pragma unroll
        for (uint32_t j = 0; j < (uint32_t)EG2VQ_VIS_MERGE; j++) {
            #pragma unroll
            for (uint32_t i = 0; i < (uint32_t)EG2VQ_VIS_MERGE; i++) {
                const uint32_t p = (x0 + i) + grid_w * (y0 + j);
                acc += (double)patch[(uint64_t)p * (uint64_t)EG2VQ_VIS_N_EMBD + d];
            }
        }
        pooled[(uint64_t)q * (uint64_t)EG2VQ_VIS_N_EMBD + d] =
            (float)(acc / (double)(EG2VQ_VIS_MERGE * EG2VQ_VIS_MERGE)) * scale;
    }
}

static inline int eg2vq_vis_pool(const float *patch, float *pooled, uint32_t grid_w,
                                 uint32_t n_soft) {
    if (!patch || !pooled || !grid_w || !n_soft) { return 0; }
    if ((grid_w % (uint32_t)EG2VQ_VIS_MERGE)) { return 0; }
    const float scale = sqrtf((float)EG2VQ_VIS_N_EMBD);   /* gemma4v.cpp:96 */
    eg2vq_vis_pool_kernel<<<n_soft, 256>>>(patch, pooled, grid_w, scale);
    return 1;
}
