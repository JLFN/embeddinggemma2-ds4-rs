/* eg2vq_ops.cuh — the gemma_vq text tower's remaining device primitives
 * (embeddinggemma2-handoff.md PHASE 7, row P3.3 step 1).
 *
 * Every kernel here mirrors one expression of the CPU reference forward
 * (ds4-rs/ds4_eg2vq_graph.inc), which is this port's own gated oracle: same
 * expression, same constant, same accumulation width where it is observable.
 * The citations are to that file, and through it to the ggml/llama.cpp source it
 * transcribes.
 *
 *   eg2vq_ops_embed   token embedding row, scaled by sqrt(n_embd)
 *                     (graph:196-208; gemma-embedding2.cpp:86)
 *   eg2vq_ops_rms     RMS over a row with an optional weight, eps inside the
 *                     square root, the sum accumulated in double (graph:47-53,
 *                     ggml ops.cpp:4224-4231)
 *   eg2vq_ops_rms_head  the same over each head vector (graph:264-277; the q and
 *                     k norms carry a weight, the v norm is weightless)
 *   eg2vq_ops_rope    NeoX rotation over the whole head, pairs (i, i + hd/2)
 *                     (graph:60-77; ops.cpp:6289-6301 and :6375-6391)
 *   eg2vq_ops_gelu_mul  gelu(gate) * up, tanh GELU at ggml's coefficients
 *                     (graph:39-41, :307-309, :322-326)
 *   eg2vq_ops_pool    mean pool over the token axis then L2 normalize
 *                     (graph:356-367; llama-graph.cpp:4049-4053;
 *                     common.cpp:1923-1941)
 *   eg2vq_ops_proj768 the output projection: per token, RMS by output_norm and a
 *                     dot with each of the 768 bf16 rows, accumulated across the
 *                     item (graph:339-356)
 *
 * Two deliberate choices, both stated because they are deviations from a literal
 * transcription of the graph's *code* and not of its arithmetic:
 *
 *  - The RoPE angles come from a table the caller builds, not from sinf/cosf in
 *    the kernel.  This build compiles with --use_fast_math, so sinf/cosf become
 *    the SFU approximations, whose argument reduction in fp32 loses ~8e-05 rad at
 *    the largest angle this tower produces (position 1414, the slowest pair) —
 *    a rotary table built and reduced in double precision does not.  The angle
 *    itself is accumulated exactly as the graph accumulates it (theta *= theta_scale
 *    in f32), so the table reproduces the reference's angles bit for bit and only
 *    the sine and cosine are computed more accurately.
 *  - Weight vectors (the norms, the layer scalars) arrive as f32, because that is
 *    the form the CPU path reads them into (graph:57-59, ds4_ref_row) and their
 *    count is a few hundred floats per layer.
 *
 * P3.3 step 2 adds the three primitives the tower sequence still needed (the
 * adapter is ds4_eg2vq_gpu.cuh): the whole-row scale and add the graph does in
 * place at :299-300, :313-318, :329-333 and :335, and a strided gelu-mul for the
 * PLE block, whose second operand is a 512-wide slice of the [n_tok][12288]
 * per-layer input at a per-layer offset (graph:321-327).  Each is one
 * expression of the same reference, gated in tests/test_eg2vq_ops.cu beside the
 * step-1 ops.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

/* tanh GELU, ggml's coefficients (graph:35-41; ggml-cpu/vec.h:963-969). */
__device__ __forceinline__ static float eg2vq_ops_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

/* IEEE-754 binary16 -> f32, by the definition of the format: a subnormal is
 * m * 2^-24, a normal is 2^(e-15) * (1 + m/1024).  This is the reading the
 * kernels get from __half2float and the reading the file's own readers use
 * (Rust's f16 decode, and the writer's go1b_fp16_to_fp32); it is deliberately
 * NOT ds4vq_f16 (ds41_vq_fmt.h:38-43), whose subnormal branch shifts one step
 * too far and returns exactly half for every subnormal input -- measured on the
 * real artifact at 8,893 of 7,553,024 codebook entries (docs/eg2vq-device-gate7.txt
 * section 2a), so the device path follows the file and the CPU path's own
 * divergence is a known, measured quantity in the parity gate, not a surprise in
 * it.  Gate: every one of the 65,536 patterns against the closed-form value in
 * double (tests/test_eg2vq_ops.cu). */
__host__ __device__ __forceinline__ static float eg2vq_ops_half_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;                                  /* +0 / -0 */
        } else {
            uint32_t lead = 9u;                      /* m != 0, so this stops at its highest set bit */
            while (!((m >> lead) & 1u)) { lead--; }
            const uint32_t frac = (m - (1u << lead)) << (23u - lead);
            f = s | ((uint32_t)(127 + (int)lead - 24) << 23) | frac;
        }
    } else if (e == 31u) {
        f = s | 0x7F800000u | (m << 13);            /* inf / NaN (payload kept) */
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

/* ---- token embedding: one q8_0 row, scaled by sqrt(n_embd) ---------------- */

/* q8_0 as this tree stores it: blocks of 32, each an f16 scale then 32 int8
 * (ds4.c's own reader; the artifact's token table is one such table of
 * [n_embd, n_vocab]). */
__global__ void eg2vq_ops_embed_kernel(
        float *__restrict__ out, const int32_t *__restrict__ ids,
        const uint8_t *__restrict__ table, uint32_t n_ids, uint32_t n_embd,
        uint64_t n_vocab, float scale) {
    const uint32_t t = blockIdx.x;
    if (t >= n_ids) { return; }
    const uint32_t nb = n_embd / 32u;
    const uint64_t id = (uint64_t)(uint32_t)ids[t];
    float *row = out + (size_t)t * n_embd;
    if (id >= n_vocab) {                     /* out-of-range id: the graph dies; the kernel zeroes */
        for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) { row[i] = 0.0f; }
        return;
    }
    const uint8_t *src = table + id * (uint64_t)nb * 34u;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        const uint8_t *blk = src + (size_t)b * 34u;
        uint16_t h;
        memcpy(&h, blk, 2);
        __half hh;
        memcpy(&hh, &h, 2);
        const float ds = __half2float(hh) * scale;
        const int8_t *q = (const int8_t *)(blk + 2);
        for (uint32_t i = 0; i < 32u; i++) { row[b * 32u + i] = ds * (float)q[i]; }
    }
}

static inline int eg2vq_ops_embed(float *out, const int32_t *ids, const uint8_t *table,
                                  uint32_t n_ids, uint32_t n_embd, uint64_t n_vocab) {
    if (!out || !ids || !table || !n_ids || !n_embd || (n_embd % 32u)) { return 0; }
    const float scale = sqrtf((float)n_embd);   /* graph:196-199: only scale the arch applies */
    eg2vq_ops_embed_kernel<<<n_ids, 128>>>(out, ids, table, n_ids, n_embd, n_vocab, scale);
    return 1;
}

/* ---- RMS, row-wise and per head ------------------------------------------ */

/* out[r][i] = x[r][i] * (1/sqrt(mean_sq + eps)) * (w ? w[i] : 1); the mean over
 * n and the cast are done exactly as the graph does them (double sum, divided in
 * double, cast to float before the eps is added). */
__global__ void eg2vq_ops_rms_kernel(
        float *__restrict__ out, const float *__restrict__ x,
        const float *__restrict__ w, uint32_t rows, uint32_t n, float eps) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) { return; }
    __shared__ double ssum[256];
    const float *xr = x + (size_t)r * n;
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = (double)xr[i];
        local += v * v;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) { ssum[threadIdx.x] += ssum[threadIdx.x + s]; }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf((float)(ssum[0] / (double)n) + eps);
    float *orow = out + (size_t)r * n;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * (w ? w[i] : 1.0f);
    }
}

static inline int eg2vq_ops_rms(float *out, const float *x, const float *w,
                                uint32_t rows, uint32_t n, float eps) {
    if (!out || !x || !rows || !n) { return 0; }
    eg2vq_ops_rms_kernel<<<rows, 256>>>(out, x, w, rows, n, eps);
    return 1;
}

/* One block per (row, head) pair; the weight vector is the head's own, so q and
 * k are normalized with their learned per-head weight and v without one. */
__global__ void eg2vq_ops_rms_head_kernel(
        float *__restrict__ x, const float *__restrict__ w,
        uint32_t rows, uint32_t heads, uint32_t hd, float eps) {
    const uint32_t idx = blockIdx.x;
    const uint32_t r = idx / heads, h = idx % heads;
    if (r >= rows) { return; }
    float *v = x + ((size_t)r * heads + h) * hd;
    __shared__ double ssum[128];
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
        const double t = (double)v[i];
        local += t * t;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) { ssum[threadIdx.x] += ssum[threadIdx.x + s]; }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf((float)(ssum[0] / (double)hd) + eps);
    for (uint32_t i = threadIdx.x; i < hd; i += blockDim.x) {
        v[i] = v[i] * scale * (w ? w[i] : 1.0f);
    }
}

static inline int eg2vq_ops_rms_head(float *x, const float *w, uint32_t rows,
                                     uint32_t heads, uint32_t hd, float eps) {
    if (!x || !rows || !heads || !hd) { return 0; }
    eg2vq_ops_rms_head_kernel<<<rows * heads, 128>>>(x, w, rows, heads, hd, eps);
    return 1;
}

/* ---- NeoX RoPE over the whole head --------------------------------------- */

/* cos_tab and sin_tab are [n_pos][hd/2], indexed by the row's own position (a
 * prefill has no cache, so position == row index, graph:133-136 with pos = t). */
__global__ void eg2vq_ops_rope_kernel(
        float *__restrict__ x, const float *__restrict__ cos_tab,
        const float *__restrict__ sin_tab, uint32_t rows, uint32_t heads, uint32_t hd) {
    const uint32_t r = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (r >= rows || h >= heads) { return; }
    const uint32_t half = hd / 2u;
    float *v = x + ((size_t)r * heads + h) * hd;
    const float *c = cos_tab + (size_t)r * half;
    const float *s = sin_tab + (size_t)r * half;
    for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * c[i] - x1 * s[i];
        v[i + half] = x0 * s[i] + x1 * c[i];
    }
}

static inline int eg2vq_ops_rope(float *x, const float *cos_tab, const float *sin_tab,
                                 uint32_t rows, uint32_t heads, uint32_t hd) {
    if (!x || !cos_tab || !sin_tab || !rows || !heads || !hd || (hd % 2u)) { return 0; }
    dim3 grid(rows, heads);
    eg2vq_ops_rope_kernel<<<grid, 128>>>(x, cos_tab, sin_tab, rows, heads, hd);
    return 1;
}

/* ---- gelu(gate) * up ----------------------------------------------------- */

__global__ void eg2vq_ops_gelu_mul_kernel(float *__restrict__ gate, const float *__restrict__ up, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { gate[i] = eg2vq_ops_gelu(gate[i]) * up[i]; }
}

static inline int eg2vq_ops_gelu_mul(float *gate, const float *up, uint64_t n) {
    if (!gate || !up || !n) { return 0; }
    const uint64_t blocks = (n + 255u) / 256u;
    eg2vq_ops_gelu_mul_kernel<<<(uint32_t)blocks, 256>>>(gate, up, n);
    return 1;
}

/* ---- the 768 projection with output_norm, and the mean pool + L2 ---------- */

/* One block per token: e = RMS(h[t], norm_w) is computed once per block, then
 * each thread owns a few of the 768 outputs and dots the bf16 weight row with e.
 * The result is accumulated (not overwritten) so the mean pool can follow without
 * a second buffer (graph:344-356). */
__global__ void eg2vq_ops_proj768_kernel(
        float *__restrict__ acc, const float *__restrict__ h,
        const uint16_t *__restrict__ w, const float *__restrict__ norm_w,
        uint32_t n_embd, uint32_t n_out, float eps) {
    const uint32_t t = blockIdx.x;
    extern __shared__ float e_s[];
    __shared__ double ssum[256];
    const float *ht = h + (size_t)t * n_embd;
    double local = 0.0;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const double v = (double)ht[i];
        local += v * v;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) { ssum[threadIdx.x] += ssum[threadIdx.x + s]; }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf((float)(ssum[0] / (double)n_embd) + eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        e_s[i] = ht[i] * scale * norm_w[i];       /* graph:341, the output_norm row */
    }
    __syncthreads();
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) {
        const uint16_t *wr = w + (size_t)o * n_embd;
        float a = 0.0f;
        for (uint32_t i = 0; i < n_embd; i++) {   /* bf16 row widened, as the CPU binder does it */
            const uint32_t u = (uint32_t)wr[i] << 16;
            float v;
            memcpy(&v, &u, 4);
            a += v * e_s[i];
        }
        atomicAdd(&acc[o], a);
    }
}

/* mean over the item's tokens, then L2 normalize with the graph's zero guard */
__global__ void eg2vq_ops_pool_l2_kernel(float *__restrict__ v, uint32_t n_out, uint32_t n_tok) {
    const float mean = 1.0f / (float)n_tok;
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) { v[o] *= mean; }
    __syncthreads();
    __shared__ double ssum[256];
    double local = 0.0;
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) {
        const double t = (double)v[o];
        local += t * t;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) { ssum[threadIdx.x] += ssum[threadIdx.x + s]; }
        __syncthreads();
    }
    const double norm = sqrt(ssum[0]);
    const float inv = norm > 0.0 ? (float)(1.0 / norm) : 0.0f;
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) { v[o] *= inv; }
}

static inline int eg2vq_ops_proj768_pool(float *out768, const float *h, const uint16_t *w,
                                         const float *norm_w, uint32_t n_tok, uint32_t n_embd,
                                         uint32_t n_out, float eps) {
    if (!out768 || !h || !w || !norm_w || !n_tok || !n_embd || !n_out) { return 0; }
    cudaMemsetAsync(out768, 0, (size_t)n_out * sizeof(float));
    eg2vq_ops_proj768_kernel<<<n_tok, 256, (size_t)n_embd * sizeof(float)>>>(
        out768, h, w, norm_w, n_embd, n_out, eps);
    eg2vq_ops_pool_l2_kernel<<<1, 256>>>(out768, n_out, n_tok);
    return 1;
}

/* ---- whole-row scale and add, and the strided PLE gelu-mul ----------------- */

/* x[i] *= s, in place: graph:335 (`h[i] *= scalar[0]`, the layer scalar) and
 * graph:221 (`pt[r] *= inv`, the per-layer input's 1/sqrt(n_embd)). */
__global__ void eg2vq_ops_scale_kernel(float *__restrict__ x, float s, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] *= s; }
}

static inline int eg2vq_ops_scale(float *x, float s, uint64_t n) {
    if (!x || !n) { return 0; }
    eg2vq_ops_scale_kernel<<<(uint32_t)((n + 255u) / 256u), 256>>>(x, s, n);
    return 1;
}

/* dst[i] += src[i]: the three residual adds that follow a norm
 * (graph:299-300, :313-318, :329-333). */
__global__ void eg2vq_ops_add_kernel(float *__restrict__ dst, const float *__restrict__ src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] += src[i]; }
}

static inline int eg2vq_ops_add(float *dst, const float *src, uint64_t n) {
    if (!dst || !src || !n) { return 0; }
    eg2vq_ops_add_kernel<<<(uint32_t)((n + 255u) / 256u), 256>>>(dst, src, n);
    return 1;
}

/* gate[r][j] = gelu(gate[r][j]) * up[r * up_stride + up_off + j], the PLE
 * block's gate (graph:321-327): `up` is the per-layer input [n_tok][12288] and
 * the layer's slice starts 512 * il floats into each token's row. */
__global__ void eg2vq_ops_gelu_mul_strided_kernel(float *__restrict__ gate, const float *__restrict__ up,
        uint32_t width, uint64_t up_stride, uint64_t up_off, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const uint64_t r = i / width, j = i - r * width;
    gate[i] = eg2vq_ops_gelu(gate[i]) * up[r * up_stride + up_off + j];
}

static inline int eg2vq_ops_gelu_mul_strided(float *gate, const float *up, uint32_t rows,
                                             uint32_t width, uint64_t up_stride, uint64_t up_off) {
    if (!gate || !up || !rows || !width || up_stride < up_off + width) { return 0; }
    const uint64_t n = (uint64_t)rows * width;
    eg2vq_ops_gelu_mul_strided_kernel<<<(uint32_t)((n + 255u) / 256u), 256>>>(
        gate, up, width, up_stride, up_off, n);
    return 1;
}
