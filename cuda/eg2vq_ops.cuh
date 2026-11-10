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
 *                     ggml ops.cpp:4224-4231); one warp per row since
 *                     performance unit 6, see "THE REDUCTION" below
 *   eg2vq_ops_rms_head  the same over each head vector (graph:264-277; the q and
 *                     k norms carry a weight, the v norm is weightless)
 *   eg2vq_ops_rope    NeoX rotation over the whole head, pairs (i, i + hd/2),
 *                     indexed by each row's position INSIDE its item
 *                     (graph:60-77; ops.cpp:6289-6301 and :6375-6391)
 *   eg2vq_ops_gelu_mul  gelu(gate) * up, tanh GELU at ggml's coefficients
 *                     (graph:39-41, :307-309, :322-326)
 *   eg2vq_ops_pool    mean pool over an ITEM's rows then L2 normalize
 *                     (graph:356-367; llama-graph.cpp:4049-4053;
 *                     common.cpp:1923-1941)
 *   eg2vq_ops_proj768 the output projection: per row, RMS by output_norm and a
 *                     dot with each of the 768 bf16 rows into that row's own
 *                     plane (graph:339-356)
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
#include <cuda_bf16.h>
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

/* The same gather with an IMAGE SPAN: the rows whose id is `image_id` take their
 * value from `soft` (UNSCALED) instead of the token table (scaled), and the span
 * must be contiguous, so the soft row for ids[t] is t - `span_start`.  The
 * unscaled raw-embedding input is llama.cpp's own rule (build_inp_embd scales the
 * token lookup and leaves an embedding entry alone: "important: do not normalize
 * weights for raw embeddings input (i.e. encoded image embeddings)",
 * src/models/gemma-embedding2.cpp:85-86), and the CPU leg's splice
 * (ds4_eg2vq_graph.inc, eg2vq_ref_embed_core) consumes the same rows in the same
 * order.  The launcher checks the span's shape; the kernel assumes it. */
__global__ void eg2vq_ops_embed_image_kernel(
        float *__restrict__ out, const int32_t *__restrict__ ids,
        const uint8_t *__restrict__ table, const float *__restrict__ soft,
        uint32_t n_ids, uint32_t n_embd, uint64_t n_vocab, float scale,
        int32_t image_id, uint32_t span_start) {
    const uint32_t t = blockIdx.x;
    if (t >= n_ids) { return; }
    float *row = out + (size_t)t * n_embd;
    if ((int32_t)ids[t] == image_id) {
        const float *src = soft + (size_t)(t - span_start) * n_embd;
        for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) { row[i] = src[i]; }
        return;
    }
    const uint32_t nb = n_embd / 32u;
    const uint64_t id = (uint64_t)(uint32_t)ids[t];
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

static inline int eg2vq_ops_embed_image(float *out, const int32_t *ids, const uint8_t *table,
                                        const float *soft, uint32_t n_ids, uint32_t n_embd,
                                        uint64_t n_vocab, int32_t image_id, uint32_t span_start,
                                        uint32_t n_soft) {
    if (!out || !ids || !table || !soft || !n_ids || !n_embd || (n_embd % 32u)) { return 0; }
    if (span_start + n_soft > n_ids) { return 0; }
    const float scale = sqrtf((float)n_embd);
    eg2vq_ops_embed_image_kernel<<<n_ids, 128>>>(out, ids, table, soft, n_ids, n_embd,
                                                 n_vocab, scale, image_id, span_start);
    return 1;
}

/* ---- RMS, row-wise and per head ------------------------------------------ */

/* THE REDUCTION (performance unit 6, 2026-11-10).
 *
 * The arithmetic is the graph's and does not change: out[r][i] = x[r][i] *
 * (1/sqrt(mean_sq + eps)) * (w ? w[i] : 1), the sum of squares accumulated in
 * double, divided in double, cast to float before the eps is added
 * (ds4_eg2vq_graph.inc:44-53; ggml/src/ggml-cpu/ops.cpp:4224-4231).
 *
 * What changed is HOW the row is reduced, because an nsys profile of one
 * 2,451-token batch showed the two norms costing 8.64 ms (121 launches) and
 * 5.44 ms (72 launches) in a 49.2 ms forward while an elementwise add of the
 * same 5 MB took 11 us -- i.e. the row kernel was ~6x slower than a plain pass
 * over the same bytes.  The old form is one 256-thread block per row, a
 * 2 KB shared double array and a nine-barrier tree: for a 512-wide row every
 * thread owns two values, so the barriers and the shared round-trip dominate
 * the two loads.
 *
 * The form below is one WARP per row: 32 lanes each read their own strided
 * slice into registers once, the sum is reduced with __shfl_xor_sync (five
 * shuffle-adds, no shared memory, no barrier), every lane computes the same
 * scale, and the registers already held are scaled on the way out -- so the row
 * is read once instead of twice.  VM is the per-lane value count, a compile-time
 * constant so the slice stays in registers (`#pragma unroll`), and the eight
 * warps of a block are eight independent rows.
 *
 * Only the summation TREE changes: the old kernel summed lanes 0..255 pairwise
 * over the row's (i, i+256) pairs, this one sums lane l's own strided slice then
 * folds the lanes by xor.  Both accumulate in double, so the two orders differ by
 * ~1e-16 relative -- and on every width this tower runs (512 for the hidden,
 * 256/512 per head, 768 and 64 in the vision tower) the f32 output is
 * BIT-IDENTICAL to the old kernel, which tests/test_eg2vq_ops.cu asserts and
 * tests/bench_eg2vq_ops reports.  Widths above 1024 values still take the old
 * block kernel (`eg2vq_ops_rms_block`), which is kept as the general fallback.
 */

/* The general form: one block per row, a double shared tree.  Kept as the
 * fallback for rows wider than the warp kernel's register budget. */
__global__ void eg2vq_ops_rms_block_kernel(
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

/* The general form's launcher (kept public so the gate can compare the two forms
 * on the same input; tests/test_eg2vq_ops.cu). */
static inline int eg2vq_ops_rms_block(float *out, const float *x, const float *w,
                                      uint32_t rows, uint32_t n, float eps) {
    eg2vq_ops_rms_block_kernel<<<rows, 256>>>(out, x, w, rows, n, eps);
    return 1;
}

/* One warp per row, the row's values held in registers between the sum and the
 * scale.  `nrow` is the number of vectors of `n` values: (rows) for the row
 * norm, (rows * heads) for the per-head one, whose vectors are the heads of a
 * row and therefore strided by `hd` rather than packed. */
template <uint32_t VM>
__global__ void eg2vq_ops_rms_warp_kernel(
        float *__restrict__ out, uint64_t out_stride,
        const float *__restrict__ x, uint64_t in_stride,
        const float *__restrict__ w, uint32_t nrow, uint32_t n, float eps) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t r = blockIdx.x * (blockDim.x >> 5u) + warp;
    if (r >= nrow) { return; }
    const float *xr = x + (size_t)r * in_stride;
    float *orow = out + (size_t)r * out_stride;
    float v[VM];
    double local = 0.0;
    #pragma unroll
    for (uint32_t j = 0; j < VM; j++) {
        const uint32_t i = lane + j * 32u;
        v[j] = (i < n) ? xr[i] : 0.0f;             /* no read past the vector */
        local += (double)v[j] * (double)v[j];
    }
    #pragma unroll
    for (uint32_t off = 16u; off; off >>= 1u) {    /* butterfly: every lane ends with the total */
        local += __shfl_xor_sync(0xffffffffu, local, off);
    }
    const float scale = 1.0f / sqrtf((float)(local / (double)n) + eps);
    #pragma unroll
    for (uint32_t j = 0; j < VM; j++) {
        const uint32_t i = lane + j * 32u;
        if (i < n) { orow[i] = v[j] * scale * (w ? w[i] : 1.0f); }
    }
}

/* 8 warps per block; nothing shared, so the block size only trades grid size
 * against the scheduler's tail. */
static inline int eg2vq_ops_rms_warp(float *out, uint64_t out_stride,
                                     const float *x, uint64_t in_stride,
                                     const float *w, uint32_t nrow, uint32_t n, float eps) {
    const uint32_t warps = 8u;
    const uint32_t blocks = (nrow + warps - 1u) / warps;
    const uint32_t vm = (n + 31u) / 32u;
    const dim3 grid(blocks), blk(warps * 32u);
    if (vm <= 2u) {
        eg2vq_ops_rms_warp_kernel<2><<<grid, blk>>>(out, out_stride, x, in_stride, w, nrow, n, eps);
    } else if (vm <= 4u) {
        eg2vq_ops_rms_warp_kernel<4><<<grid, blk>>>(out, out_stride, x, in_stride, w, nrow, n, eps);
    } else if (vm <= 8u) {
        eg2vq_ops_rms_warp_kernel<8><<<grid, blk>>>(out, out_stride, x, in_stride, w, nrow, n, eps);
    } else if (vm <= 16u) {
        eg2vq_ops_rms_warp_kernel<16><<<grid, blk>>>(out, out_stride, x, in_stride, w, nrow, n, eps);
    } else if (vm <= 32u) {
        eg2vq_ops_rms_warp_kernel<32><<<grid, blk>>>(out, out_stride, x, in_stride, w, nrow, n, eps);
    } else {
        return 0;
    }
    return 1;
}

/* The row norm (graph:47-53), rows of `n` values, optional weight. */
static inline int eg2vq_ops_rms(float *out, const float *x, const float *w,
                                uint32_t rows, uint32_t n, float eps) {
    if (!out || !x || !rows || !n) { return 0; }
    if (eg2vq_ops_rms_warp(out, n, x, n, w, rows, n, eps)) { return 1; }
    return eg2vq_ops_rms_block(out, x, w, rows, n, eps);
}

/* One vector per (row, head) pair; the weight vector is the head's own, so q and
 * k are normalized with their learned per-head weight and v without one. */
__global__ void eg2vq_ops_rms_head_block_kernel(
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

static inline int eg2vq_ops_rms_head_block(float *x, const float *w, uint32_t rows,
                                           uint32_t heads, uint32_t hd, float eps) {
    if (!x || !rows || !heads || !hd) { return 0; }
    eg2vq_ops_rms_head_block_kernel<<<rows * heads, 128>>>(x, w, rows, heads, hd, eps);
    return 1;
}

/* The shared kernel again, in place: one head vector of `hd` values per logical
 * row, consecutive heads of a row adjacent (stride `hd`). */
static inline int eg2vq_ops_rms_head(float *x, const float *w, uint32_t rows,
                                     uint32_t heads, uint32_t hd, float eps) {
    if (!x || !rows || !heads || !hd) { return 0; }
    const uint32_t nrow = rows * heads;            /* the tower's 24 x 4 x 2451 stays far below 2^32 */
    if (eg2vq_ops_rms_warp(x, hd, x, hd, w, nrow, hd, eps)) { return 1; }
    return eg2vq_ops_rms_head_block(x, w, rows, heads, hd, eps);
}

/* ---- NeoX RoPE over the whole head --------------------------------------- */

/* cos_tab and sin_tab are [n_pos][hd/2], indexed by the row's own POSITION
 * WITHIN ITS ITEM (a prefill has no cache, so position == row index inside the
 * item, graph:133-136 with pos = t).  `pos` carries that position per row: a
 * call packs one or more items into one contiguous block (unit 27), and the
 * reference's batched path likewise carries per-token pos[] with the sequences
 * packed contiguously (src/llama-batch.cpp:140-193) -- positions are
 * per-sequence, never counted from the block's start.  A single-item call passes
 * pos[t] = t, which is the pre-batch indexing exactly.  The tables have
 * DS4_EG2VQ_MAX_TOKENS positions (the item ceiling), because a position is
 * always < its item's length. */
__global__ void eg2vq_ops_rope_kernel(
        float *__restrict__ x, const float *__restrict__ cos_tab,
        const float *__restrict__ sin_tab, const int32_t *__restrict__ pos,
        uint32_t rows, uint32_t heads, uint32_t hd) {
    const uint32_t r = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (r >= rows || h >= heads) { return; }
    const uint32_t half = hd / 2u;
    const uint32_t p = (uint32_t)pos[r];
    float *v = x + ((size_t)r * heads + h) * hd;
    const float *c = cos_tab + (size_t)p * half;
    const float *s = sin_tab + (size_t)p * half;
    for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * c[i] - x1 * s[i];
        v[i + half] = x0 * s[i] + x1 * c[i];
    }
}

static inline int eg2vq_ops_rope(float *x, const float *cos_tab, const float *sin_tab,
                                 const int32_t *pos, uint32_t rows, uint32_t heads, uint32_t hd) {
    if (!x || !cos_tab || !sin_tab || !pos || !rows || !heads || !hd || (hd % 2u)) { return 0; }
    dim3 grid(rows, heads);
    eg2vq_ops_rope_kernel<<<grid, 128>>>(x, cos_tab, sin_tab, pos, rows, heads, hd);
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

/* One block per ROW: e = RMS(h[row], norm_w) is computed once per block, then
 * each thread owns a few of the 768 outputs and dots the bf16 weight row with e.
 * The result goes to this row's OWN acc row (acc + row * n_out), so every
 * element's accumulation happens inside one thread of one block and an item's
 * numbers cannot depend on which other items share the call (unit 27; before the
 * batch this kernel atomicAdd'ed every row's contribution into one 768-float
 * accumulator, whose sum order was the atomics' own).  The pool below then reads
 * those rows in order, which is the CPU reference's own sum order
 * (ds4_eg2vq_graph.inc:363-365: `out768[o] += ref_row_dot(...)` per token). */
/* The original form, kept for the A/B and as the fallback for any shape the
 * tensor-core tile cannot cover. */
__global__ void eg2vq_ops_proj768_block_kernel(
        float *__restrict__ acc, const float *__restrict__ h,
        const uint16_t *__restrict__ w, const float *__restrict__ norm_w,
        uint32_t n_embd, uint32_t n_out, float eps) {
    const uint32_t t = blockIdx.x;
    extern __shared__ float e_s[];
    __shared__ double ssum[256];
    const float *ht = h + (size_t)t * n_embd;
    float *accrow = acc + (size_t)t * n_out;
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
        accrow[o] = a;
    }
}

/* ------------------------------------------------------------------------- *
 * The same projection on the tensor cores (performance unit 7, 2026-11-10).
 *
 * WHY, measured before the change: an nsys profile of the same 8-item /
 * 2,451-token batch read 7.13 ms in ONE launch of the kernel above, 2,451 rows
 * x 512 x 768 = 1.93 GFLOP at 271 GFLOP/s, i.e. one hundredth of what this card
 * (RTX 4070 SUPER, sm_89) does on the payload GEMM. The kernel is not
 * launch-bound (one launch) and not bandwidth-bound (the 786 KB weight matrix is
 * L2-resident); it is scalar: one block of 256 threads per row, each thread
 * walking 512 bf16 weights per output with a widening memcpy in the inner loop.
 *
 * WHAT CHANGES: the dot moves to mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 * (sm_80+; the bf16 sibling of the f16 instruction this project's payload GEMM
 * and attention lane already use, cuda/eg2vq_vq.cuh:350-359 and
 * cuda/eg2vq_attn.cuh:474-477). The B operand is the weight matrix EXACTLY as
 * the file stores it: bf16, so the mma's widening is exact and B needs no
 * conversion at all -- W is [n_out][n_embd] row-major, which is the column-major
 * [k][n] the instruction wants, one uint32 load per fragment register.
 *
 * THE A OPERAND, AND WHY IT IS SPLIT (measured, /tmp/prec_proj.cu at this
 * boundary, reported in the unit's record section 2): the projection's A is the
 * f32 output_norm row, and an mma operand cannot be f32. Rounding it to one bf16
 * moves the pooled, L2-normalized 768-vector by rel_rms 1.65e-03 and rounding it
 * to f16/tf32 by 2.00e-04, both far outside the 1e-5 this kernel's own gate
 * (tests/test_eg2vq_ops.cu, "proj768 + mean pool + L2") has declared since phase
 * 7. Splitting A into hi = bf16(e) and lo = bf16(e - hi) and accumulating BOTH
 * products into the same f32 accumulators carries e to 16 significand bits and
 * puts the pooled vector at rel_rms 2.50e-06, inside the declared bound, at 2x
 * the mma work. The split is the only operand change; the k-reduction's order
 * is the tensor core's, which the unit measures against gate 7's declared bound
 * (min cos >= 0.999999, rel_rms <= 1e-03) rather than assumes.
 *
 * THE SHAPE OF THE WORK: a block owns TM = 64 tokens x TN = 64 outputs and runs
 * all K one 64-wide chunk at a time; 8 warps are 4 row groups x 2 column groups,
 * each warp 16 rows x 32 outputs (four n8 tiles). The row's own output_norm RMS
 * is the prologue (4 threads per row, double sum, the same reduction the block
 * kernel ran with 256), and the epilogue writes each row to ITS OWN plane
 * (acc + row * n_out), which is unit 27's contract: an item's numbers cannot
 * depend on which items share the call. Padding rows compute zeros and store
 * nothing.
 */
constexpr uint32_t EG2VQ_P768_TM = 64u;      /* tokens per block */
constexpr uint32_t EG2VQ_P768_TN = 64u;      /* outputs per block */
constexpr uint32_t EG2VQ_P768_BK = 64u;      /* k per shared-memory chunk */
constexpr uint32_t EG2VQ_P768_THREADS = 256u;
constexpr uint32_t EG2VQ_P768_PAD = 8u;      /* shared row pad: 4 banks a row */

/* m16n8k16, A row-major bf16, B column-major bf16, f32 accumulate. The bf16
 * form of the instruction cuda/eg2vq_vq.cuh:350-359 uses for its payload GEMM. */
__device__ __forceinline__ void eg2vq_ops_mma_bf16(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* The split pair as the two bf16 the A fragment wants: .x low, .y high. */
__device__ __forceinline__ uint32_t eg2vq_ops_bf16_pair(float lo, float hi) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&b);
}

__global__ void __launch_bounds__(EG2VQ_P768_THREADS, 2) eg2vq_ops_proj768_mma_kernel(
        float *__restrict__ acc, const float *__restrict__ h,
        const uint16_t *__restrict__ w, const float *__restrict__ norm_w,
        uint32_t n_embd, uint32_t n_out, float eps, uint32_t n_tokens) {
    constexpr uint32_t TM = EG2VQ_P768_TM, TN = EG2VQ_P768_TN, BK = EG2VQ_P768_BK;
    constexpr uint32_t P = BK + EG2VQ_P768_PAD;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, q = lane & 3u;
    const uint32_t t0 = blockIdx.x * TM, o0 = blockIdx.y * TN;
    const uint32_t trow = tid >> 2u, tpart = tid & 3u;

    __shared__ float scale_s[TM];
    __shared__ __align__(16) uint16_t as_hi[TM * P];
    __shared__ __align__(16) uint16_t as_lo[TM * P];
    __shared__ __align__(16) uint16_t ws[TN * P];

    /* The row's output_norm RMS (graph:339-345, graph:47-53): double sum, the
     * eps added before the cast, the reciprocal root in f32 -- the block form's
     * own constants, four threads per row instead of 256 per row. */
    const uint32_t rt = t0 + trow;
    double ss = 0.0;
    if (rt < n_tokens) {
        const float *hr = h + (size_t)rt * n_embd;
        for (uint32_t i = tpart; i < n_embd; i += 4u) {
            const double v = (double)hr[i];
            ss += v * v;
        }
    }
    ss += __shfl_down_sync(0xffffffffu, ss, 1);
    ss += __shfl_down_sync(0xffffffffu, ss, 2);
    if (tpart == 0u) {
        scale_s[trow] = (rt < n_tokens)
                ? 1.0f / sqrtf((float)(ss / (double)n_embd) + eps) : 0.0f;
    }
    __syncthreads();

    /* The warp's own 16 rows x 32 columns of the output. */
    const uint32_t m0 = (warp >> 1u) * 16u;
    const uint32_t n0 = (warp & 1u) * 32u;
    float ac[4][4] = {};
    ac[0][0] = ac[0][1] = ac[0][2] = ac[0][3] = 0.0f;
    ac[1][0] = ac[1][1] = ac[1][2] = ac[1][3] = 0.0f;
    ac[2][0] = ac[2][1] = ac[2][2] = ac[2][3] = 0.0f;
    ac[3][0] = ac[3][1] = ac[3][2] = ac[3][3] = 0.0f;

    for (uint32_t kc = 0; kc < n_embd / BK; kc++) {
        const uint32_t cbase = kc * BK + tpart * 16u;
        /* A: this thread's 16 columns of its row, e = h * scale * norm_w in f32
         * (graph:346-348), split into the exact bf16 hi and the residue lo. */
        uint16_t *ah = as_hi + trow * P + tpart * 16u;
        uint16_t *al = as_lo + trow * P + tpart * 16u;
        if (rt < n_tokens) {
            const float *hr = h + (size_t)rt * n_embd + cbase;
            const float sc = scale_s[trow];
            #pragma unroll
            for (uint32_t c = 0; c < 16u; c += 8u) {
                const float4 v0 = *(const float4 *)(hr + c);
                const float4 v1 = *(const float4 *)(hr + c + 4u);
                const float e0 = v0.x * sc * norm_w[cbase + c + 0u];
                const float e1 = v0.y * sc * norm_w[cbase + c + 1u];
                const float e2 = v0.z * sc * norm_w[cbase + c + 2u];
                const float e3 = v0.w * sc * norm_w[cbase + c + 3u];
                const float e4 = v1.x * sc * norm_w[cbase + c + 4u];
                const float e5 = v1.y * sc * norm_w[cbase + c + 5u];
                const float e6 = v1.z * sc * norm_w[cbase + c + 6u];
                const float e7 = v1.w * sc * norm_w[cbase + c + 7u];
                const float ev[8] = { e0, e1, e2, e3, e4, e5, e6, e7 };
                uint32_t hi4[4], lo4[4];
                #pragma unroll
                for (uint32_t p = 0; p < 4u; p++) {
                    const __nv_bfloat162 hb = __floats2bfloat162_rn(ev[2u * p], ev[2u * p + 1u]);
                    const float h0 = __bfloat162float(hb.x), h1 = __bfloat162float(hb.y);
                    hi4[p] = *reinterpret_cast<const uint32_t *>(&hb);
                    const __nv_bfloat162 lb =
                        __floats2bfloat162_rn(ev[2u * p] - h0, ev[2u * p + 1u] - h1);
                    lo4[p] = *reinterpret_cast<const uint32_t *>(&lb);
                }
                *(uint4 *)(ah + c) = make_uint4(hi4[0], hi4[1], hi4[2], hi4[3]);
                *(uint4 *)(al + c) = make_uint4(lo4[0], lo4[1], lo4[2], lo4[3]);
            }
        } else {
            *(uint4 *)ah = make_uint4(0u, 0u, 0u, 0u);
            *(uint4 *)(ah + 8u) = make_uint4(0u, 0u, 0u, 0u);
            *(uint4 *)al = make_uint4(0u, 0u, 0u, 0u);
            *(uint4 *)(al + 8u) = make_uint4(0u, 0u, 0u, 0u);
        }
        /* B: this thread's 16 columns of its output row, the file's bf16 as-is. */
        const uint32_t o = o0 + trow;
        uint16_t *wb = ws + trow * P + tpart * 16u;
        if (o < n_out) {
            const uint4 *src = (const uint4 *)(w + (size_t)o * n_embd + cbase);
            *(uint4 *)wb = src[0];
            *(uint4 *)(wb + 8u) = src[1];
        } else {
            *(uint4 *)wb = make_uint4(0u, 0u, 0u, 0u);
            *(uint4 *)(wb + 8u) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        /* A fragment order [row g, row g + 8] x [col, col + 8], the order
         * cuda/eg2vq_attn.cuh:554-575 builds for its own Q tile; B is one
         * uint32 per (k pair, output) at the same [col, col + 8] offsets. */
        #pragma unroll
        for (uint32_t kk = 0; kk < BK / 16u; kk++) {
            const uint32_t col = kk * 16u + 2u * q;
            uint32_t ahi[4], alo[4];
            ahi[0] = *(const uint32_t *)(as_hi + (m0 + g) * P + col);
            ahi[1] = *(const uint32_t *)(as_hi + (m0 + g + 8u) * P + col);
            ahi[2] = *(const uint32_t *)(as_hi + (m0 + g) * P + col + 8u);
            ahi[3] = *(const uint32_t *)(as_hi + (m0 + g + 8u) * P + col + 8u);
            alo[0] = *(const uint32_t *)(as_lo + (m0 + g) * P + col);
            alo[1] = *(const uint32_t *)(as_lo + (m0 + g + 8u) * P + col);
            alo[2] = *(const uint32_t *)(as_lo + (m0 + g) * P + col + 8u);
            alo[3] = *(const uint32_t *)(as_lo + (m0 + g + 8u) * P + col + 8u);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const uint16_t *wp = ws + (n0 + j * 8u + g) * P + col;
                const uint32_t b0 = *(const uint32_t *)wp;
                const uint32_t b1 = *(const uint32_t *)(wp + 8u);
                eg2vq_ops_mma_bf16(ac[j], ahi, b0, b1);
                eg2vq_ops_mma_bf16(ac[j], alo, b0, b1);
            }
        }
        __syncthreads();
    }

    /* The row's own plane (unit 27): C fragment c0,c1 = row g's columns 2q,2q+1,
     * c2,c3 = row g + 8's (cuda/eg2vq_attn.cuh:611-633 reads the same pair). */
    const uint32_t rlo = t0 + m0 + g, rhi = rlo + 8u;
    #pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
        const uint32_t ocol = o0 + n0 + j * 8u + 2u * q;
        if (ocol + 1u < n_out) {
            if (rlo < n_tokens) {
                acc[(size_t)rlo * n_out + ocol] = ac[j][0];
                acc[(size_t)rlo * n_out + ocol + 1u] = ac[j][1];
            }
            if (rhi < n_tokens) {
                acc[(size_t)rhi * n_out + ocol] = ac[j][2];
                acc[(size_t)rhi * n_out + ocol + 1u] = ac[j][3];
            }
        }
    }
}

/* One block per ITEM: mean over the ITEM's rows, then L2 normalize with the
 * graph's zero guard.  `acc` is the per-row projection plane and `off` the
 * batch's item spans (off[i]..off[i+1] rows), so an item's mean is its own row
 * count -- the same denominator the single-item path used (1/n_tok,
 * ds4_eg2vq_graph.inc:370-377, src/llama-graph.cpp:4049-4053 with weight 1/n),
 * and the same f32 accumulation order over its rows, which is what makes a
 * batched item's vector the single-item vector. */
__global__ void eg2vq_ops_pool_item_kernel(float *__restrict__ out, const float *__restrict__ acc,
                                           uint32_t n_out, const uint32_t *__restrict__ off,
                                           uint32_t n_items) {
    const uint32_t i = blockIdx.x;
    if (i >= n_items) { return; }
    const uint32_t r0 = off[i], r1 = off[i + 1u];
    const uint32_t n_tok = r1 - r0;
    const float mean = 1.0f / (float)n_tok;
    float *dst = out + (size_t)i * n_out;
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t r = r0; r < r1; r++) { sum += acc[(size_t)r * n_out + o]; }
        dst[o] = sum * mean;
    }
    __syncthreads();
    __shared__ double ssum[256];
    double local = 0.0;
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) {
        const double t = (double)dst[o];
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
    for (uint32_t o = threadIdx.x; o < n_out; o += blockDim.x) { dst[o] *= inv; }
}

/* The projection + pool of a whole call: `n_tokens` rows (the batch's items
 * packed contiguously), `n_items` spans in `off` (off[0] == 0, off[n_items] ==
 * n_tokens, every span non-empty -- the core refuses anything else), and
 * `out768` takes n_items * n_out floats, item i's vector at i * n_out.
 *
 * `form` selects the projection kernel: 0 is the block-per-row form, 1 the
 * tensor-core form, and -1 the shipped dispatch (the tensor-core form when the
 * shape fits its tile -- n_embd a multiple of 64 and n_out a multiple of 8, which
 * every text-tower shape is -- and the block form otherwise). The pool stage is
 * the same kernel in every case. */
static inline int eg2vq_ops_proj768_pool_form(float *out768, float *acc, const float *h,
                                              const uint16_t *w, const float *norm_w,
                                              const uint32_t *off, uint32_t n_items,
                                              uint32_t n_tokens, uint32_t n_embd,
                                              uint32_t n_out, float eps, int form) {
    if (!out768 || !acc || !h || !w || !norm_w || !off || !n_items || !n_tokens ||
        !n_embd || !n_out) { return 0; }
    const int mma_ok = (n_embd % EG2VQ_P768_BK == 0u) && (n_out % 8u == 0u);
    if (form == 1 || (form < 0 && mma_ok)) {
        if (!mma_ok) { return 0; }
        const dim3 grid((n_tokens + EG2VQ_P768_TM - 1u) / EG2VQ_P768_TM,
                        (n_out + EG2VQ_P768_TN - 1u) / EG2VQ_P768_TN);
        eg2vq_ops_proj768_mma_kernel<<<grid, EG2VQ_P768_THREADS>>>(
            acc, h, w, norm_w, n_embd, n_out, eps, n_tokens);
    } else {
        eg2vq_ops_proj768_block_kernel<<<n_tokens, 256, (size_t)n_embd * sizeof(float)>>>(
            acc, h, w, norm_w, n_embd, n_out, eps);
    }
    eg2vq_ops_pool_item_kernel<<<n_items, 256>>>(out768, acc, n_out, off, n_items);
    return 1;
}

static inline int eg2vq_ops_proj768_pool_batch(float *out768, float *acc, const float *h,
                                               const uint16_t *w, const float *norm_w,
                                               const uint32_t *off, uint32_t n_items,
                                               uint32_t n_tokens, uint32_t n_embd,
                                               uint32_t n_out, float eps) {
    return eg2vq_ops_proj768_pool_form(out768, acc, h, w, norm_w, off, n_items,
                                       n_tokens, n_embd, n_out, eps, -1);
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
