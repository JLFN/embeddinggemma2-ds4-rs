/* eg2vq_attn.cuh — the gemma_vq text tower's attention on the device
 * (embeddinggemma2-handoff.md PHASE 7, phase-doc row P3.2).
 *
 * Authority, read first-hand: the CPU reference forward's attention
 * (ds4-rs/ds4_eg2vq_graph.inc:107-166, `eg2vq_attn_worker`), which is this
 * port's own gated oracle (phase 4 put it at 768 mean cosine 0.999996809 against
 * the frozen anchor vectors), and the reference declarations it cites:
 *
 *   - scale is exactly 1.0f: gemma-embedding2.cpp:8 `f_attention_scale = 1.0f`,
 *     applied at :139-141.  Not 1/sqrt(head_dim), and not learned.
 *   - the mask is SYMMETRIC with half-window 512: gemma-embedding2.cpp:4 sets
 *     LLAMA_SWA_TYPE_SYMMETRIC, masked where pos_diff < -n_swa/2 or
 *     > n_swa/2 (src/llama-hparams.h:524-533), and the file's sliding_window is
 *     1024 (the converter doubles the checkpoint's one-sided 512).
 *   - no causal mask (:7 `causal_attn = false`): every query sees every key in
 *     its window, on both sides.
 *   - softmax subtracts the row max before exp and normalizes by the sum of the
 *     exponentials; a zero denominator yields zero (graph:148-165).
 *   - query head h reads KV head h / (n_head / n_head_kv) (ops.cpp:8981, :9044:
 *     the MHA path repeats KV heads by that ratio).
 *
 * Why the tree's existing windowed kernel is not used (phase-doc row P3.2): it
 * is a scalar one-thread-per-(query, head) kernel that refuses rows > 256 and
 * whose window is TRAILING (`first = visible > window ? visible - window : 0`),
 * which is the wrong mask here — for any item longer than 256 tokens a trailing
 * cut drops keys the reference attends to.
 *
 * Shape of the kernel: one block per query token, one warp per query head, so a
 * block's four warps share the same KV rows in L2.  Within a warp the key range
 * [k0, k1] (the window, or the whole item on the full layers) is split across the
 * 32 lanes; pass 1 writes each lane's keys' scores into shared and takes the
 * maximum, pass 2 turns those scores into exponentials and sums them, and pass 3
 * spreads the output dimensions across the lanes so the weighted sum over keys is
 * a straight accumulation.  Every lane's key loop is a stride-32 walk, so the
 * warp's rows are consecutive and the loads coalesce.
 *
 * The shared score plane is one float per key per warp, bounded by
 * EG2VQ_ATTN_KEYS_MAX; a longer item is refused by the launcher rather than
 * silently truncated (the longest anchor and frozen-set item is 1,414 tokens).
 */
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

enum {
    EG2VQ_ATTN_WARPS = 4,                          /* one warp per query head (n_head = 4) */
    EG2VQ_ATTN_MAX_HEAD = 4,
    EG2VQ_ATTN_KEYS_MAX = 2048,                    /* keys per item this kernel accepts */
    /* shared: 4 warps x 2048 keys x 4 B = 32 KB */
    EG2VQ_ATTN_SHARED = EG2VQ_ATTN_WARPS * EG2VQ_ATTN_KEYS_MAX,
};

__device__ __forceinline__ static float eg2vq_attn_dot(const float *a, const float *b, uint32_t n) {
    /* Four partial sums: the reference accumulates one f32 scalar per (query,
     * key) in f64 (ds4_eg2vq_graph.inc:145); the parity gate bounds the
     * difference this reordering can make, and four chains keep the FMA pipes
     * fed instead of serialising on one accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint32_t i = 0;
    for (; i + 4u <= n; i += 4u) {
        a0 += a[i] * b[i];
        a1 += a[i + 1u] * b[i + 1u];
        a2 += a[i + 2u] * b[i + 2u];
        a3 += a[i + 3u] * b[i + 3u];
    }
    for (; i < n; i++) { a0 += a[i] * b[i]; }
    return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ static float eg2vq_attn_wmax(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); }
    return v;
}

__device__ __forceinline__ static float eg2vq_attn_wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
    return v;
}

/* q, k, v are the tower's own layouts: q [n_tok][n_head][head_dim],
 * k and v [n_tok][n_head_kv][head_dim], all f32 and contiguous.
 * `half_swa` < 0 disables the mask (the full layers); otherwise a query at
 * position p attends to keys with |p - k| <= half_swa. */
__global__ void eg2vq_attn_kernel(
        float *__restrict__ out,
        const float *__restrict__ q,
        const float *__restrict__ k,
        const float *__restrict__ v,
        uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        int32_t half_swa) {
    __shared__ float plane[EG2VQ_ATTN_SHARED];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (warp >= n_head) { return; }
    const uint32_t qpos = blockIdx.x;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = warp / group;
    const uint64_t q_stride = (uint64_t)n_head * head_dim;
    const uint64_t kv_stride = (uint64_t)n_head_kv * head_dim;

    /* the key range this query may attend to, inclusive */
    uint32_t k0 = 0, k1 = n_tok - 1u;
    if (half_swa >= 0) {
        k0 = (qpos > (uint32_t)half_swa) ? qpos - (uint32_t)half_swa : 0u;
        const uint32_t hi = qpos + (uint32_t)half_swa;
        k1 = (hi < n_tok - 1u) ? hi : n_tok - 1u;
    }
    const uint32_t nkey = k1 - k0 + 1u;

    const float *qh = q + (uint64_t)qpos * q_stride + (uint64_t)warp * head_dim;
    const float *kh0 = k + (uint64_t)kvh * head_dim;
    float *scores = plane + (size_t)warp * EG2VQ_ATTN_KEYS_MAX;

    /* pass 1: each lane's keys' scores, into shared, and this lane's maximum */
    float m = -FLT_MAX;
    for (uint32_t kk = lane; kk < nkey; kk += 32u) {
        const float s = eg2vq_attn_dot(qh, kh0 + (uint64_t)(k0 + kk) * kv_stride, head_dim);
        scores[kk] = s;
        m = fmaxf(m, s);
    }
    const float M = eg2vq_attn_wmax(m);

    /* pass 2: exponentials (reference: expf(score - max), graph:151-153) and sum */
    float sum = 0.0f;
    for (uint32_t kk = lane; kk < nkey; kk += 32u) {
        const float p = expf(scores[kk] - M);
        scores[kk] = p;
        sum += p;
    }
    const float S = eg2vq_attn_wsum(sum);
    const float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;   /* graph:154 zero-denominator guard */

    /* pass 3: the weighted sum over keys, output dimensions spread over lanes */
    float *oh = out + (uint64_t)qpos * q_stride + (uint64_t)warp * head_dim;
    const float *vh0 = v + (uint64_t)kvh * head_dim;
    for (uint32_t d = lane; d < head_dim; d += 32u) {
        float acc = 0.0f;
        for (uint32_t kk = 0; kk < nkey; kk++) {
            acc += scores[kk] * vh0[(uint64_t)(k0 + kk) * kv_stride + d];
        }
        oh[d] = acc * inv;
    }
}

/* Returns 0 (and touches nothing) for a shape this kernel does not cover:
 * anything but 4 query heads, no tokens, keys beyond the plane, or a
 * group-unbalanced head count. */
static inline int eg2vq_attn(float *out, const float *q, const float *k, const float *v,
                             uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv,
                             uint32_t head_dim, int32_t half_swa) {
    if (!out || !q || !k || !v || !n_tok || !head_dim) { return 0; }
    if (n_head != EG2VQ_ATTN_WARPS || n_head > EG2VQ_ATTN_MAX_HEAD) { return 0; }
    if (!n_head_kv || (n_head % n_head_kv)) { return 0; }
    if (n_tok > EG2VQ_ATTN_KEYS_MAX) { return 0; }
    eg2vq_attn_kernel<<<n_tok, EG2VQ_ATTN_WARPS * 32>>>(
        out, q, k, v, n_tok, n_head, n_head_kv, head_dim, half_swa);
    return 1;
}
