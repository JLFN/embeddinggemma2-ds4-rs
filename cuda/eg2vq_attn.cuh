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
 * Shape of the kernel: one block per query ROW, one warp per query head, so a
 * block's four warps share the same KV rows in L2.  Within a warp the key range
 * [k0, k1] (the window, or the whole item on the full layers) is split across the
 * 32 lanes; pass 1 writes each lane's keys' scores into shared and takes the
 * maximum, pass 2 turns those scores into exponentials and sums them, and pass 3
 * spreads the output dimensions across the lanes so the weighted sum over keys is
 * a straight accumulation.  Every lane's key loop is a stride-32 walk, so the
 * warp's rows are consecutive and the loads coalesce.
 *
 * THE KEY RANGE IS ITEM-RELATIVE (unit 27, the batch).  A call packs one or more
 * items into ONE contiguous row block, so `pos[t]` is row t's position INSIDE its
 * item and `len[t]` that item's length: the item's first row is t - pos[t] and a
 * query reads its own item's rows only.  That is the reference's own batched
 * semantics -- llama.cpp's ubatch carries per-token seq_id[] and pos[] with the
 * sequences packed contiguously (src/llama-batch.cpp:140-193) and its mask breaks
 * attention between different sequences (src/llama-graph.cpp:482-517, `s0 != s1`)
 * while the positions stay per-sequence, so one item's rows can never enter
 * another's window.  A single-item call passes pos[t] = t and len[t] = n_tok,
 * which is the pre-batch range exactly: k0 = max(0, p - half), k1 = min(n_tok-1,
 * p + half), or the whole item on the full layers.
 *
 * THE SCORE PLANE STAYS THE ITEM CEILING (performance unit, 2026-10-20).  The
 * first form of this unit sized the plane per CALL -- the longest item of the
 * call, clamped by the layer's own window -- and measured 2.5x to 3.7x faster
 * than the scalar kernel on the shape bench, so it looked like the win.  It was
 * not: driven end to end over the 143-page corpus, the call-sized plane was
 * SLOWER than the ceiling everywhere (batch 8: 13.09 s against 11.05 s), and a
 * direct sweep of the kernel with plane size as the only variable (probe, 25
 * reps, the tower's shapes) is non-monotonic and does not follow occupancy at
 * all: at 565 tokens a 565-key plane costs 1.38 ms, 1025 costs 1.23, 2048 costs
 * 0.98, while at 294 tokens 294 costs 0.26 and 2048 costs 0.30.  Two runs of each
 * policy over the corpus could not separate 1536 from 2048 (the sign flips per
 * row, all differences inside the 2-4% run-to-run spread), so the plane is the
 * ceiling -- the size this kernel has always used, no tuning constant, no
 * plumbing through the caller -- and the VECTORIZATION below is the whole win.
 * The sweep and the QA finding that forced this are recorded in
 * ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/attention-occupancy.md.
 *
 * Note on the driver's own accounting: cuobjdump -res-usage reported SHARED:32768
 * for the scalar kernel, and sm_89 offers 100 KB of shared per SM, so exactly
 * three 128-thread blocks (12 warps of 64) could be resident.  That IS a real
 * limit, and it is why the vectorized kernel keeps a hard guard on the plane --
 * but it is not the lever the measurements found; the strided scalar loads were.
 *
 * THE LOADS AND STORES ARE VECTORIZED WHEN THE SHAPE ALLOWS IT (same unit).  Both
 * ends of the kernel were scalar: pass 1 read the K rows one float at a time
 * while 32 lanes walked 32 different rows (`a` broadcasts from q, `b` is the
 * strided K row), and pass 3 wrote one float per lane.  With head_dim a multiple
 * of 128 and 16-byte-aligned q/k/v/out, pass 1 now reads float4 and pass 3 reads
 * and writes float4.  The accumulation ORDER IS UNCHANGED, element for element:
 * the scalar loop fed accumulator chain i%4 from element i, and both vector
 * loops keep that mapping (the pair (i, i+4) inside one iteration feeds a0..a3
 * twice in the same order), and pass 3 keeps its key-major order per output
 * dimension.  The vector path is therefore bit-identical to the scalar one, and
 * the scalar path is retained verbatim as the fallback for a shape or an
 * alignment the vector path does not cover.
 */
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

enum {
    EG2VQ_ATTN_WARPS = 4,                          /* one warp per query head (n_head = 4) */
    EG2VQ_ATTN_MAX_HEAD = 4,
    EG2VQ_ATTN_KEYS_MAX = 2048,                    /* keys per item this kernel accepts */
    /* shared: 4 warps x 2048 keys x 4 B = 32 KB, the item ceiling (see the header
     * for why the plane is the ceiling and not a per-call size) */
    EG2VQ_ATTN_SHARED = EG2VQ_ATTN_WARPS * EG2VQ_ATTN_KEYS_MAX,
    /* The vector path covers a head_dim of 128k with 16-byte-aligned operands:
     * pass 3 gives each lane four consecutive output dimensions and walks the
     * warp's 32 lanes over the head in 128-float steps. */
    EG2VQ_ATTN_VEC_STEP = 32 * 4,
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

/* The same four chains, two float4 per step: element i feeds chain i%4 exactly
 * as eg2vq_attn_dot does, so the two agree bit for bit on a vectorizable shape. */
__device__ __forceinline__ static float eg2vq_attn_dot4(const float *a, const float *b, uint32_t n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint32_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        const float4 x0 = *(const float4 *)(a + i);
        const float4 y0 = *(const float4 *)(b + i);
        const float4 x1 = *(const float4 *)(a + i + 4u);
        const float4 y1 = *(const float4 *)(b + i + 4u);
        a0 += x0.x * y0.x; a1 += x0.y * y0.y; a2 += x0.z * y0.z; a3 += x0.w * y0.w;
        a0 += x1.x * y1.x; a1 += x1.y * y1.y; a2 += x1.z * y1.z; a3 += x1.w * y1.w;
    }
    for (; i + 4u <= n; i += 4u) {
        const float4 x = *(const float4 *)(a + i);
        const float4 y = *(const float4 *)(b + i);
        a0 += x.x * y.x; a1 += x.y * y.y; a2 += x.z * y.z; a3 += x.w * y.w;
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
 * `pos[t]` is row t's position inside its own item and `len[t]` that item's
 * row count (both int32, n_tok entries; a single item is pos[t] = t,
 * len[t] = n_tok).  `half_swa` < 0 disables the mask (the full layers, where a
 * query reads its whole item); otherwise a query reads the rows of its own item
 * within |pos - key_pos| <= half_swa.
 *
 * The score plane is the item ceiling, in static shared memory (the size it has
 * always had).  A dynamic allocation of the same 32,768 B was tried as part of
 * this unit and measured NOTHING over it -- see the note at the end of this
 * header on how the one corpus measurement that seemed to favour it turned out
 * to be a machine-state artifact.
 * `vec4` selects the vectorized passes; it is 1 only when the launcher verified
 * head_dim % EG2VQ_ATTN_VEC_STEP == 0 and 16-byte alignment of all four buffers,
 * and the two paths are numerically identical, so it is a throughput switch and
 * never a behaviour switch. */
__global__ void eg2vq_attn_kernel(
        float *__restrict__ out,
        const float *__restrict__ q,
        const float *__restrict__ k,
        const float *__restrict__ v,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ len,
        uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        int32_t half_swa, int vec4) {
    __shared__ float plane[EG2VQ_ATTN_SHARED];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (warp >= n_head) { return; }
    const uint32_t qrow = blockIdx.x;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = warp / group;
    const uint64_t q_stride = (uint64_t)n_head * head_dim;
    const uint64_t kv_stride = (uint64_t)n_head_kv * head_dim;

    /* the item this row belongs to, and the key range it may attend to: on the
     * sliding layers its own item's rows within the symmetric window, on the full
     * layers every row of its item (not of the block). */
    const uint32_t p = (uint32_t)pos[qrow];              /* position inside the item */
    const uint32_t nitem = (uint32_t)len[qrow];          /* the item's row count */
    const uint32_t base = qrow - p;                      /* the item's first row */
    uint32_t k0 = base, k1 = base + nitem - 1u;
    if (half_swa >= 0) {
        k0 = base + ((p > (uint32_t)half_swa) ? p - (uint32_t)half_swa : 0u);
        const uint32_t hi = p + (uint32_t)half_swa;
        k1 = base + ((hi < nitem - 1u) ? hi : nitem - 1u);
    }
    uint32_t nkey = k1 - k0 + 1u;
    /* Defensive clamp: pos and len come from the caller's span table, and the
     * block's own row count is the only bound this kernel can check, so a pair
     * that disagrees with it can never read outside the block.  The core refuses
     * such a table up front; this is the second line, not the first. */
    if (k0 > n_tok - 1u) { k0 = n_tok - 1u; }
    if (k1 > n_tok - 1u) { k1 = n_tok - 1u; }
    if (k1 < k0) { k0 = k1; }
    nkey = k1 - k0 + 1u;
    /* and the plane the block holds: a row beyond the item ceiling refuses
     * instead of writing past the block's own shared window. */
    if (nkey > (uint32_t)EG2VQ_ATTN_KEYS_MAX) { return; }

    const float *qh = q + (uint64_t)qrow * q_stride + (uint64_t)warp * head_dim;
    const float *kh0 = k + (uint64_t)kvh * head_dim;
    float *scores = plane + (size_t)warp * EG2VQ_ATTN_KEYS_MAX;

    /* pass 1: each lane's keys' scores, into shared, and this lane's maximum */
    float m = -FLT_MAX;
    if (vec4) {
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float s = eg2vq_attn_dot4(qh, kh0 + (uint64_t)(k0 + kk) * kv_stride, head_dim);
            scores[kk] = s;
            m = fmaxf(m, s);
        }
    } else {
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float s = eg2vq_attn_dot(qh, kh0 + (uint64_t)(k0 + kk) * kv_stride, head_dim);
            scores[kk] = s;
            m = fmaxf(m, s);
        }
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
    float *oh = out + (uint64_t)qrow * q_stride + (uint64_t)warp * head_dim;
    const float *vh0 = v + (uint64_t)kvh * head_dim;
    if (vec4) {
        for (uint32_t d = lane * 4u; d + 4u <= head_dim; d += (uint32_t)EG2VQ_ATTN_VEC_STEP) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (uint32_t kk = 0; kk < nkey; kk++) {
                const float p = scores[kk];
                const float4 vv = *(const float4 *)(vh0 + (uint64_t)(k0 + kk) * kv_stride + d);
                acc.x += p * vv.x;
                acc.y += p * vv.y;
                acc.z += p * vv.z;
                acc.w += p * vv.w;
            }
            acc.x *= inv; acc.y *= inv; acc.z *= inv; acc.w *= inv;
            *(float4 *)(oh + d) = acc;
        }
    } else {
        for (uint32_t d = lane; d < head_dim; d += 32u) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < nkey; kk++) {
                acc += scores[kk] * vh0[(uint64_t)(k0 + kk) * kv_stride + d];
            }
            oh[d] = acc * inv;
        }
    }
}

/* Returns 0 (and touches nothing) for a shape this kernel does not cover:
 * anything but 4 query heads, no rows, or a group-unbalanced head count.  The
 * ROW count is not bounded here -- a batch's block is as long as its items add up
 * to, and each ITEM's ceiling is DS4_EG2VQ_MAX_TOKENS, which the caller checks
 * before the ids are even uploaded (ds4_eg2vq_gpu.cuh) and the kernel's own guard
 * re-checks against the plane. */
static inline int eg2vq_attn(float *out, const float *q, const float *k, const float *v,
                             const int32_t *pos, const int32_t *len,
                             uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv,
                             uint32_t head_dim, int32_t half_swa) {
    if (!out || !q || !k || !v || !pos || !len || !n_tok || !head_dim) { return 0; }
    if (n_head != EG2VQ_ATTN_WARPS || n_head > EG2VQ_ATTN_MAX_HEAD) { return 0; }
    if (!n_head_kv || (n_head % n_head_kv)) { return 0; }
    /* the vector path needs the whole head covered by 128-float warp steps and
     * 16-byte-aligned operands; anything else keeps the scalar path. */
    const int vec4 = ((head_dim % (uint32_t)EG2VQ_ATTN_VEC_STEP) == 0u &&
                      ((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)out) % 16u == 0u) ? 1 : 0;
    eg2vq_attn_kernel<<<n_tok, EG2VQ_ATTN_WARPS * 32>>>(
        out, q, k, v, pos, len, n_tok, n_head, n_head_kv, head_dim, half_swa, vec4);
    return 1;
}

/* MEASUREMENT NOTE, for whoever tunes this next (read the record in
 * ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/attention-occupancy.md):
 * corpus-level numbers taken more than a few minutes apart on this machine are
 * NOT comparable.  Three sweeps of one build at 18:21-18:25 read 11.654, 11.718
 * and 11.658 s at batch 8, while four sweeps of two other builds at 18:07-18:17
 * read 10.596-11.045 -- and rebuilding the 18:07-18:17 configuration exactly
 * gave 11.717 s in the later window.  The isolated kernel bench moved far less
 * than that across the same period, but it is not precise either (a QA
 * re-measurement of one static binary minutes apart saw the same row 10% apart),
 * so the shift is in the machine (host path, clocks, other tenants), not in the
 * kernel: compare builds only inside one window, or interleaved.
 */
