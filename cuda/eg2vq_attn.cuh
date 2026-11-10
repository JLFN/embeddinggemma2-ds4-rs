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
 *
 * PASS 1'S K ACCESS IS WHAT THIS KERNEL WAS PAYING FOR, AND IT IS NOW GROUPED
 * (second performance unit, 2026-10-21).  Vectorizing the loads did not fix the
 * pattern: pass 1 gave each lane a whole K row, so one warp load addressed 32
 * different rows, touched 32 lines and used 16 B of each.  An ablation of the
 * shipped kernel (docs/benchmarks/eg2-embedding-2026-10-20/probes/probe-attn.cu,
 * one component removed per variant, every variant register-neutral) put the
 * cost there and nowhere else: the K address made block-uniform -- one line
 * instead of 32, same instruction count -- was 1.86x-2.25x, the K operand taken
 * out of memory altogether 2.25x-2.66x, while the exponentials (0.95x-1.05x),
 * the score plane's write side (1.01x-1.07x) and its read side (1.00x-1.08x)
 * each measured free, and pass 1 plus the row max was 1.37x-1.67x of the total.
 * So the fix is the assignment, not the arithmetic: LAB lanes cooperate on one
 * key and each lane owns GRP_GS float4 chunks of it, 4*LAB elements apart, so a
 * group's lanes read consecutive addresses and one warp load covers whole 128 B
 * lines; each lane keeps its own 64-float slice of the query row in registers
 * instead of re-reading the whole row once per key; the group's LAB partial sums
 * are reduced by a butterfly (log2(LAB) shuffles), which is what lets a lane own
 * whole keys again; and the score store is one predicated float store per chunk
 * instead of one per key.
 *
 * Measured against the shipped form in one window (probes/probe-attn-grp.cu, the
 * candidate built as a probe before it was landed; the record has the full
 * table): 2.27x and 2.39x at head_dim 256 with LAB=4 (294 and 565 tokens), 2.31x
 * at 1414, 2.06x on an 8x565 packed block, and 2.53x / 2.01x at head_dim 512
 * with LAB=8.  The (8,8) and (16,8) pairings, which trade keys in flight for a
 * shorter butterfly, measured 1.37x-1.75x and 1.50x-1.80x and were rejected.
 * The grouped form is not bit-identical to the lane-per-key form: the two agree
 * to cosine 1.000000000 with a maximum element-wise relative difference of
 * 2.3e-06-6.5e-06, inside what the gates bound (`make test-eg2vq-attn` allows a
 * per-head cosine down to 1 - 1e-6 and a relative RMS up to 1e-5; the gate-7
 * oracle declares cos >= 0.999999 / rel_rms <= 1e-03 and measures 0.999999950 /
 * 3.149e-04 on this change).
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    /* The grouped pass 1 (second performance unit): LAB lanes cooperate on one
     * key and each lane owns GRP_GS float4 chunks of it, so the lanes of a group
     * read consecutive addresses and a warp load covers whole 128 B lines
     * instead of touching one line per key and using 16 B of each.  LAB is
     * head_dim/64 -- 4 for the tower's sliding layers (head_dim 256) and 8 for
     * its full layers (512) -- and the chunk count per lane is 16 for every
     * head_dim this covers, so the lane's query slice is always 64 floats in
     * registers.  The per-lane loop below is kept for the chunk remainder and
     * for every other shape. */
    EG2VQ_ATTN_GRP_GS = 16,
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
template <int LAB>
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

    /* pass 1: each lane's keys' scores, into shared, and this lane's maximum.
     *
     * Two forms.  LAB > 0 is the grouped form (see the header): LAB lanes
     * cooperate on ONE key, each owning GRP_GS float4 chunks of it, so the
     * lanes of a group read 4*LAB consecutive floats and one warp load covers
     * whole 128 B lines.  The lane keeps its own 64-float slice of the query
     * row in registers, loaded once per row instead of once per key, and the
     * group's partial sums are reduced with a butterfly across its LAB lanes.
     * The window is consumed in chunks of KC = 32/LAB keys, and the remainder
     * (fewer than KC keys) falls through to the lane-per-key loop, which is also
     * what every other shape uses.  The two forms do NOT agree bit for bit --
     * the grouped form sums the head's terms as LAB partial sums reduced by a
     * butterfly instead of as the four chains of eg2vq_attn_dot4 alone -- and
     * the gates bound that difference (measured 2.3e-06 to 6.5e-06 maximum
     * element-wise relative, cosine 1.000000000 against the lane-per-key form).
     */
    float m = -FLT_MAX;
    uint32_t kk_grouped = 0;
    if constexpr (LAB > 0) {
        constexpr uint32_t KC = 32u / (uint32_t)LAB;
        /* the lane's chunks are 4*LAB elements apart, so LAB*SPAN == head_dim */
        constexpr uint32_t SPAN = 4u * (uint32_t)LAB;
        const uint32_t msub = lane & ((uint32_t)LAB - 1u);   /* this lane's slice of the head */
        const uint32_t gsub = lane / (uint32_t)LAB;          /* this lane's key inside the chunk */
        float4 qv[EG2VQ_ATTN_GRP_GS];
        #pragma unroll
        for (int j = 0; j < EG2VQ_ATTN_GRP_GS; j++) {
            qv[j] = *(const float4 *)(qh + 4u * msub + SPAN * (uint32_t)j);
        }
        for (; kk_grouped + KC <= nkey; kk_grouped += KC) {
            const float *krow = kh0 + (uint64_t)(k0 + kk_grouped + gsub) * kv_stride;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll
            for (int j = 0; j < EG2VQ_ATTN_GRP_GS; j++) {
                const float4 kv = *(const float4 *)(krow + 4u * msub + SPAN * (uint32_t)j);
                a0 = fmaf(qv[j].x, kv.x, a0);
                a1 = fmaf(qv[j].y, kv.y, a1);
                a2 = fmaf(qv[j].z, kv.z, a2);
                a3 = fmaf(qv[j].w, kv.w, a3);
            }
            float acc = (a0 + a1) + (a2 + a3);
            #pragma unroll
            for (int o = LAB / 2; o > 0; o >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, o);
            }
            if (msub == 0u) {
                scores[kk_grouped + gsub] = acc;
                m = fmaxf(m, acc);
            }
        }
    }
    /* the remainder, and the whole window when this instantiation has no
     * grouped form: the lane-per-key loop */
    if (vec4) {
        for (uint32_t kk = kk_grouped + lane; kk < nkey; kk += 32u) {
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

/* ---------------------------------------------------------------------------
 * THE TENSOR-CORE FORM (performance unit, 2026-11-10)
 *
 * WHY THIS UNIT, measured and not assumed: with the payload GEMM on f16 tensor
 * cores, attention is the largest item left.  One nsys profile of the same
 * 8-item / 2,451-token forward the previous three units profiled reads
 * `eg2vq_attn_kernel` 43.9 ms of 87.6 ms of kernel time (50.1%): the head_dim
 * 256 instantiation 31.5 ms over 20 launches, the 512 one 12.4 ms over 4.  The
 * GEMM is 19.4 ms (22.1%) and the small ops 24.3 ms.  Raw:
 * docs/benchmarks/eg2-embedding-2026-10-20/nsys-kern-sum-b8-mma-vq.txt.
 *
 * THE FORM, from two references read first-hand on their own branches.  The
 * dense one is the closer model, and it is the one this unit follows:
 *
 *   THE DENSE FAMILY (the shape this tower is).  PR 142, branch
 *   feat/spark2_5-family in /data/ds4-dfm-rs, cuda/spark2_5.cuh:
 *   spark_attn_mma_fp8_kernel (:887-1044) is one CTA per (KV head, row tile),
 *   8 warps, 32-key tiles, Q kept in registers, K and V both staged per tile,
 *   the swizzle at :835-837, ldmatrix at :839-848, m16n8k16 at :855-860, the
 *   f32 pair to one f16 word at :850-853, f32 online softmax and P rounded to
 *   f16.  It serves that dense model's 27 sliding AND 9 full layers with ONE
 *   kernel -- a `window` of 0 means full, because its full layers are also 256
 *   wide.  This unit's sliding kernel is that kernel's walk, tile for tile.
 *
 *   THE D=512 GEOMETRY, which the dense branch does not have.  PR 143, branch
 *   feat/gemma4-tensor-core-fp8, cuda/gemma4_attn_mma.cuh:344-583 splits the
 *   512 output dims into two 256-wide halves, one per warp of a pair, and the
 *   pair exchanges its partial 16x32 scores in shared at :450-468.  This
 *   unit's full kernel is that split, on this tower's rows.
 *
 * WHY ONE KERNEL IS NOT ENOUGH HERE, unlike the dense reference: this tower's
 * full layers differ in head_dim (512 against 256), not only in the mask, so
 * one 256-wide walk cannot cover them; and its tiles are symmetric rather than
 * trailing, so the key loop's bounds differ even where the shape does not.
 *
 * WHAT ELSE CHANGES FROM THE REFERENCES IS THE GEOMETRY, NOT THE ARITHMETIC:
 *   sliding <256, 2>  group 2: 8 tokens x 2 query heads per warp, RTILE 64,
 *                     both K and V resident (32 KiB).
 *   full    <512, 4>  one KV head and 4 query heads, so 16 M rows are 4 tokens
 *                     x 4 heads; K then V share one 32 KiB tile, 48 KiB total
 *                     with the exchange.
 * and the four things this port needs that neither reference does:
 *   - THE MASK IS SYMMETRIC, theirs are trailing-causal.  A CTA's key loop runs
 *     from the minimum begin over its rows to the maximum last (rounded to
 *     32-key tiles), and each row applies its own |pos - key_pos| <= half_swa
 *     inside the tile -- the scalar kernel's union-plus-per-row-bound shape, so
 *     the mask edge is the same gate.
 *   - THE KEY RANGE IS ITEM-RELATIVE (unit 27's batch).  A CTA's rows must all
 *     belong to ONE item, so the tile ordinal is mapped to (item, tile) by
 *     walking the item lengths (`len[t]`, the table the scalar kernel's guard
 *     already reads) instead of by dividing the row index.  The grid is the
 *     exact total tile count, which the launcher computes from the caller's
 *     spans -- the dense reference's one-sequence-per-call gets this for free.
 *   - K AND V ARE f32 IN THIS PORT'S CACHE (the dense reference's are FP8 with
 *     per-row scales, gemma4's bf16 through cp.async); staging converts f32 to
 *     f16 on the way into shared, which is also why the shared element type is
 *     __half and not bf16: this tower's tensors are f16, and bf16 would drop
 *     three significand bits (the reason the VQ unit took f16:
 *     ds41_vq_prefill_mma.cuh:24).
 *   - Q IS f32 ON ENTRY (gemma4's arrives pre-rounded to bf16, so its
 *     conversion is 0 ULP; the dense reference's is f32 and it rounds too).
 *     This is a real new rounding of the query, and the parity gate bounds it.
 * The scale is exactly 1.0f (gemma-embedding2.cpp:8), so no scale is folded
 * into the score tree -- there is nothing to fold in.
 *
 * GATE: a parity gate, not bit identity.  Both paths are compared against the
 * same f64 CPU oracle at every tower shape, and gate 7's declared bounds
 * (cos >= 0.999999, rel_rms <= 1e-03) are unchanged; the measured numbers are
 * in tests/test_eg2vq_attn.cu's own output and in the unit's record.
 * ------------------------------------------------------------------------- */

enum {
    EG2VQ_ATTN_TC_THREADS = 256u,   /* 8 warps, as both references use */
    EG2VQ_ATTN_TC_TK = 32u,         /* keys per tile, the reference's G4_MMA_TK */
    EG2VQ_ATTN_TC_RTILE_S = 64u,    /* sliding: 8 warps x 8 tokens */
    EG2VQ_ATTN_TC_RTILE_F = 16u,    /* full: 4 warps x 4 tokens, two per pair */
    EG2VQ_ATTN_TC_SMEM_S = 2u * EG2VQ_ATTN_TC_TK * 256u * 2u,                     /* K + V */
    EG2VQ_ATTN_TC_SMEM_F = EG2VQ_ATTN_TC_TK * 512u * 2u + 4u * 2u * 16u * 32u * 4u /* K/V + exchange */
};

/* The reference's swizzle (gemma4_attn_mma.cuh:43-51), unchanged: a 16-byte
 * chunk's row position XORed with the key index, so the eight rows one
 * ldmatrix reads land on eight distinct 128-byte bank groups. */
__device__ __forceinline__ static uint32_t eg2vq_attn_mma_swz(uint32_t key, uint32_t chunk, uint32_t dim) {
    return key * dim + ((chunk ^ (key & 7u)) << 3);
}

/* ldmatrix.x4 (gemma4_attn_mma.cuh:52-61): four 8x8 b16 matrices into 4 words,
 * the non-transposed form for the QK^T operand and .trans for the P*V one. */
__device__ __forceinline__ static void eg2vq_attn_mma_ldsm4(uint32_t *r, const void *p, bool trans) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    if (trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }
}

/* m16n8k16, f16 operands and an f32 accumulator (gemma4_attn_mma.cuh:63-68,
 * :87-92 for its f16 lane).  Four D registers, four A words, two B words. */
__device__ __forceinline__ static void eg2vq_attn_mma_f16(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* Two f32 to one packed f16 pair (gemma4_attn_mma.cuh:82-85's f16 arm). */
__device__ __forceinline__ static uint32_t eg2vq_attn_mma_pack(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

/* Eight f32 to the 16-byte shared slot: the reference's cp.async of a ready
 * bf16 chunk (gemma4_attn_mma.cuh:47-50) becomes a convert-and-store here,
 * because this port's K/V cache is f32.  `valid` is false for a key outside
 * the CTA's own range: the row is read anyway from a clamped (in-bounds) slot
 * and stored as zeros, and the per-row mask keeps it out of every score. */
__device__ __forceinline__ static void eg2vq_attn_mma_stage8(__half *dst, const float *src, bool valid) {
    const float4 x0 = *(const float4 *)src;
    const float4 x1 = *(const float4 *)(src + 4);
    uint4 o;
    const __half2 h0 = __floats2half2_rn(valid ? x0.x : 0.0f, valid ? x0.y : 0.0f);
    const __half2 h1 = __floats2half2_rn(valid ? x0.z : 0.0f, valid ? x0.w : 0.0f);
    const __half2 h2 = __floats2half2_rn(valid ? x1.x : 0.0f, valid ? x1.y : 0.0f);
    const __half2 h3 = __floats2half2_rn(valid ? x1.z : 0.0f, valid ? x1.w : 0.0f);
    o.x = *reinterpret_cast<const uint32_t *>(&h0);
    o.y = *reinterpret_cast<const uint32_t *>(&h1);
    o.z = *reinterpret_cast<const uint32_t *>(&h2);
    o.w = *reinterpret_cast<const uint32_t *>(&h3);
    *(uint4 *)dst = o;
}

/* The tile ordinal to (item, tile) map.  Items are contiguous and `len[t]` is
 * the row count of t's item, so walking t0 += len[t0] visits exactly the item
 * starts; every thread of the CTA runs it and reads the same two words, which
 * the 32-wide broadcast makes free.  Returns 0 past the last tile. */
template <uint32_t RTILE>
__device__ __forceinline__ static int eg2vq_attn_mma_tile_map(const int32_t *__restrict__ len,
        uint32_t n_tok, uint32_t ord, uint32_t *item_base, uint32_t *item_len, uint32_t *tile) {
    uint32_t t0 = 0;
    while (t0 < n_tok) {
        const uint32_t l = (uint32_t)len[t0];
        if (!l) { return 0; }
        const uint32_t nt = (l + RTILE - 1u) / RTILE;
        if (ord < nt) { *item_base = t0; *item_len = l; *tile = ord; return 1; }
        ord -= nt;
        t0 += l;
    }
    return 0;
}

/* The sliding geometry on tensor cores: D = 256, group 2, 2 KV heads.
 * grid = (n_head_kv, tiles), block = 256, dynamic shared = K + V tiles. */
__global__ void __launch_bounds__(EG2VQ_ATTN_TC_THREADS, 1) eg2vq_attn_mma_slide_kernel(
        float *__restrict__ out, const float *__restrict__ q,
        const float *__restrict__ k, const float *__restrict__ v,
        const int32_t *__restrict__ len, uint32_t n_tok, int32_t half_swa) {
    constexpr uint32_t D = 256u, GROUP = 2u, RTILE = EG2VQ_ATTN_TC_RTILE_S, TK = EG2VQ_ATTN_TC_TK;
    constexpr uint32_t NHEAD = 4u;
    extern __shared__ __align__(16) uint8_t smem[];
    __half *ks = (__half *)smem;
    __half *vs = ks + TK * D;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, qq = lane & 3u;
    const uint32_t kv_head = blockIdx.x;

    uint32_t item_base, item_len, tile;
    if (!eg2vq_attn_mma_tile_map<RTILE>(len, n_tok, blockIdx.y, &item_base, &item_len, &tile)) { return; }
    const uint32_t r0 = item_base + tile * RTILE;
    const uint32_t r1 = (item_base + item_len < r0 + RTILE) ? item_base + item_len : r0 + RTILE;
    const uint32_t pos_first = r0 - item_base;
    const uint32_t pos_last = r1 - 1u - item_base;
    const uint32_t half = (half_swa > 0) ? (uint32_t)half_swa : 0x7fffffffu;
    const uint32_t begin_all = (pos_first > half) ? pos_first - half : 0u;
    const uint32_t last_all = (pos_last + half < item_len - 1u) ? pos_last + half : item_len - 1u;
    if (begin_all > last_all) { return; }
    const uint32_t base_start = (begin_all / TK) * TK;
    const uint64_t kv_stride = (uint64_t)2u * D;

    /* Q fragments: 16 M rows (8 tokens x 2 heads) x 16 k16 chunks of D, the
     * A-fragment order [row g, row g+8] x [col, col+8] (gemma4_attn_mma.cuh:
     * 179-207), with the M rows this port's own (token, head) map. */
    uint32_t qa[16][4];
    #pragma unroll
    for (uint32_t kk = 0; kk < 16u; kk++) {
        const uint32_t col = kk * 16u + 2u * qq;
        #pragma unroll
        for (uint32_t m = 0; m < 2u; m++) {
            const uint32_t row_g = g + 8u * m;
            const uint32_t token = r0 + warp * 8u + (row_g >> 1);
            const bool valid = (token < r1);
            const uint32_t head = kv_head * GROUP + (row_g & 1u);
            const float *qr = q + ((uint64_t)token * NHEAD + head) * D;
            const float lo = valid ? qr[col] : 0.0f;
            const float hi = valid ? qr[col + 1u] : 0.0f;
            const float lo8 = valid ? qr[col + 8u] : 0.0f;
            const float hi8 = valid ? qr[col + 9u] : 0.0f;
            if (m == 0u) {
                qa[kk][0] = eg2vq_attn_mma_pack(lo, hi);
                qa[kk][2] = eg2vq_attn_mma_pack(lo8, hi8);
            } else {
                qa[kk][1] = eg2vq_attn_mma_pack(lo, hi);
                qa[kk][3] = eg2vq_attn_mma_pack(lo8, hi8);
            }
        }
    }

    float o[32][4] = {};
    float mrun[2] = {-INFINITY, -INFINITY}, lrun[2] = {0.0f, 0.0f};

    for (uint32_t base = base_start; base <= last_all; base += TK) {
        /* K and V for this KV head's 32 keys: each warp stages four keys, each
         * lane one 16-byte chunk of each (D/8 = 32 chunks). */
        for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
            const uint32_t k_abs = base + t;
            const bool valid = (k_abs >= begin_all && k_abs <= last_all);
            const uint32_t slot = valid ? (item_base + k_abs) : item_base;
            const float *src = k + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
            const float *vsrc = v + (uint64_t)slot * kv_stride + (uint64_t)kv_head * D;
            const uint32_t chunk = lane;
            eg2vq_attn_mma_stage8(ks + eg2vq_attn_mma_swz(t, chunk, D), src + chunk * 8u, valid);
            eg2vq_attn_mma_stage8(vs + eg2vq_attn_mma_swz(t, chunk, D), vsrc + chunk * 8u, valid);
        }
        __syncthreads();

        /* S = Q K^T over the 32-key tile: four n8 tiles, 16 k16 steps. */
        float s[4][4] = {};
        #pragma unroll
        for (uint32_t kk = 0; kk < 16u; kk++) {
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = (j + (mat >> 1)) * 8u + (lane & 7u);
                const uint32_t chunk = kk * 2u + (mat & 1u);
                eg2vq_attn_mma_ldsm4(b, ks + eg2vq_attn_mma_swz(krow, chunk, D), false);
                eg2vq_attn_mma_f16(s[j], qa[kk], b[0], b[1]);
                eg2vq_attn_mma_f16(s[j + 1u], qa[kk], b[2], b[3]);
            }
        }

        /* The symmetric mask and the online softmax, per row h = 0 (g) and
         * h = 1 (g + 8): the score set is this row's own item window. */
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t row_g = g + 8u * h;
            const uint32_t token = r0 + warp * 8u + (row_g >> 1);
            const bool rv = (token < r1);
            const uint32_t pos = token - item_base;
            const uint32_t rb = (pos > half) ? pos - half : 0u;
            const uint32_t rl = (pos + half < item_len - 1u) ? pos + half : item_len - 1u;
            float peak = -INFINITY;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t kk = base + nt * 8u + 2u * qq + e;
                    if (!rv || kk < rb || kk > rl) { s[nt][h * 2u + e] = -INFINITY; }
                    peak = fmaxf(peak, s[nt][h * 2u + e]);
                }
            }
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
            const bool tv = (peak > -INFINITY);
            const float mn = tv ? fmaxf(mrun[h], peak) : mrun[h];
            const float alpha = tv ? (mrun[h] == -INFINITY ? 0.0f : expf(mrun[h] - mn)) : 1.0f;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const float p = tv ? expf(s[nt][h * 2u + e] - mn) : 0.0f;
                    s[nt][h * 2u + e] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            lrun[h] = lrun[h] * alpha + sum;
            mrun[h] = mn;
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t++) {
                o[t][h * 2u + 0u] *= alpha;
                o[t][h * 2u + 1u] *= alpha;
            }
        }

        /* O += P V over the same 32 keys: two k16 steps, 32 n8 output tiles. */
        #pragma unroll
        for (uint32_t kk = 0; kk < 2u; kk++) {
            const uint32_t pa[4] = {
                eg2vq_attn_mma_pack(s[2u * kk][0], s[2u * kk][1]),
                eg2vq_attn_mma_pack(s[2u * kk][2], s[2u * kk][3]),
                eg2vq_attn_mma_pack(s[2u * kk + 1u][0], s[2u * kk + 1u][1]),
                eg2vq_attn_mma_pack(s[2u * kk + 1u][2], s[2u * kk + 1u][3])};
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = kk * 16u + (mat & 1u) * 8u + (lane & 7u);
                const uint32_t chunk = t + (mat >> 1);
                eg2vq_attn_mma_ldsm4(b, vs + eg2vq_attn_mma_swz(krow, chunk, D), true);
                eg2vq_attn_mma_f16(o[t], pa, b[0], b[1]);
                eg2vq_attn_mma_f16(o[t + 1u], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    const float inv[2] = {lrun[0] > 0.0f ? 1.0f / lrun[0] : 0.0f,
                          lrun[1] > 0.0f ? 1.0f / lrun[1] : 0.0f};
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t row_g = g + 8u * h;
        const uint32_t token = r0 + warp * 8u + (row_g >> 1);
        if (token >= r1) { continue; }
        const uint32_t head = kv_head * GROUP + (row_g & 1u);
        float *dst = out + ((uint64_t)token * NHEAD + head) * D;
        #pragma unroll
        for (uint32_t t = 0; t < 32u; t++) {
            const uint32_t d = t * 8u + 2u * qq;
            dst[d] = o[t][h * 2u + 0u] * inv[h];
            dst[d + 1u] = o[t][h * 2u + 1u] * inv[h];
        }
    }
}

/* The full geometry on tensor cores: D = 512, group 4, 1 KV head.
 * Each warp owns 4 tokens x 4 query heads (16 M rows) and ONE 256-dim half;
 * the two warps of an mgroup exchange their partial 16x32 scores in shared
 * before the softmax, exactly as gemma4_attn_mma.cuh:450-468 does.
 * grid = (1, tiles), block = 256, dynamic shared = K/V tile + exchange. */
__global__ void __launch_bounds__(EG2VQ_ATTN_TC_THREADS, 1) eg2vq_attn_mma_full_kernel(
        float *__restrict__ out, const float *__restrict__ q,
        const float *__restrict__ k, const float *__restrict__ v,
        const int32_t *__restrict__ len, uint32_t n_tok, int32_t half_swa) {
    constexpr uint32_t D = 512u, RTILE = EG2VQ_ATTN_TC_RTILE_F, TK = EG2VQ_ATTN_TC_TK;
    constexpr uint32_t HALF = 256u, NHEAD = 4u;
    extern __shared__ __align__(16) uint8_t smem[];
    __half *kv = (__half *)smem;
    float *exchange = (float *)(kv + TK * D);

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, qq = lane & 3u;

    uint32_t item_base, item_len, tile;
    if (!eg2vq_attn_mma_tile_map<RTILE>(len, n_tok, blockIdx.x, &item_base, &item_len, &tile)) { return; }
    const uint32_t r0 = item_base + tile * RTILE;
    const uint32_t r1 = (item_base + item_len < r0 + RTILE) ? item_base + item_len : r0 + RTILE;
    const uint32_t pos_first = r0 - item_base;
    const uint32_t pos_last = r1 - 1u - item_base;
    const uint32_t half = (half_swa > 0) ? (uint32_t)half_swa : 0x7fffffffu;
    const uint32_t begin_all = (pos_first > half) ? pos_first - half : 0u;
    const uint32_t last_all = (pos_last + half < item_len - 1u) ? pos_last + half : item_len - 1u;
    if (begin_all > last_all) { return; }
    const uint32_t base_start = (begin_all / TK) * TK;
    const uint64_t kv_stride = (uint64_t)D;          /* one KV head */
    const uint32_t mgroup = warp >> 1u, halfd = warp & 1u;

    /* Q fragments for this 256-dim half; M row = 4 tokens x 4 heads. */
    uint32_t qa[16][4];
    #pragma unroll
    for (uint32_t kk = 0; kk < 16u; kk++) {
        const uint32_t col = halfd * HALF + kk * 16u + 2u * qq;
        #pragma unroll
        for (uint32_t m = 0; m < 2u; m++) {
            const uint32_t row_g = g + 8u * m;
            const uint32_t token = r0 + mgroup * 4u + (row_g >> 2);
            const bool valid = (token < r1);
            const uint32_t head = row_g & 3u;
            const float *qr = q + ((uint64_t)token * NHEAD + head) * D;
            const float lo = valid ? qr[col] : 0.0f;
            const float hi = valid ? qr[col + 1u] : 0.0f;
            const float lo8 = valid ? qr[col + 8u] : 0.0f;
            const float hi8 = valid ? qr[col + 9u] : 0.0f;
            if (m == 0u) {
                qa[kk][0] = eg2vq_attn_mma_pack(lo, hi);
                qa[kk][2] = eg2vq_attn_mma_pack(lo8, hi8);
            } else {
                qa[kk][1] = eg2vq_attn_mma_pack(lo, hi);
                qa[kk][3] = eg2vq_attn_mma_pack(lo8, hi8);
            }
        }
    }

    float o[32][4] = {};
    float mrun[2] = {-INFINITY, -INFINITY}, lrun[2] = {0.0f, 0.0f};

    for (uint32_t base = base_start; base <= last_all; base += TK) {
        /* K for the tile: each warp four keys, each lane two 16-byte chunks
         * (D/8 = 64 chunks). */
        for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
            const uint32_t k_abs = base + t;
            const bool valid = (k_abs >= begin_all && k_abs <= last_all);
            const uint32_t slot = valid ? (item_base + k_abs) : item_base;
            const float *src = k + (uint64_t)slot * kv_stride;
            for (uint32_t c = lane; c < D / 8u; c += 32u) {
                eg2vq_attn_mma_stage8(kv + eg2vq_attn_mma_swz(t, c, D), src + c * 8u, valid);
            }
        }
        __syncthreads();

        /* Partial S over this warp's half: four n8 tiles, 16 k16 steps. */
        float s[4][4] = {};
        #pragma unroll
        for (uint32_t kk = 0; kk < 16u; kk++) {
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = (j + (mat >> 1)) * 8u + (lane & 7u);
                const uint32_t chunk = halfd * 32u + kk * 2u + (mat & 1u);
                eg2vq_attn_mma_ldsm4(b, kv + eg2vq_attn_mma_swz(krow, chunk, D), false);
                eg2vq_attn_mma_f16(s[j], qa[kk], b[0], b[1]);
                eg2vq_attn_mma_f16(s[j + 1u], qa[kk], b[2], b[3]);
            }
        }

        /* Publish the half's scores and add both halves in half order
         * (gemma4_attn_mma.cuh:450-468). */
        float *mine = exchange + (mgroup * 2u + halfd) * 16u * 32u;
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
            *(float2 *)(mine + g * 32u + nt * 8u + 2u * qq) = make_float2(s[nt][0], s[nt][1]);
            *(float2 *)(mine + (g + 8u) * 32u + nt * 8u + 2u * qq) = make_float2(s[nt][2], s[nt][3]);
        }
        __syncthreads();
        const float *ex0 = exchange + (mgroup * 2u + 0u) * 16u * 32u;
        const float *ex1 = exchange + (mgroup * 2u + 1u) * 16u * 32u;
        #pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
            const float2 a0 = *(const float2 *)(ex0 + g * 32u + nt * 8u + 2u * qq);
            const float2 b0 = *(const float2 *)(ex1 + g * 32u + nt * 8u + 2u * qq);
            const float2 a1 = *(const float2 *)(ex0 + (g + 8u) * 32u + nt * 8u + 2u * qq);
            const float2 b1 = *(const float2 *)(ex1 + (g + 8u) * 32u + nt * 8u + 2u * qq);
            s[nt][0] = a0.x + b0.x; s[nt][1] = a0.y + b0.y;
            s[nt][2] = a1.x + b1.x; s[nt][3] = a1.y + b1.y;
        }

        /* The symmetric mask and the online softmax, per row. */
        #pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t row_g = g + 8u * h;
            const uint32_t token = r0 + mgroup * 4u + (row_g >> 2);
            const bool rv = (token < r1);
            const uint32_t pos = token - item_base;
            const uint32_t rb = (pos > half) ? pos - half : 0u;
            const uint32_t rl = (pos + half < item_len - 1u) ? pos + half : item_len - 1u;
            float peak = -INFINITY;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const uint32_t kk = base + nt * 8u + 2u * qq + e;
                    if (!rv || kk < rb || kk > rl) { s[nt][h * 2u + e] = -INFINITY; }
                    peak = fmaxf(peak, s[nt][h * 2u + e]);
                }
            }
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
            const bool tv = (peak > -INFINITY);
            const float mn = tv ? fmaxf(mrun[h], peak) : mrun[h];
            const float alpha = tv ? (mrun[h] == -INFINITY ? 0.0f : expf(mrun[h] - mn)) : 1.0f;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; e++) {
                    const float p = tv ? expf(s[nt][h * 2u + e] - mn) : 0.0f;
                    s[nt][h * 2u + e] = p;
                    sum += p;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            lrun[h] = lrun[h] * alpha + sum;
            mrun[h] = mn;
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t++) {
                o[t][h * 2u + 0u] *= alpha;
                o[t][h * 2u + 1u] *= alpha;
            }
        }

        /* V over the same tile (K is dead after the score barrier), then
         * O += P V over this warp's 256 output dims. */
        __syncthreads();
        for (uint32_t t = warp * 4u; t < warp * 4u + 4u; t++) {
            const uint32_t k_abs = base + t;
            const bool valid = (k_abs >= begin_all && k_abs <= last_all);
            const uint32_t slot = valid ? (item_base + k_abs) : item_base;
            const float *src = v + (uint64_t)slot * kv_stride;
            for (uint32_t c = lane; c < D / 8u; c += 32u) {
                eg2vq_attn_mma_stage8(kv + eg2vq_attn_mma_swz(t, c, D), src + c * 8u, valid);
            }
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t kk = 0; kk < 2u; kk++) {
            const uint32_t pa[4] = {
                eg2vq_attn_mma_pack(s[2u * kk][0], s[2u * kk][1]),
                eg2vq_attn_mma_pack(s[2u * kk][2], s[2u * kk][3]),
                eg2vq_attn_mma_pack(s[2u * kk + 1u][0], s[2u * kk + 1u][1]),
                eg2vq_attn_mma_pack(s[2u * kk + 1u][2], s[2u * kk + 1u][3])};
            #pragma unroll
            for (uint32_t t = 0; t < 32u; t += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3;
                const uint32_t krow = kk * 16u + (mat & 1u) * 8u + (lane & 7u);
                const uint32_t chunk = halfd * 32u + t + (mat >> 1);
                eg2vq_attn_mma_ldsm4(b, kv + eg2vq_attn_mma_swz(krow, chunk, D), true);
                eg2vq_attn_mma_f16(o[t], pa, b[0], b[1]);
                eg2vq_attn_mma_f16(o[t + 1u], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    const float inv[2] = {lrun[0] > 0.0f ? 1.0f / lrun[0] : 0.0f,
                          lrun[1] > 0.0f ? 1.0f / lrun[1] : 0.0f};
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t row_g = g + 8u * h;
        const uint32_t token = r0 + mgroup * 4u + (row_g >> 2);
        if (token >= r1) { continue; }
        const uint32_t head = row_g & 3u;
        float *dst = out + ((uint64_t)token * NHEAD + head) * D + halfd * HALF;
        #pragma unroll
        for (uint32_t t = 0; t < 32u; t++) {
            const uint32_t d = t * 8u + 2u * qq;
            dst[d] = o[t][h * 2u + 0u] * inv[h];
            dst[d + 1u] = o[t][h * 2u + 1u] * inv[h];
        }
    }
}

/* The tensor-core launcher.  Returns 0 for a shape it does not cover, and the
 * caller (eg2vq_attn below) then takes the scalar path; `off` is the caller's
 * own span table (n_items + 1 entries) and is used only to size the grid,
 * while the kernels derive the same mapping from `len` on the device. */
static inline int eg2vq_attn_mma(float *out, const float *q, const float *k, const float *v,
                                 const int32_t *pos, const int32_t *len,
                                 const uint32_t *off, uint32_t n_items,
                                 uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv,
                                 uint32_t head_dim, int32_t half_swa) {
    if (!out || !q || !k || !v || !pos || !len || !off || !n_items || !n_tok || !head_dim) { return 0; }
    if (n_head != 4u || n_head_kv > 2u) { return 0; }
    int slide = (n_head_kv == 2u && head_dim == 256u);
    const int full = (n_head_kv == 1u && head_dim == 512u);
    if (!slide && !full) { return 0; }
    /* Half the window on the sliding layers; the full layers read the item.
     * A negative half_swa on the sliding geometry is accepted as "no mask"
     * only because the kernel's bound then covers the item, which is what the
     * scalar kernel does too. */
    uint64_t tiles = 0;
    const uint32_t rtile = slide ? EG2VQ_ATTN_TC_RTILE_S : EG2VQ_ATTN_TC_RTILE_F;
    for (uint32_t i = 0; i < n_items; i++) {
        const uint32_t l = off[i + 1u] - off[i];
        tiles += (l + rtile - 1u) / rtile;
    }
    if (!tiles || tiles > 0x7fffffffull) { return 0; }
    if (slide) {
        static int smem_set = 0;
        if (!smem_set) {
            if (cudaFuncSetAttribute(eg2vq_attn_mma_slide_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)EG2VQ_ATTN_TC_SMEM_S) != cudaSuccess) { return 0; }
            smem_set = 1;
        }
        const dim3 grid(2u, (unsigned)tiles), block(EG2VQ_ATTN_TC_THREADS);
        eg2vq_attn_mma_slide_kernel<<<grid, block, EG2VQ_ATTN_TC_SMEM_S>>>(out, q, k, v, len, n_tok, half_swa);
    } else {
        static int smem_set = 0;
        if (!smem_set) {
            if (cudaFuncSetAttribute(eg2vq_attn_mma_full_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)EG2VQ_ATTN_TC_SMEM_F) != cudaSuccess) { return 0; }
            smem_set = 1;
        }
        const dim3 grid((unsigned)tiles), block(EG2VQ_ATTN_TC_THREADS);
        eg2vq_attn_mma_full_kernel<<<grid, block, EG2VQ_ATTN_TC_SMEM_F>>>(out, q, k, v, len, n_tok, half_swa);
    }
    return 1;
}

/* The scalar launcher: this kernel's original body, unchanged, kept as the
 * fallback for every shape the tensor-core path does not cover. */
static inline int eg2vq_attn_scalar(float *out, const float *q, const float *k, const float *v,
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
    /* the grouped pass 1 needs head_dim == 64*LAB for a LAB of 2, 4, 8 or 16
     * (both halves of the tower's head dims, 256 and 512); every other shape
     * takes the LAB = 0 instantiation, which is the lane-per-key loop alone. */
    const uint32_t lab = (vec4 && (head_dim % 64u) == 0u) ? (head_dim / 64u) : 0u;
    const dim3 grid(n_tok), block(EG2VQ_ATTN_WARPS * 32);
    #define EG2VQ_ATTN_LAUNCH(L) \
        eg2vq_attn_kernel<L><<<grid, block>>>(out, q, k, v, pos, len, n_tok, n_head, \
                                              n_head_kv, head_dim, half_swa, vec4)
    switch (lab) {
        case  2u: EG2VQ_ATTN_LAUNCH( 2); break;
        case  4u: EG2VQ_ATTN_LAUNCH( 4); break;
        case  8u: EG2VQ_ATTN_LAUNCH( 8); break;
        case 16u: EG2VQ_ATTN_LAUNCH(16); break;
        default:  EG2VQ_ATTN_LAUNCH( 0); break;
    }
    #undef EG2VQ_ATTN_LAUNCH
    return 1;
}

/* The public entry: the tensor-core path when the shape and the span table fit
 * it, the scalar kernel otherwise.  `off` is the caller's span table (n_items +
 * 1 rows), the same table the caller already derives `pos` and `len` from; it
 * reaches this launcher only to size the grid exactly, and passing it NULL (or
 * zero items) keeps the scalar path, which is what a shape the tensor-core
 * kernels do not cover needs anyway.  A batch must pass its real table: the
 * tensor-core kernels map a CTA to exactly one item, so the tile count they are
 * launched with is that table's, not n_tok's. */
/* THE DISPATCH HAS A ROW THRESHOLD ONLY IN THE BENCH'S MEASUREMENTS, NOT IN THE
 * CODE, and this is deliberate.  Measured (bench_eg2vq_attn, scalar against
 * tensor cores in one process, 20 repeats a row): 27 rows 0.97x, 115 rows
 * 1.01x, 184 rows 1.88x, 294 rows 2.26x, 565 rows 4.74x, 1,414 rows 9.26x; a
 * 64-item block of 20-token items (1,280 rows) 1.56x and an 8-item block of
 * 115-token items (920 rows) 4.26x -- so a call below ~184 rows gains nothing
 * and loses at most 3%, and the axis is the call's ROW count, not the item
 * length.  A threshold on it CANNOT be shipped: unit 27's contract is that a
 * batched item's vector is its single-item vector BYTE FOR BYTE, and a
 * call-sized threshold picks the arithmetic from the whole call's length, so a
 * 20-token item would take the scalar path alone and the tensor-core one inside
 * a 64-item batch.  One arithmetic per shape, call-independent, is worth more
 * than the 3% the smallest single item would save. */
static inline int eg2vq_attn(float *out, const float *q, const float *k, const float *v,
                             const int32_t *pos, const int32_t *len,
                             uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv,
                             uint32_t head_dim, int32_t half_swa,
                             const uint32_t *off, uint32_t n_items) {
    if (!out || !q || !k || !v || !pos || !len || !n_tok || !head_dim) { return 0; }
    if (n_head != EG2VQ_ATTN_WARPS || n_head > EG2VQ_ATTN_MAX_HEAD) { return 0; }
    if (!n_head_kv || (n_head % n_head_kv)) { return 0; }
    /* The tensor-core kernels are built for this tower's two geometries and
     * 16-byte-aligned operands (their fragment loads are float4). */
    if (((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)out) % 16u == 0u &&
        eg2vq_attn_mma(out, q, k, v, pos, len, off, n_items, n_tok, n_head, n_head_kv,
                       head_dim, half_swa)) {
        return 1;
    }
    return eg2vq_attn_scalar(out, q, k, v, pos, len, n_tok, n_head, n_head_kv, head_dim, half_swa);
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
