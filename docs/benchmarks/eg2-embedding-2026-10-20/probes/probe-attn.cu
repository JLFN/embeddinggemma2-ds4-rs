/* probe-attn.cu — ablation of eg2vq_attn_kernel (the gemma_vq text tower's
 * attention, cuda/eg2vq_attn.cuh).
 *
 * WHAT IT IS: the shipped kernel's body, copied into a template<int V>, so each
 * variant gets its own clean codegen, with exactly ONE component changed per
 * variant and the register footprint held constant.  Nothing is removed by
 * deleting arithmetic: a memory operand is replaced by a register value (or by
 * a differently addressed one), so every variant keeps the shipped loop counts,
 * the FMA counts and the four-chain accumulator mapping of the shipped body.
 * The numbers are WRONG for every V != 0; the only thing read from this file is
 * the time.  The method is the one probes/probe-vq.cu established for the VQ
 * GEMM: read that file's header first if this is your first contact with it.
 *
 *   V0  baseline: byte-for-byte the body of cuda/eg2vq_attn.cuh's vector path
 *       (pass 1 scores into the shared plane + the lane's max; pass 2 expf and
 *       the sum; pass 3 the weighted sum over keys)
 *   V1  pass 1's K address is block-uniform (the item's first key row instead of
 *       kh0 + (k0+kk)*kv_stride): same instruction count, one 128 B line per
 *       warp load instead of 32, which is the sector/line question the GEMM's
 *       own V5 asked
 *   V2  pass 1's K operand comes from a register instead of memory: no K
 *       traffic at all, same FMA count and the same chain mapping
 *   V3  pass 2 without expf (the linear value); the loop, the shared read and
 *       the shared write are unchanged, so this isolates the SFU exponentials
 *   V4  pass 3's V operand comes from a register (same FMA count) — the K/V
 *       load pattern's pass-3 half
 *   V5  pass 3 reads no score from the shared plane (the p factor is a register
 *       derived from kk, so the inner loop cannot collapse); the V load and the
 *       four FMAs are unchanged.  Pass 2 still reads pass 1's stores, so the
 *       plane stays live (SHARED:32768); what dies is pass 2's own store, which
 *       nothing reads any more — the plane's read side, in isolation
 *   V6  pass 1 stores nothing into the plane (the dot product, the max, the K
 *       traffic and the loop are unchanged); pass 2 then reads uninitialized
 *       shared, which is the point — the plane's write side
 *   V7  pass 1 and the row max only (no passes 2 and 3).  Nothing reads the
 *       plane afterwards, so ptxas drops the stores with it (SHARED:0) — read
 *       this row as pass 1's arithmetic-plus-K-traffic bound, and V6 as the
 *       isolated write side
 *   V8  passes 1 and 2 only (no pass 3)
 *   V9  every component the other variants isolate, removed at once (V2 + V3 +
 *       V4 + V5 + V6).  NOT a live kernel: an upper bound on what removing all
 *       of them together could buy.  The q read of pass 1 stays, and that
 *       address is uniform across the warp and across kk, so L1 serves it.
 *
 * Two things this file pins deliberately, both from the GEMM unit's lessons:
 *
 *  - Its own geometry (PA_* below), with static_asserts against the header.  A
 *    probe that silently follows the header stops being the instrument its
 *    numbers were taken with (probe-vq.cu crashed that way after the header
 *    moved).  If a static_assert fires because the header changed, the probe
 *    must be RE-DERIVED from the new body and re-measured — not re-pointed.
 *  - Its V0 against the SHIPPED launcher, on the device, byte for byte: the
 *    run prints "V0 vs shipped: identical" over the whole output, so a variant
 *    table's baseline can never be a copy of something else.
 *
 * usage: probe-attn [repeats]     (default 20; build it against the shipped
 *             header — see the probe's compile line in vq-gemm-rows.md section 7
 *             for the flags and include path this tree uses)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cuda/eg2vq_attn.cuh"

/* PINNED GEOMETRY (see the note above): if these asserts fire, the header
 * moved and this probe's numbers no longer describe it. */
#define PA_WARPS     4
#define PA_KEYS_MAX  2048
#define PA_VEC_STEP  128
static_assert(EG2VQ_ATTN_WARPS == PA_WARPS, "header warps moved: re-derive the probe");
static_assert(EG2VQ_ATTN_KEYS_MAX == PA_KEYS_MAX, "header plane moved: re-derive the probe");
static_assert(EG2VQ_ATTN_VEC_STEP == PA_VEC_STEP, "header vec step moved: re-derive the probe");

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

/* which components a variant removes */
template <int V> struct F {
    static constexpr bool UNI_K    = (V == 1);
    static constexpr bool REG_K    = (V == 2) || (V == 9);
    static constexpr bool NO_EXP   = (V == 3) || (V == 9);
    static constexpr bool REG_V    = (V == 4) || (V == 9);
    static constexpr bool REG_P3   = (V == 5) || (V == 9);   /* pass 3's score */
    static constexpr bool REG_P2   = (V == 9);               /* pass 2's score */
    static constexpr bool NO_STORE = (V == 6) || (V == 9);
    static constexpr bool P1_ONLY  = (V == 7);
    static constexpr bool P12_ONLY = (V == 8);
};

/* the K operand from a register: identical FMA count, identical chain mapping
 * (element i feeds chain i%4) to eg2vq_attn_dot4, so only the K traffic moves */
__device__ __forceinline__ static float pa_dot4_regk(const float *a, float4 ky, uint32_t n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint32_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        const float4 x0 = *(const float4 *)(a + i);
        const float4 x1 = *(const float4 *)(a + i + 4u);
        a0 += x0.x * ky.x; a1 += x0.y * ky.y; a2 += x0.z * ky.z; a3 += x0.w * ky.w;
        a0 += x1.x * ky.x; a1 += x1.y * ky.y; a2 += x1.z * ky.z; a3 += x1.w * ky.w;
    }
    for (; i + 4u <= n; i += 4u) {
        const float4 x = *(const float4 *)(a + i);
        a0 += x.x * ky.x; a1 += x.y * ky.y; a2 += x.z * ky.z; a3 += x.w * ky.w;
    }
    for (; i < n; i++) { a0 += a[i] * ky.x; }
    return (a0 + a1) + (a2 + a3);
}

template <int V>
__global__ void pa_kernel(
        float *__restrict__ out,
        const float *__restrict__ q,
        const float *__restrict__ k,
        const float *__restrict__ v,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ len,
        uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        int32_t half_swa) {
    __shared__ float plane[PA_WARPS * PA_KEYS_MAX];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (warp >= n_head) { return; }
    const uint32_t qrow = blockIdx.x;
    const uint32_t group = n_head / n_head_kv;
    const uint32_t kvh = warp / group;
    const uint64_t kv_stride = (uint64_t)n_head_kv * head_dim;

    /* the window, exactly as the shipped kernel computes it */
    const uint32_t p = (uint32_t)pos[qrow];
    const uint32_t nitem = (uint32_t)len[qrow];
    const uint32_t base = qrow - p;
    uint32_t k0 = base, k1 = base + nitem - 1u;
    if (half_swa >= 0) {
        k0 = base + ((p > (uint32_t)half_swa) ? p - (uint32_t)half_swa : 0u);
        const uint32_t hi = p + (uint32_t)half_swa;
        k1 = base + ((hi < nitem - 1u) ? hi : nitem - 1u);
    }
    if (k0 > n_tok - 1u) { k0 = n_tok - 1u; }
    if (k1 > n_tok - 1u) { k1 = n_tok - 1u; }
    if (k1 < k0) { k1 = k0; }
    const uint32_t nkey = k1 - k0 + 1u;
    if (nkey > (uint32_t)PA_KEYS_MAX) { return; }

    const float *qh = q + (uint64_t)qrow * (uint64_t)n_head * head_dim + (uint64_t)warp * head_dim;
    const float *kh0 = k + (uint64_t)kvh * head_dim;
    float *scores = plane + (size_t)warp * PA_KEYS_MAX;

    /* pass 1: each lane's keys' scores, into shared, and this lane's maximum */
    float m = -FLT_MAX;
    if (F<V>::REG_K) {
        const float4 ky = *(const float4 *)kh0;          /* the K operand, once */
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float s = pa_dot4_regk(qh, ky, head_dim);
            if (!F<V>::NO_STORE) { scores[kk] = s; }
            m = fmaxf(m, s);
        }
    } else if (F<V>::UNI_K) {
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float s = eg2vq_attn_dot4(qh, kh0, head_dim);  /* one line, same count */
            scores[kk] = s;
            m = fmaxf(m, s);
        }
    } else {
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float s = eg2vq_attn_dot4(qh, kh0 + (uint64_t)(k0 + kk) * kv_stride, head_dim);
            if (!F<V>::NO_STORE) { scores[kk] = s; }
            m = fmaxf(m, s);
        }
    }
    const float M = eg2vq_attn_wmax(m);
    if (F<V>::P1_ONLY) { if (lane == 0u) { out[qrow] = M; } return; }

    /* pass 2: exponentials and their sum */
    float sum = 0.0f;
    for (uint32_t kk = lane; kk < nkey; kk += 32u) {
        const float sc = F<V>::REG_P2 ? (float)(kk & 1023u) * 1.0e-3f : scores[kk];
        const float pp = F<V>::NO_EXP ? sc : expf(sc - M);
        if (!F<V>::NO_STORE) { scores[kk] = pp; }
        sum += pp;
    }
    const float S = eg2vq_attn_wsum(sum);
    const float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    if (F<V>::P12_ONLY) { if (lane == 0u) { out[qrow] = S; } return; }

    /* pass 3: the weighted sum over keys, output dimensions spread over lanes */
    float *oh = out + (uint64_t)qrow * (uint64_t)n_head * head_dim + (uint64_t)warp * head_dim;
    const float *vh0 = v + (uint64_t)kvh * head_dim;
    for (uint32_t d = lane * 4u; d + 4u <= head_dim; d += (uint32_t)PA_VEC_STEP) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t kk = 0; kk < nkey; kk++) {
            const float pp = F<V>::REG_P3 ? (float)(kk & 1023u) * 1.0e-3f : scores[kk];
            float4 vv;
            if (F<V>::REG_V) {
                vv = make_float4((float)(kk & 1023u) * 1.0e-3f, 0.5f, 0.25f, 0.75f);
            } else {
                vv = *(const float4 *)(vh0 + (uint64_t)(k0 + kk) * kv_stride + d);
            }
            acc.x += pp * vv.x;
            acc.y += pp * vv.y;
            acc.z += pp * vv.z;
            acc.w += pp * vv.w;
        }
        acc.x *= inv; acc.y *= inv; acc.z *= inv; acc.w *= inv;
        *(float4 *)(oh + d) = acc;
    }
}

template <int V>
static bool pa_launch(float *out, const float *q, const float *k, const float *v,
                      const int32_t *pos, const int32_t *len, uint32_t n_tok,
                      uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
                      int32_t half_swa) {
    if (!n_tok || n_head != PA_WARPS || !n_head_kv || (n_head % n_head_kv)) { return false; }
    if (head_dim % PA_VEC_STEP) { return false; }                 /* the vector path only */
    if ((((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)out) % 16u) != 0u) { return false; }
    pa_kernel<V><<<n_tok, PA_WARPS * 32>>>(out, q, k, v, pos, len, n_tok, n_head, n_head_kv,
                                           head_dim, half_swa);
    return true;
}

/* the probe's shapes: the tower's own (see tests/bench_eg2vq_attn.cu for why
 * exactly these) */
struct shape {
    const char *tag;
    std::vector<uint32_t> item_len;
    uint32_t head_dim, n_head_kv;
    int32_t half_swa;
};

static uint32_t s_rng = 20261021u;
static float rnd_unit(void) {
    s_rng ^= s_rng << 13u;
    s_rng ^= s_rng >> 17u;
    s_rng ^= s_rng << 5u;
    return ((float)(s_rng & 0xffffu) / 32768.0f) - 1.0f;
}

static uint32_t keys_of(uint32_t p, uint32_t nitem, int32_t half_swa) {
    uint32_t k0 = 0, k1 = nitem - 1u;
    if (half_swa >= 0) {
        k0 = (p > (uint32_t)half_swa) ? p - (uint32_t)half_swa : 0u;
        const uint32_t hi = p + (uint32_t)half_swa;
        k1 = (hi < nitem - 1u) ? hi : nitem - 1u;
    }
    return k1 - k0 + 1u;
}

/* one variant's median of `reps` timed launches on one resident input */
template <int V>
static double pa_time(const float *dq, const float *dk, const float *dv, float *dy,
                      const int32_t *dpos, const int32_t *dlen, uint32_t n_tok,
                      uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
                      int32_t half_swa, int reps, bool &ok) {
    ok = true;
    std::vector<float> ms;
    for (int rep = 0; rep < reps + 3; rep++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        if (!pa_launch<V>(dy, dq, dk, dv, dpos, dlen, n_tok, n_head, n_head_kv, head_dim, half_swa)) {
            ok = false;
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            break;
        }
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float msec = 0.0f;
        CK(cudaEventElapsedTime(&msec, a, b));
        if (rep >= 3) { ms.push_back(msec); }
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    if (ms.empty()) { ok = false; return 0.0; }
    for (size_t i = 0; i + 1 < ms.size(); i++) {
        for (size_t j = i + 1; j < ms.size(); j++) {
            if (ms[j] < ms[i]) { const float t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
        }
    }
    return (double)ms[ms.size() / 2];
}

int main(int argc, char **argv) {
    const int reps = (argc > 1) ? atoi(argv[1]) : 20;
    const uint32_t n_head = PA_WARPS;

    const shape shapes[] = {
        { "page294_s",   { 294 }, 256, 2, 512 },
        { "page565_s",   { 565 }, 256, 2, 512 },
        { "max1414_s",   { 1414 }, 256, 2, 512 },
        { "batch8x565_s", { 565, 565, 565, 565, 565, 565, 565, 565 }, 256, 2, 512 },
        { "batch8x565_f", { 565, 565, 565, 565, 565, 565, 565, 565 }, 512, 1, -1 },
    };

    printf("# probe-attn — ablation of eg2vq_attn_kernel, %d repeats per row, vector path only\n", reps);
    printf("# V0 base | V1 K one line | V2 K in register | V3 no expf | V4 V in register\n");
    printf("# V5 no plane read | V6 no plane store | V7 pass1 only | V8 pass1+2 | V9 all removed (UB)\n");
    printf("%-14s %7s %7s %5s %12s %8s\n", "shape", "rows", "keys", "var", "ms_med", "xV0");

    int fail = 0;
    for (const shape &s : shapes) {
        std::vector<int32_t> pos, len;
        for (uint32_t l : s.item_len) {
            for (uint32_t t = 0; t < l; t++) { pos.push_back((int32_t)t); len.push_back((int32_t)l); }
        }
        const uint32_t n_tok = (uint32_t)pos.size();
        const uint64_t q_stride = (uint64_t)n_head * s.head_dim;
        const uint64_t kv_stride = (uint64_t)s.n_head_kv * s.head_dim;
        std::vector<float> hq((size_t)n_tok * q_stride), hk((size_t)n_tok * kv_stride),
                           hv((size_t)n_tok * kv_stride);
        for (float &x : hq) { x = rnd_unit(); }
        for (float &x : hk) { x = rnd_unit(); }
        for (float &x : hv) { x = rnd_unit(); }
        std::vector<float> hy((size_t)n_tok * q_stride);

        float *dq = NULL, *dk = NULL, *dv = NULL, *dy = NULL;
        int32_t *dpos = NULL, *dlen = NULL;
        CK(cudaMalloc(&dq, hq.size() * 4)); CK(cudaMalloc(&dk, hk.size() * 4));
        CK(cudaMalloc(&dv, hv.size() * 4)); CK(cudaMalloc(&dy, hy.size() * 4));
        CK(cudaMalloc(&dpos, pos.size() * 4)); CK(cudaMalloc(&dlen, len.size() * 4));
        CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dlen, len.data(), len.size() * 4, cudaMemcpyHostToDevice));

        uint64_t keys = 0;
        for (uint32_t r = 0; r < n_tok; r++) {
            keys += keys_of((uint32_t)pos[r], (uint32_t)len[r], s.half_swa);
        }
        const double flop = 4.0 * (double)s.head_dim * (double)keys * (double)n_head;

        /* V0 against the SHIPPED launcher, byte for byte (the anchor: this
         * table's baseline must be the shipped kernel, not something else) */
        bool ok = false;
        const double v0a = pa_time<0>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv,
                                     s.head_dim, s.half_swa, reps, ok);
        if (!ok) { printf("%-14s REFUSED\n", s.tag); cudaFree(dq); cudaFree(dk); cudaFree(dv);
                   cudaFree(dy); cudaFree(dpos); cudaFree(dlen); fail = 1; continue; }
        std::vector<float> yprobe(hy.size()), yship(hy.size());
        CK(cudaMemcpy(yprobe.data(), dy, yprobe.size() * 4, cudaMemcpyDeviceToHost));
        if (!eg2vq_attn(dy, dq, dk, dv, dpos, dlen, n_tok, n_head, s.n_head_kv,
                        s.head_dim, s.half_swa)) {
            printf("%-14s the shipped launcher refused a shape the probe accepts\n", s.tag);
            fail = 1;
        }
        CK(cudaMemcpy(yship.data(), dy, yship.size() * 4, cudaMemcpyDeviceToHost));
        if (memcmp(yprobe.data(), yship.data(), yprobe.size() * 4) != 0) {
            printf("%-14s V0 IS NOT THE SHIPPED KERNEL (outputs differ)\n", s.tag);
            fail = 1;
        }

        /* the variant ladder for this shape */
        struct row { const char *var; double ms; int v; };
        const row rows[] = {
            { "V0a", v0a, 0 },
            { "V1", 0.0, 1 }, { "V2", 0.0, 2 }, { "V3", 0.0, 3 }, { "V4", 0.0, 4 },
            { "V5", 0.0, 5 }, { "V6", 0.0, 6 }, { "V7", 0.0, 7 }, { "V8", 0.0, 8 },
            { "V9", 0.0, 9 },
        };
        double t[10];
        t[0] = v0a;
        t[1] = pa_time<1>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[2] = pa_time<2>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[3] = pa_time<3>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[4] = pa_time<4>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[5] = pa_time<5>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[6] = pa_time<6>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[7] = pa_time<7>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[8] = pa_time<8>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        t[9] = pa_time<9>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa, reps, ok);
        const double v0b = pa_time<0>(dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv,
                                     s.head_dim, s.half_swa, reps, ok);
        const double v0 = 0.5 * (v0a + v0b);

        for (const row &r : rows) {
            const double ms = t[r.v];
            printf("%-14s %7u %7llu %5s %12.4f %8.2f  %7.1f gflop/s\n",
                   s.tag, n_tok, (unsigned long long)keys, r.var, ms, v0 / ms,
                   flop / (ms * 1.0e6));
        }
        printf("%-14s %7u %7llu %5s %12.4f %8.2f  (V0 round 2; V0a = %.4f)\n",
               s.tag, n_tok, (unsigned long long)keys, "V0b", v0b, v0 / v0b, v0a);

        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
        cudaFree(dpos); cudaFree(dlen);
    }
    printf("%s\n", fail ? "PROBE_ATTN_FAIL" : "PROBE_ATTN_OK");
    return fail;
}
