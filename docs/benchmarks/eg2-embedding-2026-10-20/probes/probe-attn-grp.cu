/* probe-attn-grp.cu — the candidate fix for what probe-attn.cu named: pass 1's
 * K access pattern.
 *
 * WHY (all measured, see probe-attn.txt and the record that cites it): the
 * shipped pass 1 gives each lane a whole K row, so one warp load instruction
 * touches 32 different 128 B lines and uses 16 B of each.  Making that address
 * block-uniform (one line, same instruction count) was 2.05x-2.25x; taking the
 * K operand out of memory altogether was 2.25x-2.66x.  The exponentials, the
 * shared score plane's write side and its read side each measured 1.00x-1.08x,
 * i.e. free.  So the K line pattern is the limiter and nothing else in pass 1 is.
 *
 * WHAT THIS PROBE IS: the shipped kernel, copied verbatim (passes 2 and 3, the
 * window arithmetic, the plane, the launcher geometry), with pass 1 offered in
 * two forms selected by a template parameter:
 *
 *   ON=0  the shipped pass 1 (lane owns a whole key row), which must reproduce
 *         the shipped launcher's output byte for byte — the probe's anchor,
 *         printed as the "V0 vs shipped" line
 *   ON=1  the GROUPED pass 1: LAB lanes cooperate on one key and each lane owns
 *         a 4*GS-element slice of it, so the LAB lanes of a group read
 *         consecutive float4s and one warp load covers whole 128 B lines.  Each
 *         lane keeps its own slice of the query row in registers (qv[GS]),
 *         loaded once per row instead of once per key, and the lane's LAB
 *         partial sums are reduced with a butterfly inside the group.  The
 *         window is consumed in chunks of KC = 32/LAB keys; the remainder (at
 *         most KC-1 keys) uses the shipped lane-per-key path, which is kept
 *         verbatim for exactly that purpose.
 *
 * The grouped form's arithmetic is NOT bit-identical to the shipped form: the
 * dot product's 256 (or 512) terms are summed as LAB partial sums reduced by a
 * butterfly instead of by the shipped four chains alone.  The gates tolerate
 * no reordering at all (the shipped kernel's own four chains already differ
 * from the f64 CPU reference by ~1e-7 relative and pass), so this probe prints
 * the cosine and the maximum relative difference of every variant against the
 * shipped output, per shape.  A throughput number from a variant whose cos is
 * not 1.000000000 is still read for time only, but a variant that does not
 * agree to ~1e-6 would not be landable in the first place.
 *
 * The parameter pairs are fixed by head_dim: the lane's float4 chunks are 4*LAB
 * elements apart, so LAB*4*GS == head_dim and GS is the number of chunks one
 * lane owns.  (8, 8) and (4, 16) therefore cover head_dim 256, and (16, 8) and
 * (8, 16) cover 512.  A larger GS means longer-lived query registers, but LAB is
 * what sets the coalescing: the LAB lanes of a group cover 4*LAB consecutive
 * elements, so LAB=8 reads exactly one 128 B line per warp load and LAB=4 reads
 * two 32 B sectors of one line.
 *
 * usage: probe-attn-grp [repeats]   (default 20)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cuda/eg2vq_attn.cuh"

#define PA_WARPS     4
#define PA_KEYS_MAX  2048
#define PA_VEC_STEP  128
static_assert(EG2VQ_ATTN_WARPS == PA_WARPS, "header warps moved: re-derive the probe");
static_assert(EG2VQ_ATTN_KEYS_MAX == PA_KEYS_MAX, "header plane moved: re-derive the probe");
static_assert(EG2VQ_ATTN_VEC_STEP == PA_VEC_STEP, "header vec step moved: re-derive the probe");

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

/* LAB*4*GS must equal head_dim */
template <int LAB, int GS, int ON>
__global__ void gr_kernel(
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
    const uint64_t q_stride = (uint64_t)n_head * head_dim;
    const uint64_t kv_stride = (uint64_t)n_head_kv * head_dim;

    const uint32_t p = (uint32_t)pos[qrow];
    const uint32_t nitem = (uint32_t)len[qrow];
    const uint32_t base = qrow - p;
    uint32_t k0 = base, k1 = base + nitem - 1u;
    if (half_swa >= 0) {
        k0 = base + ((p > (uint32_t)half_swa) ? p - (uint32_t)half_swa : 0u);
        const uint32_t hi = p + (uint32_t)half_swa;
        k1 = base + ((hi < nitem - 1u) ? hi : nitem - 1u);
    }
    uint32_t nkey = k1 - k0 + 1u;
    if (k0 > n_tok - 1u) { k0 = n_tok - 1u; }
    if (k1 > n_tok - 1u) { k1 = n_tok - 1u; }
    if (k1 < k0) { k0 = k1; }
    nkey = k1 - k0 + 1u;
    if (nkey > (uint32_t)PA_KEYS_MAX) { return; }

    const float *qh = q + (uint64_t)qrow * q_stride + (uint64_t)warp * head_dim;
    const float *kh0 = k + (uint64_t)kvh * head_dim;
    float *scores = plane + (size_t)warp * PA_KEYS_MAX;

    /* ---- pass 1 ---------------------------------------------------------- */
    float m = -FLT_MAX;
    if (ON) {
        constexpr uint32_t KC   = 32u / (uint32_t)LAB;
        /* The lane's slice: its float4 chunks are 4*LAB elements apart, so the
         * LAB lanes of a group cover 4*LAB consecutive elements and each warp
         * load touches whole lines.  LAB*SPAN == head_dim, i.e. GS = head_dim /
         * (4*LAB) chunks per lane. */
        constexpr uint32_t SPAN = 4u * (uint32_t)LAB;
        const uint32_t msub = lane & ((uint32_t)LAB - 1u);   /* this lane's slice */
        const uint32_t gsub = lane / (uint32_t)LAB;          /* this lane's key in the chunk */
        float4 qv[GS];
        #pragma unroll
        for (int j = 0; j < GS; j++) {
            qv[j] = *(const float4 *)(qh + 4u * msub + SPAN * (uint32_t)j);
        }
        uint32_t kk = 0;
        for (; kk + KC <= nkey; kk += KC) {
            const float *krow = kh0 + (uint64_t)(k0 + kk + gsub) * kv_stride;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll
            for (int j = 0; j < GS; j++) {
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
                scores[kk + gsub] = acc;
                m = fmaxf(m, acc);
            }
        }
        for (uint32_t r = kk + lane; r < nkey; r += 32u) {
            const float s = eg2vq_attn_dot4(qh, kh0 + (uint64_t)(k0 + r) * kv_stride, head_dim);
            scores[r] = s;
            m = fmaxf(m, s);
        }
    } else {
        for (uint32_t kk = lane; kk < nkey; kk += 32u) {
            const float s = eg2vq_attn_dot4(qh, kh0 + (uint64_t)(k0 + kk) * kv_stride, head_dim);
            scores[kk] = s;
            m = fmaxf(m, s);
        }
    }
    const float M = eg2vq_attn_wmax(m);

    /* ---- pass 2 and pass 3, verbatim from the shipped kernel ------------- */
    float sum = 0.0f;
    for (uint32_t kk = lane; kk < nkey; kk += 32u) {
        const float p = expf(scores[kk] - M);
        scores[kk] = p;
        sum += p;
    }
    const float S = eg2vq_attn_wsum(sum);
    const float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;

    float *oh = out + (uint64_t)qrow * q_stride + (uint64_t)warp * head_dim;
    const float *vh0 = v + (uint64_t)kvh * head_dim;
    for (uint32_t d = lane * 4u; d + 4u <= head_dim; d += (uint32_t)PA_VEC_STEP) {
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
}

struct variant { const char *tag; int on, lab, gs; };

static bool gr_launch(const variant &vr, float *out, const float *q, const float *k, const float *v,
                      const int32_t *pos, const int32_t *len, uint32_t n_tok, uint32_t n_head,
                      uint32_t n_head_kv, uint32_t head_dim, int32_t half_swa) {
    if (!n_tok || n_head != PA_WARPS || !n_head_kv || (n_head % n_head_kv)) { return false; }
    if (head_dim % PA_VEC_STEP) { return false; }
    if ((((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)out) % 16u) != 0u) { return false; }
    const dim3 blk(PA_WARPS * 32);
    #define GO(LAB, GS, ON) \
        gr_kernel<LAB, GS, ON><<<n_tok, blk>>>(out, q, k, v, pos, len, n_tok, n_head, \
                                               n_head_kv, head_dim, half_swa); return true;
    if (vr.on == 0) {
        if (head_dim == 256u) { GO(8, 8, 0) }
        if (head_dim == 512u) { GO(8, 16, 0) }
        return false;
    }
    if (head_dim == 256u) {
        if (vr.lab == 8 && vr.gs == 8)  { GO(8, 8, 1) }
        if (vr.lab == 4 && vr.gs == 16) { GO(4, 16, 1) }
    }
    if (head_dim == 512u) {
        if (vr.lab == 16 && vr.gs == 8) { GO(16, 8, 1) }
        if (vr.lab == 8 && vr.gs == 16) { GO(8, 16, 1) }
    }
    #undef GO
    return false;
}

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

static void agree(const std::vector<float> &a, const std::vector<float> &b,
                  double &cos, double &maxrel) {
    double dot = 0.0, na = 0.0, nb = 0.0, mx = 0.0, mref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > mx) { mx = d; }
        if (fabs((double)a[i]) > mref) { mref = fabs((double)a[i]); }
    }
    cos = (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 0.0;
    maxrel = (mref > 0.0) ? mx / mref : 0.0;
}

static double gr_time(const variant &var, const float *dq, const float *dk, const float *dv,
                      float *dy, const int32_t *dpos, const int32_t *dlen, uint32_t n_tok,
                      uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, int32_t half_swa,
                      int reps, std::vector<float> *copy) {
    std::vector<float> ms;
    for (int rep = 0; rep < reps + 3; rep++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        if (!gr_launch(var, dy, dq, dk, dv, dpos, dlen, n_tok, n_head, n_head_kv, head_dim, half_swa)) {
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            return -1.0;
        }
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float msec = 0.0f;
        CK(cudaEventElapsedTime(&msec, a, b));
        if (rep >= 3) { ms.push_back(msec); }
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    if (ms.empty()) { return -1.0; }
    for (size_t i = 0; i + 1 < ms.size(); i++) {
        for (size_t j = i + 1; j < ms.size(); j++) {
            if (ms[j] < ms[i]) { const float t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
        }
    }
    if (copy) {
        copy->resize((size_t)n_tok * n_head * head_dim);
        CK(cudaMemcpy(copy->data(), dy, copy->size() * 4, cudaMemcpyDeviceToHost));
    }
    return (double)ms[ms.size() / 2];
}

int main(int argc, char **argv) {
    const int reps = (argc > 1) ? atoi(argv[1]) : 20;
    const uint32_t n_head = PA_WARPS;

    const shape shapes[] = {
        { "page294_s",    { 294 }, 256, 2, 512 },
        { "page565_s",    { 565 }, 256, 2, 512 },
        { "max1414_s",    { 1414 }, 256, 2, 512 },
        { "batch8x565_s", { 565, 565, 565, 565, 565, 565, 565, 565 }, 256, 2, 512 },
        { "max1414_f",    { 1414 }, 512, 1, -1 },
        { "batch8x565_f", { 565, 565, 565, 565, 565, 565, 565, 565 }, 512, 1, -1 },
    };
    /* The two grouped forms per head_dim, and the shipped form as the baseline. */
    const variant vs_s[] = { { "S", 0, 0, 0 } };
    const variant vs_g8_256[]  = { { "G8x8",  1, 8,  8  } };
    const variant vs_g4_256[]  = { { "G4x16", 1, 4,  16 } };
    const variant vs_g16_512[] = { { "G16x8", 1, 16, 8  } };
    const variant vs_g8_512[]  = { { "G8x16", 1, 8,  16 } };

    printf("# probe-attn-grp — shipped pass 1 vs the grouped pass 1, %d repeats per row\n", reps);
    printf("# LAB*4*GS == head_dim; KC = 32/LAB keys per warp per chunk; cos/rel = vs the shipped output\n");
    printf("%-14s %7s %7s %6s %12s %8s %14s %10s\n",
           "shape", "rows", "keys", "var", "ms_med", "xS", "cos", "maxrel");

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

        /* the shipped kernel itself: the probe's ON=0 must match it byte for
         * byte, and every grouped variant is compared against its output */
        std::vector<float> shipped((size_t)n_tok * q_stride);
        double t_ship = gr_time(vs_s[0], dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv,
                                s.head_dim, s.half_swa, reps, &shipped);
        if (t_ship < 0.0) {
            printf("%-14s REFUSED\n", s.tag);
            fail = 1;
            cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
            cudaFree(dpos); cudaFree(dlen);
            continue;
        }
        {
            std::vector<float> yship((size_t)n_tok * q_stride);
            if (!eg2vq_attn(dy, dq, dk, dv, dpos, dlen, n_tok, n_head, s.n_head_kv,
                            s.head_dim, s.half_swa)) {
                printf("%-14s the shipped launcher refused\n", s.tag);
                fail = 1;
            }
            CK(cudaMemcpy(yship.data(), dy, yship.size() * 4, cudaMemcpyDeviceToHost));
            if (memcmp(yship.data(), shipped.data(), shipped.size() * 4) != 0) {
                printf("%-14s ON=0 IS NOT THE SHIPPED KERNEL\n", s.tag);
                fail = 1;
            }
        }
        /* the shipped form again AFTER the variants: one interleaved window */
        std::vector<double> rtimes;
        double cos = 1.0, rel = 0.0;
        agree(shipped, shipped, cos, rel);
        printf("%-14s %7u %7llu %6s %12.4f %8.2f %14.9f %10.1e\n",
               s.tag, n_tok, (unsigned long long)keys, "S-a", t_ship, 1.0, cos, rel);

        const variant *list = NULL;
        size_t nlist = 0;
        if (s.head_dim == 256u) {
            list = vs_g8_256; nlist = 1;
        } else {
            list = vs_g16_512; nlist = 1;
        }
        for (size_t i = 0; i < nlist; i++) {
            std::vector<float> got;
            const double t = gr_time(list[i], dq, dk, dv, dy, dpos, dlen, n_tok, n_head,
                                     s.n_head_kv, s.head_dim, s.half_swa, reps, &got);
            if (t < 0.0) { printf("%-14s %6s REFUSED\n", s.tag, list[i].tag); continue; }
            agree(got, shipped, cos, rel);
            printf("%-14s %7u %7llu %6s %12.4f %8.2f %14.9f %10.1e\n",
                   s.tag, n_tok, (unsigned long long)keys, list[i].tag, t, t_ship / t, cos, rel);
        }
        /* the second grouped form for the same head_dim, measured after */
        const variant alt = (s.head_dim == 256u) ? vs_g4_256[0] : vs_g8_512[0];
        {
            std::vector<float> got;
            const double t = gr_time(alt, dq, dk, dv, dy, dpos, dlen, n_tok, n_head, s.n_head_kv,
                                     s.head_dim, s.half_swa, reps, &got);
            if (t < 0.0) {
                printf("%-14s %6s REFUSED\n", s.tag, alt.tag);
            } else {
                agree(got, shipped, cos, rel);
                printf("%-14s %7u %7llu %6s %12.4f %8.2f %14.9f %10.1e\n",
                       s.tag, n_tok, (unsigned long long)keys, alt.tag, t, t_ship / t, cos, rel);
            }
        }
        /* and the shipped form once more, to bound this window's drift */
        {
            std::vector<float> got;
            const double t = gr_time(vs_s[0], dq, dk, dv, dy, dpos, dlen, n_tok, n_head,
                                     s.n_head_kv, s.head_dim, s.half_swa, reps, &got);
            printf("%-14s %7u %7llu %6s %12.4f %8.2f\n",
                   s.tag, n_tok, (unsigned long long)keys, "S-b", t, t_ship / t);
            if (memcmp(got.data(), shipped.data(), shipped.size() * 4) != 0) {
                printf("%-14s S-b differs from S-a: the shipped kernel is not stable\n", s.tag);
                fail = 1;
            }
        }
        (void)flop;

        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
        cudaFree(dpos); cudaFree(dlen);
    }
    printf("%s\n", fail ? "PROBE_ATTN_GRP_FAIL" : "PROBE_ATTN_GRP_OK");
    return fail;
}
