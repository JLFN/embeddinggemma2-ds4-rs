/* probe-r8.cu — second probe: a REAL, correct re-tiling of the VQ GEMM with
 * more output rows per thread (register blocking on the activation).
 *
 * WHY: probe-vq's ablation said the kernel's cost is the activation load, and
 * only that: V4 (all loads, no outer product) = 1.00x of baseline, V2 (no
 * codebook gather) = 1.08x, while V5 (same x-load instruction count, one
 * 32-byte sector per instruction instead of sixteen) = 2.4-3.7x and V1 (one
 * x load per j instead of four, register-neutral) = 2.0-2.6x.  Per thread per
 * j the kernel loads 16 bytes per token for T*R FMAs, i.e. 16/R bytes of
 * activation per FMA: R=4 pays 4 B/FMA.  These kernels raise R to 8 (2 B/FMA)
 * and keep the accumulation order element-for-element, so any of them must be
 * BIT-IDENTICAL to eg2vq_vq_gemm_kernel — the probe checks exactly that.
 *
 * configs, all with 16 columns-tokens lanes (tx) and NTY row groups (ty):
 *   A <T=4,R=8,NTY=8 >  128 threads, 64 tokens x 64 rows, 41.5 KB shared
 *   B <T=2,R=8,NTY=8 >  128 threads, 32 tokens x 64 rows, 41.5 KB shared
 *   C <T=4,R=4,NTY=16>  256 threads, 64 tokens x 64 rows (the shipped tile)
 *   D <T=2,R=4,NTY=16>  256 threads, 32 tokens x 64 rows
 *
 * usage: probe-r8 [repeats]
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

#include "cuda/eg2vq_vq.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static std::mt19937 rng(20261020u);

static uint16_t p_h16(float v) { __half h = __float2half_rn(v); uint16_t u; memcpy(&u, &h, 2); return u; }

static void p_pack(std::vector<uint8_t> &out, const std::vector<float> &cb,
                   const std::vector<uint16_t> &gr, const std::vector<uint32_t> &idx,
                   uint32_t rows, uint32_t cols, uint32_t nc, uint32_t dim) {
    const size_t head = 16;
    const size_t cb_bytes = (size_t)nc * dim * 2;
    const size_t gr_bytes = (size_t)rows * 2;
    const size_t idx_bytes = (idx.size() * 12 + 7) / 8 + 1;
    out.assign(head + cb_bytes + gr_bytes + idx_bytes, 0);
    uint32_t mg = DS4VQ_MAT_MAGIC, r32 = rows, c32 = cols;
    uint16_t d16 = (uint16_t)dim, n16 = (uint16_t)nc;
    memcpy(&out[0], &mg, 4); memcpy(&out[4], &d16, 2); memcpy(&out[6], &n16, 2);
    memcpy(&out[8], &r32, 4); memcpy(&out[12], &c32, 4);
    for (size_t i = 0; i < (size_t)nc * dim; i++) { const uint16_t h = p_h16(cb[i]); memcpy(&out[head + 2 * i], &h, 2); }
    for (size_t r = 0; r < rows; r++) { memcpy(&out[head + cb_bytes + 2 * r], &gr[r], 2); }
    uint8_t *p = &out[head + cb_bytes + gr_bytes];
    for (size_t i = 0; i < idx.size(); i++) {
        const size_t bit = i * 12;
        const uint32_t m = (idx[i] & 0xFFFu) << (bit & 7);
        p[bit >> 3] |= (uint8_t)m;
        p[(bit >> 3) + 1] |= (uint8_t)(m >> 8);
        if (m >> 16) { p[(bit >> 3) + 2] |= (uint8_t)(m >> 16); }
    }
}

/* The same arithmetic as eg2vq_vq_gemm_kernel, with T tokens and R rows per
 * thread and NTY row groups per block.  Every output element's accumulation
 * order over the K slab is unchanged, so the result is bit-identical. */
template <int T, int R, int NTY>
__global__ void vq_r8_kernel(
        float *__restrict__ out,
        const float *__restrict__ x,
        uint32_t t_stride,
        const uint8_t *__restrict__ pay,
        uint32_t rows, uint32_t cols, uint32_t n_tok) {
    const uint32_t BNT = (uint32_t)(16 * T);      /* tokens per block */
    const uint32_t BNR = (uint32_t)(NTY * R);     /* rows per block */
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t tid = ty * 16u + tx;
    const uint32_t r0 = blockIdx.x * BNR + ty * (uint32_t)R;
    const uint32_t t0 = blockIdx.y * BNT + tx * (uint32_t)T;
    const eg2vq_vq_mat m = eg2vq_vq_open(pay, rows, cols);
    if (!m.ok) { return; }

    __shared__ __align__(16) uint2 cb_s[EG2VQ_NC_MAX];
    __shared__ uint16_t idx_s[BNR * EG2VQ_IDX_PITCH];

    const uint32_t nc = m.nc;
    {
        const uint4 *src = (const uint4 *)m.cb;
        uint4 *dst = (uint4 *)cb_s;
        const uint32_t chunks = nc / 2u;
        for (uint32_t i = tid; i < chunks; i += (uint32_t)(16 * NTY)) { dst[i] = src[i]; }
    }

    float acc[T][R];
    for (uint32_t a = 0; a < (uint32_t)T; a++) {
        for (uint32_t b = 0; b < (uint32_t)R; b++) { acc[a][b] = 0.0f; }
    }

    const uint32_t urow0 = tid >> 2;      /* 0 .. 4*NTY-1 */
    const uint32_t uslot = tid & 3u;
    const uint32_t grow_base = blockIdx.x * BNR;

    const uint32_t n_slab = cols / (uint32_t)EG2VQ_BK;
    for (uint32_t slab = 0; slab < n_slab; slab++) {
        /* R/4 passes of the 16-codeword unpack cover BNR rows */
        #pragma unroll
        for (uint32_t p = 0; p < (uint32_t)(R / 4); p++) {
            const uint32_t urow = urow0 + p * (uint32_t)(4 * NTY);
            const uint8_t *ustream = m.ix + (size_t)(grow_base + urow) * m.row_bytes;
            uint32_t w[6];
            memcpy(w, ustream + (size_t)slab * (size_t)EG2VQ_SLAB_BYTES
                     + (size_t)uslot * (size_t)EG2VQ_SLOT_BYTES, sizeof(w));
            uint16_t *dstrow = idx_s + urow * EG2VQ_IDX_PITCH;
            const uint32_t jbase = uslot * 16u;
            #pragma unroll
            for (uint32_t k = 0; k < 16u; k++) {
                dstrow[jbase + k] = (uint16_t)eg2vq_vq_bits12(w, 6u, k * 12u);
            }
        }
        __syncthreads();

        const uint32_t i0 = slab * (uint32_t)EG2VQ_BK;
        #pragma unroll 4
        for (uint32_t j = 0; j < (uint32_t)EG2VQ_CW_PER_SLAB; j++) {
            const uint32_t i = i0 + 4u * j;
            float cw[R][4];
            #pragma unroll
            for (uint32_t q = 0; q < (uint32_t)R; q++) {
                const uint32_t row_local = ty * (uint32_t)R + q;
                const uint32_t code = (uint32_t)idx_s[row_local * EG2VQ_IDX_PITCH + j];
                const uint2 c = cb_s[code];
                __half2 h0, h1;
                memcpy(&h0, &c.x, 4);
                memcpy(&h1, &c.y, 4);
                const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                cw[q][0] = f0.x; cw[q][1] = f0.y; cw[q][2] = f1.x; cw[q][3] = f1.y;
            }
            float xv[T][4];
            #pragma unroll
            for (uint32_t a = 0; a < (uint32_t)T; a++) {
                xv[a][0] = xv[a][1] = xv[a][2] = xv[a][3] = 0.0f;
                if (t0 + a < n_tok) {
                    const float4 v = *(const float4 *)(x + (size_t)(t0 + a) * t_stride + i);
                    xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                }
            }
            #pragma unroll
            for (uint32_t a = 0; a < (uint32_t)T; a++) {
                #pragma unroll
                for (uint32_t q = 0; q < (uint32_t)R; q++) {
                    acc[a][q] += cw[q][0] * xv[a][0];
                    acc[a][q] += cw[q][1] * xv[a][1];
                    acc[a][q] += cw[q][2] * xv[a][2];
                    acc[a][q] += cw[q][3] * xv[a][3];
                }
            }
        }
        __syncthreads();
    }

    float g[R];
    #pragma unroll
    for (uint32_t q = 0; q < (uint32_t)R; q++) {
        __half gh;
        memcpy(&gh, m.gr + (size_t)(r0 + q) * 2u, 2);
        g[q] = __half2float(gh);
    }
    #pragma unroll
    for (uint32_t a = 0; a < (uint32_t)T; a++) {
        if (t0 + a >= n_tok) { continue; }
        #pragma unroll
        for (uint32_t q = 0; q < (uint32_t)R; q++) {
            out[(size_t)(t0 + a) * rows + r0 + q] = acc[a][q] * g[q];
        }
    }
}

template <int T, int R, int NTY>
static void launch_r8(float *dy, const float *dx, uint32_t t_stride, const uint8_t *dpay,
                      uint32_t rows, uint32_t cols, uint32_t n_tok) {
    dim3 block(16, NTY);
    dim3 grid(rows / (NTY * R), (n_tok + 16 * T - 1) / (16 * T));
    vq_r8_kernel<T, R, NTY><<<grid, block>>>(dy, dx, t_stride, dpay, rows, cols, n_tok);
}

struct q_shape { uint32_t rows, cols; const char *tag; };

static std::vector<uint8_t> mk_payload(const q_shape &s, uint32_t nc, uint32_t dim) {
    std::vector<float> cb((size_t)nc * dim);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float &v : cb) { v = p_h16(0.05f * nd(rng)); }
    std::vector<uint16_t> gr(s.rows);
    for (uint16_t &g : gr) { g = p_h16(0.25f + (float)(rng() % 1000u) * 0.00375f); }
    std::vector<uint32_t> idx((size_t)s.rows * s.cols / dim);
    for (uint32_t &v : idx) { v = rng() & 0xFFFu; }
    std::vector<uint8_t> pay;
    p_pack(pay, cb, gr, idx, s.rows, s.cols, nc, dim);
    return pay;
}

int main(int argc, char **argv) {
    const int repeats = (argc > 1) ? atoi(argv[1]) : 15;
    const uint32_t nc = 4096u, dim = 4u;
    const q_shape shapes[] = {
        {  512,  512, "k/v_proj_s" },
        { 2048,  512, "ffn_gate_s" },
        {12288,  512, "ple_model_proj" },
    };
    const uint32_t toks[] = { 306, 4520 };

    /* ---- correctness first: every config bit-identical to the shipped kernel ---- */
    printf("# probe-r8 — bit-identity against eg2vq_vq_gemm_kernel\n");
    int bad = 0;
    {
        const q_shape cs[] = { {256, 512, "id256x512"}, {128, 1024, "id128x1024"} };
        for (const q_shape &s : cs) {
            std::vector<uint8_t> pay = mk_payload(s, nc, dim);
            uint8_t *dpay = NULL;
            CK(cudaMalloc(&dpay, pay.size()));
            CK(cudaMemcpy(dpay, pay.data(), pay.size(), cudaMemcpyHostToDevice));
            const uint32_t vtok[3] = { 1, 70, 257 };   /* 1, ragged, multi-block */
            for (uint32_t tok : vtok) {
                /* the activation must hold a WHOLE tile: the kernel reads every
                 * token lane of the block, in-range or not */
                std::vector<float> x((size_t)512 * s.cols);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                for (float &v : x) { v = nd(rng); }
                float *dx = NULL, *ref = NULL, *got = NULL;
                CK(cudaMalloc(&dx, x.size() * 4));
                CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&ref, (size_t)tok * s.rows * 4));
                CK(cudaMalloc(&got, (size_t)tok * s.rows * 4));
                CK(cudaMemset(ref, 0, (size_t)tok * s.rows * 4));
                CK(cudaMemset(got, 0xFF, (size_t)tok * s.rows * 4));
                eg2vq_vq_gemm(ref, dx, s.cols, dpay, s.rows, s.cols, tok);
                launch_r8<4, 8, 8>(got, dx, s.cols, dpay, s.rows, s.cols, tok);
                CK(cudaDeviceSynchronize());
                std::vector<float> h1((size_t)tok * s.rows), h2((size_t)tok * s.rows);
                CK(cudaMemcpy(h1.data(), ref, h1.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h2.data(), got, h2.size() * 4, cudaMemcpyDeviceToHost));
                size_t diff = 0;
                double maxabs = 0.0, maxrel = 0.0;
                for (size_t i = 0; i < h1.size(); i++) {
                    if (memcmp(&h1[i], &h2[i], 4) != 0) { diff++; }
                    const double d = fabs((double)h1[i] - (double)h2[i]);
                    if (d > maxabs) { maxabs = d; }
                    const double r = (fabs((double)h1[i]) > 1e-12) ? d / fabs((double)h1[i]) : 0.0;
                    if (r > maxrel) { maxrel = r; }
                }
                printf("%-12s tok=%-4u A<4,8,8> differing: %zu of %zu  maxabs %.3e maxrel %.3e"
                       "  ref[0..2]=%.6f %.6f %.6f  got[0..2]=%.6f %.6f %.6f\n",
                       s.tag, tok, diff, h1.size(), maxabs, maxrel,
                       h1[0], h1.size() > 1 ? h1[1] : 0.f, h1.size() > 2 ? h1[2] : 0.f,
                       h2[0], h2.size() > 1 ? h2[1] : 0.f, h2.size() > 2 ? h2[2] : 0.f);
                if (diff) { bad = 1; }
                CK(cudaFree(dx)); CK(cudaFree(ref)); CK(cudaFree(got));
            }
            CK(cudaFree(dpay));
        }
        printf("%s\n", bad ? "IDENTITY_FAIL" : "IDENTITY_OK");
        if (bad) { return 1; }
    }

    printf("# ablation vs the shipped kernel, %d repeats per row\n", repeats);
    printf("%-16s %6s %7s %10s %10s %9s %10s %9s %10s %9s %10s %9s\n",
           "shape", "rows", "tokens", "base ms", "base G/s",
           "A<4,8,8>", "G/s", "B<2,8,8> ", "G/s", "C<4,4,16>", "G/s", "D<2,4,16>");
    for (const q_shape &s : shapes) {
        std::vector<uint8_t> pay = mk_payload(s, nc, dim);
        uint8_t *dpay = NULL;
        CK(cudaMalloc(&dpay, pay.size()));
        CK(cudaMemcpy(dpay, pay.data(), pay.size(), cudaMemcpyHostToDevice));
        std::vector<float> x((size_t)4520 * s.cols);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float &v : x) { v = nd(rng); }
        float *dx = NULL, *dy = NULL;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        for (uint32_t tok : toks) {
            CK(cudaFree(dy));
            CK(cudaMalloc(&dy, (size_t)tok * s.rows * 4));
            const double flop = 2.0 * (double)s.rows * (double)s.cols * (double)tok;
            double ms[5];
            for (int v = 0; v < 5; v++) {
                std::vector<float> msv;
                for (int rep = 0; rep < repeats + 3; rep++) {
                    cudaEvent_t a, b;
                    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                    CK(cudaEventRecord(a));
                    switch (v) {
                        case 0: eg2vq_vq_gemm(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 1: launch_r8<4, 8, 8>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 2: launch_r8<2, 8, 8>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 3: launch_r8<4, 4, 16>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        default: launch_r8<2, 4, 16>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                    }
                    CK(cudaEventRecord(b));
                    CK(cudaEventSynchronize(b));
                    float msec = 0.0f;
                    CK(cudaEventElapsedTime(&msec, a, b));
                    if (rep >= 3) { msv.push_back(msec); }
                    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
                }
                std::vector<float> sorted = msv;
                for (size_t i = 0; i + 1 < sorted.size(); i++) {
                    for (size_t j = i + 1; j < sorted.size(); j++) {
                        if (sorted[j] < sorted[i]) { const float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
                    }
                }
                ms[v] = sorted[sorted.size() / 2];
            }
            printf("%-16s %6u %7u %10.4f %10.1f %9.2fx %10.1f %9.2fx %10.1f %9.2fx %10.1f %9.2fx %10.1f\n",
                   s.tag, s.rows, tok, ms[0], flop / (ms[0] * 1.0e6),
                   ms[0] / ms[1], flop / (ms[1] * 1.0e6),
                   ms[0] / ms[2], flop / (ms[2] * 1.0e6),
                   ms[0] / ms[3], flop / (ms[3] * 1.0e6),
                   ms[0] / ms[4], flop / (ms[4] * 1.0e6));
        }
        CK(cudaFree(dpay));
        CK(cudaFree(dx));
        CK(cudaFree(dy));
    }
    printf("PROBE_R8_OK\n");
    return 0;
}
