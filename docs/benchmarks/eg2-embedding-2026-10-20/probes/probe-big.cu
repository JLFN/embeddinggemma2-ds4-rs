/* probe-big.cu — the R=8 tile with 256 threads: BN=128 rows x BM=64 tokens,
 * which needs 50,176 B of shared and therefore the dynamic-shared opt-in.
 * Compared against the shipped kernel and against <T=4,R=8,NTY=8> (128
 * threads, BN=64, the static-shared R=8 config).
 *
 * usage: probe-big [repeats]
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

#define BIG_T 4
#define BIG_R 8
#define BIG_NTY 16
#define BIG_BNR (BIG_NTY * BIG_R)          /* 128 rows per block */
#define BIG_CB_BYTES (EG2VQ_NC_MAX * 8u)  /* 32 KB */
#define BIG_IDX_BYTES (BIG_BNR * EG2VQ_IDX_PITCH * 2)
#define BIG_SHMEM (BIG_CB_BYTES + BIG_IDX_BYTES)

__global__ void vq_big_kernel(
        float *__restrict__ out, const float *__restrict__ x, uint32_t t_stride,
        const uint8_t *__restrict__ pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint2 *cb_s = (uint2 *)smem;
    uint16_t *idx_s = (uint16_t *)(smem + BIG_CB_BYTES);

    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t tid = ty * 16u + tx;
    const uint32_t r0 = blockIdx.x * (uint32_t)BIG_BNR + ty * (uint32_t)BIG_R;
    const uint32_t t0 = blockIdx.y * (uint32_t)(16 * BIG_T) + tx * (uint32_t)BIG_T;
    const eg2vq_vq_mat m = eg2vq_vq_open(pay, rows, cols);
    if (!m.ok) { return; }

    const uint32_t nc = m.nc;
    {
        const uint4 *src = (const uint4 *)m.cb;
        uint4 *dst = (uint4 *)cb_s;
        const uint32_t chunks = nc / 2u;
        for (uint32_t i = tid; i < chunks; i += 256u) { dst[i] = src[i]; }
    }

    float acc[BIG_T][BIG_R];
    for (uint32_t a = 0; a < BIG_T; a++) { for (uint32_t b = 0; b < BIG_R; b++) { acc[a][b] = 0.0f; } }

    const uint32_t urow0 = tid >> 2;
    const uint32_t uslot = tid & 3u;
    const uint32_t grow_base = blockIdx.x * (uint32_t)BIG_BNR;

    const uint32_t n_slab = cols / (uint32_t)EG2VQ_BK;
    for (uint32_t slab = 0; slab < n_slab; slab++) {
        #pragma unroll
        for (uint32_t p = 0; p < (uint32_t)(BIG_R / 4); p++) {
            const uint32_t urow = urow0 + p * (uint32_t)(4 * BIG_NTY);
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
            float cw[BIG_R][4];
            #pragma unroll
            for (uint32_t q = 0; q < (uint32_t)BIG_R; q++) {
                const uint32_t row_local = ty * (uint32_t)BIG_R + q;
                const uint32_t code = (uint32_t)idx_s[row_local * EG2VQ_IDX_PITCH + j];
                const uint2 c = cb_s[code];
                __half2 h0, h1;
                memcpy(&h0, &c.x, 4);
                memcpy(&h1, &c.y, 4);
                const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                cw[q][0] = f0.x; cw[q][1] = f0.y; cw[q][2] = f1.x; cw[q][3] = f1.y;
            }
            float xv[BIG_T][4];
            #pragma unroll
            for (uint32_t a = 0; a < (uint32_t)BIG_T; a++) {
                xv[a][0] = xv[a][1] = xv[a][2] = xv[a][3] = 0.0f;
                if (t0 + a < n_tok) {
                    const float4 v = *(const float4 *)(x + (size_t)(t0 + a) * t_stride + i);
                    xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                }
            }
            #pragma unroll
            for (uint32_t a = 0; a < (uint32_t)BIG_T; a++) {
                #pragma unroll
                for (uint32_t q = 0; q < (uint32_t)BIG_R; q++) {
                    acc[a][q] += cw[q][0] * xv[a][0];
                    acc[a][q] += cw[q][1] * xv[a][1];
                    acc[a][q] += cw[q][2] * xv[a][2];
                    acc[a][q] += cw[q][3] * xv[a][3];
                }
            }
        }
        __syncthreads();
    }

    float g[BIG_R];
    #pragma unroll
    for (uint32_t q = 0; q < (uint32_t)BIG_R; q++) {
        __half gh;
        memcpy(&gh, m.gr + (size_t)(r0 + q) * 2u, 2);
        g[q] = __half2float(gh);
    }
    #pragma unroll
    for (uint32_t a = 0; a < (uint32_t)BIG_T; a++) {
        if (t0 + a >= n_tok) { continue; }
        #pragma unroll
        for (uint32_t q = 0; q < (uint32_t)BIG_R; q++) {
            out[(size_t)(t0 + a) * rows + r0 + q] = acc[a][q] * g[q];
        }
    }
}

/* the static-shared R=8 config A, for the head-to-head */
template <int T, int R, int NTY>
__global__ void vq_r8_kernel(
        float *__restrict__ out, const float *__restrict__ x, uint32_t t_stride,
        const uint8_t *__restrict__ pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    const uint32_t BNT = (uint32_t)(16 * T);
    const uint32_t BNR = (uint32_t)(NTY * R);
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
    for (uint32_t a = 0; a < (uint32_t)T; a++) { for (uint32_t b = 0; b < (uint32_t)R; b++) { acc[a][b] = 0.0f; } }
    const uint32_t urow0 = tid >> 2, uslot = tid & 3u;
    const uint32_t grow_base = blockIdx.x * BNR;
    const uint32_t n_slab = cols / (uint32_t)EG2VQ_BK;
    for (uint32_t slab = 0; slab < n_slab; slab++) {
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
            for (uint32_t k = 0; k < 16u; k++) { dstrow[jbase + k] = (uint16_t)eg2vq_vq_bits12(w, 6u, k * 12u); }
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
                memcpy(&h0, &c.x, 4); memcpy(&h1, &c.y, 4);
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
        __half gh; memcpy(&gh, m.gr + (size_t)(r0 + q) * 2u, 2); g[q] = __half2float(gh);
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

struct q_shape { uint32_t rows, cols; const char *tag; };

int main(int argc, char **argv) {
    const int repeats = (argc > 1) ? atoi(argv[1]) : 15;
    const uint32_t nc = 4096u, dim = 4u;
    cudaFuncSetAttribute(vq_big_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BIG_SHMEM);
    cudaFuncSetAttribute(vq_r8_kernel<4, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, BIG_SHMEM);
    const q_shape shapes[] = {
        {  512,  512, "k/v_proj_s" },
        { 2048,  512, "ffn_gate_s" },
        {12288,  512, "ple_model_proj" },
    };
    const uint32_t toks[] = { 306, 4520 };
    printf("# probe-big — BN=128 dynamic vs BN=64 static, both R=8, %d repeats\n", repeats);

    /* identity first */
    int bad = 0;
    {
        const q_shape s = { 256, 512, "id256x512" };
        std::vector<float> cb((size_t)nc * dim);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float &v : cb) { v = p_h16(0.05f * nd(rng)); }
        std::vector<uint16_t> gr(s.rows);
        for (uint16_t &g : gr) { g = p_h16(0.25f); }
        std::vector<uint32_t> idx((size_t)s.rows * s.cols / dim);
        for (uint32_t &v : idx) { v = rng() & 0xFFFu; }
        std::vector<uint8_t> pay;
        p_pack(pay, cb, gr, idx, s.rows, s.cols, nc, dim);
        uint8_t *dpay = NULL;
        CK(cudaMalloc(&dpay, pay.size()));
        CK(cudaMemcpy(dpay, pay.data(), pay.size(), cudaMemcpyHostToDevice));
        std::vector<float> xh((size_t)512 * s.cols);
        for (float &v : xh) { v = nd(rng); }
        float *dx = NULL, *ref = NULL, *got = NULL;
        CK(cudaMalloc(&dx, xh.size() * 4));
        CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&ref, (size_t)257 * s.rows * 4));
        CK(cudaMalloc(&got, (size_t)257 * s.rows * 4));
        for (uint32_t tok : { 1u, 70u, 257u }) {
            CK(cudaMemset(ref, 0, (size_t)tok * s.rows * 4));
            CK(cudaMemset(got, 0xFF, (size_t)tok * s.rows * 4));
            eg2vq_vq_gemm(ref, dx, s.cols, dpay, s.rows, s.cols, tok);
            dim3 blk(16, BIG_NTY);
            dim3 grd(s.rows / BIG_BNR, (tok + 16 * BIG_T - 1) / (16 * BIG_T));
            vq_big_kernel<<<grd, blk, BIG_SHMEM>>>(got, dx, s.cols, dpay, s.rows, s.cols, tok);
            CK(cudaDeviceSynchronize());
            std::vector<float> h1((size_t)tok * s.rows), h2((size_t)tok * s.rows);
            CK(cudaMemcpy(h1.data(), ref, h1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2.data(), got, h2.size() * 4, cudaMemcpyDeviceToHost));
            size_t diff = 0;
            for (size_t i = 0; i < h1.size(); i++) { if (memcmp(&h1[i], &h2[i], 4) != 0) { diff++; } }
            printf("id256x512 tok=%-4u big<4,8,16> differing: %zu of %zu\n", tok, diff, h1.size());
            if (diff) { bad = 1; }
        }
        CK(cudaFree(dx)); CK(cudaFree(ref)); CK(cudaFree(got)); CK(cudaFree(dpay));
        printf("%s\n", bad ? "IDENTITY_FAIL" : "IDENTITY_OK");
        if (bad) { return 1; }
    }

    printf("%-16s %6s %7s %11s %10s %11s %10s %11s %10s\n",
           "shape", "rows", "tokens", "base ms", "G/s", "A<4,8,8>", "G/s", "BIG<4,8,16>", "G/s");
    for (const q_shape &s : shapes) {
        std::vector<float> cb((size_t)nc * dim);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float &v : cb) { v = p_h16(0.05f * nd(rng)); }
        std::vector<uint16_t> gr(s.rows);
        for (uint16_t &g : gr) { g = p_h16(0.25f + (float)(rng() % 1000u) * 0.00375f); }
        std::vector<uint32_t> idx((size_t)s.rows * s.cols / dim);
        for (uint32_t &v : idx) { v = rng() & 0xFFFu; }
        std::vector<uint8_t> pay;
        p_pack(pay, cb, gr, idx, s.rows, s.cols, nc, dim);
        uint8_t *dpay = NULL;
        CK(cudaMalloc(&dpay, pay.size()));
        CK(cudaMemcpy(dpay, pay.data(), pay.size(), cudaMemcpyHostToDevice));
        std::vector<float> x((size_t)4520 * s.cols);
        for (float &v : x) { v = nd(rng); }
        float *dx = NULL, *dy = NULL;
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        for (uint32_t tok : toks) {
            CK(cudaFree(dy));
            CK(cudaMalloc(&dy, (size_t)tok * s.rows * 4));
            const double flop = 2.0 * (double)s.rows * (double)s.cols * (double)tok;
            double ms[3];
            for (int v = 0; v < 3; v++) {
                std::vector<float> msv;
                for (int rep = 0; rep < repeats + 3; rep++) {
                    cudaEvent_t a, b;
                    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                    CK(cudaEventRecord(a));
                    if (v == 0) { eg2vq_vq_gemm(dy, dx, s.cols, dpay, s.rows, s.cols, tok); }
                    if (v == 1) {
                        dim3 blk(16, 8);
                        dim3 grd(s.rows / 64, (tok + 63) / 64);
                        vq_r8_kernel<4, 8, 8><<<grd, blk>>>(dy, dx, s.cols, dpay, s.rows, s.cols, tok);
                    }
                    if (v == 2) {
                        dim3 blk(16, BIG_NTY);
                        dim3 grd(s.rows / BIG_BNR, (tok + 63) / 64);
                        vq_big_kernel<<<grd, blk, BIG_SHMEM>>>(dy, dx, s.cols, dpay, s.rows, s.cols, tok);
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
            printf("%-16s %6u %7u %11.4f %10.1f %10.2fx %10.1f %10.2fx %10.1f\n",
                   s.tag, s.rows, tok, ms[0], flop / (ms[0] * 1.0e6),
                   ms[0] / ms[1], flop / (ms[1] * 1.0e6),
                   ms[0] / ms[2], flop / (ms[2] * 1.0e6));
        }
        CK(cudaFree(dpay)); CK(cudaFree(dx)); CK(cudaFree(dy));
    }
    printf("PROBE_BIG_OK\n");
    return 0;
}
