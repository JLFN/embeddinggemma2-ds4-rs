/* probe-vq.cu — ablation probe for eg2vq_vq_gemm_kernel.
 *
 * WHAT IT IS: the real kernel, copied into a template<int V> so each variant
 * gets its own clean codegen, with exactly one component changed per variant
 * and the register footprint held constant (v1 of this file hoisted a whole
 * 256-column activation slab into registers and spilled; every variant here is
 * register-neutral by construction).  The numbers are WRONG for every V != 0;
 * the only thing read from this file is the time.
 *
 *   V0  baseline, byte-for-byte the body of cuda/eg2vq_vq.cuh
 *   V1  one x load per j, its four values reused by all four tokens (removes
 *       3/4 of the LDG.E.128, keeps the index reads, the gathers, the
 *       conversions and all 16 FFMA per j)
 *   V2  codebook gather + index read replaced by a constant codeword (keeps
 *       the four x loads and all 16 FFMA per j)
 *   V6  the whole 256-column activation slab per token hoisted into
 *       registers (spills; the first probe version shipped this by accident)
 *   V3  FMA only: no loads, a loop-carried scalar keeps FFMA alive.  NOT a
 *       clean ceiling: with the operands identical ptxas CSEs the outer
 *       product, so the reported rate is an upper bound only.
 *   V4  loads only: every load kept and consumed by a few FADDs instead of the
 *       FFMA outer product (the load-side floor)
 *   V5  the four x loads kept, but addressed block-uniformly (every thread
 *       loads the block's first token row): same instruction count as V0, one
 *       32-byte sector per instruction instead of sixteen
 *
 * usage: probe-vq [repeats]  (build it against the PRE-change header: see the
 *                             PINNED note below the includes)
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

/* PINNED TO THE PRE-CHANGE TILE (2026-10-21, after the QA pass): this probe's
 * V0 must stay the R=4 body it was measured against, whatever the shipped
 * header says today.  Commit 8d618ecb made eg2vq_vq_gemm_kernel 16x8 with 8
 * rows per thread, so the probe cannot take the tile from the header any more:
 * a rebuilt probe would silently become an R=8 baseline with a 16x16 launcher
 * and read past its own tiles (illegal memory access, observed).  The block
 * shape, EG2VQ_BN, EG2VQ_BM, EG2VQ_BK and the slab layout are unchanged in the
 * header and are still taken from it. */
#define PVQ_T 4
#define PVQ_R 4

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

template <int V>
__global__ void vq_probe_kernel(
        float *__restrict__ out,
        const float *__restrict__ x,
        uint32_t t_stride,
        const uint8_t *__restrict__ pay,
        uint32_t rows, uint32_t cols, uint32_t n_tok) {
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t tid = ty * 16u + tx;
    const uint32_t r0 = blockIdx.x * (uint32_t)EG2VQ_BN + ty * (uint32_t)PVQ_R;
    const uint32_t t0 = blockIdx.y * (uint32_t)EG2VQ_BM + tx * (uint32_t)PVQ_T;
    const eg2vq_vq_mat m = eg2vq_vq_open(pay, rows, cols);
    if (!m.ok) { return; }

    __shared__ __align__(16) uint2 cb_s[EG2VQ_NC_MAX];
    __shared__ uint16_t idx_s[EG2VQ_BN * EG2VQ_IDX_PITCH];

    const uint32_t nc = m.nc;
    {
        const uint4 *src = (const uint4 *)m.cb;
        uint4 *dst = (uint4 *)cb_s;
        const uint32_t chunks = nc / 2u;
        for (uint32_t i = tid; i < chunks; i += 256u) { dst[i] = src[i]; }
    }

    float acc[PVQ_T][PVQ_R];
    const uint32_t t_ok = (t0 + (uint32_t)PVQ_T <= n_tok) ? 1u : 0u;
    for (uint32_t a = 0; a < PVQ_T; a++) {
        for (uint32_t b = 0; b < PVQ_R; b++) { acc[a][b] = 0.0f; }
    }

    const uint32_t urow = tid >> 2;
    const uint32_t uslot = tid & 3u;
    const uint32_t grow = blockIdx.x * (uint32_t)EG2VQ_BN + urow;
    const uint8_t *ustream = m.ix + (size_t)grow * m.row_bytes;

    float carry = 1.0f;   /* V3 only: keeps the FFMA chain loop-carried */
    float lsum = 0.0f;    /* V4 only: consumes every load */

    /* V6 only: the whole 256-column activation slab of each of this thread's
     * tokens, held in registers.  This is the variant the FIRST version of this
     * probe shipped by accident; it spills to local memory and is measured here
     * so that the record's claim about it has an artifact of its own. */
    float xh[PVQ_T][EG2VQ_BK];
    if (V == 6) {
        for (uint32_t a = 0; a < PVQ_T; a++) {
            for (uint32_t c = 0; c < (uint32_t)EG2VQ_BK; c += 4u) {
                const float4 v = (t0 + a < n_tok)
                                     ? *(const float4 *)(x + (size_t)(t0 + a) * t_stride + c)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
                xh[a][c] = v.x; xh[a][c + 1] = v.y; xh[a][c + 2] = v.z; xh[a][c + 3] = v.w;
            }
        }
    }

    const uint32_t n_slab = cols / (uint32_t)EG2VQ_BK;
    for (uint32_t slab = 0; slab < n_slab; slab++) {
        if (V != 2) {
            uint32_t w[6];
            memcpy(w, ustream + (size_t)slab * (size_t)EG2VQ_SLAB_BYTES
                     + (size_t)uslot * (size_t)EG2VQ_SLOT_BYTES, sizeof(w));
            const uint32_t jbase = uslot * 16u;
            uint16_t *dstrow = idx_s + urow * EG2VQ_IDX_PITCH;
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
            float cw[PVQ_R][4];
            float xv[PVQ_T][4];

            if (V == 3) {
                carry = fmaf(carry, 1.0000001f, 1.0e-9f);
                #pragma unroll
                for (uint32_t q = 0; q < PVQ_R; q++) {
                    #pragma unroll
                    for (uint32_t c = 0; c < 4u; c++) { cw[q][c] = carry; }
                }
                #pragma unroll
                for (uint32_t a = 0; a < PVQ_T; a++) {
                    #pragma unroll
                    for (uint32_t c = 0; c < 4u; c++) { xv[a][c] = carry; }
                }
            } else if (V == 4) {
                #pragma unroll
                for (uint32_t q = 0; q < PVQ_R; q++) {
                    const uint32_t row_local = ty * (uint32_t)PVQ_R + q;
                    const uint32_t code = (uint32_t)idx_s[row_local * EG2VQ_IDX_PITCH + j];
                    const uint2 c = cb_s[code];
                    __half2 h0, h1;
                    memcpy(&h0, &c.x, 4);
                    memcpy(&h1, &c.y, 4);
                    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                    lsum += (f0.x + f0.y) + (f1.x + f1.y);
                }
                #pragma unroll
                for (uint32_t a = 0; a < PVQ_T; a++) {
                    if (t0 + a < n_tok) {
                        const float4 v = *(const float4 *)(x + (size_t)(t0 + a) * t_stride + i);
                        lsum += (v.x + v.y) + (v.z + v.w);
                    }
                }
            } else {
                #pragma unroll
                for (uint32_t q = 0; q < PVQ_R; q++) {
                    const uint32_t row_local = ty * (uint32_t)PVQ_R + q;
                    const uint32_t code = (V == 2) ? (uint32_t)(row_local)
                                                   : (uint32_t)idx_s[row_local * EG2VQ_IDX_PITCH + j];
                    const uint2 c = cb_s[code];
                    __half2 h0, h1;
                    memcpy(&h0, &c.x, 4);
                    memcpy(&h1, &c.y, 4);
                    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                    cw[q][0] = f0.x; cw[q][1] = f0.y; cw[q][2] = f1.x; cw[q][3] = f1.y;
                }
                if (V == 6) {
                    /* every token's slab already in registers: no x load here */
                    const uint32_t cbase = j * 4u;
                    #pragma unroll
                    for (uint32_t a = 0; a < PVQ_T; a++) {
                        xv[a][0] = xh[a][cbase]; xv[a][1] = xh[a][cbase + 1u];
                        xv[a][2] = xh[a][cbase + 2u]; xv[a][3] = xh[a][cbase + 3u];
                    }
                } else if (V == 1) {
                    /* one load, four tokens: register-neutral, 1/4 the LDG */
                    const float4 v = (t0 < n_tok) ? *(const float4 *)(x + (size_t)t0 * t_stride + i)
                                                  : make_float4(0.f, 0.f, 0.f, 0.f);
                    #pragma unroll
                    for (uint32_t a = 0; a < PVQ_T; a++) {
                        xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                    }
                } else if (V == 5) {
                    /* four loads, block-uniform address: 1 sector per instruction */
                    const uint32_t tu = blockIdx.y * (uint32_t)EG2VQ_BM;
                    #pragma unroll
                    for (uint32_t a = 0; a < PVQ_T; a++) {
                        const float4 v = (tu < n_tok) ? *(const float4 *)(x + (size_t)tu * t_stride + i)
                                                      : make_float4(0.f, 0.f, 0.f, 0.f);
                        xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                    }
                } else if (t_ok) {
                    #pragma unroll
                    for (uint32_t a = 0; a < PVQ_T; a++) {
                        const float4 v = *(const float4 *)(x + (size_t)(t0 + a) * t_stride + i);
                        xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (uint32_t a = 0; a < PVQ_T; a++) {
                        xv[a][0] = xv[a][1] = xv[a][2] = xv[a][3] = 0.0f;
                        if (t0 + a < n_tok) {
                            const float4 v = *(const float4 *)(x + (size_t)(t0 + a) * t_stride + i);
                            xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                        }
                    }
                }
                #pragma unroll
                for (uint32_t a = 0; a < PVQ_T; a++) {
                    #pragma unroll
                    for (uint32_t q = 0; q < PVQ_R; q++) {
                        acc[a][q] += cw[q][0] * xv[a][0];
                        acc[a][q] += cw[q][1] * xv[a][1];
                        acc[a][q] += cw[q][2] * xv[a][2];
                        acc[a][q] += cw[q][3] * xv[a][3];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (V == 3) { acc[0][0] += carry; }
    if (V == 4) { acc[0][0] += lsum; }

    float g[PVQ_R];
    #pragma unroll
    for (uint32_t q = 0; q < PVQ_R; q++) {
        __half gh;
        memcpy(&gh, m.gr + (size_t)(r0 + q) * 2u, 2);
        g[q] = __half2float(gh);
    }
    #pragma unroll
    for (uint32_t a = 0; a < PVQ_T; a++) {
        if (t0 + a >= n_tok) { continue; }
        #pragma unroll
        for (uint32_t q = 0; q < PVQ_R; q++) {
            out[(size_t)(t0 + a) * rows + r0 + q] = acc[a][q] * g[q];
        }
    }
}

template <int V>
static void run_variant(float *dy, const float *dx, uint32_t t_stride, const uint8_t *dpay,
                        uint32_t rows, uint32_t cols, uint32_t n_tok) {
    dim3 block(16, 16);
    dim3 grid(rows / EG2VQ_BN, (n_tok + EG2VQ_BM - 1) / EG2VQ_BM);
    vq_probe_kernel<V><<<grid, block>>>(dy, dx, t_stride, dpay, rows, cols, n_tok);
}

struct q_shape { uint32_t rows, cols; const char *tag; };

int main(int argc, char **argv) {
    const int repeats = (argc > 1) ? atoi(argv[1]) : 15;
    const uint32_t nc = 4096u, dim = 4u;
    const q_shape shapes[] = {
        {  512,  512, "k/v_proj_s" },
        { 2048,  512, "ffn_gate_s" },
        {12288,  512, "ple_model_proj" },
    };
    const uint32_t toks[] = { 306, 4520 };

    printf("# probe-vq — ablation of eg2vq_vq_gemm_kernel, %d repeats per row\n", repeats);
    printf("# V0 base | V1 one x load/4 tokens | V2 no gather | V3 FMA-only(UB) | V4 loads-only | V5 x block-uniform\n");
    printf("%-16s %6s %7s %4s %10s %10s %10s %10s %10s %10s\n",
           "shape", "rows", "tokens", "var", "ms_med", "gflop/s", "V1", "V2", "V4", "V5");
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
            float md[7];
            for (int v = 0; v < 7; v++) {
                std::vector<float> ms;
                for (int rep = 0; rep < repeats + 3; rep++) {
                    cudaEvent_t a, b;
                    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                    CK(cudaEventRecord(a));
                    switch (v) {
                        case 0: run_variant<0>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 1: run_variant<1>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 2: run_variant<2>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 3: run_variant<3>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 4: run_variant<4>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        case 5: run_variant<5>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                        default: run_variant<6>(dy, dx, s.cols, dpay, s.rows, s.cols, tok); break;
                    }
                    CK(cudaEventRecord(b));
                    CK(cudaEventSynchronize(b));
                    float msec = 0.0f;
                    CK(cudaEventElapsedTime(&msec, a, b));
                    if (rep >= 3) { ms.push_back(msec); }
                    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
                }
                std::vector<float> sorted = ms;
                for (size_t i = 0; i + 1 < sorted.size(); i++) {
                    for (size_t j = i + 1; j < sorted.size(); j++) {
                        if (sorted[j] < sorted[i]) { const float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
                    }
                }
                md[v] = sorted[sorted.size() / 2];
            }
            printf("%-16s %6u %7u %4s %10.4f %10.1f %9.2fx %9.2fx %9.2fx %9.2fx\n",
                   s.tag, s.rows, tok, "all", md[0], flop / ((double)md[0] * 1.0e6),
                   md[0] / md[1], md[0] / md[2], md[0] / md[4], md[0] / md[5]);
            printf("%-16s %6u %7u %4s %10.4f %10.1f %9.2fx\n", s.tag, s.rows, tok, "V6",
                   md[6], flop / ((double)md[6] * 1.0e6), md[0] / md[6]);
            printf("%-16s %6u %7u %4s %10.4f %10.1f   (V3 = %s)\n", s.tag, s.rows, tok, "V3",
                   md[3], flop / ((double)md[3] * 1.0e6), "upper bound only, ptxas CSEs it");
        }
        CK(cudaFree(dpay));
        CK(cudaFree(dx));
        CK(cudaFree(dy));
    }
    printf("PROBE_VQ_OK\n");
    return 0;
}
