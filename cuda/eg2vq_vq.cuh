/* eg2vq_vq.cuh — the dim-4 DQVQ payload GEMM, the gemma_vq device path's
 * weight consumer (embeddinggemma2-handoff.md PHASE 7, phase-doc row P3.1).
 *
 * Authority, read first-hand (global rule 0.2):
 *   - the payload layout and the value rule: ds41_vq_fmt.h:73-105
 *     (`ds4vq_dequant_f32`), unchanged here except that the codeword is
 *     multiplied by the row gain AFTER the dot instead of per element:
 *     value[row][col] = codebook[idx][col % dim] * gain[row], so
 *     sum_i value[r][i] * x[i] = gain[r] * sum_i codebook[...][i % 4] * x[i].
 *   - the on-device consumption pattern: cuda/ds41_vq_row.cuh (V4.1's row
 *     kernel) for what carries over — payload open with the geometry checked
 *     against the caller's expectation (`v41_vq_open`, :44-72), the codebook
 *     resident in shared memory, the bit stream read as whole words with
 *     __funnelshift (there: 12-bit main stream + plane; here: one plain
 *     12-bit stream), and the EOF note that the multiply-add expression is
 *     the one place arithmetic identity with the reference is decided.
 *
 * What differs from the V4.1 kernel, and why (measured geometry, not taste):
 *   - dim 4, not 8: our whole text-tower tier is `towers=vq4x4096` (the
 *     contract fixes dim 4 and nc 4096, ds4_eg2vq.h:29-38), so one 12-bit
 *     index covers four weights and a codeword is 8 B, not 16.
 *   - a GEMM, not a per-row dot: the tower runs whole items (26 to 1414
 *     tokens) through one matrix at a time, so the kernels here keep a
 *     64-token by 64-row tile resident and reuse each decoded codeword
 *     across the tile's tokens.  Per-row kernels would re-read the index
 *     stream once per token.
 *   - f32 activations and f32 accumulation: the CPU reference path is f32
 *     (ds4_eg2vq_graph.inc), the activation is orders of magnitude smaller
 *     than the weights, and the whole point of this family is that the
 *     artifact is not less smart — so no activation rounding is introduced
 *     on the device that the reference does not have.
 *
 * The shared-memory budget is deliberately under the 48 KB static limit:
 * codebook 32 KB (4096 x 4 f16) + one 64-row index slab 8.5 KB = 41,472 B, no
 * dynamic-shared opt-in, so occupancy is not traded for it.  A 128-row slab
 * (50,176 B, 256 threads) was built and measured in the same sitting: it is
 * equal or 5% better at the largest shapes but 0.83x at 512 x 512 / 306 tokens
 * (half the blocks of the shipped tile), and it needs the dynamic-shared
 * opt-in.  It is not carried.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

#include "../ds41_vq_fmt.h"

/* Tile geometry.  16 x 8 threads; each thread owns 4 tokens (tx) x 8 rows
 * (ty), which is 32 accumulators and 8 codeword groups per K step.
 *
 * WHY 8 ROWS AND NOT 4 (measured, embeddinggemma2 performance unit, 2026-10-21):
 * the kernel is limited by the ACTIVATION LOAD, not by arithmetic and not by
 * the codebook gather.  The ablation that says so is committed with this unit
 * (docs/benchmarks/eg2-embedding-2026-10-20/probes/probe-vq.cu, its output in
 * probe-ablation.txt, the record in vq-gemm-rows.md); it times the shipped
 * body with one component removed at a time on the tower's own shapes: with
 * the 16 FFMA per j replaced by a handful of FADDs and every load kept, the
 * kernel runs at 1.00x of baseline (the arithmetic is entirely hidden); with
 * the codebook gather and index read cut it runs at 1.08x; with the four x
 * loads replaced by one reused load it runs 2.0-2.6x faster, and with the four
 * x loads kept but addressed block-uniformly (one 32-byte sector per
 * instruction instead of sixteen) 2.4-3.7x faster.  Per thread per j the kernel
 * loads 16 bytes per token for T*R FMAs, so it pays 16/R bytes of activation
 * per FMA: R = 4 pays 4 B/FMA and R = 8 pays 2 B/FMA.  Measured against the
 * shipped body, R = 8 is 1.35x-1.66x on the same shapes and BIT-IDENTICAL
 * (accumulation order per output element is unchanged; the probe compared
 * 65,792 floats on a 256 x 512 case and the ragged 70- and 257-token tails with
 * zero differing bytes). */
enum {
    EG2VQ_BM = 64,          /* tokens per block */
    EG2VQ_BN = 64,          /* output rows per block */
    EG2VQ_BK = 256,         /* columns per K slab = 64 codewords */
    EG2VQ_CW_PER_SLAB = EG2VQ_BK / 4,
    EG2VQ_NTX = 16,         /* token lanes (each owns EG2VQ_T_PER_THREAD tokens) */
    EG2VQ_NTY = 8,          /* row groups (each owns EG2VQ_R_PER_THREAD rows) */
    EG2VQ_T_PER_THREAD = 4,
    EG2VQ_R_PER_THREAD = 8,
    EG2VQ_IDX_PITCH = EG2VQ_CW_PER_SLAB + 4,   /* uint16 rows: bank decorrelation */
    EG2VQ_NC_MAX = 4096,
    EG2VQ_IDX_BITS = 12,
    EG2VQ_IDX_MASK = (1u << EG2VQ_IDX_BITS) - 1u,
    /* Packed bytes of one K slab: BK columns / 4 = 64 codewords x 12 bits. */
    EG2VQ_SLAB_BYTES = EG2VQ_CW_PER_SLAB * EG2VQ_IDX_BITS / 8,   /* 96 */
    /* Packed bytes each thread unpacks per slab: 16 codewords x 12 bits. */
    EG2VQ_SLOT_BYTES = 16 * EG2VQ_IDX_BITS / 8,                  /* 24 */
};

/* One payload as the kernel reads it: the header fields are re-checked here
 * (the same five fields ds4_eg2vq_bind.inc re-reads on the CPU path) and the
 * geometry must be the caller's, so a payload bound to the wrong layer stops
 * instead of decoding into another layer's shape. */
typedef struct {
    const uint8_t *cb;      /* codebook: nc * 4 f16 */
    const uint8_t *gr;      /* row gains: rows f16 */
    const uint8_t *ix;      /* 12-bit index stream, one row after another */
    uint32_t rows, cols, cw_row;   /* cw_row = cols / 4 codewords per row */
    uint32_t row_bytes;     /* packed index bytes per row = 3 * cols / 8 */
    uint32_t nc, nbit;
    int ok;
} eg2vq_vq_mat;

__device__ __forceinline__ static eg2vq_vq_mat eg2vq_vq_open(const uint8_t *pay, uint32_t rows, uint32_t cols) {
    eg2vq_vq_mat m;
    m.cb = m.gr = m.ix = NULL;
    m.rows = m.cols = m.cw_row = m.row_bytes = m.nc = m.nbit = 0u;
    m.ok = 0;
    if (pay == NULL || rows == 0u || cols == 0u) { return m; }
    uint32_t mg;
    uint16_t dim, nc;
    uint32_t hrows, hcols;
    memcpy(&mg, pay, 4);
    memcpy(&dim, pay + 4, 2);
    memcpy(&nc, pay + 6, 2);
    memcpy(&hrows, pay + 8, 4);
    memcpy(&hcols, pay + 12, 4);
    if (mg != (uint32_t)DS4VQ_MAT_MAGIC || dim != 4u || nc > (uint16_t)EG2VQ_NC_MAX) { return m; }
    if (hrows != rows || hcols != cols) { return m; }
    m.cb = pay + 16;
    m.gr = m.cb + (size_t)nc * 4u * 2u;
    m.ix = m.gr + (size_t)rows * 2u;
    m.rows = rows;
    m.cols = cols;
    m.cw_row = cols / 4u;
    m.row_bytes = (cols / 4u) * (uint32_t)EG2VQ_IDX_BITS / 8u;
    m.nc = nc;
    /* The extracted width follows nc exactly as the vendored decoder derives it
     * (ds41_vq_fmt.h:85-86).  This kernel's extractor is the 12-bit one the
     * vq4x4096 tier needs, so any other width is refused here instead of being
     * decoded under the wrong layout. */
    uint32_t nbit = 0;
    while ((1u << nbit) < (uint32_t)nc) { nbit++; }
    if (nbit != (uint32_t)EG2VQ_IDX_BITS) { return m; }
    m.nbit = nbit;
    m.ok = 1;
    return m;
}

/* 12-bit little-endian extract from a word window: the stream is one
 * continuous bit stream per row (ds4vq_dequant_f32:88-99 computes the bit
 * position as i * nbit over the row's own indices, so each row starts on a
 * byte boundary because 3 * cols bits is a whole number of bytes for every
 * width this tier uses). */
__device__ __forceinline__ static uint32_t eg2vq_vq_bits12(const uint32_t *w, uint32_t nw, uint32_t bitpos) {
    const uint32_t i = bitpos >> 5, s = bitpos & 31u;
    const uint32_t lo = w[i];
    const uint32_t hi = (i + 1u < nw) ? w[i + 1u] : 0u;
    return __funnelshift_r(lo, hi, s) & (uint32_t)EG2VQ_IDX_MASK;
}

/* The GEMM: out[t][r] = gain[r] * sum_i cb[idx[r][i/4]][i%4] * x[t][i], with
 * x row stride `t_stride` (>= cols) and out row stride exactly `rows`.
 *
 * Grid: x = rows / EG2VQ_BN, y = ceil(n_tok / EG2VQ_BM).  Block: 16 x 8.
 * `n_tok` may be any positive number: token lanes past the item's last token
 * neither load x nor store out.
 */
__global__ void eg2vq_vq_gemm_kernel(
        float *__restrict__ out,
        const float *__restrict__ x,
        uint32_t t_stride,
        const uint8_t *__restrict__ pay,
        uint32_t rows, uint32_t cols, uint32_t n_tok) {
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;   /* block: 16 x 8 */
    const uint32_t tid = ty * (uint32_t)EG2VQ_NTX + tx;  /* 0..127, the flat thread id */
    const uint32_t r0 = blockIdx.x * (uint32_t)EG2VQ_BN + ty * (uint32_t)EG2VQ_R_PER_THREAD;
    const uint32_t t0 = blockIdx.y * (uint32_t)EG2VQ_BM + tx * (uint32_t)EG2VQ_T_PER_THREAD;
    const eg2vq_vq_mat m = eg2vq_vq_open(pay, rows, cols);
    if (!m.ok) { return; }   /* the CPU path refuses by name; here nothing decodes */

    __shared__ __align__(16) uint2 cb_s[EG2VQ_NC_MAX];
    __shared__ uint16_t idx_s[EG2VQ_BN * EG2VQ_IDX_PITCH];

    const uint32_t nc = m.nc;
    {   /* the codebook pane: whole 16 B chunks, coalesced */
        const uint4 *src = (const uint4 *)m.cb;
        uint4 *dst = (uint4 *)cb_s;
        const uint32_t chunks = nc / 2u;
        for (uint32_t i = tid; i < chunks; i += (uint32_t)(EG2VQ_NTX * EG2VQ_NTY)) { dst[i] = src[i]; }
    }

    float acc[EG2VQ_T_PER_THREAD][EG2VQ_R_PER_THREAD];
    for (uint32_t a = 0; a < EG2VQ_T_PER_THREAD; a++) {
        for (uint32_t b = 0; b < EG2VQ_R_PER_THREAD; b++) { acc[a][b] = 0.0f; }
    }

    /* This thread's share of the index slab, and the K loop over 256-column
     * slabs.  Within a slab, thread `tid` unpacks 16 codewords of slab row
     * (tid / 4), slot (tid % 4): 16 codewords = 192 bits = 24 B, read as six
     * words through memcpy because the row stream's 4-byte alignment is not
     * guaranteed by the payload contract (row_bytes = 3 * cols / 8 is a
     * multiple of 4 for every width in this tier, but nothing in the layout
     * promises it).  The block's 128 threads cover 32 rows per pass, so the
     * slab's EG2VQ_BN rows take EG2VQ_R_PER_THREAD / 4 passes.
     */
    const uint32_t urow0 = tid >> 2;            /* 0 .. 32-1 within a pass */
    const uint32_t uslot = tid & 3u;
    /* The stream belongs to the matrix row this block's slab row maps to, which
     * is block-local only for the first row block: urow is 0..EG2VQ_BN-1 in
     * EVERY block, so the global row must be added here or every later block
     * re-reads the first block's rows. */
    const uint32_t grow_base = blockIdx.x * (uint32_t)EG2VQ_BN;

    const uint32_t n_slab = cols / (uint32_t)EG2VQ_BK;
    for (uint32_t slab = 0; slab < n_slab; slab++) {
        /* ---- unpack this slab's 64 x 64 indices into shared ---- */
        #pragma unroll
        for (uint32_t p = 0; p < (uint32_t)(EG2VQ_R_PER_THREAD / 4); p++) {
            const uint32_t urow = urow0 + p * (uint32_t)(4 * EG2VQ_NTY);
            const uint8_t *ustream = m.ix + (size_t)(grow_base + urow) * m.row_bytes;
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

        /* ---- the slab's 64 codewords ---- */
        const uint32_t i0 = slab * (uint32_t)EG2VQ_BK;
        #pragma unroll 4
        for (uint32_t j = 0; j < (uint32_t)EG2VQ_CW_PER_SLAB; j++) {
            const uint32_t i = i0 + 4u * j;
            /* this thread's eight codewords: rows ty*8 + q */
            float cw[EG2VQ_R_PER_THREAD][4];
            #pragma unroll
            for (uint32_t q = 0; q < EG2VQ_R_PER_THREAD; q++) {
                const uint32_t row_local = ty * (uint32_t)EG2VQ_R_PER_THREAD + q;
                const uint32_t code = (uint32_t)idx_s[row_local * EG2VQ_IDX_PITCH + j];
                const uint2 c = cb_s[code];
                __half2 h0, h1;
                memcpy(&h0, &c.x, 4);
                memcpy(&h1, &c.y, 4);
                const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
                cw[q][0] = f0.x; cw[q][1] = f0.y; cw[q][2] = f1.x; cw[q][3] = f1.y;
            }
            /* this thread's four tokens' four values at this codeword; the
             * token predicate does not depend on j, so the compiler hoists it
             * and predicates the loads. */
            float xv[EG2VQ_T_PER_THREAD][4];
            #pragma unroll
            for (uint32_t a = 0; a < EG2VQ_T_PER_THREAD; a++) {
                xv[a][0] = xv[a][1] = xv[a][2] = xv[a][3] = 0.0f;
                if (t0 + a < n_tok) {
                    const float4 v = *(const float4 *)(x + (size_t)(t0 + a) * t_stride + i);
                    xv[a][0] = v.x; xv[a][1] = v.y; xv[a][2] = v.z; xv[a][3] = v.w;
                }
            }
            #pragma unroll
            for (uint32_t a = 0; a < EG2VQ_T_PER_THREAD; a++) {
                #pragma unroll
                for (uint32_t q = 0; q < EG2VQ_R_PER_THREAD; q++) {
                    acc[a][q] += cw[q][0] * xv[a][0];
                    acc[a][q] += cw[q][1] * xv[a][1];
                    acc[a][q] += cw[q][2] * xv[a][2];
                    acc[a][q] += cw[q][3] * xv[a][3];
                }
            }
        }
        __syncthreads();
    }

    /* row gain, applied once per output element (see the file header) */
    float g[EG2VQ_R_PER_THREAD];
    #pragma unroll
    for (uint32_t q = 0; q < EG2VQ_R_PER_THREAD; q++) {
        __half gh;
        memcpy(&gh, m.gr + (size_t)(r0 + q) * 2u, 2);
        g[q] = __half2float(gh);
    }
    #pragma unroll
    for (uint32_t a = 0; a < EG2VQ_T_PER_THREAD; a++) {
        if (t0 + a >= n_tok) { continue; }
        #pragma unroll
        for (uint32_t q = 0; q < EG2VQ_R_PER_THREAD; q++) {
            out[(size_t)(t0 + a) * rows + r0 + q] = acc[a][q] * g[q];
        }
    }
}

/* `cols` must be a whole number of 256-column slabs and `rows` a whole number
 * of 64-row tiles: every matrix in this family's tier is (512, 1024, 2048)
 * wide and (512 ... 12288) tall, all of them multiples.  A shape that is not
 * returns 0 rather than silently computing a partial tile. */
static inline int eg2vq_vq_gemm(float *out, const float *x, uint32_t t_stride,
                                const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (!out || !x || !pay || !n_tok || !rows || !cols) { return 0; }
    if ((rows % EG2VQ_BN) || (cols % EG2VQ_BK) || (t_stride < cols)) { return 0; }
    dim3 block(EG2VQ_NTX, EG2VQ_NTY);
    dim3 grid(rows / EG2VQ_BN, (n_tok + EG2VQ_BM - 1) / EG2VQ_BM);
    eg2vq_vq_gemm_kernel<<<grid, block>>>(out, x, t_stride, pay, rows, cols, n_tok);
    return 1;
}
