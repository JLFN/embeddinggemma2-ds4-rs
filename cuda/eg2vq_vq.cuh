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
#include <mma.h>
#include <cstdint>
#include <cstring>

#include "../ds41_vq_fmt.h"

namespace eg2wmma = nvcuda::wmma;

/* Tile geometry.  16 x 8 threads; each thread owns 4 tokens (tx) x 8 rows
 * (ty), which is 32 accumulators and 8 codeword groups per K step.
 *
 * WHY 8 ROWS AND NOT 4 (measured, embeddinggemma2 performance unit, 2026-10-21):
 * the kernel is limited by the ACTIVATION LOAD, not by arithmetic and not by
 * the codebook gather.  The ablation that says so is committed with this unit
 * (docs/benchmarks/eg2-embedding-2026-10-20/probes/probe-vq.cu, its output in
 * probe-ablation.txt, the record in vq-gemm-rows.md); it times the pre-change
 * body with one component removed at a time on the tower's own shapes: with the
 * 16 FFMA per j replaced by a handful of FADDs and every load kept, the kernel
 * runs at 1.00x-1.12x of baseline (the arithmetic is hidden); with the codebook
 * gather and index read cut it runs at 1.08x-1.18x; with the four x loads
 * replaced by one reused load it runs 2.0x-3.1x faster, and with the four x
 * loads kept but addressed block-uniformly (one 32-byte sector per instruction
 * instead of sixteen) 2.4x-4.2x faster.  Per K step a thread loads 16 bytes per
 * token and spends T*R*4 scalar multiply-adds on it, i.e. 4/R bytes of
 * activation per scalar multiply-add (1 B at R = 4, 0.5 B at R = 8; counted with
 * the codeword's four-element group as one multiply-add the same quantity is
 * 16/R, i.e. 4 B then 2 B -- the ratio is what matters, and the convention is
 * named here because an earlier version of this comment mixed the two).
 * Measured on the tower's shapes, R = 8 is 1.35x-1.66x over the pre-change body
 * (probe-bn128.txt, column A); measured as the shipped A/B with the two
 * binaries interleaved, 1.26x-1.79x on one round pairing and 1.23x-1.75x on the
 * other.  It is BIT-IDENTICAL: the committed driver
 * (docs/benchmarks/eg2-embedding-2026-10-20/qa-identity) feeds the same payload
 * and activation bytes to the old and the new kernel in one process and compares
 * 12,492,416 floats over 15 cases, ragged tails included: zero differ. */
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

/* ------------------------------------------------------------------------- *
 * The tensor-core path: the same GEMM on f16 x f16 -> f32 mma.
 *
 * WHY THIS FORM (read first-hand, global rule 0.2; every line below was
 * re-read in /data/ds4-dfm-rs at the branch it is cited on):
 *   - the A fragment is gathered straight from the payload's f16 codebook, so
 *     the weights are never expanded: one code word is 4 f16 = 8 B and the
 *     fragment wants exactly the 4 B halves the lane owns, which is the
 *     register-direct form of the sibling VQ engines —
 *     cuda/ds41_vq_prefill_mma.cuh:259 ("the codebook lookup ... that IS the
 *     mma A fragment: no A/B tile writes, no ldmatrix") with the gather itself
 *     at :411-421 (four code words -> four fragment words), and
 *     cuda/glm_vq_mma.cuh:190-210 (the same code-word gather for a 12-bit
 *     stream, whose B operand is a direct uint2 read at :210).
 *   - f16, not bf16: the codebook is f16 (dim 4 x f16 per entry), so the
 *     weight side is exact; bf16 would drop 3 mantissa bits from every weight,
 *     which is the case ds4-dfm-rs names and rejects for an f16 codebook
 *     (ds41_vq_prefill_mma.cuh:24: "v2's f16 codebook would lose 3 mantissa
 *     bits to bf16, so v2 stays on the fused kernels"). The other gemma4's
 *     tensor-core VQ prefill, cuda/gemma_vq_prefill.cuh on the local branch
 *     feat/gemma4-tensor-core-fp8 (PR 143), stages its tile as bf16 because
 *     its codebook is E4M3 and that conversion is exact; its own f16 arm
 *     (:154-165) rounds through bf16, which is why this kernel takes the
 *     register-direct f16 form instead. The activation is the only value
 *     rounded, once, to f16 (11 significand bits; the same trade gemma4's f16
 *     P V lane documents, cuda/gemma4_attn_mma.cuh:76-91).
 *   - the activation tile and its B operand follow the bf16 tile of
 *     cuda/glm_vq_mma.cuh:143-166 (staging) and :210 (the direct B read);
 *     the swizzled 16-B chunk layout and the x4 ldmatrix that yields two n8
 *     tiles are the vqm form of ds41_vq_prefill_mma.cuh:82 (vqm_swz), :172
 *     (the B store) and :187 (the x4 ldmatrix).
 *   - f32 accumulation throughout; the row gain is applied after the sum, the
 *     same point as the scalar kernel (cuda/eg2vq_vq.cuh:4-9), so gain is
 *     never rounded into a weight.
 *
 * The only arithmetic difference from the scalar kernel is the f16 rounding
 * of x and the tensor core's own k16 summation order; the unit's gate
 * (tests/test_eg2vq_vq.cu) measures both against FP64 and declares the bound
 * this path must hold, and the commentary there is the record. Bit identity
 * with the scalar path is not claimed and cannot be: the tensor core sums 16
 * products in its own order.
 * ------------------------------------------------------------------------- */
enum {
    EG2VQ_MMA_BM = 128,        /* weight rows per block: 8 warps x 16 */
    EG2VQ_MMA_BK = 64,         /* columns per slab: 16 code words */
    EG2VQ_MMA_CW = EG2VQ_MMA_BK / 4,
    EG2VQ_MMA_PITCH = EG2VQ_MMA_CW + 4,          /* u16 rows: bank decorrelation */
    EG2VQ_MMA_WARPS = 8,
    EG2VQ_MMA_THREADS = EG2VQ_MMA_WARPS * 32,
    EG2VQ_MMA_SLAB_BYTES = EG2VQ_MMA_CW * EG2VQ_IDX_BITS / 8,   /* 24 */
    /* The two token tiles the dispatcher instantiates (measured crossover is
     * recorded with the unit): 64 tokens when the tile is latency-bound,
     * 128 when there is enough work to fill the card. */
    EG2VQ_MMA_BN_S = 64,
    EG2VQ_MMA_BN_L = 128,
};

/* m16n8k16, A row-major f16, B column-major f16, f32 accumulate. The exact
 * instruction gemma4's f16 attention lane uses (cuda/gemma4_attn_mma.cuh:87-91)
 * and the VQ prefill's register-direct form uses
 * (cuda/ds41_vq_prefill_mma.cuh:83-86). */
__device__ __forceinline__ void eg2vq_mma_f16(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t eg2vq_mma_pack(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}

/* One 16-B chunk of a token row: the eight f16 the ldmatrix x4 reads at once.
 * `chunk ^ (row & 7)` puts the eight rows of one matrix on eight bank groups
 * (cuda/glm_vq_mma.cuh:43-47, vqm_swz). */
__device__ __forceinline__ __half *eg2vq_mma_bs(__half *xs, uint32_t row, uint32_t chunk) {
    return xs + row * (uint32_t)EG2VQ_MMA_BK + ((chunk ^ (row & 7u)) << 3);
}

__device__ __forceinline__ void eg2vq_mma_ldsm4(uint32_t *r, const __half *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

/* The 12-bit extractor for one 24-byte slab row, in scalar registers (the
 * array form of eg2vq_vq_bits12 spills under the 128-register cap the
 * 128-token tile needs). `bitpos` is a compile-time multiple of 12 in the
 * unrolled loop, so the switch folds to the right two words. */
__device__ __forceinline__ uint32_t eg2vq_vq_bits12_6(
        uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t w4, uint32_t w5,
        uint32_t bitpos) {
    const uint32_t i = bitpos >> 5, s = bitpos & 31u;
    uint32_t lo, hi;
    switch (i) {
        case 0: lo = w0; hi = w1; break;
        case 1: lo = w1; hi = w2; break;
        case 2: lo = w2; hi = w3; break;
        case 3: lo = w3; hi = w4; break;
        case 4: lo = w4; hi = w5; break;
        default: lo = w5; hi = 0u; break;
    }
    return __funnelshift_r(lo, hi, s) & (uint32_t)EG2VQ_IDX_MASK;
}

/* One code word as the two 32-bit halves the A fragment wants: .x is
 * elements 0,1 and .y is elements 2,3 of the 4-value f16 vector. */
__device__ __forceinline__ uint2 eg2vq_mma_cw(const uint8_t *cb, uint32_t code) {
    return __ldg((const uint2 *)(cb + (size_t)code * 8u));
}

/* Grid: x = rows / 128, y = ceil(n_tok / BN); block: 256 threads = 8 warps,
 * each warp owning 16 weight rows and every token tile of the block. */
template <uint32_t BN>
__global__ void __launch_bounds__(EG2VQ_MMA_THREADS, 2) eg2vq_vq_gemm_mma_kernel(
        float *__restrict__ out,
        const float *__restrict__ x,
        uint32_t t_stride,
        const uint8_t *__restrict__ pay,
        uint32_t rows, uint32_t cols, uint32_t n_tok) {
    static_assert(BN == (uint32_t)EG2VQ_MMA_BN_S || BN == (uint32_t)EG2VQ_MMA_BN_L, "the two instantiated tiles");
    constexpr uint32_t TILES = BN / 8u;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, q = lane & 3u;
    const uint32_t r0 = blockIdx.x * (uint32_t)EG2VQ_MMA_BM;
    const uint32_t t0 = blockIdx.y * BN;
    const eg2vq_vq_mat m = eg2vq_vq_open(pay, rows, cols);
    if (!m.ok) { return; }

    __shared__ uint16_t idx_s[EG2VQ_MMA_BM * EG2VQ_MMA_PITCH];
    __shared__ __align__(16) __half xs[BN * (uint32_t)EG2VQ_MMA_BK];

    const uint32_t nt = (n_tok - t0 < BN) ? n_tok - t0 : BN;
    const uint32_t n_slab = cols / (uint32_t)EG2VQ_MMA_BK;
    const uint32_t la = warp * 16u + g, lb = la + 8u;
    const uint32_t row_a = r0 + la, row_b = r0 + lb;

    float acc[TILES][4];
    #pragma unroll
    for (uint32_t j = 0; j < TILES; j++) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }

    /* The 256 threads stage both operands: threads 0..127 decode one weight
     * row's 16 indices each (24 B, six words), and all 256 convert the 64 x 64
     * activation slab to f16, four threads per token and two 16-B chunks each. */
    for (uint32_t slab = 0; slab < n_slab; slab++) {
        if (tid < (uint32_t)EG2VQ_MMA_BM) {
            const uint32_t *p = (const uint32_t *)(m.ix + (size_t)(r0 + tid) * m.row_bytes
                             + (size_t)slab * (size_t)EG2VQ_MMA_SLAB_BYTES);
            const uint32_t w0 = __ldg(p + 0), w1 = __ldg(p + 1), w2 = __ldg(p + 2);
            const uint32_t w3 = __ldg(p + 3), w4 = __ldg(p + 4), w5 = __ldg(p + 5);
            uint16_t *dst = idx_s + tid * (uint32_t)EG2VQ_MMA_PITCH;
            #pragma unroll
            for (uint32_t k = 0; k < (uint32_t)EG2VQ_MMA_CW; k++) {
                dst[k] = (uint16_t)eg2vq_vq_bits12_6(w0, w1, w2, w3, w4, w5, k * 12u);
            }
        }
        #pragma unroll
        for (uint32_t h = 0; h < BN / 64u; h++) {
            const uint32_t tok = h * 64u + (tid >> 2u);   /* 0..BN-1, four threads per token */
            const uint32_t part = tid & 3u;               /* 0..3 */
            const uint32_t col = slab * (uint32_t)EG2VQ_MMA_BK + part * 16u;
            #pragma unroll
            for (uint32_t c = 0; c < 2u; c++) {
                float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (tok < nt) {
                    const float *src = x + (size_t)(t0 + tok) * t_stride + col + c * 8u;
                    v0 = *(const float4 *)src;
                    v1 = *(const float4 *)(src + 4u);
                }
                uint4 w;
                w.x = eg2vq_mma_pack(v0.x, v0.y);
                w.y = eg2vq_mma_pack(v0.z, v0.w);
                w.z = eg2vq_mma_pack(v1.x, v1.y);
                w.w = eg2vq_mma_pack(v1.z, v1.w);
                *(uint4 *)eg2vq_mma_bs(xs, tok, part * 2u + c) = w;
            }
        }
        __syncthreads();

        /* Four k16 steps per 64-column slab. The lane's A fragment is rows
         * la and lb at columns 2q,2q+1 and 2q+8,2q+9: code words 4kk + q/2 and
         * 4kk + 2 + q/2, word half q&1 -- the mapping
         * ds41_vq_prefill_mma.cuh:344-352 states for its own code words. */
        #pragma unroll
        for (uint32_t kk = 0; kk < (uint32_t)(EG2VQ_MMA_BK / 16u); kk++) {
            const uint32_t c0 = 4u * kk + (q >> 1u);
            const uint32_t c1 = c0 + 2u;
            const uint2 wa = eg2vq_mma_cw(m.cb, idx_s[la * (uint32_t)EG2VQ_MMA_PITCH + c0]);
            const uint2 wb = eg2vq_mma_cw(m.cb, idx_s[lb * (uint32_t)EG2VQ_MMA_PITCH + c0]);
            const uint2 wc = eg2vq_mma_cw(m.cb, idx_s[la * (uint32_t)EG2VQ_MMA_PITCH + c1]);
            const uint2 wd = eg2vq_mma_cw(m.cb, idx_s[lb * (uint32_t)EG2VQ_MMA_PITCH + c1]);
            const uint32_t sel = q & 1u;
            const uint32_t a[4] = {sel ? wa.y : wa.x, sel ? wb.y : wb.x,
                                   sel ? wc.y : wc.x, sel ? wd.y : wd.x};
            #pragma unroll
            for (uint32_t j = 0; j < TILES; j += 2u) {
                uint32_t b[4];
                const uint32_t mat = lane >> 3u;
                const uint32_t tok = (j + (mat >> 1u)) * 8u + (lane & 7u);
                eg2vq_mma_ldsm4(b, eg2vq_mma_bs(xs, tok, kk * 2u + (mat & 1u)));
                eg2vq_mma_f16(acc[j], a, b[0], b[1]);
                eg2vq_mma_f16(acc[j + 1u], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    /* Row gain once per output element, the scalar kernel's point
     * (cuda/eg2vq_vq.cuh:4-9): out = gain * sum, never a rounded weight. */
    __half gh;
    memcpy(&gh, m.gr + (size_t)row_a * 2u, 2);
    const float ga = __half2float(gh);
    memcpy(&gh, m.gr + (size_t)row_b * 2u, 2);
    const float gb = __half2float(gh);
    #pragma unroll
    for (uint32_t j = 0; j < TILES; j++) {
        const uint32_t tok = t0 + j * 8u + (lane & 3u) * 2u;
        if (tok < n_tok) {
            out[(size_t)tok * rows + row_a] = acc[j][0] * ga;
            out[(size_t)tok * rows + row_b] = acc[j][2] * gb;
        }
        if (tok + 1u < n_tok) {
            out[(size_t)(tok + 1u) * rows + row_a] = acc[j][1] * ga;
            out[(size_t)(tok + 1u) * rows + row_b] = acc[j][3] * gb;
        }
    }
}

/* ------------------------------------------------------------------------- *
 * The same GEMM in the form the other gemma4 uses (PR 143, local branch
 * feat/gemma4-tensor-core-fp8): restore the weight tile into shared and run
 * nvcuda::wmma 16x16x16 with a 16-byte row stride. Read first-hand from that
 * branch: cuda/gemma_vq_prefill.cuh:54-59 (the tile and PAD = 8),
 * :121-166 (the restored weight tile), :233-304 (the USE_MMA arm: eight
 * warps, 16 rows each, two n16 tiles, eg2wmma::load/mma/store, the scalar
 * fallback knob). The one deliberate difference is the element type: that
 * codebook is E4M3, so its tile is bf16 and the conversion is exact; this
 * codebook is f16, and the branch's own f16 arm (:154-165) would round every
 * weight through bf16, losing 3 mantissa bits. So the tile here is f16, the
 * weight is exact, and only the activation is rounded once. The row gain
 * stays after the sum, unlike the branch, because here it must not be
 * rounded into a weight.
 *
 * Kept beside the register-direct kernel so the two are measured in one
 * window (bench_eg2vq_vq); the public entry uses whichever that measurement
 * chose, and the record names it.
 * ------------------------------------------------------------------------- */
template <uint32_t BN>
__global__ void __launch_bounds__(EG2VQ_MMA_THREADS, 2) eg2vq_vq_gemm_wmma_kernel(
        float *__restrict__ out,
        const float *__restrict__ x,
        uint32_t t_stride,
        const uint8_t *__restrict__ pay,
        uint32_t rows, uint32_t cols, uint32_t n_tok) {
    static_assert(BN == 32u || BN == 64u, "the WMMA tiles are 32 or 64 tokens");
    constexpr uint32_t BM = EG2VQ_MMA_BM, BK = EG2VQ_MMA_BK, PAD = 8u;
    constexpr uint32_t STRIDE = BK + PAD;         /* 72 halves = 144 B: 16-B aligned */
    constexpr uint32_t THREADS = EG2VQ_MMA_THREADS;
    const eg2vq_vq_mat m = eg2vq_vq_open(pay, rows, cols);
    if (!m.ok) { return; }
    const uint32_t tid = threadIdx.x, warp = tid >> 5u;
    const uint32_t r0 = blockIdx.x * BM;
    const uint32_t t0 = blockIdx.y * BN;

    __shared__ __half ws[BM * STRIDE];           /* the restored weight tile */
    __shared__ __half xs[BN * STRIDE];           /* the activation tile */
    __shared__ uint16_t idx_s[BM * (uint32_t)EG2VQ_MMA_PITCH];

    eg2wmma::fragment<eg2wmma::accumulator, 16, 16, 16, float> acc[BN / 16u];
    #pragma unroll
    for (uint32_t j = 0; j < BN / 16u; j++) { eg2wmma::fill_fragment(acc[j], 0.0f); }
    const uint32_t m_base = warp * 16u;

    const uint32_t n_slab = cols / BK;
    for (uint32_t slab = 0; slab < n_slab; slab++) {
        /* indices: one thread per weight row, 16 code words (24 B) */
        if (tid < BM) {
            const uint32_t *p = (const uint32_t *)(m.ix + (size_t)(r0 + tid) * m.row_bytes
                             + (size_t)slab * (size_t)EG2VQ_MMA_SLAB_BYTES);
            const uint32_t w0 = __ldg(p + 0), w1 = __ldg(p + 1), w2 = __ldg(p + 2);
            const uint32_t w3 = __ldg(p + 3), w4 = __ldg(p + 4), w5 = __ldg(p + 5);
            uint16_t *dst = idx_s + tid * (uint32_t)EG2VQ_MMA_PITCH;
            #pragma unroll
            for (uint32_t k = 0; k < (uint32_t)EG2VQ_MMA_CW; k++) {
                dst[k] = (uint16_t)eg2vq_vq_bits12_6(w0, w1, w2, w3, w4, w5, k * 12u);
            }
        }
        __syncthreads();   /* the weight stage reads another thread's indices */
        /* weights: two threads per row, eight code words each, copied as the
         * codebook's own f16 words (no conversion, so the weight is exact) */
        {
            const uint32_t r = tid / 2u, part = tid % 2u;
            __half *dst = ws + r * STRIDE + part * 32u;
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const uint32_t code = idx_s[r * (uint32_t)EG2VQ_MMA_PITCH + part * 8u + j];
                const uint2 cw = __ldg((const uint2 *)(m.cb + (size_t)code * 8u));
                *(uint2 *)(dst + j * 4u) = cw;
            }
        }
        /* activations: BN x BK floats -> f16, one round */
        for (uint32_t i = tid * 8u; i < BN * BK; i += THREADS * 8u) {
            const uint32_t t = i / BK, k = i % BK;
            const uint32_t token = t0 + t;
            const float *src = x + (size_t)token * t_stride + slab * BK + k;
            __half *dst = xs + t * STRIDE + k;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; e++) {
                dst[e] = (token < n_tok) ? __float2half_rn(src[e]) : __half(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t ks = 0; ks < BK; ks += 16u) {
            eg2wmma::fragment<eg2wmma::matrix_a, 16, 16, 16, __half, eg2wmma::row_major> a_frag;
            eg2wmma::load_matrix_sync(a_frag, ws + m_base * STRIDE + ks, STRIDE);
            #pragma unroll
            for (uint32_t j = 0; j < BN / 16u; j++) {
                eg2wmma::fragment<eg2wmma::matrix_b, 16, 16, 16, __half, eg2wmma::col_major> b_frag;
                eg2wmma::load_matrix_sync(b_frag, xs + j * 16u * STRIDE + ks, STRIDE);
                eg2wmma::mma_sync(acc[j], a_frag, b_frag, acc[j]);
            }
        }
        __syncthreads();
    }

    /* Store-back through shared, one 32-token half at a time, then a
     * coalesced write with the row gain applied after the sum. */
    const float gain = __half2float(*(const __half *)(m.gr + (size_t)(r0 + tid % BM) * 2u));
    float *out_s = (float *)ws;
    #pragma unroll
    for (uint32_t hh = 0; hh < BN / 32u; hh++) {
        #pragma unroll
        for (uint32_t j = 0; j < 2u; j++) {
            eg2wmma::store_matrix_sync(out_s + j * 16u * BM + m_base, acc[hh * 2u + j],
                                    (int)BM, eg2wmma::mem_col_major);
        }
        __syncthreads();
        for (uint32_t idx = tid; idx < 32u * BM; idx += THREADS) {
            const uint32_t t_local = idx / BM, r_local = idx % BM;
            const uint32_t token = t0 + hh * 32u + t_local;
            if (token < n_tok) {
                out[(size_t)token * rows + r0 + r_local] = out_s[idx] * gain;
            }
        }
        __syncthreads();
    }
}

/* The WMMA tile cannot take fewer than 32 tokens usefully, and its 32-token
 * tile is the branch's own choice; 64 is the same kernel with two halves. */
static inline int eg2vq_vq_gemm_wmma32(float *out, const float *x, uint32_t t_stride,
                                       const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (!out || !x || !pay || !n_tok || !rows || !cols) { return 0; }
    if ((rows % EG2VQ_MMA_BM) || (cols % EG2VQ_MMA_BK) || (t_stride < cols)) { return 0; }
    dim3 block((uint32_t)EG2VQ_MMA_THREADS);
    dim3 grid(rows / (uint32_t)EG2VQ_MMA_BM, (n_tok + 31u) / 32u);
    eg2vq_vq_gemm_wmma_kernel<32><<<grid, block>>>(out, x, t_stride, pay, rows, cols, n_tok);
    return 1;
}

static inline int eg2vq_vq_gemm_wmma64(float *out, const float *x, uint32_t t_stride,
                                       const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (!out || !x || !pay || !n_tok || !rows || !cols) { return 0; }
    if ((rows % EG2VQ_MMA_BM) || (cols % EG2VQ_MMA_BK) || (t_stride < cols)) { return 0; }
    dim3 block((uint32_t)EG2VQ_MMA_THREADS);
    dim3 grid(rows / (uint32_t)EG2VQ_MMA_BM, (n_tok + 63u) / 64u);
    eg2vq_vq_gemm_wmma_kernel<64><<<grid, block>>>(out, x, t_stride, pay, rows, cols, n_tok);
    return 1;
}

/* `cols` must be a whole number of 64-column slabs and `rows` a whole number
 * of 128-row tiles (the tower's shapes are). The token tile is chosen from
 * the measured crossover (bench_eg2vq_vq, both tiles over the tower's shapes
 * and token counts): 128 tokens when the call carries at least 512 tokens and
 * its 128-token grid still fills one wave of 128 blocks, 64 tokens otherwise.
 * The 128-token tile halves the block count and the index traffic per token,
 * which wins from about one wave up; below that the call cannot fill the card
 * and the longer tile only lengthens the wave (measured 1.5x for the 64-token
 * tile at 115-306 tokens). */
enum {
    /* The MMA tile fits one token too and still beats the scalar kernel there
     * (the call is weight-bandwidth bound, and the MMA block covers twice the
     * rows), so there is no minimum token count: only a shape the tiles cannot
     * cover falls back to the scalar kernel. */
    EG2VQ_MMA_MIN_TOK = 1,
    /* The 128-token tile is chosen only once the call is at least one full
     * wave of its blocks (128 SMs) and has enough tokens to amortize the
     * longer per-block critical path; the crossover is measured in
     * bench_eg2vq_vq and recorded with the unit. */
    EG2VQ_MMA_BIG_TOK = 512,
    EG2VQ_MMA_WAVE_BLOCKS = 128,
};

template <uint32_t BN>
static inline int eg2vq_vq_gemm_mma_t(float *out, const float *x, uint32_t t_stride,
                                      const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (!out || !x || !pay || !n_tok || !rows || !cols) { return 0; }
    if ((rows % EG2VQ_MMA_BM) || (cols % EG2VQ_MMA_BK) || (t_stride < cols)) { return 0; }
    dim3 block((uint32_t)EG2VQ_MMA_THREADS);
    dim3 grid(rows / (uint32_t)EG2VQ_MMA_BM, (n_tok + BN - 1u) / BN);
    eg2vq_vq_gemm_mma_kernel<BN><<<grid, block>>>(out, x, t_stride, pay, rows, cols, n_tok);
    return 1;
}

static inline int eg2vq_vq_gemm_mma64(float *out, const float *x, uint32_t t_stride,
                                      const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    return eg2vq_vq_gemm_mma_t<EG2VQ_MMA_BN_S>(out, x, t_stride, pay, rows, cols, n_tok);
}

static inline int eg2vq_vq_gemm_mma128(float *out, const float *x, uint32_t t_stride,
                                       const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    return eg2vq_vq_gemm_mma_t<EG2VQ_MMA_BN_L>(out, x, t_stride, pay, rows, cols, n_tok);
}

static inline int eg2vq_vq_gemm_mma(float *out, const float *x, uint32_t t_stride,
                                    const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (!out || !x || !pay || !n_tok || !rows || !cols) { return 0; }
    if ((rows % EG2VQ_MMA_BM) || (cols % EG2VQ_MMA_BK) || (t_stride < cols)) { return 0; }
    const uint32_t row_tiles = rows / (uint32_t)EG2VQ_MMA_BM;
    const uint32_t blocks_l = row_tiles * ((n_tok + (uint32_t)EG2VQ_MMA_BN_L - 1u) / (uint32_t)EG2VQ_MMA_BN_L);
    return (n_tok >= (uint32_t)EG2VQ_MMA_BIG_TOK && blocks_l >= (uint32_t)EG2VQ_MMA_WAVE_BLOCKS)
        ? eg2vq_vq_gemm_mma_t<EG2VQ_MMA_BN_L>(out, x, t_stride, pay, rows, cols, n_tok)
        : eg2vq_vq_gemm_mma_t<EG2VQ_MMA_BN_S>(out, x, t_stride, pay, rows, cols, n_tok);
}

/* `cols` must be a whole number of 256-column slabs and `rows` a whole number
 * of 64-row tiles: every matrix in this family's tier is (512, 1024, 2048)
 * wide and (512 ... 12288) tall, all of them multiples.  A shape that is not
 * returns 0 rather than silently computing a partial tile. */
static inline int eg2vq_vq_gemm_scalar(float *out, const float *x, uint32_t t_stride,
                                       const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (!out || !x || !pay || !n_tok || !rows || !cols) { return 0; }
    if ((rows % EG2VQ_BN) || (cols % EG2VQ_BK) || (t_stride < cols)) { return 0; }
    dim3 block(EG2VQ_NTX, EG2VQ_NTY);
    dim3 grid(rows / EG2VQ_BN, (n_tok + EG2VQ_BM - 1) / EG2VQ_BM);
    eg2vq_vq_gemm_kernel<<<grid, block>>>(out, x, t_stride, pay, rows, cols, n_tok);
    return 1;
}

/* The public entry: the tensor-core path for a token count that can fill its
 * tile, the scalar kernel otherwise (and for any shape the MMA tiles cannot
 * cover). Both paths read the same payload and write the same layout. */
static inline int eg2vq_vq_gemm(float *out, const float *x, uint32_t t_stride,
                                const uint8_t *pay, uint32_t rows, uint32_t cols, uint32_t n_tok) {
    if (n_tok >= (uint32_t)EG2VQ_MMA_MIN_TOK &&
        eg2vq_vq_gemm_mma(out, x, t_stride, pay, rows, cols, n_tok)) {
        return 1;
    }
    return eg2vq_vq_gemm_scalar(out, x, t_stride, pay, rows, cols, n_tok);
}
