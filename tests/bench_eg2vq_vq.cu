/* bench_eg2vq_vq.cu — the gemma_vq text tower's dim-4 DQVQ payload GEMM,
 * measured across the tower's own matrices and token counts (embeddinggemma2
 * performance unit, phase 7 row P3.1's kernel).
 *
 * WHY THIS EXISTS: the nsys attribution of a real packed-batch forward puts
 * eg2vq_vq_gemm_kernel at 36.5% of the kernel time before the attention unit's
 * change and 53.0% after it (212.6 ms of 401.1 ms for 8 items / 2451 tokens, 434
 * launches), at ~2.9 TFLOP/s against this part's ~35.5 TFLOP/s of FP32.  A
 * whole-forward profile cannot say why, and cannot say whether a change helped:
 * this harness times the kernel alone, one matrix at a time, on the shapes the
 * tower actually runs, and reports the effective GFLOP/s beside the
 * milliseconds.
 *
 * The shapes are the tower's own (ds4_eg2vq.h pins them): the sliding layers'
 * q/k/v (1024/512/512 rows at 512 columns), o_proj (512 x 1024), the FFN's
 * gate/up (2048 x 512) and down (512 x 2048), the per-layer embedding's own
 * pair (512 x 512) and the 12288 x 512 prologue, plus the full layers' q
 * (2048 x 512) and o (512 x 2048).  The token counts are the workload's: 1 (the
 * prologue's case), 115, 306, 565, 1414 and 4520 (a batch of eight 565-token
 * items).  The tile is 64 tokens, so a count that is not a multiple of 64 also
 * exercises the ragged tail.
 *
 * The payload is packed here by the same writer the gate uses
 * (tests/test_eg2vq_vq.cu's p_pack, which mirrors /data/YoungAi's vq_pack
 * :242-266): [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols], codebook nc*dim
 * f16, rows f16 gains, then one continuous little-endian 12-bit index stream.
 * Its contents are random, which is what a throughput measurement wants: the
 * kernel's work does not depend on the values, and every codeword is over the
 * whole book.  Correctness is NOT this harness's job -- tests/test_eg2vq_vq.cu
 * (make test-eg2vq-vq) owns that, against the payload bytes and against FP64.
 *
 * usage: bench_eg2vq_vq [repeats]        (default 20)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

#include "../cuda/eg2vq_vq.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static std::mt19937 rng(20261020u);

static uint16_t p_h16(float v) { __half h = __float2half_rn(v); uint16_t u; memcpy(&u, &h, 2); return u; }

/* the payload writer, mirroring vq_pack (:242-266) as the gate's p_pack does */
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

struct q_shape {
    uint32_t rows, cols;
    const char *tag;
};

int main(int argc, char **argv) {
    const int repeats = (argc > 1) ? atoi(argv[1]) : 20;
    const uint32_t nc = 4096u, dim = 4u;

    /* the tower's own matrices (ds4_eg2vq.h: the sliding layers' q/k/v/o, the
     * FFN, the per-layer embedding's pair, the 12288-row prologue, and the full
     * layers' q/o) */
    const q_shape shapes[] = {
        {  512,  512, "k/v_proj_s" },
        { 1024,  512, "q_proj_s" },
        {  512, 1024, "o_proj_s" },
        { 2048,  512, "ffn_gate_s" },
        { 2048,  512, "ffn_up_s" },
        {  512, 2048, "ffn_down_s" },
        {  512,  512, "ple_gate_s" },
        {  512,  512, "ple_proj_s" },
        { 2048,  512, "q_proj_f" },
        {  512, 2048, "o_proj_f" },
        { 12288, 512, "ple_model_proj" },
    };
    const uint32_t toks[] = { 1, 115, 306, 565, 1414, 4520 };

    printf("# bench_eg2vq_vq — %d repeats per row, one kernel launch per measurement\n", repeats);
    printf("%-16s %6s %6s %7s %10s %10s %10s %9s\n",
           "shape", "rows", "cols", "tokens", "ms_min", "ms_med", "ms_max", "gflop/s");
    int fail = 0;
    for (const q_shape &s : shapes) {
        /* one payload per shape, reused across the token counts */
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
            const size_t out_elems = (size_t)tok * s.rows;
            CK(cudaFree(dy));
            CK(cudaMalloc(&dy, out_elems * 4));
            const double flop = 2.0 * (double)s.rows * (double)s.cols * (double)tok;
            if (!eg2vq_vq_gemm(dy, dx, s.cols, dpay, s.rows, s.cols, tok)) {
                printf("%-16s REFUSED by the launcher\n", s.tag);
                fail = 1;
                continue;
            }
            std::vector<float> ms;
            for (int rep = 0; rep < repeats + 3; rep++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                CK(cudaEventRecord(a));
                if (!eg2vq_vq_gemm(dy, dx, s.cols, dpay, s.rows, s.cols, tok)) { fail = 1; break; }
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float msec = 0.0f;
                CK(cudaEventElapsedTime(&msec, a, b));
                if (rep >= 3) { ms.push_back(msec); }
                CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            }
            if (ms.empty()) { continue; }
            std::vector<float> sorted = ms;
            for (size_t i = 0; i + 1 < sorted.size(); i++) {
                for (size_t j = i + 1; j < sorted.size(); j++) {
                    if (sorted[j] < sorted[i]) { const float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
                }
            }
            const float mn = sorted.front(), mx = sorted.back(), md = sorted[sorted.size() / 2];
            printf("%-16s %6u %6u %7u %10.4f %10.4f %10.4f %9.1f\n",
                   s.tag, s.rows, s.cols, tok, mn, md, mx, flop / ((double)md * 1.0e6));
        }
        CK(cudaFree(dpay));
        CK(cudaFree(dx));
        CK(cudaFree(dy));
    }
    printf("%s\n", fail ? "BENCH_EG2VQ_VQ_FAIL" : "BENCH_EG2VQ_VQ_OK");
    return fail;
}
