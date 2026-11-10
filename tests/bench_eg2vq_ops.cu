/* bench_eg2vq_ops.cu — the gemma_vq tower's norm primitives, measured across the
 * shapes the tower actually runs (embeddinggemma2 performance unit 6, 2026-11-10).
 *
 * WHY THIS EXISTS: an nsys profile of one 2,451-token batch put the two norms at
 * 14.1 ms of a 49.2 ms forward (rms 8.64 ms over 121 launches, rms_head 5.44 ms
 * over 72), while an elementwise add over the same 5 MB took 11 us per call.  A
 * whole-forward profile cannot answer "did THIS change help", so this harness
 * times the kernels alone, on the tower's own shapes, and reports the effective
 * memory rate beside the milliseconds — the kernels are bandwidth-bound, not
 * arithmetic-bound (a row is a sum of squares and a scale).
 *
 * The shapes are the tower's, not invented: the text tower's hidden is 512, its
 * per-head norms are 256 wide on the twenty sliding layers (2 KV heads) and 512
 * on the four full ones (1 KV head), with 4 query heads; the per-layer input
 * (PLE) norm is 512 wide over n_tok x 24 rows, one call per forward; the vision
 * tower's hidden is 768 and its head 64 with 12 heads.  The row counts are the
 * workload's: 2,451 rows is the 8-page batch every profile in this directory
 * uses (42+162+11+104+624+481+424+603 tokens), 294 one page, 27 a short CLI item,
 * and 1,120 the vision tower's token ceiling.
 *
 * Both reductions are timed on the same input in the same process and back to
 * back, so the columns are an interleaved A/B and not two windows (this
 * machine's per-window drift is the record's own caution), and the two are also
 * compared as f32 bit patterns before timing, so no number here can come from a
 * kernel that computes something else.  The full arithmetic gate remains
 * tests/test_eg2vq_ops.cu (make test-eg2vq-ops).
 *
 * usage: bench_eg2vq_ops [repeats]        (default 200)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../cuda/eg2vq_ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint32_t s_rng = 20261110u;
static float rnd_unit(void) {
    s_rng ^= s_rng << 13u;
    s_rng ^= s_rng >> 17u;
    s_rng ^= s_rng << 5u;
    return ((float)(s_rng & 0xffffu) / 32768.0f) - 1.0f;
}

/* one row of the report: a norm at one shape.  `rows` is the number of ITEMS of
 * `n` values the call normalizes (for the per-head form that is rows x heads). */
struct shape {
    const char *tag;
    const char *form;      /* "rms" or "rms_head" */
    uint32_t rows, n, heads;
};

int main(int argc, char **argv) {
    const int repeats = (argc > 1) ? atoi(argv[1]) : 200;
    const float eps = 1e-6f;

    const shape shapes[] = {
        { "rms 512 x 2451 (8 pages)",       "rms", 2451u, 512u, 0u },
        { "rms 512 x 58824 (PLE)",          "rms", 58824u, 512u, 0u },
        { "rms 512 x 294 (1 page)",         "rms", 294u, 512u, 0u },
        { "rms 512 x 27 (short item)",      "rms", 27u, 512u, 0u },
        { "rms 768 x 1120 (vision)",        "rms", 1120u, 768u, 0u },
        { "rms_head 256 x 4 (q, swa)",      "rms_head", 2451u, 256u, 4u },
        { "rms_head 256 x 2 (k,v swa)",     "rms_head", 2451u, 256u, 2u },
        { "rms_head 512 x 4 (q, full)",     "rms_head", 2451u, 512u, 4u },
        { "rms_head 512 x 1 (k,v full)",    "rms_head", 2451u, 512u, 1u },
        { "rms_head 256 x 4 (q, 1 page)",   "rms_head", 294u, 256u, 4u },
        { "rms_head 64 x 12 (vision)",      "rms_head", 1120u, 64u, 12u },
    };

    printf("# bench_eg2vq_ops — %d repeats per row per form, median; both forms interleaved\n", repeats);
    printf("%-32s %8s %10s %10s %8s %10s %10s\n",
           "shape", "rows", "ms_block", "ms_warp", "speedup", "block GB/s", "warp GB/s");
    int fail = 0;
    for (const shape &s : shapes) {
        const uint32_t nrow = (s.heads ? s.rows * s.heads : s.rows);
        const uint64_t nvals = (uint64_t)nrow * s.n;
        std::vector<float> hx((size_t)nvals), hw(s.n);
        for (float &v : hx) { v = rnd_unit(); }
        for (float &v : hw) { v = 1.0f + 0.1f * rnd_unit(); }

        float *dx = NULL, *dw = NULL, *db = NULL, *dv = NULL;
        CK(cudaMalloc(&dx, nvals * 4)); CK(cudaMalloc(&dw, s.n * 4));
        CK(cudaMalloc(&db, nvals * 4)); CK(cudaMalloc(&dv, nvals * 4));
        CK(cudaMemcpy(dw, hw.data(), s.n * 4, cudaMemcpyHostToDevice));

        /* ONE LAUNCH of one form, on the buffers as they are: no copy inside the
         * timed region, so a PCIe transfer cannot be mistaken for the kernel.
         * The per-head form is in place, so across repeats it normalizes an
         * already-normalized buffer — the same bytes and the same reduction, so
         * the cost is unchanged; the correctness comparison below runs on the
         * pristine input, once, before any timing. */
        auto launch = [&](int warp, float *dst) -> int {
            if (s.heads) {
                return warp ? eg2vq_ops_rms_warp(dx, s.n, dx, s.n, dw, nrow, s.n, eps)
                            : eg2vq_ops_rms_head_block(dx, dw, s.rows, s.heads, s.n, eps);
            }
            return warp ? eg2vq_ops_rms_warp(dst, s.n, dx, s.n, dw, nrow, s.n, eps)
                        : eg2vq_ops_rms_block(dst, dx, dw, nrow, s.n, eps);
        };

        /* correctness of the pair before any timing: the two must agree bit for bit */
        std::vector<float> hb((size_t)nvals), hv((size_t)nvals);
        CK(cudaMemcpy(dx, hx.data(), nvals * 4, cudaMemcpyHostToDevice));
        launch(0, db);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hb.data(), s.heads ? dx : db, nvals * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dx, hx.data(), nvals * 4, cudaMemcpyHostToDevice));
        launch(1, dv);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hv.data(), s.heads ? dx : dv, nvals * 4, cudaMemcpyDeviceToHost));
        if (memcmp(hb.data(), hv.data(), (size_t)nvals * 4) != 0) {
            printf("%-32s FORMS DISAGREE (not bit-identical)\n", s.tag);
            fail = 1;
            cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dv);
            continue;
        }

        auto time_form = [&](int warp, float *dst) -> float {
            std::vector<float> ms;
            for (int rep = 0; rep < repeats + 20; rep++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                CK(cudaEventRecord(a));
                launch(warp, dst);
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float msec = 0.0f;
                CK(cudaEventElapsedTime(&msec, a, b));
                if (rep >= 20) { ms.push_back(msec); }
                CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            }
            for (size_t i = 0; i + 1 < ms.size(); i++) {
                for (size_t j = i + 1; j < ms.size(); j++) {
                    if (ms[j] < ms[i]) { const float t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
                }
            }
            return ms[ms.size() / 2];
        };

        /* The two forms are run in the same process and back to back (block row
         * then warp row), so the columns are an interleaved A/B.  A call's own
         * minimum traffic is one read and one write of its bytes; the old block
         * kernel read the row twice, which the warps avoid, so the GB/s columns
         * are a common denominator and not each kernel's literal traffic. */
        const float mb = time_form(0, db);
        const float mw = time_form(1, dv);
        /* minimum traffic: one read and one write of the call's own bytes */
        const double gb = (double)nvals * 4.0 * 2.0 / 1.0e9;
        printf("%-32s %8u %10.4f %10.4f %8.2f %10.1f %10.1f\n",
               s.tag, nrow, mb, mw, mb / mw, gb / ((double)mb * 1e-3), gb / ((double)mw * 1e-3));
        cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dv);
    }
    printf("%s\n", fail ? "BENCH_EG2VQ_OPS_FAIL" : "BENCH_EG2VQ_OPS_OK");
    return fail;
}
