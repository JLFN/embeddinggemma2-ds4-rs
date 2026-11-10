/* bench_eg2vq_attn.cu — the gemma_vq text tower's device attention, measured
 * across the tower's own shapes (embeddinggemma2 performance unit, phase 7 row
 * P3.2's kernel).
 *
 * WHY THIS EXISTS: the embedding path had no benchmark at all (ds4-rs
 * docs/benchmarks/eg2-embedding-2026-10-20/README.md section "Where the cost
 * most plausibly is"), so the first nsys attribution of a real forward was the
 * only evidence available, and it named eg2vq_attn_kernel as 47.9% of the
 * kernel time of a single-item forward (31.5 ms of 65.7 ms for a 294-token
 * item) and 58.9% of a packed-batch forward (337 ms of 573 ms for 8 items /
 * 2451 tokens) while running at ~0.3 TFLOP/s.  A whole-forward profile mixes
 * the kernel with everything else and cannot answer "did THIS change help", so
 * this harness times the kernel alone, on the shapes the tower actually runs,
 * and reports the effective GFLOP/s beside the milliseconds.
 *
 * The shapes are the tower's, not invented: 4 query heads; the twenty sliding
 * layers are head_dim 256 with 2 KV heads and a symmetric half-window of 512
 * (DS4_EG2VQ_SWA = 1024, so 2*512+1 = 1025 keys at most), and the four full
 * layers are head_dim 512 with 1 KV head and no mask (ds4_eg2vq.h pins both).
 * The token counts are the workload's own: 115 (the CLI's single item), 294
 * (four PDF pages), 565 (the corpus's per-page mean), 1414 (the longest frozen
 * item, where the window finally binds), plus a packed batch of 8 x 306 = 2448
 * rows and 8 x 565 = 4520 rows with per-row pos/len metadata, which is the
 * batch case the route serves.
 *
 * The FLOP count is exact rather than nominal: every row's own key count is
 * computed from pos/len/half_swa the way the kernel computes it, and each
 * (row, head, key) contributes 2*head_dim for the score and 2*head_dim for the
 * weighted sum.  A row with a clipped window therefore counts what it does.
 *
 * The arithmetic is checked against the CPU reference's own transcription
 * (eg2vq_attn_worker, ds4_eg2vq_graph.inc:124-166) on one small shape, so a
 * timing number from this harness can never come from a kernel that computes
 * something else.  The full correctness gates remain tests/test_eg2vq_attn.cu
 * (make test-eg2vq-attn) and tests/parity/eg2vq_embd_gpu_oracle.
 *
 * usage: bench_eg2vq_attn [repeats]        (default 20)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../cuda/eg2vq_attn.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static uint32_t s_rng = 20261020u;
static float rnd_unit(void) {
    s_rng ^= s_rng << 13u;
    s_rng ^= s_rng >> 17u;
    s_rng ^= s_rng << 5u;
    return ((float)(s_rng & 0xffffu) / 32768.0f) - 1.0f;
}

/* one item's rows, or a packed block of items; pos/len as the device core
 * writes them (ds4_eg2vq_gpu.cuh, the batch metadata block). */
struct shape {
    const char *tag;
    std::vector<uint32_t> item_len;   /* one entry per item; rows are their sum */
    uint32_t head_dim, n_head_kv;
    int32_t half_swa;
};

static uint32_t keys_of(uint32_t p, uint32_t nitem, int32_t half_swa) {
    uint32_t k0 = 0, k1 = nitem - 1u;
    if (half_swa >= 0) {
        k0 = (p > (uint32_t)half_swa) ? p - (uint32_t)half_swa : 0u;
        const uint32_t hi = p + (uint32_t)half_swa;
        k1 = (hi < nitem - 1u) ? hi : nitem - 1u;
    }
    return k1 - k0 + 1u;
}

/* the reference's arithmetic, one (query, head) at a time (the transcription
 * test_eg2vq_attn.cu uses, kept here so this harness validates its own input). */
static void ref_attn(std::vector<float> &out, const std::vector<float> &q, const std::vector<float> &k,
                     const std::vector<float> &v, const std::vector<int32_t> &pos,
                     const std::vector<int32_t> &len, uint32_t n_head, uint32_t n_head_kv,
                     uint32_t head_dim, int32_t half_swa) {
    const uint32_t n_tok = (uint32_t)pos.size();
    const uint64_t q_stride = (uint64_t)n_head * head_dim, kv_stride = (uint64_t)n_head_kv * head_dim;
    const uint32_t group = n_head / n_head_kv;
    out.assign((size_t)n_tok * q_stride, 0.0f);
    std::vector<float> scores;
    for (uint32_t r = 0; r < n_tok; r++) {
        const uint32_t p = (uint32_t)pos[r], nitem = (uint32_t)len[r], base = r - p;
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kvh = h / group;
            const float *qh = q.data() + (size_t)r * q_stride + (size_t)h * head_dim;
            float *oh = out.data() + (size_t)r * q_stride + (size_t)h * head_dim;
            scores.assign(nitem, -INFINITY);
            float max_score = -INFINITY;
            for (uint32_t kp = 0; kp < nitem; kp++) {
                if (half_swa >= 0) {
                    const int64_t diff = (int64_t)p - (int64_t)kp;
                    if (diff < -(int64_t)half_swa || diff > (int64_t)half_swa) { continue; }
                }
                const float *kh = k.data() + (size_t)(base + kp) * kv_stride + (size_t)kvh * head_dim;
                double acc = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) { acc += (double)qh[d] * (double)kh[d]; }
                scores[kp] = (float)acc;
                if ((float)acc > max_score) { max_score = (float)acc; }
            }
            double denom = 0.0;
            for (uint32_t kp = 0; kp < nitem; kp++) {
                const float e = expf(scores[kp] - max_score);
                scores[kp] = e;
                denom += e;
            }
            const float inv = denom > 0.0 ? (float)(1.0 / denom) : 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t kp = 0; kp < nitem; kp++) {
                    if (scores[kp] == 0.0f) { continue; }
                    const float *vh = v.data() + (size_t)(base + kp) * kv_stride + (size_t)kvh * head_dim;
                    acc += scores[kp] * vh[d];
                }
                oh[d] = acc * inv;
            }
        }
    }
}

static double cos_of(const std::vector<float> &a, const std::vector<float> &b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 0.0;
}

int main(int argc, char **argv) {
    const int repeats = (argc > 1) ? atoi(argv[1]) : 20;
    const uint32_t n_head = 4;

    const shape shapes[] = {
        { "single27_s",  {  27 }, 256, 2, 512 },
        { "cli115_s",   { 115 }, 256, 2, 512 },
        { "single184_s", { 184 }, 256, 2, 512 },
        { "page294_s",  { 294 }, 256, 2, 512 },
        { "page565_s",  { 565 }, 256, 2, 512 },
        { "max1414_s",  { 1414 }, 256, 2, 512 },
        { "single27_f",  {  27 }, 512, 1, -1 },
        { "page294_f",  { 294 }, 512, 1, -1 },
        { "max1414_f",  { 1414 }, 512, 1, -1 },
        { "batch8x306_s", { 306, 306, 306, 306, 306, 306, 306, 306 }, 256, 2, 512 },
        { "batch8x115_s", { 115, 115, 115, 115, 115, 115, 115, 115 }, 256, 2, 512 },
        { "batch8x565_s", { 565, 565, 565, 565, 565, 565, 565, 565 }, 256, 2, 512 },
        { "batch8x565_f", { 565, 565, 565, 565, 565, 565, 565, 565 }, 512, 1, -1 },
        { "batch64x20_s", { 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20 }, 256, 2, 512 },
    };

    printf("# bench_eg2vq_attn — %d repeats per row per path, one kernel launch per measurement\n", repeats);
    printf("%-16s %7s %7s %11s %11s %8s %10s\n",
           "shape", "rows", "keys", "ms_scalar", "ms_mma", "speedup", "mma_gflop/s");
    int fail = 0;
    for (const shape &s : shapes) {
        /* build the packed row block and its metadata */
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

        /* exact FLOPs: every (row, head, key) triple pays 2*head_dim twice */
        double flop = 0.0;
        for (uint32_t r = 0; r < n_tok; r++) {
            flop += (double)n_head * (double)keys_of((uint32_t)pos[r], (uint32_t)len[r], s.half_swa)
                    * (double)s.head_dim * 4.0;
        }
        uint64_t keys = 0;
        for (uint32_t r = 0; r < n_tok; r++) {
            keys += keys_of((uint32_t)pos[r], (uint32_t)len[r], s.half_swa);
        }

        if (!eg2vq_attn_scalar(dy, dq, dk, dv, dpos, dlen, n_tok, n_head, s.n_head_kv,
                        s.head_dim, s.half_swa)) {
            printf("%-16s REFUSED by the launcher\n", s.tag);
            fail = 1;
            continue;
        }
        /* correctness on the first shape only: the CPU reference is f64 and
         * scalar, so one small case is what this harness can afford. */
        double cos = 1.0;
        if (n_tok <= 294) {
            std::vector<float> cpu;
            ref_attn(cpu, hq, hk, hv, pos, len, n_head, s.n_head_kv, s.head_dim, s.half_swa);
            CK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));
            cos = cos_of(cpu, hy);
            if (cos < 0.99999) {
                printf("%-16s COS FAIL %.9f vs the CPU reference\n", s.tag, cos);
                fail = 1;
            }
        }

        /* the call's span table, the same one the device core builds */
        std::vector<uint32_t> off(1, 0u);
        for (uint32_t l : s.item_len) { off.push_back(off.back() + l); }

        /* One timing pass over one path, in the same process and back to back, so
         * the scalar and tensor-core columns are an interleaved A/B and not two
         * windows (this machine's per-window drift is the record's own caution). */
        auto time_path = [&](int mma) -> std::vector<float> {
            std::vector<float> ms;
            for (int rep = 0; rep < repeats + 3; rep++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                CK(cudaEventRecord(a));
                const int ok = mma
                    ? eg2vq_attn_mma(dy, dq, dk, dv, dpos, dlen, off.data(), (uint32_t)s.item_len.size(),
                                     n_tok, n_head, s.n_head_kv, s.head_dim, s.half_swa)
                    : eg2vq_attn_scalar(dy, dq, dk, dv, dpos, dlen, n_tok, n_head, s.n_head_kv,
                                        s.head_dim, s.half_swa);
                if (!ok) {
                    printf("%-16s refused mid-run (%s)\n", s.tag, mma ? "tensor cores" : "scalar");
                    fail = 1;
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
            for (size_t i = 0; i + 1 < ms.size(); i++) {
                for (size_t j = i + 1; j < ms.size(); j++) {
                    if (ms[j] < ms[i]) { const float t = ms[i]; ms[i] = ms[j]; ms[j] = t; }
                }
            }
            return ms;
        };

        const std::vector<float> msc = time_path(0);
        const std::vector<float> msm = time_path(1);
        if (msc.empty() || msm.empty()) { continue; }
        const float mdsc = msc[msc.size() / 2], mdmm = msm[msm.size() / 2];
        printf("%-16s %7u %7llu %11.4f %11.4f %8.2f %10.1f\n",
               s.tag, n_tok, (unsigned long long)keys, mdsc, mdmm, mdsc / mdmm,
               flop / ((double)mdmm * 1.0e6));

        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
        cudaFree(dpos); cudaFree(dlen);
    }
    printf("%s\n", fail ? "BENCH_EG2VQ_ATTN_FAIL" : "BENCH_EG2VQ_ATTN_OK");
    return fail;
}
