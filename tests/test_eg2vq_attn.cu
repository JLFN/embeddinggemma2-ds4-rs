// test_eg2vq_attn.cu — the gemma_vq device attention against the CPU reference's
// own arithmetic (embeddinggemma2-handoff.md PHASE 7, row P3.2).
//
// The reference is a transcription of eg2vq_attn_worker
// (ds4-rs/ds4_eg2vq_graph.inc:124-166), which is itself the gated oracle of this
// port (phase 4: 768 mean cosine 0.999996809 against the frozen anchor).  It is
// deliberately written the same way the graph writes it — row max, exp, sum,
// weighted sum, per (query, head) — so a disagreement is a disagreement about the
// algorithm, not about software architecture:
//
//   scale exactly 1.0f      (no 1/sqrt(head_dim))
//   SYMMETRIC mask          |q_pos - k_pos| <= half_swa; half_swa < 0 = no mask
//   no causal mask          every query sees both sides
//   kvh = h / (n_head/n_head_kv)
//   zero denominator        -> zero output
//
// Cases are the tower's own shapes: 4 query heads; head_dim 256 with 2 KV heads
// on the twenty sliding layers (half-window 512), head_dim 512 with 1 KV head on
// the four full ones (no mask); token counts from a query (26) to the longest
// frozen-set document (1414), which also exercises the mask (n_tok > 1025).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

#include "../cuda/eg2vq_attn.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static std::mt19937 rng(20261019u);

// The reference arithmetic, one (query, head) at a time.
static void ref_attn(std::vector<float> &out, const std::vector<float> &q, const std::vector<float> &k,
                     const std::vector<float> &v, uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv,
                     uint32_t head_dim, int32_t half_swa) {
    const uint64_t q_stride = (uint64_t)n_head * head_dim, kv_stride = (uint64_t)n_head_kv * head_dim;
    const uint32_t group = n_head / n_head_kv;
    out.assign((size_t)n_tok * q_stride, 0.0f);
    std::vector<float> scores(n_tok);
    for (uint32_t p = 0; p < n_tok; p++) {
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kvh = h / group;
            const float *qh = q.data() + (size_t)p * q_stride + (size_t)h * head_dim;
            float *oh = out.data() + (size_t)p * q_stride + (size_t)h * head_dim;
            float max_score = -INFINITY;
            for (uint32_t kp = 0; kp < n_tok; kp++) {
                if (half_swa >= 0) {
                    const int64_t diff = (int64_t)p - (int64_t)kp;
                    if (diff < -(int64_t)half_swa || diff > (int64_t)half_swa) { scores[kp] = -INFINITY; continue; }
                }
                const float *kh = k.data() + (size_t)kp * kv_stride + (size_t)kvh * head_dim;
                double acc = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) { acc += (double)qh[d] * (double)kh[d]; }
                scores[kp] = (float)acc;
                if ((float)acc > max_score) { max_score = (float)acc; }
            }
            double denom = 0.0;
            for (uint32_t kp = 0; kp < n_tok; kp++) {
                const float e = expf(scores[kp] - max_score);
                scores[kp] = e;
                denom += e;
            }
            const float inv = denom > 0.0 ? (float)(1.0 / denom) : 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t kp = 0; kp < n_tok; kp++) {
                    if (scores[kp] == 0.0f) { continue; }
                    const float *vh = v.data() + (size_t)kp * kv_stride + (size_t)kvh * head_dim;
                    acc += scores[kp] * vh[d];
                }
                oh[d] = acc * inv;
            }
        }
    }
}

struct a_case {
    uint32_t tokens, head_dim, n_head_kv;
    int32_t half_swa;
    const char *tag;
};

int main(void) {
    const a_case cases[] = {
        {   26, 256, 2,  512, "sliding: 26 tokens (a query)" },
        {  184, 256, 2,  512, "sliding: 184 tokens (mean document)" },
        {  513, 256, 2,  512, "sliding: 513 tokens (window equals item)" },
        {  514, 256, 2,  512, "sliding: 514 tokens (mask starts to bite)" },
        { 1414, 256, 2,  512, "sliding: 1414 tokens (longest frozen item)" },
        {  349, 512, 1,   -1, "full: 349 tokens, no mask" },
        { 1414, 512, 1,   -1, "full: 1414 tokens, no mask" },
    };
    const uint32_t n_head = 4;
    int fails = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const a_case &c : cases) {
        const uint64_t q_stride = (uint64_t)n_head * c.head_dim, kv_stride = (uint64_t)c.n_head_kv * c.head_dim;
        std::vector<float> q((size_t)c.tokens * q_stride), k((size_t)c.tokens * kv_stride), v((size_t)c.tokens * kv_stride);
        for (float &x : q) { x = nd(rng); }
        for (float &x : k) { x = nd(rng); }
        for (float &x : v) { x = nd(rng); }

        std::vector<float> yref;
        ref_attn(yref, q, k, v, c.tokens, n_head, c.n_head_kv, c.head_dim, c.half_swa);

        float *dq = NULL, *dk = NULL, *dv = NULL, *dy = NULL;
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, yref.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, yref.size() * 4));
        if (!eg2vq_attn(dy, dq, dk, dv, c.tokens, n_head, c.n_head_kv, c.head_dim, c.half_swa)) {
            printf("FAIL %s: the launcher refused the shape\n", c.tag);
            fails++;
            cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
            continue;
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> ydev(yref.size());
        CK(cudaMemcpy(ydev.data(), dy, ydev.size() * 4, cudaMemcpyDeviceToHost));

        double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0, min_cos = 1.0, scale = 0.0;
        for (size_t i = 0; i < yref.size(); i++) {
            const double d = (double)ydev[i] - (double)yref[i];
            if (fabs(d) > max_abs) { max_abs = fabs(d); }
            sum_sq += d * d;
            ref_sq += (double)yref[i] * (double)yref[i];
            if (fabs((double)yref[i]) > scale) { scale = fabs((double)yref[i]); }
        }
        // per (query, head) cosine, because a whole-tensor cosine hides a head
        std::vector<float> hv((size_t)c.head_dim), hr((size_t)c.head_dim);
        for (uint32_t p = 0; p < c.tokens; p++) {
            for (uint32_t h = 0; h < n_head; h++) {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (uint32_t d = 0; d < c.head_dim; d++) {
                    const double a = yref[(size_t)p * q_stride + (size_t)h * c.head_dim + d];
                    const double b = ydev[(size_t)p * q_stride + (size_t)h * c.head_dim + d];
                    dot += a * b; na += a * a; nb += b * b;
                }
                if (na > 0.0 && nb > 0.0) {
                    const double cos = dot / sqrt(na * nb);
                    if (cos < min_cos) { min_cos = cos; }
                }
            }
        }
        (void)hv; (void)hr;
        const double rel = ref_sq > 0.0 ? sqrt(sum_sq / ref_sq) : 0.0;
        const int bad = (rel > 1e-5 || min_cos < 1.0 - 1e-6);
        printf("%-44s tok %4u hd %3u kv %u half %5d | maxabs %.2e rel %.2e mincos %.9f | %s\n",
               c.tag, c.tokens, c.head_dim, c.n_head_kv, c.half_swa, max_abs, rel, min_cos,
               bad ? "OUT OF BOUND" : "ok");
        if (bad) { fails++; }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
    }

    /* The mask must actually cut: with one value row set far outside the window,
     * a query at the far end must not see it.  q is zero here on purpose, so
     * every in-window key gets the same weight (1/nkey) exactly and the test
     * cannot be defeated by a legitimately underflowed softmax weight on the one
     * row that carries a value. */
    {
        const uint32_t tok = 1200, hd = 256, nkv = 2, half = 512;
        const uint64_t q_stride = (uint64_t)n_head * hd, kv_stride = (uint64_t)nkv * hd;
        std::vector<float> q((size_t)tok * q_stride, 0.0f), k((size_t)tok * kv_stride, 0.0f), v((size_t)tok * kv_stride, 0.0f);
        for (float &x : k) { x = nd(rng); }
        v[(size_t)1100 * kv_stride] = 7.0f;          /* only row 1100 carries a value */
        std::vector<float> yref;
        ref_attn(yref, q, k, v, tok, n_head, nkv, hd, half);
        float *dq = NULL, *dk = NULL, *dv = NULL, *dy = NULL;
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, yref.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, yref.size() * 4));
        eg2vq_attn(dy, dq, dk, dv, tok, n_head, nkv, hd, half);
        CK(cudaDeviceSynchronize());
        std::vector<float> ydev(yref.size());
        CK(cudaMemcpy(ydev.data(), dy, ydev.size() * 4, cudaMemcpyDeviceToHost));
        /* Row 1100 is the only value row, so the whole test is the mask's edge:
         * query 588 is at distance exactly 512 (inside), query 587 at 513
         * (outside), and every other query's window must miss it too.  Both the
         * reference and the device must agree element by element, and the device
         * output for a blind query must be exactly zero, not merely close. */
        int edge_ok = 1, blind_ok = 1;
        const uint32_t inside[2] = {1100u, 588u}, outside[2] = {587u, 0u};
        for (uint32_t i = 0; i < 2; i++) {
            const size_t at = (size_t)inside[i] * q_stride;
            if (yref[at] == 0.0f || ydev[at] == 0.0f || fabsf(ydev[at] - yref[at]) > 1e-5f * fabsf(yref[at])) {
                printf("   inside query %u: ref %.6g dev %.6g\n", inside[i], (double)yref[at], (double)ydev[at]);
                edge_ok = 0;
            }
            const size_t bt = (size_t)outside[i] * q_stride;
            if (yref[bt] != 0.0f) {
                printf("   outside query %u: reference gives %.6g (should be 0)\n", outside[i], (double)yref[bt]);
                edge_ok = 0;
            }
            for (uint32_t h = 0; h < n_head && blind_ok; h++) {
                for (uint32_t d = 0; d < hd; d++) {
                    if (ydev[bt + (size_t)h * hd + d] != 0.0f) {
                        printf("   outside query %u head %u dim %u: device gives %.6g (should be 0)\n",
                               outside[i], h, d, (double)ydev[bt + (size_t)h * hd + d]);
                        blind_ok = 0;
                    }
                }
            }
        }
        printf("%-44s %s\n", "mask edge (512 sees / 513 blind)",
               (edge_ok && blind_ok) ? "distance 512 sees row 1100, 513 and beyond blind PASS" : "FAIL");
        if (!(edge_ok && blind_ok)) { fails++; }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
    }

    /* A shape the kernel does not cover must be refused, not half-done. */
    {
        const uint32_t tok = 4096, hd = 256, nkv = 2;
        std::vector<float> q((size_t)tok * n_head * hd, 0.0f), k((size_t)tok * nkv * hd, 0.0f), v((size_t)tok * nkv * hd, 0.0f);
        std::vector<float> y((size_t)tok * n_head * hd, 5.0f);
        float *dq = NULL, *dk = NULL, *dv = NULL, *dy = NULL;
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, y.size() * 4));
        CK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
        const int took = eg2vq_attn(dy, dq, dk, dv, tok, n_head, nkv, hd, 512);
        CK(cudaDeviceSynchronize());
        std::vector<float> back(64);
        CK(cudaMemcpy(back.data(), dy, back.size() * 4, cudaMemcpyDeviceToHost));
        int untouched = 1;
        for (float x : back) { if (x != 5.0f) { untouched = 0; } }
        printf("%-44s %s\n", "4096 tokens (beyond the plane)",
               (!took && untouched) ? "refused, output untouched PASS" : "FAIL");
        if (took || !untouched) { fails++; }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
    }

    printf(fails ? "EG2VQ_ATTN_FAIL (%d)\n" : "EG2VQ_ATTN_PASS\n", fails);
    return fails ? 1 : 0;
}
