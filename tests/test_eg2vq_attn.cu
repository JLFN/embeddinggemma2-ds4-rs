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
//
// Unit 27 adds the two batch cases at the end: the metadata upload (`pos`/`len`,
// which the device core derives from the item spans) is stated for one item by
// upload_one_item_meta, and a packed block of two very different items is
// compared against the two single-item runs.
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

// Unit 27: the batch's per-row metadata.  `pos[t]` is row t's position inside
// its item and `len[t]` that item's length -- the two arrays the device core
// derives from the item spans and the kernel reads for its key range.  This
// helper states the ONE-item case, which is what the pre-batch tests exercised:
// pos[t] = t and len[t] = n_tok.  The packed case below states its own.
struct attn_meta {
    int32_t *pos = NULL;
    int32_t *len = NULL;
};

static attn_meta upload_one_item_meta(uint32_t n_tok) {
    std::vector<int32_t> pos(n_tok), len(n_tok, (int32_t)n_tok);
    for (uint32_t t = 0; t < n_tok; t++) { pos[t] = (int32_t)t; }
    attn_meta m;
    CK(cudaMalloc(&m.pos, (size_t)n_tok * 4));
    CK(cudaMalloc(&m.len, (size_t)n_tok * 4));
    CK(cudaMemcpy(m.pos, pos.data(), (size_t)n_tok * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.len, len.data(), (size_t)n_tok * 4, cudaMemcpyHostToDevice));
    return m;
}

static void free_meta(attn_meta m) {
    if (m.pos) { cudaFree(m.pos); }
    if (m.len) { cudaFree(m.len); }
}

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
        const attn_meta meta = upload_one_item_meta(c.tokens);
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, yref.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, yref.size() * 4));
        if (!eg2vq_attn(dy, dq, dk, dv, meta.pos, meta.len, c.tokens, n_head, c.n_head_kv, c.head_dim, c.half_swa)) {
            printf("FAIL %s: the launcher refused the shape\n", c.tag);
            fails++;
            free_meta(meta);
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
        free_meta(meta);
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
        const attn_meta meta = upload_one_item_meta(tok);
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, yref.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, yref.size() * 4));
        eg2vq_attn(dy, dq, dk, dv, meta.pos, meta.len, tok, n_head, nkv, hd, half);
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
        free_meta(meta);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
    }

    /* ---- unit 27: two items packed into one block ---------------------------
     * Items of very different lengths (40 and 1,200 rows) in ONE call: each row
     * must read the rows of its OWN item only, so the packed run must equal the
     * two single-item runs element for element.  A mask over the whole block
     * would let the 1,200-row item's rows into the 40-row item's window (and a
     * position counted from the block's start would rotate the second item's
     * rows with the wrong table entries), so this is the batch's core claim at
     * the kernel level. */
    {
        const uint32_t hd = 256, nkv = 2, half = 512;
        const uint32_t lens[2] = { 40u, 1200u };
        const uint32_t tok = lens[0] + lens[1];
        const uint64_t q_stride = (uint64_t)n_head * hd, kv_stride = (uint64_t)nkv * hd;
        std::vector<float> q((size_t)tok * q_stride), k((size_t)tok * kv_stride), v((size_t)tok * kv_stride);
        for (float &x : q) { x = nd(rng); }
        for (float &x : k) { x = nd(rng); }
        for (float &x : v) { x = nd(rng); }

        std::vector<float> yref((size_t)tok * q_stride, 0.0f);
        for (uint32_t i = 0, r0 = 0; i < 2; i++) {
            const uint32_t len = lens[i];
            std::vector<float> qq(q.begin() + (size_t)r0 * q_stride,
                                  q.begin() + (size_t)(r0 + len) * q_stride);
            std::vector<float> kk(k.begin() + (size_t)r0 * kv_stride,
                                  k.begin() + (size_t)(r0 + len) * kv_stride);
            std::vector<float> vv(v.begin() + (size_t)r0 * kv_stride,
                                  v.begin() + (size_t)(r0 + len) * kv_stride);
            std::vector<float> yy;
            ref_attn(yy, qq, kk, vv, len, n_head, nkv, hd, half);
            for (size_t x = 0; x < yy.size(); x++) { yref[(size_t)r0 * q_stride + x] = yy[x]; }
            r0 += len;
        }

        std::vector<int32_t> pos(tok), lenv(tok);
        for (uint32_t i = 0, r0 = 0; i < 2; i++) {
            for (uint32_t t = r0; t < r0 + lens[i]; t++) {
                pos[t] = (int32_t)(t - r0);
                lenv[t] = (int32_t)lens[i];
            }
            r0 += lens[i];
        }
        float *dq = NULL, *dk = NULL, *dv = NULL, *dy = NULL;
        int32_t *dpos = NULL, *dlen = NULL;
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, yref.size() * 4));
        CK(cudaMalloc(&dpos, (size_t)tok * 4)); CK(cudaMalloc(&dlen, (size_t)tok * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), (size_t)tok * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dlen, lenv.data(), (size_t)tok * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, yref.size() * 4));
        int packed_ok = 1;
        /* the plane is the item ceiling, so this case's own bound is the item
         * lengths being different, not a cap the caller passes. */
        if (!eg2vq_attn(dy, dq, dk, dv, dpos, dlen, tok, n_head, nkv, hd, half)) {
            printf("FAIL: the launcher refused the packed block\n");
            fails++;
            packed_ok = 0;
        }
        if (packed_ok) {
            CK(cudaDeviceSynchronize());
            std::vector<float> ydev(yref.size());
            CK(cudaMemcpy(ydev.data(), dy, ydev.size() * 4, cudaMemcpyDeviceToHost));
            /* the same two bounds every case above is held to (this kernel's
             * four-way f32 accumulation against the reference's f64 scalar) */
            double sum_sq = 0.0, ref_sq = 0.0, min_cos = 1.0;
            for (size_t i = 0; i < yref.size(); i++) {
                const double d = (double)ydev[i] - (double)yref[i];
                sum_sq += d * d;
                ref_sq += (double)yref[i] * (double)yref[i];
            }
            for (uint32_t r = 0; r < tok; r++) {
                for (uint32_t h = 0; h < n_head; h++) {
                    double dot = 0.0, na = 0.0, nb = 0.0;
                    for (uint32_t d = 0; d < hd; d++) {
                        const double a = yref[(size_t)r * q_stride + (size_t)h * hd + d];
                        const double b = ydev[(size_t)r * q_stride + (size_t)h * hd + d];
                        dot += a * b; na += a * a; nb += b * b;
                    }
                    if (na > 0.0 && nb > 0.0) {
                        const double cos = dot / sqrt(na * nb);
                        if (cos < min_cos) { min_cos = cos; }
                    }
                }
            }
            const double rel = ref_sq > 0.0 ? sqrt(sum_sq / ref_sq) : 0.0;
            printf("%-44s rel %.2e mincos %.9f | %s\n", "two items packed (40 + 1200 rows)",
                   rel, min_cos,
                   (rel <= 1e-5 && min_cos >= 1.0 - 1e-6) ? "each row reads its own item PASS" : "FAIL");
            if (!(rel <= 1e-5 && min_cos >= 1.0 - 1e-6)) { fails++; }
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
        cudaFree(dpos); cudaFree(dlen);
    }

    /* A shape the kernel does not cover must be refused, not half-done.  The ROW
     * COUNT is deliberately no longer a refusal here: a batch's block is as long
     * as its items add up to, and what must stay inside the key plane is each
     * ITEM's length -- the device core checks that before it launches (unit 27),
     * and tools/eg2_embed_batch_gate27.sh exercises it with a 2049-token item.
     * The head geometry is still the launcher's own to refuse. */
    {
        const uint32_t tok = 64, hd = 256, nkv = 2;
        std::vector<float> q((size_t)tok * n_head * hd, 0.0f), k((size_t)tok * nkv * hd, 0.0f), v((size_t)tok * nkv * hd, 0.0f);
        std::vector<float> y((size_t)tok * n_head * hd, 5.0f);
        float *dq = NULL, *dk = NULL, *dv = NULL, *dy = NULL;
        const attn_meta meta = upload_one_item_meta(tok);
        CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4)); CK(cudaMalloc(&dy, y.size() * 4));
        CK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
        const int took = eg2vq_attn(dy, dq, dk, dv, meta.pos, meta.len, tok, n_head + 1, nkv, hd, 512);
        CK(cudaDeviceSynchronize());
        std::vector<float> back(64);
        CK(cudaMemcpy(back.data(), dy, back.size() * 4, cudaMemcpyDeviceToHost));
        int untouched = 1;
        for (float x : back) { if (x != 5.0f) { untouched = 0; } }
        printf("%-44s %s\n", "5 query heads (not the tower's 4)",
               (!took && untouched) ? "refused, output untouched PASS" : "FAIL");
        if (took || !untouched) { fails++; }
        free_meta(meta);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
    }

    printf(fails ? "EG2VQ_ATTN_FAIL (%d)\n" : "EG2VQ_ATTN_PASS\n", fails);
    return fails ? 1 : 0;
}
