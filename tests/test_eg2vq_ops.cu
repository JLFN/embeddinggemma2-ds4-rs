// test_eg2vq_ops.cu — the gemma_vq tower's small device ops against the CPU
// reference's own arithmetic (embeddinggemma2-handoff.md PHASE 7, row P3.3 step 1).
//
// Each case is a transcription of the graph expression the kernel mirrors, at the
// tower's own widths (ds4-rs/ds4_eg2vq_graph.inc; the citations are in the kernel
// header).  The RoPE tables are built the way the graph accumulates its angles —
// theta starts at the position and multiplies by theta_scale in f32, once per
// pair (graph:60-77) — with the sine and cosine taken in double precision, which
// is the one thing the device-side table deviates from the graph's cosf/sinf on
// purpose and is stated in the kernel header.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

#include "../cuda/eg2vq_ops.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static std::mt19937 rng(20261019u);
static const float EG2VQ_EPS = 1e-6f;      /* attention.layer_norm_rms_epsilon */

static uint16_t h16(float v) { __half h = __float2half_rn(v); uint16_t u; memcpy(&u, &h, 2); return u; }
static float f16v(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

// graph:47-53
static void ref_rms(std::vector<float> &out, const std::vector<float> &x, const std::vector<float> *w,
                    uint32_t rows, uint32_t n) {
    out.assign((size_t)rows * n, 0.0f);
    for (uint32_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) { ss += (double)x[(size_t)r * n + i] * (double)x[(size_t)r * n + i]; }
        const float scale = 1.0f / sqrtf((float)(ss / (double)n) + EG2VQ_EPS);
        for (uint32_t i = 0; i < n; i++) {
            out[(size_t)r * n + i] = x[(size_t)r * n + i] * scale * (w ? (*w)[i] : 1.0f);
        }
    }
}

// graph:264-277 (the per-head norms, in place)
static void ref_rms_head(std::vector<float> &x, const std::vector<float> *w, uint32_t rows,
                         uint32_t heads, uint32_t hd) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t h = 0; h < heads; h++) {
            float *v = &x[((size_t)r * heads + h) * hd];
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) { ss += (double)v[i] * (double)v[i]; }
            const float scale = 1.0f / sqrtf((float)(ss / (double)hd) + EG2VQ_EPS);
            for (uint32_t i = 0; i < hd; i++) { v[i] = v[i] * scale * (w ? (*w)[i] : 1.0f); }
        }
    }
}

// graph:339-367: the output_norm RMS and the bf16 projection per row, summed
// over the rows of ONE item, then that item's mean and L2 normalization --
// item-relative because the device pool is (unit 27, cuda/eg2vq_ops.cuh's
// eg2vq_ops_pool_item_kernel), so `want` is built the same way for a single item
// and for one item of a packed batch.
static void ref_proj_pool(const std::vector<float> &h, uint32_t r0, uint32_t r1,
                          const std::vector<uint16_t> &w, const std::vector<float> &nw,
                          uint32_t n_embd, uint32_t n_out, std::vector<float> &want) {
    want.assign(n_out, 0.0f);
    for (uint32_t t = r0; t < r1; t++) {
        std::vector<float> e(n_embd);
        double ss = 0.0;
        for (uint32_t i = 0; i < n_embd; i++) {
            ss += (double)h[(size_t)t * n_embd + i] * (double)h[(size_t)t * n_embd + i];
        }
        const float scale = 1.0f / sqrtf((float)(ss / (double)n_embd) + EG2VQ_EPS);
        for (uint32_t i = 0; i < n_embd; i++) { e[i] = h[(size_t)t * n_embd + i] * scale * nw[i]; }
        for (uint32_t o = 0; o < n_out; o++) {
            float a = 0.0f;
            for (uint32_t i = 0; i < n_embd; i++) {
                const uint32_t u = (uint32_t)w[(size_t)o * n_embd + i] << 16;
                float f;
                memcpy(&f, &u, 4);
                a += f * e[i];
            }
            want[o] += a;
        }
    }
    const float mean = 1.0f / (float)(r1 - r0);
    for (float &v : want) { v *= mean; }
    double ss = 0.0;
    for (float v : want) { ss += (double)v * (double)v; }
    const double norm = sqrt(ss);
    const float inv = norm > 0.0 ? (float)(1.0 / norm) : 0.0f;
    for (float &v : want) { v *= inv; }
}

// graph:60-77, with the angle accumulation the graph itself does
static void rope_tables(std::vector<float> &cos_tab, std::vector<float> &sin_tab,
                        uint32_t n_pos, uint32_t hd, float base) {
    const uint32_t half = hd / 2u;
    const float theta_scale = powf(base, -2.0f / (float)hd);
    cos_tab.assign((size_t)n_pos * half, 0.0f);
    sin_tab.assign((size_t)n_pos * half, 0.0f);
    for (uint32_t p = 0; p < n_pos; p++) {
        float theta = (float)p;
        for (uint32_t i = 0; i < half; i++) {
            cos_tab[(size_t)p * half + i] = (float)cos((double)theta);
            sin_tab[(size_t)p * half + i] = (float)sin((double)theta);
            theta *= theta_scale;
        }
    }
}

// Rotate `rows` rows starting at r0, reading the table at positions p0..p0+rows
// (unit 27: the device kernel indexes the table by each row's position INSIDE its
// item, and one item is the p0 == 0, r0 == 0 case).
static void ref_rope_at(std::vector<float> &x, uint32_t r0, uint32_t rows, uint32_t heads, uint32_t hd,
                        const std::vector<float> &cos_tab, const std::vector<float> &sin_tab,
                        uint32_t p0) {
    const uint32_t half = hd / 2u;
    for (uint32_t r = r0; r < r0 + rows; r++) {
        const uint32_t p = p0 + (r - r0);
        for (uint32_t h = 0; h < heads; h++) {
            float *v = &x[((size_t)r * heads + h) * hd];
            for (uint32_t i = 0; i < half; i++) {
                const float c = cos_tab[(size_t)p * half + i], s = sin_tab[(size_t)p * half + i];
                const float x0 = v[i], x1 = v[i + half];
                v[i] = x0 * c - x1 * s;
                v[i + half] = x0 * s + x1 * c;
            }
        }
    }
}

static void ref_rope(std::vector<float> &x, uint32_t rows, uint32_t heads, uint32_t hd,
                     const std::vector<float> &cos_tab, const std::vector<float> &sin_tab) {
    ref_rope_at(x, 0, rows, heads, hd, cos_tab, sin_tab, 0);
}

static double rel_rms(const std::vector<float> &a, const std::vector<float> &b, double *maxabs) {
    double ss = 0.0, rr = 0.0, ma = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
        rr += (double)b[i] * (double)b[i];
        if (fabs(d) > ma) { ma = fabs(d); }
    }
    if (maxabs) { *maxabs = ma; }
    return rr > 0.0 ? sqrt(ss / rr) : 0.0;
}

static int report(const char *tag, double rel, double maxabs, double bound) {
    const int bad = rel > bound;
    printf("%-40s rel RMS %.3e   max abs %.3e   bound %.0e   %s\n", tag, rel, maxabs, bound,
           bad ? "OUT OF BOUND" : "ok");
    return bad ? 1 : 0;
}

int main(void) {
    int fails = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const uint32_t n_embd = 512, n_head = 4;

    /* ---- row RMS, with and without a weight (the 512-wide hidden) ---------- */
    {
        const uint32_t rows = 184;
        std::vector<float> x((size_t)rows * n_embd), w(n_embd);
        for (float &v : x) { v = nd(rng); }
        for (float &v : w) { v = 1.0f + 0.1f * nd(rng); }
        std::vector<float> want;
        ref_rms(want, x, &w, rows, n_embd);
        float *dx = NULL, *dw = NULL, *dy = NULL;
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dw, w.size() * 4)); CK(cudaMalloc(&dy, want.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        eg2vq_ops_rms(dy, dx, dw, rows, n_embd, EG2VQ_EPS);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(want.size());
        CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
        double ma; const double r1 = rel_rms(got, want, &ma); fails += report("rms rows, weighted (512 wide)", r1, ma, 1e-6);
        // weightless (the V norm's shape)
        ref_rms(want, x, NULL, rows, n_embd);
        eg2vq_ops_rms(dy, dx, NULL, rows, n_embd, EG2VQ_EPS);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
        const double r2 = rel_rms(got, want, &ma); fails += report("rms rows, weightless", r2, ma, 1e-6);
        cudaFree(dx); cudaFree(dw); cudaFree(dy);
    }

    /* ---- per-head RMS at both head widths, weighted (q/k) and not (v) ------- */
    for (uint32_t hd : {256u, 512u}) {
        const uint32_t rows = 73;
        std::vector<float> x((size_t)rows * n_head * hd), w(hd);
        for (float &v : x) { v = nd(rng); }
        for (float &v : w) { v = 1.0f + 0.1f * nd(rng); }
        std::vector<float> want = x;
        ref_rms_head(want, &w, rows, n_head, hd);
        float *dx = NULL, *dw = NULL;
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dw, w.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        eg2vq_ops_rms_head(dx, dw, rows, n_head, hd, EG2VQ_EPS);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(x.size());
        CK(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
        char tag[80];
        snprintf(tag, sizeof(tag), "rms per head, weighted (hd %u)", hd);
        double ma; const double r3 = rel_rms(got, want, &ma); fails += report(tag, r3, ma, 1e-6);
        want = x;
        ref_rms_head(want, NULL, rows, n_head, hd);
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        eg2vq_ops_rms_head(dx, NULL, rows, n_head, hd, EG2VQ_EPS);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
        snprintf(tag, sizeof(tag), "rms per head, weightless (hd %u)", hd);
        const double r4 = rel_rms(got, want, &ma); fails += report(tag, r4, ma, 1e-6);
        cudaFree(dx); cudaFree(dw);
    }

    /* ---- RoPE at both bases, over a full item (the mask layers' base 1e4 and
     *      the full layers' 1e6, ds4_eg2vq.h:66-67) ------------------------- */
    {
        const uint32_t rows = 1414, hd = 256;
        std::vector<float> cos_tab, sin_tab;
        std::vector<float> x((size_t)rows * n_head * hd);
        for (float &v : x) { v = nd(rng); }
        float *dx = NULL, *dc = NULL, *ds = NULL;
        int32_t *dpos = NULL;
        CK(cudaMalloc(&dx, x.size() * 4));
        for (float base : {1.0e4f, 1.0e6f}) {
            rope_tables(cos_tab, sin_tab, rows, hd, base);
            std::vector<float> want = x;
            ref_rope(want, rows, n_head, hd, cos_tab, sin_tab);
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            if (!dc) { CK(cudaMalloc(&dc, cos_tab.size() * 4)); }
            if (!ds) { CK(cudaMalloc(&ds, sin_tab.size() * 4)); }
            CK(cudaMemcpy(dc, cos_tab.data(), cos_tab.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ds, sin_tab.data(), sin_tab.size() * 4, cudaMemcpyHostToDevice));
            /* one item: a row's position IS its row index (unit 27's `pos`) */
            std::vector<int32_t> pos(rows);
            for (uint32_t t = 0; t < rows; t++) { pos[t] = (int32_t)t; }
            if (!dpos) { CK(cudaMalloc(&dpos, (size_t)rows * 4)); }
            CK(cudaMemcpy(dpos, pos.data(), (size_t)rows * 4, cudaMemcpyHostToDevice));
            eg2vq_ops_rope(dx, dc, ds, dpos, rows, n_head, hd);
            CK(cudaDeviceSynchronize());
            std::vector<float> got(x.size());
            CK(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
            char tag[80];
            snprintf(tag, sizeof(tag), "rope 1414x4x256, base %.0e", (double)base);
            double ma; const double r3 = rel_rms(got, want, &ma); fails += report(tag, r3, ma, 1e-6);
        }
        cudaFree(dx); cudaFree(dc); cudaFree(ds); cudaFree(dpos);
    }

    /* ---- unit 27: RoPE's position is per ITEM -------------------------------
     * Two items (40 and 600 rows) packed into one block: the second item's rows
     * must be rotated with the table entries of THEIR own positions 0..599, not
     * with the block's row indices 40..639, which is what a batch call's `pos`
     * carries.  The reference is the same table read twice, once per item. */
    {
        const uint32_t hd = 256, half = 40, other = 600;
        const uint32_t rows = half + other;
        std::vector<float> cos_tab, sin_tab, x((size_t)rows * n_head * hd);
        for (float &v : x) { v = nd(rng); }
        rope_tables(cos_tab, sin_tab, rows, hd, 1.0e4f);
        std::vector<float> want = x;
        ref_rope_at(want, 0, half, n_head, hd, cos_tab, sin_tab, 0);
        ref_rope_at(want, half, other, n_head, hd, cos_tab, sin_tab, 0);
        std::vector<int32_t> pos(rows);
        for (uint32_t t = 0; t < rows; t++) {
            pos[t] = (int32_t)(t < half ? t : t - half);
        }
        float *dx = NULL, *dc = NULL, *ds = NULL;
        int32_t *dpos = NULL;
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dc, cos_tab.size() * 4));
        CK(cudaMalloc(&ds, sin_tab.size() * 4)); CK(cudaMalloc(&dpos, (size_t)rows * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dc, cos_tab.data(), cos_tab.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds, sin_tab.data(), sin_tab.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), (size_t)rows * 4, cudaMemcpyHostToDevice));
        eg2vq_ops_rope(dx, dc, ds, dpos, rows, n_head, hd);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(x.size());
        CK(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
        // exact comparison against the per-item reference: the table entry is the
        // only input, so a different `pos` shows up immediately
        double ma = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            const double d = fabs((double)got[i] - (double)want[i]);
            if (d > ma) { ma = d; }
        }
        printf("%-44s maxabs %.2e | %s\n", "rope pos restarts per item",
               ma, ma == 0.0 ? "ok" : "OUT OF BOUND");
        if (ma != 0.0) { fails++; }
        cudaFree(dx); cudaFree(dc); cudaFree(ds); cudaFree(dpos);
    }

    /* ---- gelu(gate) * up ---------------------------------------------------- */
    {
        const uint32_t rows = 184, nf = 2048;
        std::vector<float> g((size_t)rows * nf), u((size_t)rows * nf), want;
        for (float &v : g) { v = nd(rng); }
        for (float &v : u) { v = nd(rng); }
        want.resize(g.size());
        for (size_t i = 0; i < g.size(); i++) {
            const float x = g[i];
            want[i] = 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x))) * u[i];
        }
        float *dg = NULL, *du = NULL;
        CK(cudaMalloc(&dg, g.size() * 4)); CK(cudaMalloc(&du, u.size() * 4));
        CK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(du, u.data(), u.size() * 4, cudaMemcpyHostToDevice));
        eg2vq_ops_gelu_mul(dg, du, g.size());
        CK(cudaDeviceSynchronize());
        std::vector<float> got(g.size());
        CK(cudaMemcpy(got.data(), dg, got.size() * 4, cudaMemcpyDeviceToHost));
        double ma; const double r5 = rel_rms(got, want, &ma); fails += report("gelu(gate) * up", r5, ma, 1e-5);
        cudaFree(dg); cudaFree(du);
    }

    /* ---- the token embedding: a q8_0 table row, scaled by sqrt(n_embd) ------ */
    {
        const uint32_t vocab = 64, n_ids = 5;
        const uint32_t nb = n_embd / 32u;
        std::vector<uint8_t> table((size_t)vocab * nb * 34u);
        for (uint32_t r = 0; r < vocab; r++) {
            for (uint32_t b = 0; b < nb; b++) {
                uint8_t *blk = &table[((size_t)r * nb + b) * 34u];
                const uint16_t d = h16(0.005f + 0.001f * (float)(rng() % 97u));
                memcpy(blk, &d, 2);
                for (uint32_t i = 0; i < 32u; i++) { blk[2 + i] = (uint8_t)(int8_t)((int)(rng() % 255u) - 127); }
            }
        }
        std::vector<int32_t> ids = {3, 17, 0, 63, 41};
        std::vector<float> want((size_t)n_ids * n_embd);
        for (uint32_t t = 0; t < n_ids; t++) {
            const uint8_t *src = &table[(size_t)(uint32_t)ids[t] * nb * 34u];
            for (uint32_t b = 0; b < nb; b++) {
                const uint8_t *blk = src + (size_t)b * 34u;
                uint16_t d; memcpy(&d, blk, 2);
                const float ds = f16v(d) * sqrtf((float)n_embd);
                for (uint32_t i = 0; i < 32u; i++) {
                    want[(size_t)t * n_embd + b * 32u + i] = ds * (float)(int8_t)blk[2 + i];
                }
            }
        }
        float *dy = NULL;
        uint8_t *dt = NULL;
        int32_t *di = NULL;
        CK(cudaMalloc(&dy, want.size() * 4));
        CK(cudaMalloc(&dt, table.size()));
        CK(cudaMalloc(&di, ids.size() * 4));
        CK(cudaMemcpy(dt, table.data(), table.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(di, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        eg2vq_ops_embed(dy, di, dt, n_ids, n_embd, vocab);
        CK(cudaDeviceSynchronize());
        std::vector<float> got(want.size());
        CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
        double ma; const double r6 = rel_rms(got, want, &ma); fails += report("q8_0 token row x sqrt(512)", r6, ma, 1e-6);
        cudaFree(dy); cudaFree(dt); cudaFree(di);
    }

    /* ---- 768 projection + mean pool + L2 ----------------------------------- */
    {
        const uint32_t n_tok = 349, n_out = 768;
        std::vector<float> h((size_t)n_tok * n_embd), nw(n_embd);
        // The artifact's embedding_projection is BF16 [n_out][n_embd]; the device
        // widens it with the same shift the CPU binder uses (u << 16), so the test
        // stores real bf16 patterns and the reference widens them identically.
        std::vector<uint16_t> w((size_t)n_out * n_embd);
        for (float &v : h) { v = nd(rng); }
        for (float &v : nw) { v = 1.0f + 0.1f * nd(rng); }
        for (uint16_t &v : w) {
            __nv_bfloat16 b = __float2bfloat16(0.02f * nd(rng));
            memcpy(&v, &b, 2);
        }
        auto bf16v = [](uint16_t u) { const uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; };
        (void)bf16v;
        std::vector<float> want;
        ref_proj_pool(h, 0, n_tok, w, nw, n_embd, n_out, want);

        float *dh = NULL, *dnw = NULL, *dy = NULL, *dacc = NULL;
        uint16_t *dw = NULL;
        uint32_t *doff = NULL;
        const uint32_t off_one[2] = { 0u, n_tok };
        CK(cudaMalloc(&dh, h.size() * 4)); CK(cudaMalloc(&dnw, nw.size() * 4));
        CK(cudaMalloc(&dy, n_out * 4)); CK(cudaMalloc(&dw, w.size() * 2));
        CK(cudaMalloc(&dacc, (size_t)n_tok * n_out * 4));
        CK(cudaMalloc(&doff, sizeof(off_one)));
        CK(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnw, nw.data(), nw.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(doff, off_one, sizeof(off_one), cudaMemcpyHostToDevice));
        if (!eg2vq_ops_proj768_pool_batch(dy, dacc, dh, dw, dnw, doff, 1u, n_tok, n_embd, n_out, EG2VQ_EPS)) {
            printf("FAIL: the 768 projection launcher refused the shape\n");
            fails++;
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> got(n_out);
        CK(cudaMemcpy(got.data(), dy, n_out * 4, cudaMemcpyDeviceToHost));
        double ma;
        const double rel = rel_rms(got, want, &ma);
        fails += report("proj768 + mean pool + L2", rel, ma, 1e-5);
        // and the pooled vector must be unit length
        double ns = 0.0;
        for (float v : got) { ns += (double)v * (double)v; }
        printf("%-40s |v| = %.9f   %s\n", "pooled vector norm", sqrt(ns),
               fabs(sqrt(ns) - 1.0) <= 1e-6 ? "ok" : "OUT OF BOUND");
        if (fabs(sqrt(ns) - 1.0) > 1e-6) { fails++; }
        cudaFree(dh); cudaFree(dnw); cudaFree(dy); cudaFree(dw);
        cudaFree(dacc); cudaFree(doff);
    }

    /* ---- unit 27: the pool is per ITEM -------------------------------------
     * Three items of very different lengths (1, 349 and 40 rows) packed into one
     * block: the same kernel pair must return each item's OWN mean/L2 vector, so
     * every item is compared against the reference run for it alone.  A pool that
     * averaged the block, or that let one item's rows into another's, cannot
     * pass this. */
    {
        const uint32_t n_out = 768;
        const uint32_t lens[3] = { 1u, 349u, 40u };
        const uint32_t n_tok = lens[0] + lens[1] + lens[2];
        std::vector<float> h((size_t)n_tok * n_embd), nw(n_embd);
        std::vector<uint16_t> w((size_t)n_out * n_embd);
        for (float &v : h) { v = nd(rng); }
        for (float &v : nw) { v = 1.0f + 0.1f * nd(rng); }
        for (uint16_t &v : w) {
            __nv_bfloat16 b = __float2bfloat16(0.02f * nd(rng));
            memcpy(&v, &b, 2);
        }
        uint32_t off_h[4] = { 0u, 0u, 0u, 0u };
        for (uint32_t i = 0; i < 3; i++) { off_h[i + 1] = off_h[i] + lens[i]; }

        float *dh = NULL, *dnw = NULL, *dy = NULL, *dacc = NULL;
        uint16_t *dw = NULL;
        uint32_t *doff = NULL;
        CK(cudaMalloc(&dh, h.size() * 4)); CK(cudaMalloc(&dnw, nw.size() * 4));
        CK(cudaMalloc(&dy, (size_t)3 * n_out * 4)); CK(cudaMalloc(&dw, w.size() * 2));
        CK(cudaMalloc(&dacc, (size_t)n_tok * n_out * 4));
        CK(cudaMalloc(&doff, sizeof(off_h)));
        CK(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnw, nw.data(), nw.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(doff, off_h, sizeof(off_h), cudaMemcpyHostToDevice));
        int ok = 1;
        if (!eg2vq_ops_proj768_pool_batch(dy, dacc, dh, dw, dnw, doff, 3u, n_tok, n_embd, n_out, EG2VQ_EPS)) {
            printf("FAIL: the 3-item pool launcher refused the shape\n");
            fails++;
            ok = 0;
        }
        if (ok) {
            CK(cudaDeviceSynchronize());
            std::vector<float> got((size_t)3 * n_out);
            CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0.0;
            for (uint32_t i = 0; i < 3; i++) {
                std::vector<float> want;
                ref_proj_pool(h, off_h[i], off_h[i + 1], w, nw, n_embd, n_out, want);
                double ma;
                const double rel = rel_rms(std::vector<float>(got.begin() + (size_t)i * n_out,
                                                              got.begin() + (size_t)(i + 1) * n_out),
                                           want, &ma);
                if (rel > worst) { worst = rel; }
            }
            printf("%-44s worst rel_rms %.2e | %s\n", "3 items packed (1/349/40 rows)",
                   worst, worst <= 1e-5 ? "ok" : "OUT OF BOUND");
            if (!(worst <= 1e-5)) { fails++; }
        }
        cudaFree(dh); cudaFree(dnw); cudaFree(dy); cudaFree(dw);
        cudaFree(dacc); cudaFree(doff);
    }

    /* ---- the IEEE half widening (P3.3 step 2) ------------------------------- */
    // Every one of the 65,536 patterns against the format's definition: a
    // subnormal is m * 2^-24, a normal 2^(e-15) * (1 + m/1024).  Both the
    // adapter's host-side widening and the device's __half2float are checked:
    // the two must agree with the format and with each other, because the CPU
    // path's vendored decoder halves subnormals (ds41_vq_fmt.h:38-43) and the
    // device deliberately follows the file instead.
    {
        uint32_t bad_host = 0, bad_dev = 0, nan_inf = 0;
        double worst = 0.0;
        for (uint32_t u = 0; u < 65536u; u++) {
            const uint32_t e = (u >> 10) & 31u, m = u & 1023u;
            if (e == 31u) { nan_inf++; continue; }
            const double mag = (e == 0u)
                ? (double)m * pow(2.0, -24.0)
                : pow(2.0, (double)e - 15.0) * (1.0 + (double)m / 1024.0);
            const double want = (u & 0x8000u) ? -mag : mag;
            const float host = eg2vq_ops_half_to_f32((uint16_t)u);
            const float dev = __half2float(__ushort_as_half((unsigned short)u));
            // Both values are exactly representable in f32, so this is bit
            // equality -- and bit equality is also what catches -0.0.
            const float wf = (float)want;
            uint32_t hb, db, wb;
            memcpy(&hb, &host, 4); memcpy(&db, &dev, 4); memcpy(&wb, &wf, 4);
            if (hb != wb) { bad_host++; const double d = fabs((double)host - want); if (d > worst) { worst = d; } }
            if (db != wb) { bad_dev++; }
        }
        printf("%-40s %u/65536 exact host, %u exact device (%u nan/inf skipped), worst |diff| %.3e   %s\n",
               "half -> f32 (IEEE, all patterns)", 65536u - nan_inf - bad_host, 65536u - nan_inf - bad_dev,
               nan_inf, worst, (bad_host || bad_dev) ? "OUT OF BOUND" : "ok");
        if (bad_host || bad_dev) { fails++; }
    }

    /* ---- whole-row scale and add (P3.3 step 2) ------------------------------ */
    {
        const uint64_t sizes[3] = {(uint64_t)184 * 512u, (uint64_t)1414 * 512u, (uint64_t)768};
        for (uint64_t n : sizes) {
            std::vector<float> x((size_t)n), want((size_t)n);
            for (float &v : x) { v = nd(rng); }
            want = x;
            const float s = 0.013f;                         // graph:335, a layer scalar's magnitude
            for (float &v : want) { v *= s; }
            float *dx = NULL;
            CK(cudaMalloc(&dx, x.size() * 4));
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            if (!eg2vq_ops_scale(dx, s, n)) { printf("FAIL: the scale launcher refused n=%llu\n", (unsigned long long)n); fails++; }
            CK(cudaDeviceSynchronize());
            std::vector<float> got(x.size());
            CK(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
            char tag[64];
            snprintf(tag, sizeof(tag), "scale x*%.3f, n=%llu", (double)s, (unsigned long long)n);
            double ma; const double r = rel_rms(got, want, &ma); fails += report(tag, r, ma, 1e-7);
            // and the add of the same shape
            std::vector<float> dst = x;
            for (float &v : dst) { v = nd(rng); }
            std::vector<float> want2 = dst;
            for (size_t i = 0; i < want2.size(); i++) { want2[i] += x[i]; }
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            float *dd = NULL;
            CK(cudaMalloc(&dd, dst.size() * 4));
            CK(cudaMemcpy(dd, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice));
            if (!eg2vq_ops_add(dd, dx, n)) { printf("FAIL: the add launcher refused n=%llu\n", (unsigned long long)n); fails++; }
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got.data(), dd, got.size() * 4, cudaMemcpyDeviceToHost));
            snprintf(tag, sizeof(tag), "add dst+=src, n=%llu", (unsigned long long)n);
            const double r2 = rel_rms(got, want2, &ma); fails += report(tag, r2, ma, 1e-7);
            cudaFree(dd);
            cudaFree(dx);
        }
    }

    /* ---- the PLE gate: gelu(gate) * a 512-slice of the per-layer input ------ */
    {
        const uint32_t n_tok = 184, pl = 512, n_layer = 24, il = 7;
        std::vector<float> g((size_t)n_tok * pl), ple((size_t)n_tok * pl * n_layer), want;
        for (float &v : g) { v = nd(rng); }
        for (float &v : ple) { v = nd(rng); }
        want.resize(g.size());
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t j = 0; j < pl; j++) {
                const float x = g[(size_t)t * pl + j];
                const float gl = 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
                want[(size_t)t * pl + j] = gl * ple[((size_t)t * n_layer + il) * pl + j];
            }
        }
        float *dg = NULL, *dp = NULL;
        CK(cudaMalloc(&dg, g.size() * 4)); CK(cudaMalloc(&dp, ple.size() * 4));
        CK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp, ple.data(), ple.size() * 4, cudaMemcpyHostToDevice));
        if (!eg2vq_ops_gelu_mul_strided(dg, dp, n_tok, pl, (uint64_t)pl * n_layer, (uint64_t)il * pl)) {
            printf("FAIL: the strided gelu-mul launcher refused the shape\n");
            fails++;
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> got(g.size());
        CK(cudaMemcpy(got.data(), dg, got.size() * 4, cudaMemcpyDeviceToHost));
        double ma; const double r = rel_rms(got, want, &ma);
        fails += report("gelu(gate) * ple[il] (strided 512 of 12288)", r, ma, 1e-5);
        // a wrong stride must be refused rather than read out of bounds
        if (eg2vq_ops_gelu_mul_strided(dg, dp, n_tok, pl, (uint64_t)pl, (uint64_t)pl)) {
            printf("FAIL: the strided gelu-mul accepted a stride its offset cannot fit\n");
            fails++;
        }
        cudaFree(dg); cudaFree(dp);
    }

    printf(fails ? "EG2VQ_OPS_FAIL (%d)\n" : "EG2VQ_OPS_PASS\n", fails);
    return fails ? 1 : 0;
}
