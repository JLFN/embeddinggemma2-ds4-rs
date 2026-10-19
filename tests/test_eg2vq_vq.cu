// test_eg2vq_vq.cu — the dim-4 DQVQ device GEMM against the payload's own
// semantics and against FP64 (embeddinggemma2-handoff.md PHASE 7, row P3.1).
//
// Three references, because the kernel can be wrong in three different ways:
//
//   the stream  the payload bytes are decoded here in the test by a second
//               implementation (p_decode_ieee) that reads the 12-bit stream and
//               the codeword table directly from the bytes p_pack wrote.  A
//               kernel that extracts the index stream differently, reads the
//               wrong row pitch or takes a codeword from the wrong index
//               disagrees by a lot.  This test's decoder and the tree's
//               vendored one (ds4vq_dequant_f32) must agree except on the one
//               case below.
//   arithmetic  a round trip through f32 can still be mis-summed, so the decode
//               is dotted in double precision and that is where the kernel's f32
//               accumulation must land.  Inputs are f32 and the codeword and
//               gain values are exact f16 -> f32, so the ONLY error in the
//               kernel is summation order: the bound is the f32 rounding of a
//               dot, not a quantization bound.
//   the known divergence  ds4vq_f16 (ds41_vq_fmt.h:38-43, the converter the CPU
//               path's binder uses) returns HALF the IEEE value for every
//               subnormal f16 input, while the payload writer's own reader
//               (onebit_quant.h:141-159) and this test's decoder follow IEEE.
//               The test asserts that this is the ONLY disagreement, and
//               reports how many entries it covers; see tools/eg2_subnorm.c for
//               what it costs on the real artifact (8,893 of 7,553,024 codebook
//               entries, 2.3e-09 of the decoded weight mass).
//
// The synthetic payloads are packed by p_pack below, which mirrors the
// project's own writer byte for byte: /data/YoungAi/gguf-tools/quantize/vq_qc.h
// `vq_pack` (:242-266) — [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols],
// codebook nc*dim f16, rows f16 gains, then one continuous little-endian bit
// stream of ceil(log2 nc)-bit indices (12 here), padded to whole bytes plus one
// guard byte.  Geometries are the tower's own (ds4_eg2vq.h:52-68): widths 512,
// 1024 and 2048, heights 512 to 12288.
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

static std::mt19937 rng(20261019u);

// ---- f16 helpers (the writer's own conversion is go1b_fp32_to_fp16; the test
// only needs a value that is exactly representable, so it rounds through
// __half and reads it back, which is what the payload carries) ----------------
static uint16_t p_h16(float v) { __half h = __float2half_rn(v); uint16_t u; memcpy(&u, &h, 2); return u; }
static float p_f16(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

// ---- the payload writer, mirroring vq_pack (:242-266) ----------------------
static void p_pack(std::vector<uint8_t> &out, const std::vector<float> &cb,
                   const std::vector<uint16_t> &gr, const std::vector<uint32_t> &idx,
                   uint32_t rows, uint32_t cols, uint32_t nc, uint32_t dim) {
    const size_t head = 16;
    const size_t cb_bytes = (size_t)nc * dim * 2;
    const size_t gr_bytes = (size_t)rows * 2;
    const size_t nidx = idx.size();
    const size_t idx_bytes = (nidx * 12 + 7) / 8 + 1;
    out.assign(head + cb_bytes + gr_bytes + idx_bytes, 0);
    uint32_t mg = DS4VQ_MAT_MAGIC, r32 = rows, c32 = cols;
    uint16_t d16 = (uint16_t)dim, n16 = (uint16_t)nc;
    memcpy(&out[0], &mg, 4); memcpy(&out[4], &d16, 2); memcpy(&out[6], &n16, 2);
    memcpy(&out[8], &r32, 4); memcpy(&out[12], &c32, 4);
    for (size_t i = 0; i < (size_t)nc * dim; i++) { const uint16_t h = p_h16(cb[i]); memcpy(&out[head + 2 * i], &h, 2); }
    for (size_t r = 0; r < rows; r++) { memcpy(&out[head + cb_bytes + 2 * r], &gr[r], 2); }
    uint8_t *p = &out[head + cb_bytes + gr_bytes];
    for (size_t i = 0; i < nidx; i++) {
        const size_t bit = i * 12;
        const uint32_t m = (idx[i] & 0xFFFu) << (bit & 7);
        p[bit >> 3] |= (uint8_t)m;
        p[(bit >> 3) + 1] |= (uint8_t)(m >> 8);
        if (m >> 16) { p[(bit >> 3) + 2] |= (uint8_t)(m >> 16); }
    }
}

struct p_case {
    uint32_t rows, cols, tokens;
    const char *tag;
};

// The payload's value rule, read straight from the bytes: value[r][i] =
// IEEE(codebook word idx[r][i/4]][i%4]) * IEEE(gain[r]), with the index stream
// one continuous little-endian 12-bit stream (ds4vq_dequant_f32:88-99 states the
// same rule; this implementation follows IEC 60559 for the half conversion).
static void p_decode_ieee(const std::vector<uint8_t> &pay, uint32_t rows, uint32_t cols,
                          uint32_t nc, uint32_t dim, std::vector<float> &out,
                          std::vector<float> *entries = NULL) {
    const uint8_t *cb = pay.data() + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const size_t nidx = (size_t)rows * cols / dim;
    out.assign((size_t)rows * cols, 0.0f);
    if (entries) { entries->assign(out.size(), 0.0f); }
    for (size_t i = 0; i < nidx; i++) {
        const size_t bit = i * 12;
        uint32_t v = (uint32_t)ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8)
                   | ((uint32_t)ix[(bit >> 3) + 2] << 16);
        v = (v >> (bit & 7)) & 0xFFFu;
        const size_t r = i * dim / cols;
        const float g = p_f16((uint16_t)(gr[2 * r] | ((uint16_t)gr[2 * r + 1] << 8)));
        float *o = out.data() + i * dim;
        for (uint32_t d = 0; d < dim; d++) {
            const size_t w = (size_t)v * dim + d;
            const uint16_t h = (uint16_t)(cb[2 * w] | ((uint16_t)cb[2 * w + 1] << 8));
            o[d] = p_f16(h) * g;
            if (entries) { (*entries)[i * dim + d] = p_f16(h); }
        }
    }
}

int main(void) {
    // The tower's shapes: q/k/v/o and the FFN at both head widths, the 12288-row
    // per-layer projection, and the token extremes (a 26-token query, a
    // 1414-token document, plus ragged counts that leave a partial 64-token tile).
    const p_case cases[] = {
        {  512,  512,   26, "o_proj swa x 26 tokens" },
        { 1024,  512,   73, "q_proj swa x 73 tokens" },
        { 2048,  512,  184, "ffn_gate x 184 tokens" },
        { 12288,  512,    1, "per_layer_model_proj x 1 token" },
        { 1024, 1024,  349, "o_proj full x 349 tokens" },
        {  512, 2048,  1414, "ffn_down x 1414 tokens" },
    };
    const uint32_t nc = 4096u, dim = 4u;
    int fails = 0;
    size_t n_subnormal_reported = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const p_case &c : cases) {
        // ---- the payload: a codebook of f16 values, gains in the writer's
        // clamped range, random indices over the whole book ------------------
        std::vector<float> cb((size_t)nc * dim);
        for (float &v : cb) { v = p_f16(p_h16(0.05f * nd(rng))); }
        std::vector<uint16_t> gr(c.rows);
        for (uint16_t &g : gr) { g = p_h16(0.25f + (float)(rng() % 1000u) * 0.00375f); }
        const size_t nidx = (size_t)c.rows * c.cols / dim;
        std::vector<uint32_t> idx(nidx);
        for (uint32_t &v : idx) { v = rng() & 0xFFFu; }
        std::vector<uint8_t> pay;
        p_pack(pay, cb, gr, idx, c.rows, c.cols, nc, dim);

        // ---- activations and the two host references ------------------------
        std::vector<float> x((size_t)c.tokens * c.cols);
        for (float &v : x) { v = nd(rng); }

        std::vector<float> dense, entries;               // this test's own decode, IEEE halves
        p_decode_ieee(pay, c.rows, c.cols, nc, dim, dense, &entries);
        // p_pack is checked against the values that went in, so a packing bug
        // cannot hide behind the decode.
        {
            size_t bad = 0;
            for (uint32_t r = 0; r < c.rows && bad < 4; r++) {
                for (uint32_t i = 0; i < c.cols && bad < 4; i++) {
                    const uint32_t v = idx[(size_t)r * c.cols / dim + i / dim];
                    const float want = cb[(size_t)v * dim + (i % dim)] * p_f16(gr[r]);
                    if (dense[(size_t)r * c.cols + i] != want) {
                        if (!bad) {
                            printf("  first mismatch r %u i %u idx %u want %.9g got %.9g\n",
                                   r, i, v, want, dense[(size_t)r * c.cols + i]);
                        }
                        bad++;
                    }
                }
            }
            if (bad) { printf("FAIL %s: p_pack and p_decode_ieee disagree (%zu elements)\n", c.tag, bad); fails++; continue; }
        }

        // The tree's vendored decoder must agree with this test's decode
        // everywhere except on subnormal codebook entries, where it halves the
        // value (the divergence named in this file's header).  Anything else
        // would be a second, unexplained disagreement between two readers of the
        // same bytes.
        std::vector<float> dvend((size_t)c.rows * c.cols);
        if (ds4vq_dequant_f32(pay.data(), dvend.data(), (int)c.rows, (int)c.cols) != 0) {
            printf("FAIL %s: the vendored decoder refused the payload\n", c.tag);
            fails++;
            continue;
        }
        {
            size_t only_subnormal = 1, n_diff = 0;
            for (size_t i = 0; i < dvend.size(); i++) {
                if (dvend[i] == dense[i]) { continue; }
                n_diff++;
                const bool subnormal = (entries[i] != 0.0f && fabsf(entries[i]) < 6.10352e-05f);
                if (!(subnormal && dense[i] == 2.0f * dvend[i])) {
                    if (only_subnormal) {
                        printf("  first unexplained diff at %zu: ieee %.9g vendored %.9g entry %.9g\n",
                               i, dense[i], dvend[i], entries[i]);
                    }
                    only_subnormal = 0;
                }
            }
            if (!only_subnormal) {
                printf("FAIL %s: the vendored decoder differs from IEEE outside the subnormal rule\n", c.tag);
                fails++;
                continue;
            }
            n_subnormal_reported += n_diff;
        }

        // FP64 reference on the IEEE decode. The full product is
        // rows*cols*tokens; the two largest cases are checked on a sub-tile
        // instead so the gate stays quick, and the print names the sub-tile.
        const uint32_t ref_tokens = (c.rows == 12288u) ? 1u : ((c.tokens > 256u) ? 256u : c.tokens);
        const uint32_t ref_rows = (c.rows > 2048u) ? 2048u : c.rows;
        std::vector<double> yref((size_t)ref_tokens * ref_rows);
        for (uint32_t t = 0; t < ref_tokens; t++) {
            for (uint32_t r = 0; r < ref_rows; r++) {
                const float *w = dense.data() + (size_t)r * c.cols;
                const float *xx = x.data() + (size_t)t * c.cols;
                double s = 0.0;
                for (uint32_t i = 0; i < c.cols; i++) { s += (double)w[i] * (double)xx[i]; }
                yref[(size_t)t * ref_rows + r] = s;
            }
        }

        // ---- the device ------------------------------------------------------
        float *dp = NULL, *dx = NULL, *dy = NULL;
        CK(cudaMalloc(&dp, pay.size()));
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dy, (size_t)c.tokens * c.rows * 4));
        CK(cudaMemcpy(dp, pay.data(), pay.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dy, 0, (size_t)c.tokens * c.rows * 4));
        if (!eg2vq_vq_gemm(dy, dx, c.cols, (const uint8_t *)dp, c.rows, c.cols, c.tokens)) {
            printf("FAIL %s: the launcher refused the shape\n", c.tag);
            fails++;
            cudaFree(dp); cudaFree(dx); cudaFree(dy);
            continue;
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> ydev((size_t)c.tokens * c.rows);
        CK(cudaMemcpy(ydev.data(), dy, ydev.size() * 4, cudaMemcpyDeviceToHost));

        // ---- compare ---------------------------------------------------------
        double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0, min_cos = 1.0;
        for (uint32_t t = 0; t < ref_tokens; t++) {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (uint32_t r = 0; r < ref_rows; r++) {
                const double a = yref[(size_t)t * ref_rows + r];
                const double b = ydev[(size_t)t * c.rows + r];
                const double d = b - a;
                if (fabs(d) > max_abs) { max_abs = fabs(d); }
                sum_sq += d * d;
                ref_sq += a * a;
                dot += a * b; na += a * a; nb += b * b;
            }
            const double cos = (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 1.0;
            if (cos < min_cos) { min_cos = cos; }
        }
        const double rel = (ref_sq > 0.0) ? sqrt(sum_sq / ref_sq) : 0.0;

        // ---- timing (one matrix, best of 3 warm launches) --------------------
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float ms = 0.0f, best = 1e30f;
        for (int rep = 0; rep < 4; rep++) {
            CK(cudaEventRecord(e0));
            eg2vq_vq_gemm(dy, dx, c.cols, (const uint8_t *)dp, c.rows, c.cols, c.tokens);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (rep && ms < best) { best = ms; }
        }
        const double flop = 2.0 * (double)c.rows * (double)c.cols * (double)c.tokens;
        const double gflops = flop / (best * 1e6);
        const double gbs = (double)pay.size() * ((c.tokens + 63) / 64) / (best * 1e6);

        printf("%-34s rows %5u cols %4u tok %4u | ref %ux%u | maxabs %.2e rel %.2e mincos %.9f | %7.3f ms %6.1f GFLOP/s %6.1f GB/s%s\n",
               c.tag, c.rows, c.cols, c.tokens, ref_rows, ref_tokens, max_abs, rel, min_cos,
               best, gflops, gbs, (rel > 1e-5 || min_cos < 1.0 - 1e-6) ? "  <-- OUT OF BOUND" : "");
        if (rel > 1e-5 || min_cos < 1.0 - 1e-6) { fails++; }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(dp); cudaFree(dx); cudaFree(dy);
    }

    // ---- a payload bound to the wrong geometry must decode nothing ----------
    {
        std::vector<float> cb((size_t)nc * dim, 0.0f);
        std::vector<uint16_t> gr(512, p_h16(1.0f));
        std::vector<uint32_t> idx((size_t)512 * 512 / dim, 0u);
        std::vector<uint8_t> pay;
        p_pack(pay, cb, gr, idx, 512, 512, nc, dim);
        std::vector<float> x((size_t)26 * 512, 1.0f), y((size_t)26 * 512, 7.0f);
        float *dp = NULL, *dx = NULL, *dy = NULL;
        CK(cudaMalloc(&dp, pay.size())); CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, y.size() * 4));
        CK(cudaMemcpy(dp, pay.data(), pay.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
        // rows 512 in the payload, 1024 asked for: the kernel's open() must refuse.
        eg2vq_vq_gemm(dy, dx, 512, (const uint8_t *)dp, 1024, 512, 26);
        CK(cudaDeviceSynchronize());
        std::vector<float> back(y.size());
        CK(cudaMemcpy(back.data(), dy, back.size() * 4, cudaMemcpyDeviceToHost));
        int untouched = 1;
        for (float v : back) { if (v != 7.0f) { untouched = 0; break; } }
        printf("%-34s %s\n", "wrong-geometry payload", untouched ? "refused, output untouched PASS"
                                                               : "WROTE SOMETHING FAIL");
        if (!untouched) { fails++; }
        cudaFree(dp); cudaFree(dx); cudaFree(dy);
    }

    printf("vendored decoder vs IEEE: %zu decoded elements differ, all of them subnormal entries at exactly half value\n",
           n_subnormal_reported);
    printf(fails ? "EG2VQ_VQ_FAIL (%d)\n" : "EG2VQ_VQ_PASS\n", fails);
    return fails ? 1 : 0;
}
