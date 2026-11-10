/* QA independent bit-identity driver: shared payload/activation generator.
 * Deterministic; identical bytes fed to both the old (31285531) and the new
 * (8d618ecb) kernels in ONE process. */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include "ds41_vq_fmt.h"

static inline uint16_t qa_f2h(float v) {
    __half h = __float2half_rn(v);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

struct QaRng {
    uint64_t s;
    explicit QaRng(uint64_t seed) : s(seed ? seed : 1u) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float unit() { return (float)((double)(int64_t)(next() >> 11) / 9007199254740992.0 - 1.0); }
};

static inline std::vector<uint8_t> qa_payload(uint32_t rows, uint32_t cols, uint32_t nc, uint64_t seed) {
    const uint32_t dim = 4;
    QaRng rng(seed);
    const size_t head = 16;
    const size_t cb_bytes = (size_t)nc * dim * 2;
    const size_t gr_bytes = (size_t)rows * 2;
    const uint32_t cw_row = cols / 4u;
    const size_t idx_bytes = (size_t)rows * cw_row * 12u / 8u;
    std::vector<uint8_t> out(head + cb_bytes + gr_bytes + idx_bytes, 0);
    uint32_t mg = (uint32_t)DS4VQ_MAT_MAGIC, r32 = rows, c32 = cols;
    uint16_t d16 = (uint16_t)dim, n16 = (uint16_t)nc;
    std::memcpy(&out[0], &mg, 4); std::memcpy(&out[4], &d16, 2); std::memcpy(&out[6], &n16, 2);
    std::memcpy(&out[8], &r32, 4); std::memcpy(&out[12], &c32, 4);
    for (size_t i = 0; i < (size_t)nc * dim; i++) {
        const uint16_t h = qa_f2h(rng.unit() * 0.05f);
        std::memcpy(&out[head + 2 * i], &h, 2);
    }
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t h = qa_f2h(0.25f + (rng.unit() + 1.0f) * 0.05f);
        std::memcpy(&out[head + cb_bytes + 2 * r], &h, 2);
    }
    uint8_t *p = &out[head + cb_bytes + gr_bytes];
    for (uint32_t r = 0; r < rows; r++) {
        uint8_t *row = p + (size_t)r * cw_row * 12u / 8u;
        for (uint32_t i = 0; i < cw_row; i++) {
            const uint32_t v = (uint32_t)(rng.next() & 0xFFFu);
            const size_t bit = (size_t)i * 12u;
            const uint32_t m = v << (bit & 7u);
            row[bit >> 3] |= (uint8_t)m;
            row[(bit >> 3) + 1] |= (uint8_t)(m >> 8);
            if (m >> 16) { row[(bit >> 3) + 2] |= (uint8_t)(m >> 16); }
        }
    }
    return out;
}

static inline std::vector<float> qa_x(uint32_t rows, uint32_t ts, uint64_t seed) {
    QaRng rng(seed);
    std::vector<float> x((size_t)rows * ts);
    for (float &v : x) { v = rng.unit(); }
    return x;
}

extern "C" {
int qa_run_old(float *out, const float *x, uint32_t ts, const uint8_t *pay,
               uint32_t rows, uint32_t cols, uint32_t ntok);
int qa_run_new(float *out, const float *x, uint32_t ts, const uint8_t *pay,
               uint32_t rows, uint32_t cols, uint32_t ntok);
int qa_refuse_old(void);
int qa_refuse_new(void);
}
