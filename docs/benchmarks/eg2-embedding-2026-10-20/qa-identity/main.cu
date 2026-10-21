/* QA independent bit-identity driver: runs the old kernel (31285531) and the
 * new kernel (8d618ecb) on the SAME payload/activation bytes in one process
 * and compares the outputs byte for byte. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>

#include "common.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

struct Case { const char *tag; uint32_t rows, cols, ntok, ts; };

static int run_case(const Case &c) {
    const uint32_t nc = 4096u;
    std::vector<uint8_t> pay = qa_payload(c.rows, c.cols, nc, 1000u + c.rows + c.cols + c.ntok);
    std::vector<float> x = qa_x(c.ntok, c.ts, 77u + c.rows + c.cols + c.ntok);
    const size_t out_bytes = (size_t)c.ntok * c.rows * 4u;

    uint8_t *dp = NULL; float *dx = NULL, *dold = NULL, *dnew = NULL;
    CK(cudaMalloc(&dp, pay.size()));
    CK(cudaMalloc(&dx, x.size() * 4u));
    CK(cudaMalloc(&dold, out_bytes));
    CK(cudaMalloc(&dnew, out_bytes));
    CK(cudaMemcpy(dp, pay.data(), pay.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4u, cudaMemcpyHostToDevice));
    CK(cudaMemset(dold, 0xCD, out_bytes));
    CK(cudaMemset(dnew, 0xCD, out_bytes));

    const int ro = qa_run_old(dold, dx, c.ts, dp, c.rows, c.cols, c.ntok);
    const int rn = qa_run_new(dnew, dx, c.ts, dp, c.rows, c.cols, c.ntok);

    std::vector<float> ho((size_t)c.ntok * c.rows), hn((size_t)c.ntok * c.rows);
    CK(cudaMemcpy(ho.data(), dold, out_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hn.data(), dnew, out_bytes, cudaMemcpyDeviceToHost));

    size_t diff = 0, diffbytes = 0;
    double maxabs = 0.0, maxrel = 0.0;
    for (size_t i = 0; i < ho.size(); i++) {
        if (std::memcmp(&ho[i], &hn[i], 4) != 0) {
            diff++;
            diffbytes += 4;
            const double d = std::fabs((double)ho[i] - (double)hn[i]);
            if (d > maxabs) { maxabs = d; }
            const double a = std::fabs((double)ho[i]);
            if (a > 1e-30) { const double r = d / a; if (r > maxrel) { maxrel = r; } }
        }
    }
    printf("%-22s rows=%-6u cols=%-5u tok=%-4u ts=%-5u  old_rc=%d new_rc=%d  differing %zu of %zu floats"
           "  maxabs %.3e maxrel %.3e\n",
           c.tag, c.rows, c.cols, c.ntok, c.ts, ro, rn, diff, ho.size(), maxabs, maxrel);

    cudaFree(dp); cudaFree(dx); cudaFree(dold); cudaFree(dnew);
    return diff ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "san") {
        /* one small ragged case for compute-sanitizer */
        Case c = {"san-ragged", 256u, 512u, 70u, 512u};
        run_case(c);
        Case c2 = {"san-rows64-t1", 64u, 512u, 1u, 512u};
        run_case(c2);
        printf("SAN_DONE\n");
        return 0;
    }

    printf("# qa identity: old(31285531) vs new(8d618ecb), same bytes in one process\n");
    const Case cases[] = {
        {"id256x512-t1",        256u,  512u,   1u,  512u},
        {"id256x512-t70",       256u,  512u,  70u,  512u},
        {"id256x512-t257",      256u,  512u, 257u,  512u},
        {"id256x512-t512",      256u,  512u, 512u,  512u},
        {"id256x512-t64",       256u,  512u,  64u,  512u},
        {"big12288x2048-t1",  12288u, 2048u,   1u, 2048u},
        {"big12288x2048-t64", 12288u, 2048u,  64u, 2048u},
        {"big12288x2048-t512",12288u, 2048u, 512u, 2048u},
        {"rows64x512-t1",        64u,  512u,   1u,  512u},
        {"rows64x512-t65",       64u,  512u,  65u,  512u},
        {"big12288x2048-t70", 12288u, 2048u,  70u, 2048u},
        {"big12288x2048-t257",12288u, 2048u, 257u, 2048u},
        {"rows512x512-t65",     512u,  512u,  65u,  512u},
        {"strided512x512-t130", 512u,  512u, 130u,  576u},
        {"rows2048x1024-t512", 2048u, 1024u, 512u, 1024u},
    };
    int bad = 0;
    for (const Case &c : cases) { if (run_case(c)) { bad++; } }
    const int ro = qa_refuse_old(), rn = qa_refuse_new();
    printf("launcher refusal path: old refused %d of 9, new refused %d of 9\n", ro, rn);
    if (ro != 9 || rn != 9) { bad++; }
    printf("cases with differing floats: %d of %d\n", bad, (int)(sizeof(cases) / sizeof(cases[0])));
    printf("%s\n", bad ? "QA_IDENTITY_FAIL" : "QA_IDENTITY_OK");
    return bad ? 1 : 0;
}
