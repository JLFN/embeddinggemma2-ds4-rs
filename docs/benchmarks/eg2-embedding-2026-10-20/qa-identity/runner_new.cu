/* shipped kernel (8d618ecb) behind extern "C" */
#include <cuda_runtime.h>
#include "cuda/eg2vq_vq.cuh"   /* resolved via -I<ds4-rs>/ */

extern "C" int qa_run_new(float *out, const float *x, uint32_t ts, const uint8_t *pay,
                          uint32_t rows, uint32_t cols, uint32_t ntok) {
    const int ok = eg2vq_vq_gemm(out, x, ts, pay, rows, cols, ntok);
    if (!ok) { return 0; }
    const cudaError_t e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 1 : -1;
}

extern "C" int qa_refuse_new(void) {
    int n = 0;
    static float o[64 * 64];
    static float xx[64 * 64];
    uint8_t dummy[64] = {0};
    if (!eg2vq_vq_gemm(o, xx, 512, dummy, 100, 512, 1)) { n++; }
    if (!eg2vq_vq_gemm(o, xx, 100, dummy, 64, 100, 1)) { n++; }
    if (!eg2vq_vq_gemm(o, xx, 256, dummy, 64, 512, 1)) { n++; }
    if (!eg2vq_vq_gemm(o, xx, 512, dummy, 64, 512, 0)) { n++; }
    if (!eg2vq_vq_gemm(o, xx, 512, dummy, 0, 512, 1)) { n++; }
    if (!eg2vq_vq_gemm(o, xx, 0, dummy, 64, 0, 1)) { n++; }
    if (!eg2vq_vq_gemm(NULL, xx, 512, dummy, 64, 512, 1)) { n++; }
    if (!eg2vq_vq_gemm(o, NULL, 512, dummy, 64, 512, 1)) { n++; }
    if (!eg2vq_vq_gemm(o, xx, 512, NULL, 64, 512, 1)) { n++; }
    return n;
}
