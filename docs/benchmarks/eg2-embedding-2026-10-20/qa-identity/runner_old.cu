/* old kernel (parent commit 31285531) behind extern "C"; kernel renamed so the
 * two header definitions can coexist in one link. */
#include <cuda_runtime.h>
#define eg2vq_vq_gemm_kernel qa_eg2vq_vq_gemm_kernel_old
#include "old/cuda/eg2vq_vq.cuh"
#undef eg2vq_vq_gemm_kernel

extern "C" int qa_run_old(float *out, const float *x, uint32_t ts, const uint8_t *pay,
                          uint32_t rows, uint32_t cols, uint32_t ntok) {
    const int ok = eg2vq_vq_gemm(out, x, ts, pay, rows, cols, ntok);
    if (!ok) { return 0; }
    const cudaError_t e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 1 : -1;
}

extern "C" int qa_refuse_old(void) {
    int n = 0;
    static float o[64 * 64];
    static float xx[64 * 64];
    uint8_t dummy[64] = {0};
    if (!eg2vq_vq_gemm(o, xx, 512, dummy, 100, 512, 1)) { n++; }   /* rows % 64 != 0 */
    if (!eg2vq_vq_gemm(o, xx, 100, dummy, 64, 100, 1)) { n++; }    /* cols % 256 != 0 */
    if (!eg2vq_vq_gemm(o, xx, 256, dummy, 64, 512, 1)) { n++; }    /* t_stride < cols */
    if (!eg2vq_vq_gemm(o, xx, 512, dummy, 64, 512, 0)) { n++; }    /* n_tok 0 */
    if (!eg2vq_vq_gemm(o, xx, 512, dummy, 0, 512, 1)) { n++; }     /* rows 0 */
    if (!eg2vq_vq_gemm(o, xx, 0, dummy, 64, 0, 1)) { n++; }        /* cols 0 */
    if (!eg2vq_vq_gemm(NULL, xx, 512, dummy, 64, 512, 1)) { n++; } /* out NULL */
    if (!eg2vq_vq_gemm(o, NULL, 512, dummy, 64, 512, 1)) { n++; }  /* x NULL */
    if (!eg2vq_vq_gemm(o, xx, 512, NULL, 64, 512, 1)) { n++; }     /* pay NULL */
    return n;
}
