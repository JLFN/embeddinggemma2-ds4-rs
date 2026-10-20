/* ds4_resize_pillow.inc — Pillow-compatible separable resampling, SHARED and
 * CPU-ONLY.  Extracted verbatim from the Qwen4exp phone-path section of ds4.c
 * on 2026-10-20 (unit 23) for one reason: that section sits inside the
 * `#ifndef DS4_NO_GPU` block that opens at ds4.c:18444, so the CPU-only
 * reference oracle could not call the resizer even though the code itself is
 * plain C with no CUDA in it.  Nothing here changed: the bodies below are the
 * bytes that were at ds4.c:26636-26779, and both callers (the Qwen4exp image
 * path and the gemma_vq vision tower) now share them.
 *
 * Authority: this IS llama.cpp mtmd's `img_tool::resize_pillow`
 * (/data/llama.cpp @ 31288c9c tools/mtmd/mtmd-image.cpp:204-390), compared
 * clause by clause — bicubic (filter support 2.0) with Pillow's a = -0.5,
 * filterscale = max(1, scale), support = 2*filterscale, center = (i + 0.5)*scale,
 * first = (int)(center - support + 0.5), weights normalized by their sum,
 * fixed point 2^22 with the "+/-0.5 then truncate toward zero" rounding, two
 * separable passes, accumulator bias 1<<21 and a >>22 clip.  Keep it OUTSIDE
 * the DS4_NO_GPU guard, and keep it the only resizer in this tree.
 */

typedef struct {
    int ksize;
    int *bounds;
    int32_t *weights;
} ds4_qwen_resize_axis;

static void qwen4exp_resize_axis_free(ds4_qwen_resize_axis *a) {
    if (!a) return;
    free(a->weights);
    free(a->bounds);
    memset(a, 0, sizeof(*a));
}

static double qwen4exp_bicubic_filter(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

/* Antialiased BICUBIC coefficients adapted from the already-vendored
 * llama.cpp mtmd image path.  Keeping this local avoids a second image
 * dependency while matching the Qwen processor's explicit resample mode. */
static bool qwen4exp_resize_axis_build(int in_size, int out_size,
                                       ds4_qwen_resize_axis *a) {
    if (!a || in_size <= 0 || out_size <= 0) return false;
    memset(a, 0, sizeof(*a));
    const double scale = (double)in_size / (double)out_size;
    const double filter_scale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * filter_scale;
    const int ksize = (int)ceil(support) * 2 + 1;
    if (ksize <= 0 || (size_t)out_size > SIZE_MAX / 2u / sizeof(int) ||
        (size_t)out_size > SIZE_MAX / (size_t)ksize / sizeof(int32_t))
        return false;
    a->bounds = xmalloc((size_t)out_size * 2u * sizeof(int));
    a->weights = xmalloc(
        (size_t)out_size * (size_t)ksize * sizeof(int32_t));
    double *tmp = xmalloc((size_t)ksize * sizeof(double));
    const double fixed_scale = ldexp(1.0, 22);
    for (int dst = 0; dst < out_size; dst++) {
        const double center = ((double)dst + 0.5) * scale;
        int first = (int)(center - support + 0.5);
        int last = (int)(center + support + 0.5);
        if (first < 0) first = 0;
        if (last > in_size) last = in_size;
        const int count = last - first;
        double sum = 0.0;
        for (int k = 0; k < count; k++) {
            tmp[k] = qwen4exp_bicubic_filter(
                ((double)(first + k) - center + 0.5) / filter_scale);
            sum += tmp[k];
        }
        a->bounds[2 * dst] = first;
        a->bounds[2 * dst + 1] = count;
        for (int k = 0; k < ksize; k++) {
            const double normalized = k < count && sum != 0.0
                ? tmp[k] / sum : 0.0;
            double fixed = normalized * fixed_scale;
            fixed += normalized < 0.0 ? -0.5 : 0.5;
            if (fixed > INT32_MAX) fixed = INT32_MAX;
            if (fixed < INT32_MIN) fixed = INT32_MIN;
            a->weights[(size_t)dst * ksize + k] = (int32_t)round(fixed);
        }
    }
    free(tmp);
    a->ksize = ksize;
    return true;
}

static uint8_t qwen4exp_clip_u8(int64_t value) {
    value >>= 22;
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (uint8_t)value;
}

static uint8_t *qwen4exp_resize_rgb_pillow(
        const uint8_t *src, uint32_t src_w, uint32_t src_h,
        uint32_t dst_w, uint32_t dst_h) {
    if (!src || !src_w || !src_h || !dst_w || !dst_h ||
        (uint64_t)dst_w * dst_h > SIZE_MAX / 3u) return NULL;
    if (src_w == dst_w && src_h == dst_h) {
        const size_t bytes = (size_t)src_w * src_h * 3u;
        uint8_t *copy = xmalloc(bytes);
        memcpy(copy, src, bytes);
        return copy;
    }
    ds4_qwen_resize_axis horizontal = {0}, vertical = {0};
    if (!qwen4exp_resize_axis_build((int)src_w, (int)dst_w, &horizontal) ||
        !qwen4exp_resize_axis_build((int)src_h, (int)dst_h, &vertical)) {
        qwen4exp_resize_axis_free(&horizontal);
        qwen4exp_resize_axis_free(&vertical);
        return NULL;
    }
    if ((uint64_t)dst_w * src_h > SIZE_MAX / 3u ||
        (uint64_t)dst_w * dst_h > SIZE_MAX / 3u) {
        qwen4exp_resize_axis_free(&horizontal);
        qwen4exp_resize_axis_free(&vertical);
        return NULL;
    }
    uint8_t *tmp = xmalloc((size_t)dst_w * src_h * 3u);
    uint8_t *dst = xmalloc((size_t)dst_w * dst_h * 3u);
    for (uint32_t y = 0; y < src_h; y++) {
        for (uint32_t x = 0; x < dst_w; x++) {
            const int first = horizontal.bounds[2u * x];
            const int count = horizontal.bounds[2u * x + 1u];
            int64_t acc[3] = {1 << 21, 1 << 21, 1 << 21};
            for (int k = 0; k < count; k++) {
                const int32_t w = horizontal.weights[
                    (size_t)x * horizontal.ksize + k];
                const uint8_t *pixel = src +
                    ((size_t)y * src_w + (uint32_t)(first + k)) * 3u;
                for (uint32_t c = 0; c < 3u; c++)
                    acc[c] += (int64_t)pixel[c] * w;
            }
            uint8_t *pixel = tmp + ((size_t)y * dst_w + x) * 3u;
            for (uint32_t c = 0; c < 3u; c++)
                pixel[c] = qwen4exp_clip_u8(acc[c]);
        }
    }
    for (uint32_t y = 0; y < dst_h; y++) {
        const int first = vertical.bounds[2u * y];
        const int count = vertical.bounds[2u * y + 1u];
        for (uint32_t x = 0; x < dst_w; x++) {
            int64_t acc[3] = {1 << 21, 1 << 21, 1 << 21};
            for (int k = 0; k < count; k++) {
                const int32_t w = vertical.weights[
                    (size_t)y * vertical.ksize + k];
                const uint8_t *pixel = tmp +
                    ((size_t)(first + k) * dst_w + x) * 3u;
                for (uint32_t c = 0; c < 3u; c++)
                    acc[c] += (int64_t)pixel[c] * w;
            }
            uint8_t *pixel = dst + ((size_t)y * dst_w + x) * 3u;
            for (uint32_t c = 0; c < 3u; c++)
                pixel[c] = qwen4exp_clip_u8(acc[c]);
        }
    }
    free(tmp);
    qwen4exp_resize_axis_free(&horizontal);
    qwen4exp_resize_axis_free(&vertical);
    return dst;
}
