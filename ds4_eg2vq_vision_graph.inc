/* ds4_eg2vq_vision_graph.inc — the EmbeddingGemma-2 vision tower as this
 * project's CPU reference forward (unit 23).  Included from ds4.c beside
 * ds4_eg2vq_vision_bind.inc, after ds4_eg2vq_graph.inc, so it reuses that
 * file's helpers (eg2vq_rms, eg2vq_rope, eg2vq_matvec, eg2vq_read_f32) and
 * the reference path's own idioms (ds4_parallel_for,
 * qwen4exp_resize_rgb_pillow, f16_to_f32/f32_to_f16).
 *
 * Authority, first-hand: /data/llama.cpp @ 31288c9c
 * tools/mtmd/models/gemma4v.cpp:5-117 (the graph) and the builders it calls in
 * tools/mtmd/clip.cpp (build_vit :337-560, build_norm :607-628, build_ffn
 * :628-700, build_attn :762-830), the preprocessing in tools/mtmd/mtmd-image.cpp,
 * and — where the graph alone is not enough — the KERNELS the tensors of this
 * shape actually reach (ggml/src/ggml-cpu).  Every mechanism below carries its
 * file:line, and the ones that were found by measurement rather than by reading
 * the graph say so.
 *
 * Skeleton, in the order the port reproduces it:
 *   image bytes -> decode (stb_image, 3 channels, NO EXIF rotation)
 *     -> size solver: align 48, then min/max pixels        mtmd-image.cpp:775-782
 *     -> PAD_CEIL: scale = min(sw,sh), ceil the region, black canvas, centered
 *        composite with integer division                     mtmd-image.cpp:57-96
 *     -> (v/255 - mean)/std with mean 0 std 1               mtmd-image.cpp:7-14
 *   graph: patches * 2 - 1                                   gemma4v.cpp:9
 *     -> one 16x16 stride-16 conv, NO bias                   gemma4v.cpp:12-16
 *     -> + position table[row]: x slice then y slice          gemma4v.cpp:26-43
 *     -> 16 blocks:
 *          x = RMS(inpL, ln1)                                 clip.cpp:376
 *          q,k,v = Wq/Wk/Wv x                                 clip.cpp:424-434
 *          per-head RMS on q,k ([64] norms => norm_per_head)  clip.cpp:440-465
 *          axial NeoX RoPE: dims [0,32) on pos_x, [32,64) on
 *            pos_y at offset 32, theta 100                     gemma4v.cpp:46-73
 *          weightless V RMS                                    clip.cpp:481-483
 *          attention: kq_scale exactly 1.0, NO mask            gemma4v.cpp:75,
 *                                                             clip.cpp:787-802
 *          o proj -> attn_post_norm -> +residual               clip.cpp:486-500
 *          ln2 -> GEGLU (gelu_quick) -> ffn_post_norm          clip.cpp:519-529
 *          +residual                                           clip.cpp:537
 *     -> 3x3 stride-3 average pool over the (grid_w,grid_h) grid, then
 *        scale sqrt(n_embd)                                    gemma4v.cpp:83-99
 *     -> weightless RMS (eps 1e-6) + 768->512 projection        gemma4v.cpp:104-114
 *
 * Patch index convention: i = y*grid_w + x, pos_x = i % grid_w, pos_y = i /
 * grid_w (clip.cpp:5562-5568, n_cols = image_size_width / patch_size).
 * Pool row: q = x' + (grid_w/3)*y' (clip.cpp:254-256 for the grid, then the
 * reshape in gemma4v.cpp:93-95).
 *
 * WHAT THE GRAPH DOES NOT TELL YOU (both found by measuring against the frozen
 * dumps, 2026-10-20; the numbers are in docs/eg2vq-vision-tower-gate23.txt):
 *
 *  1. EVERY bf16-weight matmul QUANTIZES ITS ACTIVATION TO BF16.  ggml's CPU
 *     mul_mat converts src1 to the weight's own vec_dot_type
 *     (ggml/src/ggml-cpu/ggml-cpu.c:1328-1340), and this revision's BF16 entry
 *     declares `.vec_dot_type = GGML_TYPE_BF16`
 *     (ggml/src/ggml-cpu/ggml-cpu.c:395-400).  The port reproduces that
 *     conversion (eg2vq_vision_bf16) wherever the activation is src1 — the
 *     seven per-block matrices and the tie-in projection.
 *  2. The FFN activation is FFN_GELU_QUICK, not the tanh GELU, because the
 *     mmproj sets neither clip.use_gelu nor clip.use_silu (clip.cpp:1414-1429)
 *     and the reference printed "load_hparams: ffn_op: gelu_quick" — and ggml's
 *     f32 gelu_quick is itself an f16 TABLE lookup, quantizing the input and the
 *     output (vec.h:47, :1047-1057, ggml-cpu.c:3884-3891).
 *
 * Two more things that are easy to get wrong and are NOT free choices:
 *  - the conv is the one bf16-weight product whose activation is NOT quantized,
 *    because ggml_conv_2d passes (im2col, weight) as (src0, src1): the f32 im2col
 *    drives vec_dot_type and the bf16 weight is merely widened (ggml.c:4785-4789,
 *    ggml-cpu.c:1332-1340).  Rounding it moved both fixtures AWAY from the
 *    frozen dumps (see the evidence doc's discrimination table);
 *  - the reference's encode ran with FLASH ATTENTION ENABLED, whose branch casts
 *    K and V to f16 before the kernel (clip.cpp:787-790); see
 *    ds4_eg2vq_vision_opts.
 */

/* The FFN activation the reference actually ran: ggml's gelu_quick,
 * x * sigmoid(1.702 x) (ggml/src/ggml-cpu/vec.h:964, :1031) — computed through
 * the f16 table as ggml does it (GGML_GELU_QUICK_FP16 is defined
 * unconditionally at vec.h:47): the input is quantized to f16, the value is
 * looked up in ggml_table_gelu_quick_f16 (built as
 * FP32_TO_FP16(gelu_quick_f32(FP16_TO_FP32(i))), ggml-cpu.c:3884-3891), and the
 * entry is widened back to f32 — so the output is f16-quantized too.  Carrying
 * the 65536-entry table is pointless here: the same four conversions in the same
 * order give the identical value. */
#define EG2VQ_VISION_GELU_QUICK_COEF (-1.702f)

static inline float eg2vq_vision_gelu_quick(float x) {
    const float xf = f16_to_f32(f32_to_f16(x));
    const float y = xf * (1.0f / (1.0f + expf(EG2VQ_VISION_GELU_QUICK_COEF * xf)));
    return f16_to_f32(f32_to_f16(y));
}

/* Round-to-nearest-even f32 -> bf16 -> f32, ggml's own conversion
 * (ggml_compute_fp32_to_bf16, ggml/src/ggml-impl.h:623-635: add 0x7fff plus the
 * kept LSB, then truncate; a quiet-NaN guard the finite data here never takes).
 * See note 1 above for why the port must apply it to the activations. */
static inline float eg2vq_vision_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) { return x; }
    bits = (bits + (0x7fffu + ((bits >> 16) & 1u))) >> 16;
    bits <<= 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* ------------------------------------------------------------------------- *
 * The preprocessing (mtmd_image_preprocessor_dyn_size, mtmd-image.cpp:767-791)
 * ------------------------------------------------------------------------- */

/* calc_size_preserved_ratio (mtmd-image.cpp:122-156) with align_size =
 * patch * merge, min_pixels = 70 * patch_area, max_pixels = 1120 * patch_area
 * (clip-model.h:207-212, clip.cpp:1734).  The alignment is applied BEFORE any
 * limit and the limit branches are the reference's own, in float. */
static int eg2vq_vision_target_size(uint32_t w, uint32_t h,
                                    uint32_t *out_w, uint32_t *out_h) {
    if (!w || !h) { return 0; }
    const float align = (float)DS4_EG2VQ_VISION_ALIGN;
    const uint32_t rw = (uint32_t)(roundf((float)w / align) * align);
    const uint32_t rh = (uint32_t)(roundf((float)h / align) * align);
    uint32_t w_bar = rw < DS4_EG2VQ_VISION_ALIGN ? DS4_EG2VQ_VISION_ALIGN : rw;
    uint32_t h_bar = rh < DS4_EG2VQ_VISION_ALIGN ? DS4_EG2VQ_VISION_ALIGN : rh;

    if ((uint64_t)h_bar * w_bar > (uint64_t)DS4_EG2VQ_VISION_MAX_PIXELS) {
        const float beta = sqrtf((float)h * (float)w / (float)DS4_EG2VQ_VISION_MAX_PIXELS);
        h_bar = (uint32_t)(floorf((float)h / beta / align) * align);
        w_bar = (uint32_t)(floorf((float)w / beta / align) * align);
        if (h_bar < DS4_EG2VQ_VISION_ALIGN) { h_bar = DS4_EG2VQ_VISION_ALIGN; }
        if (w_bar < DS4_EG2VQ_VISION_ALIGN) { w_bar = DS4_EG2VQ_VISION_ALIGN; }
    } else if ((uint64_t)h_bar * w_bar < (uint64_t)DS4_EG2VQ_VISION_MIN_PIXELS) {
        const float beta = sqrtf((float)DS4_EG2VQ_VISION_MIN_PIXELS / ((float)h * (float)w));
        h_bar = (uint32_t)(ceilf((float)h * beta / align) * align);
        w_bar = (uint32_t)(ceilf((float)w * beta / align) * align);
    }
    /* A canvas the patch grid cannot tile is a different image, not a smaller
     * one: refuse rather than truncate. */
    if (!w_bar || !h_bar ||
        (w_bar % DS4_EG2VQ_VISION_PATCH) || (h_bar % DS4_EG2VQ_VISION_PATCH)) {
        return 0;
    }
    if ((uint64_t)(w_bar / DS4_EG2VQ_VISION_PATCH) * (h_bar / DS4_EG2VQ_VISION_PATCH)
            > (uint64_t)DS4_EG2VQ_VISION_MAX_PATCHES) {
        return 0;
    }
    *out_w = w_bar;
    *out_h = h_bar;
    return 1;
}

int ds4_eg2vq_vision_prepare(const uint8_t *data, size_t len,
                             ds4_eg2vq_vision_geometry *geo,
                             uint32_t *pos_x, uint32_t *pos_y,
                             float *canvas, uint64_t canvas_cap,
                             char *err, size_t errlen) {
    if (err && errlen) { err[0] = '\0'; }
    if (!data || !len || len > (size_t)INT_MAX || !geo) {
        if (err && errlen) { snprintf(err, errlen, "invalid image payload"); }
        return 0;
    }
    /* stb_image, 3 channels, and NO EXIF orientation: the reference decodes
     * exactly this way (mtmd-helper.cpp:404) and parses no APP1/EXIF at all. */
    int src_w = 0, src_h = 0, src_c = 0;
    uint8_t *rgb = stbi_load_from_memory(data, (int)len, &src_w, &src_h, &src_c, 3);
    if (!rgb || src_w <= 0 || src_h <= 0) {
        if (err && errlen) {
            snprintf(err, errlen, "cannot decode the image: %s",
                     stbi_failure_reason() ? stbi_failure_reason() : "invalid image");
        }
        return 0;
    }

    uint32_t canvas_w = 0, canvas_h = 0;
    if (!eg2vq_vision_target_size((uint32_t)src_w, (uint32_t)src_h, &canvas_w, &canvas_h)) {
        free(rgb);
        if (err && errlen) { snprintf(err, errlen, "the size solver refused the image"); }
        return 0;
    }
    const uint64_t canvas_px = (uint64_t)canvas_w * canvas_h;
    if (canvas && canvas_px * 3u > canvas_cap) {
        free(rgb);
        if (err && errlen) { snprintf(err, errlen, "the canvas buffer is too small"); }
        return 0;
    }

    /* PAD_CEIL (clip-model.h:67 is the default and gemma4v does not override
     * it): scale = min(target/src) per side, ceil the region, resample into it,
     * fill the canvas with the pad colour {0,0,0}, composite CENTERED with
     * integer division (mtmd-image.cpp:57-96).  Where the alignment does not
     * exceed the aspect scale — the frozen fixtures — scale is 1.0 and the
     * resample is a bit-exact identity (the bicubic taps at integer distances
     * are [0,1,0,0]), so the region IS the source and the padding is all the
     * alignment added.  This is why the port must not stretch. */
    const float scale_w = (float)canvas_w / (float)src_w;
    const float scale_h = (float)canvas_h / (float)src_h;
    const float scale = scale_w < scale_h ? scale_w : scale_h;
    uint32_t region_w = (uint32_t)(int)ceilf((float)src_w * scale);
    uint32_t region_h = (uint32_t)(int)ceilf((float)src_h * scale);
    if (region_w > canvas_w) { region_w = canvas_w; }
    if (region_h > canvas_h) { region_h = canvas_h; }
    const int32_t offset_x = (int32_t)((canvas_w - region_w) / 2u);
    const int32_t offset_y = (int32_t)((canvas_h - region_h) / 2u);

    uint8_t *resized = qwen4exp_resize_rgb_pillow(rgb, (uint32_t)src_w, (uint32_t)src_h,
                                                 region_w, region_h);
    free(rgb);
    if (!resized) {
        if (err && errlen) { snprintf(err, errlen, "the resizer failed"); }
        return 0;
    }

    const uint32_t grid_w = canvas_w / DS4_EG2VQ_VISION_PATCH;
    const uint32_t grid_h = canvas_h / DS4_EG2VQ_VISION_PATCH;
    const uint32_t out_x = grid_w / DS4_EG2VQ_VISION_MERGE;
    const uint32_t out_y = grid_h / DS4_EG2VQ_VISION_MERGE;
    if (!out_x || !out_y || (grid_w % DS4_EG2VQ_VISION_MERGE) ||
        (grid_h % DS4_EG2VQ_VISION_MERGE)) {
        free(resized);
        if (err && errlen) { snprintf(err, errlen, "the patch grid does not tile by the pooler"); }
        return 0;
    }

    if (canvas) {
        /* inp_raw: [w, h, 3] with x fastest, pad colour (from_u8's v/255 then
         * gemma4v.cpp:9's *2 - 1, two roundings in that order). */
        for (uint64_t i = 0; i < canvas_px * 3u; i++) { canvas[i] = -1.0f; }
        for (uint32_t y = 0; y < region_h; y++) {
            for (uint32_t x = 0; x < region_w; x++) {
                const uint8_t *p = resized + ((size_t)y * region_w + x) * 3u;
                for (uint32_t c = 0; c < 3u; c++) {
                    float f = (float)p[c] / 255.0f;
                    f = f * 2.0f - 1.0f;
                    canvas[(uint64_t)(x + (uint32_t)offset_x)
                           + (uint64_t)canvas_w * ((uint64_t)(y + (uint32_t)offset_y)
                                                   + (uint64_t)canvas_h * c)] = f;
                }
            }
        }
    }
    free(resized);

    /* The positions the reference's own encoder fills in (clip.cpp:5562-5568). */
    for (uint32_t i = 0; i < grid_w * grid_h; i++) {
        pos_x[i] = i % grid_w;
        pos_y[i] = i / grid_w;
    }

    geo->source_w = (uint32_t)src_w;
    geo->source_h = (uint32_t)src_h;
    geo->canvas_w = canvas_w;
    geo->canvas_h = canvas_h;
    geo->region_w = region_w;
    geo->region_h = region_h;
    geo->offset_x = offset_x;
    geo->offset_y = offset_y;
    geo->grid_w = grid_w;
    geo->grid_h = grid_h;
    geo->n_patches = grid_w * grid_h;
    geo->n_soft = out_x * out_y;
    return 1;
}

/* ------------------------------------------------------------------------- *
 * The tower
 * ------------------------------------------------------------------------- */

/* The patch convolution: one 16x16 stride-16 window, no bias (gemma4v.cpp:
 * 12-16).  Weight order (ggml ne0 fastest): kw + 16*kh + 256*ic + 768*oc.
 * Input order (the graph's inp_raw): x + canvas_w*(y + canvas_h*ic). */
typedef struct {
    const float *w;
    const float *img;
    float       *out;
    uint32_t     grid_w, canvas_w, canvas_h;
} eg2vq_vision_conv_ctx;

static void eg2vq_vision_conv_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const eg2vq_vision_conv_ctx *c = vctx;
    const uint32_t E = DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t P = DS4_EG2VQ_VISION_PATCH;
    const uint64_t plane = (uint64_t)c->canvas_w * c->canvas_h;
    for (uint64_t i = r0; i < r1; i++) {
        const uint32_t p = (uint32_t)(i / E);
        const uint32_t oc = (uint32_t)(i % E);
        const uint32_t px = p % c->grid_w;
        const uint32_t py = p / c->grid_w;
        const float *wp = c->w + (uint64_t)oc * P * P * 3u;
        const float *base = c->img + (uint64_t)(px * P)
                          + (uint64_t)c->canvas_w * (uint64_t)(py * P);
        double acc = 0.0;
        for (uint32_t ic = 0; ic < 3u; ic++) {
            for (uint32_t kh = 0; kh < P; kh++) {
                const float *row = base + (uint64_t)c->canvas_w * kh + plane * ic;
                for (uint32_t kw = 0; kw < P; kw++) {
                    acc += (double)wp[ic * P * P + kh * P + kw] * (double)row[kw];
                }
            }
        }
        c->out[(uint64_t)p * E + oc] = (float)acc;
    }
}

/* Per-head RMS then axial RoPE on q and k, and the weightless V RMS, all for
 * one patch (clip.cpp:457-483, gemma4v.cpp:46-73).  The theta step is
 * powf(rope_theta, -2.0f/n_dims) with n_dims = head_dim/2 = 32
 * (ggml-cpu/ops.cpp:6455, where n_dims is the op's own 32, not the full head),
 * and each rope call rotates pairs (i, i + 16) INSIDE its own 32 dims — the
 * second call at the +32 offset the graph sets with ggml_rope_set_offset
 * (gemma4v.cpp:70, ggml.c:4547-4557). */
typedef struct {
    float        *q, *k, *v;
    const float  *qn, *kn;
    const uint32_t *pos_x, *pos_y;
    float         theta_scale;
} eg2vq_vision_qkv_ctx;

static void eg2vq_vision_qkv_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const eg2vq_vision_qkv_ctx *c = vctx;
    const uint32_t E = DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t H = DS4_EG2VQ_VISION_N_HEAD;
    const uint32_t D = DS4_EG2VQ_VISION_HEAD_DIM;
    for (uint64_t p = r0; p < r1; p++) {
        const float px = (float)c->pos_x[p];
        const float py = (float)c->pos_y[p];
        for (uint32_t h = 0; h < H; h++) {
            float *qh = c->q + p * E + (uint64_t)h * D;
            float *kh = c->k + p * E + (uint64_t)h * D;
            float *vh = c->v + p * E + (uint64_t)h * D;
            eg2vq_rms(qh, qh, c->qn, D, DS4_EG2VQ_VISION_RMS_EPS);
            eg2vq_rope(qh, D / 2u, px, c->theta_scale);
            eg2vq_rope(qh + D / 2u, D / 2u, py, c->theta_scale);
            eg2vq_rms(kh, kh, c->kn, D, DS4_EG2VQ_VISION_RMS_EPS);
            eg2vq_rope(kh, D / 2u, px, c->theta_scale);
            eg2vq_rope(kh + D / 2u, D / 2u, py, c->theta_scale);
            eg2vq_rms(vh, vh, NULL, D, DS4_EG2VQ_VISION_RMS_EPS);
        }
    }
}

/* Attention over the whole patch grid: kq_scale exactly 1.0, no mask, no
 * causal window, 12 query heads over 12 KV heads (gemma4v.cpp:75,
 * clip.cpp:787-802 with kq_mask = nullptr).  The reference's kernel multiplies
 * the raw dot by kq_scale = 1.0 — a no-op — so the port does not carry it. */
typedef struct {
    const float *q, *k, *v;
    float       *out;
    uint32_t     n_patches;
} eg2vq_vision_attn_ctx;

static void eg2vq_vision_attn_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const eg2vq_vision_attn_ctx *c = vctx;
    const uint32_t E = DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t H = DS4_EG2VQ_VISION_N_HEAD;
    const uint32_t D = DS4_EG2VQ_VISION_HEAD_DIM;
    float *scores = xmalloc((size_t)c->n_patches * sizeof(float));
    for (uint64_t qp = r0; qp < r1; qp++) {
        for (uint32_t h = 0; h < H; h++) {
            const float *qh = c->q + qp * E + (uint64_t)h * D;
            float *oh = c->out + qp * E + (uint64_t)h * D;
            float max_score = -INFINITY;
            for (uint32_t kp = 0; kp < c->n_patches; kp++) {
                const float *kh = c->k + (uint64_t)kp * E + (uint64_t)h * D;
                double acc = 0.0;
                for (uint32_t d = 0; d < D; d++) acc += (double)qh[d] * (double)kh[d];
                scores[kp] = (float)acc;
                if (scores[kp] > max_score) { max_score = scores[kp]; }
            }
            double denom = 0.0;
            for (uint32_t kp = 0; kp < c->n_patches; kp++) {
                const float e = expf(scores[kp] - max_score);
                scores[kp] = e;
                denom += e;
            }
            const float inv = denom > 0.0 ? (float)(1.0 / denom) : 0.0f;
            for (uint32_t d = 0; d < D; d++) { oh[d] = 0.0f; }
            for (uint32_t kp = 0; kp < c->n_patches; kp++) {
                const float p = scores[kp];
                if (p == 0.0f) { continue; }
                const float *vh = c->v + (uint64_t)kp * E + (uint64_t)h * D;
                for (uint32_t d = 0; d < D; d++) oh[d] += p * vh[d];
            }
            for (uint32_t d = 0; d < D; d++) oh[d] *= inv;
        }
    }
    free(scores);
}

/* The 3x3 stride-3 average pool over the patch grid, then the sqrt(n_embd)
 * scale (gemma4v.cpp:83-99).  The grid is [grid_w, grid_h] with the patch
 * index x + grid_w*y, so output row q = x' + (grid_w/3)*y'. */
typedef struct {
    const float *patch;
    float       *pooled;
    uint32_t     grid_w;
    float        scale;
} eg2vq_vision_pool_ctx;

static void eg2vq_vision_pool_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const eg2vq_vision_pool_ctx *c = vctx;
    const uint32_t E = DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t K = DS4_EG2VQ_VISION_MERGE;
    const uint32_t out_x = c->grid_w / K;
    for (uint64_t q = r0; q < r1; q++) {
        const uint32_t x0 = (uint32_t)(q % out_x) * K;
        const uint32_t y0 = (uint32_t)(q / out_x) * K;
        for (uint32_t d = 0; d < E; d++) {
            double acc = 0.0;
            for (uint32_t j = 0; j < K; j++) {
                for (uint32_t i = 0; i < K; i++) {
                    const uint32_t p = (x0 + i) + c->grid_w * (y0 + j);
                    acc += (double)c->patch[(uint64_t)p * E + d];
                }
            }
            c->pooled[(uint64_t)q * E + d] = (float)acc / (float)(K * K) * c->scale;
        }
    }
}

/* One matvec through the family's own batched worker, so the vision tower and
 * the text tower share the reference path's parallel unit and accumulation —
 * with the one thing the kernels add on top: the ACTIVATION is bf16, because
 * that is what every caller here feeds a bf16-weight matmul (see note 1 at the
 * top of this file). */
static void eg2vq_vision_matvec(const float *w, uint32_t rows, uint32_t cols,
                                const float *x, uint64_t x_stride,
                                float *out, uint32_t n_tokens) {
    ds4_eg2vq_matrix mx;
    mx.tensor = NULL;
    mx.dense = (float *)w;
    mx.rows = rows;
    mx.cols = cols;
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *row = (float *)x + (uint64_t)t * x_stride;
        for (uint32_t i = 0; i < cols; i++) { row[i] = eg2vq_vision_bf16(row[i]); }
    }
    eg2vq_matvec(&mx, x, x_stride, out, n_tokens);
}

int ds4_eg2vq_vision_forward(const ds4_model *m,
                             const ds4_eg2vq_vision_weights *v,
                             const uint8_t *data, size_t len,
                             const ds4_eg2vq_vision_opts *opts,
                             float *soft, uint64_t soft_cap,
                             ds4_eg2vq_vision_geometry *geo,
                             ds4_eg2vq_vision_stages *stages,
                             char *err, size_t errlen) {
    if (err && errlen) { err[0] = '\0'; }
    if (!m || !v || !v->dense || !data || !len || !soft || !geo) {
        if (err && errlen) { snprintf(err, errlen, "invalid vision arguments"); }
        return 0;
    }
    const int f16_kv = opts ? opts->f16_kv : 1;

    /* The preprocessing: the canvas, the geometry and the positions. */
    uint32_t *pos_x = xmalloc((size_t)DS4_EG2VQ_VISION_MAX_PATCHES * sizeof(uint32_t));
    uint32_t *pos_y = xmalloc((size_t)DS4_EG2VQ_VISION_MAX_PATCHES * sizeof(uint32_t));
    float *canvas = xmalloc((size_t)DS4_EG2VQ_VISION_MAX_PIXELS * 3u * sizeof(float));
    ds4_eg2vq_vision_geometry g;
    if (!ds4_eg2vq_vision_prepare(data, len, &g, pos_x, pos_y, canvas,
                                 (uint64_t)DS4_EG2VQ_VISION_MAX_PIXELS * 3u,
                                 err, errlen)) {
        free(canvas); free(pos_y); free(pos_x);
        return 0;
    }
    if (g.n_soft > DS4_EG2VQ_VISION_SOFT_MAX ||
        (uint64_t)g.n_soft * DS4_EG2VQ_VISION_N_OUT > soft_cap) {
        free(canvas); free(pos_y); free(pos_x);
        if (err && errlen) { snprintf(err, errlen, "the soft-token buffer is too small"); }
        return 0;
    }

    const uint32_t E  = DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t NF = DS4_EG2VQ_VISION_N_FF;
    const uint32_t NP = g.n_patches;
    const float eps = DS4_EG2VQ_VISION_RMS_EPS;

    float *h       = xmalloc((size_t)NP * E * sizeof(float));
    float *x       = xmalloc((size_t)NP * E * sizeof(float));
    float *q       = xmalloc((size_t)NP * E * sizeof(float));
    float *k       = xmalloc((size_t)NP * E * sizeof(float));
    float *vv      = xmalloc((size_t)NP * E * sizeof(float));
    float *attn    = xmalloc((size_t)NP * E * sizeof(float));
    float *f       = xmalloc((size_t)NP * E * sizeof(float));
    float *ffn_g   = xmalloc((size_t)NP * NF * sizeof(float));
    float *ffn_u   = xmalloc((size_t)NP * NF * sizeof(float));
    float *ffn_o   = xmalloc((size_t)NP * E * sizeof(float));
    float *pooled  = xmalloc((size_t)g.n_soft * E * sizeof(float));

    /* 1. the patch conv, then the two position tables added in the reference's
     * order: + emb_x, then + emb_y (gemma4v.cpp:26-43).  The tables are f32 in
     * the mmproj and addressed row-major within their slice: slice 0 is X at
     * offset 0, slice 1 is Y at pos_size * n_embd (gemma4v.cpp:30-34). */
    {
        eg2vq_vision_conv_ctx cc;
        cc.w = v->patch_embd;
        cc.img = canvas;
        cc.out = h;
        cc.grid_w = g.grid_w;
        cc.canvas_w = g.canvas_w;
        cc.canvas_h = g.canvas_h;
        ds4_parallel_for((uint64_t)NP * E, eg2vq_vision_conv_worker, &cc);

        const float *tbl = (const float *)tensor_data(m, v->position_embd);
        const uint64_t slice = (uint64_t)DS4_EG2VQ_VISION_POS_SIZE * E;
        for (uint32_t p = 0; p < NP; p++) {
            const float *ex = tbl + (uint64_t)pos_x[p] * E;
            const float *ey = tbl + slice + (uint64_t)pos_y[p] * E;
            float *hp = h + (uint64_t)p * E;
            for (uint32_t d = 0; d < E; d++) { hp[d] += ex[d]; }
            for (uint32_t d = 0; d < E; d++) { hp[d] += ey[d]; }
        }
        /* the gate's attribution stage 1: the tower's input (see
         * ds4_eg2vq_vision_stages) */
        if (stages && stages->patch) {
            memcpy(stages->patch, h, (size_t)NP * E * sizeof(float));
        }
    }

    const float theta_scale = powf(DS4_EG2VQ_VISION_ROPE_THETA,
                                   -2.0f / (float)(DS4_EG2VQ_VISION_HEAD_DIM / 2u));

    for (uint32_t il = 0; il < (uint32_t)DS4_EG2VQ_VISION_N_LAYER; il++) {
        const ds4_eg2vq_vision_layer *l = &v->layer[il];
        float ln1[DS4_EG2VQ_VISION_N_EMBD], ln_ap[DS4_EG2VQ_VISION_N_EMBD];
        float ln2[DS4_EG2VQ_VISION_N_EMBD], ln_fp[DS4_EG2VQ_VISION_N_EMBD];
        float qn[DS4_EG2VQ_VISION_HEAD_DIM], kn[DS4_EG2VQ_VISION_HEAD_DIM];
        eg2vq_read_f32(m, l->ln1, ln1);
        eg2vq_read_f32(m, l->attn_post_norm, ln_ap);
        eg2vq_read_f32(m, l->ln2, ln2);
        eg2vq_read_f32(m, l->ffn_post_norm, ln_fp);
        eg2vq_read_f32(m, l->q_norm, qn);
        eg2vq_read_f32(m, l->k_norm, kn);

        /* a. ln1, then the three projections (their activation is bf16 inside
         * eg2vq_vision_matvec, exactly as the reference's mul_mat makes it). */
        for (uint32_t p = 0; p < NP; p++) {
            eg2vq_rms(x + (uint64_t)p * E, h + (uint64_t)p * E, ln1, E, eps);
        }
        eg2vq_vision_matvec(l->q, E, E, x, E, q, NP);
        eg2vq_vision_matvec(l->k, E, E, x, E, k, NP);
        eg2vq_vision_matvec(l->v, E, E, x, E, vv, NP);

        /* b. per-head RMS + axial RoPE on q,k; weightless RMS on v. */
        {
            eg2vq_vision_qkv_ctx qc;
            qc.q = q; qc.k = k; qc.v = vv;
            qc.qn = qn; qc.kn = kn;
            qc.pos_x = pos_x; qc.pos_y = pos_y;
            qc.theta_scale = theta_scale;
            ds4_parallel_for(NP, eg2vq_vision_qkv_worker, &qc);
        }
        /* The reference's executed attention rounds K and V to f16 (the flash
         * branch of build_attn, clip.cpp:787-790).  f16_kv = 1 mirrors that;
         * 0 is the graph in f32 and exists so the gate can attribute the
         * difference it measures. */
        if (f16_kv) {
            f16_round_inplace_cpu(k, NP * E);
            f16_round_inplace_cpu(vv, NP * E);
        }

        /* c. attention, scale 1.0, no mask. */
        {
            eg2vq_vision_attn_ctx ac;
            ac.q = q; ac.k = k; ac.v = vv; ac.out = attn;
            ac.n_patches = NP;
            ds4_parallel_for(NP, eg2vq_vision_attn_worker, &ac);
        }

        /* d. o projection, attn_post_norm, residual. */
        eg2vq_vision_matvec(l->o, E, E, attn, E, f, NP);
        for (uint32_t p = 0; p < NP; p++) {
            float *fp = f + (uint64_t)p * E;
            eg2vq_rms(fp, fp, ln_ap, E, eps);
            float *hp = h + (uint64_t)p * E;
            for (uint32_t d = 0; d < E; d++) { hp[d] += fp[d]; }
        }

        /* e. ln2, the GEGLU FFN (gelu_quick(gate) * up), ffn_post_norm, residual. */
        for (uint32_t p = 0; p < NP; p++) {
            eg2vq_rms(x + (uint64_t)p * E, h + (uint64_t)p * E, ln2, E, eps);
        }
        eg2vq_vision_matvec(l->ffn_gate, NF, E, x, E, ffn_g, NP);
        eg2vq_vision_matvec(l->ffn_up, NF, E, x, E, ffn_u, NP);
        for (size_t i = 0; i < (size_t)NP * NF; i++) {
            ffn_g[i] = eg2vq_vision_gelu_quick(ffn_g[i]) * ffn_u[i];
        }
        eg2vq_vision_matvec(l->ffn_down, E, NF, ffn_g, NF, ffn_o, NP);
        for (uint32_t p = 0; p < NP; p++) {
            float *op = ffn_o + (uint64_t)p * E;
            eg2vq_rms(op, op, ln_fp, E, eps);
            float *hp = h + (uint64_t)p * E;
            for (uint32_t d = 0; d < E; d++) { hp[d] += op[d]; }
        }
        /* the gate's attribution stage 2: the residual stream after block 0 */
        if (il == 0u && stages && stages->l0) {
            memcpy(stages->l0, h, (size_t)NP * E * sizeof(float));
        }
    }

    /* 2. the pooler: 3x3 stride 3, then sqrt(n_embd). */
    {
        eg2vq_vision_pool_ctx pc;
        pc.patch = h;
        pc.pooled = pooled;
        pc.grid_w = g.grid_w;
        pc.scale = sqrtf((float)E);
        ds4_parallel_for(g.n_soft, eg2vq_vision_pool_worker, &pc);
        /* the gate's attribution stage 3: the pooled rows the embedder reads */
        if (stages && stages->pooled) {
            memcpy(stages->pooled, pooled, (size_t)g.n_soft * E * sizeof(float));
        }
    }

    /* 3. the multimodal embedder: weightless RMS (eps 1e-6) then the 768->512
     * projection (gemma4v.cpp:104-114), one row per soft token. */
    for (uint32_t t = 0; t < g.n_soft; t++) {
        float *pt = pooled + (uint64_t)t * E;
        eg2vq_rms(pt, pt, NULL, E, eps);
    }
    eg2vq_vision_matvec(v->input_projection, DS4_EG2VQ_VISION_N_OUT, E,
                        pooled, E, soft, g.n_soft);

    free(pooled); free(ffn_o); free(ffn_u); free(ffn_g);
    free(f); free(attn); free(vv); free(k); free(q); free(x); free(h);
    free(canvas); free(pos_y); free(pos_x);
    *geo = g;
    return 1;
}
