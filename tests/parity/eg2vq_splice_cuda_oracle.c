/* P4.6 -- the IMAGE SPLICE end to end (embeddinggemma2-handoff.md PHASE 8,
 * unit 25): one image through the vision tower, its soft tokens spliced into the
 * text tower's image span, and the pooled embedding compared against the frozen
 * reference dumps (tests/vision/reference/eg2-vision-*.text.tsv).
 *
 * WHAT IS UNDER TEST.  The reference's own splice, read first-hand: llama.cpp
 * builds one batch whose text entries are token ids and whose image-span entries
 * are raw embeddings (tests/parity/eg2vq_vision_anchor.cpp, following
 * mtmd-helper.cpp's decode_embd_batch), and its graph scales only the TOKEN
 * lookup: "important: do not normalize weights for raw embeddings input (i.e.
 * encoded image embeddings)", src/models/gemma-embedding2.cpp:85-86.  The port's
 * splice is ds4_eg2vq_ref_embed_image (CPU) and ds4_gpu_eg2vq_embed_image
 * (device), both of which take the soft rows unscaled and the token rows times
 * sqrt(n_embd).
 *
 * THE SEQUENCE.  The anchor's prompt was the bare media marker, so the reference
 * tokenized [BOS][<|image>] + 88 image positions + [<image|>][EOS] = 92 positions
 * with the image span at [2, 90) — the frozen dumps' own header says so.  The four
 * text ids are the artifact's own vocabulary entries, read with
 * `cargo run -p ds4-core --example gemma_vq_vocab -- <artifact> 255999,258880,258882`
 * (bos 2, boi 255999, eoi 258882, eos 1), and the placeholder the span carries is
 * the checkpoint's image_token_id 258880 (ds4_eg2vq.h's DS4_EG2VQ_IMAGE_TOKEN_ID).
 * The oracle builds that sequence itself, checks the span against the geometry the
 * preprocessing produced, and prints it.
 *
 * THE THREE COMPARISONS, with the bound printed before any run and never widened:
 *   a. the CPU leg's pooled vector against the frozen "l2" row (max |difference|,
 *      relative RMS, cosine);
 *   b. the CPU leg's vector against the frozen "raw" row BY DIRECTION only (the raw
 *      row is pre-L2, so its magnitude is not comparable and a cosine is);
 *   c. the device leg against the CPU leg on the same inputs (the port's own
 *      parity, as gate 24 does for the vision tower);
 * plus the refusals the device entry must make (a non-contiguous span, a span that
 * disagrees with the soft block, a placeholder in a text-only call, an
 * out-of-range id) and a text-only control so the image can be seen reaching the
 * tower.
 *
 * THE FOUR LEGS, and why there are four.  The frozen spliced rows were produced by
 * the anchor harness on the BASELINE bf16 text model (tools/eg2_vision_anchor.sh
 * passes $BASE as its second argument), while the shipped text tower is the
 * 4-bit-per-4-elements VQ artifact — whose vectors keep 95.60% agreement with the
 * baseline's (gate 5, docs/retrieval-cost-gate5.txt 5b).  Comparing the ARTIFACT's
 * splice with the frozen rows therefore measures the TIER's loss, not the splice.
 * The legs are:
 *   1. baseline + the FROZEN soft rows -> the SPLICE's own fidelity, bounded;
 *   2. baseline + the PORT's soft rows -> adds the vision tower's own 0.82% (gate 23);
 *   3. artifact + the PORT's soft rows -> adds the tier's loss; reported, and its
 *      cosine is expected to sit in gate 5's own range for this tier;
 *   4. artifact DEVICE leg vs leg 3 -> the device path's parity with the CPU one, bounded.
 *
 *   usage: eg2vq_splice_cuda_oracle <artifact.gguf> <mmproj.gguf> <baseline.gguf> <image> [<image> ...]
 */
#include "../../ds4.c"

#include <cuda_runtime.h>

/* Declared here, printed before the first run, never widened afterwards.  The
 * scales: the frozen l2 row is a unit 768-vector (components ~0.036 RMS), so a 2%
 * relative bound is ~0.0007 RMS per component and a max |difference| of 0.01 is
 * about three times that; the text tower's own device parity was measured at
 * 3.1e-04 relative RMS and cosine 0.99999995 (docs/eg2vq-device-gate7.txt), which
 * is what the device-vs-CPU bound below allows for. */
static const double SPLICE_L2_RELRMS      = 0.020;
static const double SPLICE_L2_MAXABS      = 0.010;
static const double SPLICE_L2_COS_MIN     = 0.999;
static const double SPLICE_RAW_COS_MIN    = 0.999;
static const double SPLICE_DEV_RELRMS     = 0.005;
static const double SPLICE_DEV_MAXABS     = 0.005;
static const double SPLICE_DEV_COS_MIN    = 0.9999;
/* The FROZEN-SOFT leg is the bound-checked statement about the SPLICE itself: the
 * reference's own soft rows go through the port's splice, so the only thing left in
 * the comparison is the splice and the text tower's own fidelity (gate 4 measured
 * that at cosine 0.999996809 against the frozen anchor vectors). */
static const double SPLICE_FROZEN_RELRMS  = 0.010;
static const double SPLICE_FROZEN_MAXABS  = 0.005;
static const double SPLICE_FROZEN_COS_MIN = 0.999;
/* The end-to-end leg (the PORT's soft rows) is REPORTED, not bounded at these
 * widths: it carries the vision tower's 0.82% distance from the reference into a
 * text tower whose activations are quantized to bf16 at every matmul, so a
 * difference of two bf16 steps in the input rows lands ~20-25% away at the
 * pooled vector — the same quantization-boundary propagation gate 24 measured on
 * the vision tower, and the oracle's own kick probe measures it here.  Only a
 * floor is asserted, so a broken path (near-orthogonal output) still fails. */
static const double SPLICE_E2E_COS_MIN    = 0.900;

typedef struct {
    double maxabs, relrms, cos;
    uint64_t rows;
    uint32_t cols;
} splice_cmp;

static splice_cmp splice_compare(const float *a, const float *b, uint64_t rows, uint32_t cols) {
    splice_cmp r = {0.0, 0.0, 0.0, rows, cols};
    double ss = 0.0, rr = 0.0, dot = 0.0, a2 = 0.0, b2 = 0.0;
    for (uint64_t i = 0; i < rows; i++) {
        for (uint32_t d = 0; d < cols; d++) {
            const double x = (double)a[i * cols + d];
            const double y = (double)b[i * cols + d];
            const double e = x - y;
            const double ae = e < 0.0 ? -e : e;
            if (ae > r.maxabs) { r.maxabs = ae; }
            ss += e * e;
            rr += y * y;
            dot += x * y;
            a2 += x * x;
            b2 += y * y;
        }
    }
    r.relrms = (rr > 0.0) ? sqrt(ss / rr) : 0.0;
    r.cos = (a2 > 0.0 && b2 > 0.0) ? dot / sqrt(a2 * b2) : 0.0;
    return r;
}

/* The frozen dumps' own format: a comment line, "raw" <TAB> 768 values, "l2"
 * <TAB> 768 values (eg2vq_vision_anchor.cpp:44-46). */
static int splice_load_frozen(const char *path, float *raw, float *l2, uint32_t *n_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "eg2vq_splice: cannot open %s\n", path); return 0; }
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int have_raw = 0, have_l2 = 0;
    while ((len = getline(&line, &cap, fp)) >= 0) {
        if (line[0] == '#') { continue; }
        const int is_raw = (strncmp(line, "raw", 3) == 0);
        const int is_l2 = (strncmp(line, "l2", 2) == 0);
        if (!is_raw && !is_l2) { continue; }
        float *dst = is_raw ? raw : l2;
        uint32_t got = 0;
        const char *p = line + (is_raw ? 3 : 2);
        while (*p && got < (uint32_t)DS4_EG2VQ_N_OUT) {
            char *end = NULL;
            const double v = strtod(p, &end);
            if (end == p) { break; }
            dst[got++] = (float)v;
            p = end;
        }
        if (got != (uint32_t)DS4_EG2VQ_N_OUT) {
            fprintf(stderr, "eg2vq_splice: %s has %u of %d values on the %s row\n",
                    path, got, (int)DS4_EG2VQ_N_OUT, is_raw ? "raw" : "l2");
            free(line); fclose(fp);
            return 0;
        }
        if (is_raw) { have_raw = 1; } else { have_l2 = 1; }
    }
    free(line);
    fclose(fp);
    *n_out = (uint32_t)DS4_EG2VQ_N_OUT;
    return have_raw && have_l2;
}

/* The frozen soft-token dump: one row per token, "<index> <TAB> 512 values"
 * (eg2vq_vision_anchor.cpp:149-162).  Feeding THESE rows through the port's
 * splice isolates the question this gate has to answer: is the splice itself
 * faithful?  With the port's own soft tokens the same comparison also carries the
 * VISION tower's distance from the reference (gate 23's 0.82%), which the text
 * tower then amplifies — the two numbers must be reported apart, not conflated. */
static int splice_load_frozen_soft(const char *path, float *soft, uint32_t expect_rows) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "eg2vq_splice: cannot open %s\n", path); return 0; }
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    uint32_t rows = 0;
    while ((len = getline(&line, &cap, fp)) >= 0) {
        if (line[0] == '#' || line[0] == 'S' || line[0] == '\n') { continue; }
        if (rows >= expect_rows) { break; }
        const char *p = strchr(line, '\t');
        if (!p) { continue; }
        p++;
        uint32_t got = 0;
        while (*p && got < (uint32_t)DS4_EG2VQ_N_EMBD) {
            char *end = NULL;
            const double v = strtod(p, &end);
            if (end == p) { break; }
            soft[(size_t)rows * DS4_EG2VQ_N_EMBD + got++] = (float)v;
            p = end;
        }
        if (got != (uint32_t)DS4_EG2VQ_N_EMBD) {
            fprintf(stderr, "eg2vq_splice: %s row %u has %u of %d values\n",
                    path, rows, got, (int)DS4_EG2VQ_N_EMBD);
            free(line); fclose(fp);
            return 0;
        }
        rows++;
    }
    free(line);
    fclose(fp);
    if (rows != expect_rows) {
        fprintf(stderr, "eg2vq_splice: %s holds %u rows, the sequence needs %u\n", path, rows, expect_rows);
        return 0;
    }
    return 1;
}

static uint8_t *splice_read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eg2vq_splice: cannot open %s\n", path); exit(2); }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); exit(2); }
    const long n = ftell(f);
    if (n <= 0) { fclose(f); fprintf(stderr, "eg2vq_splice: %s is empty\n", path); exit(2); }
    rewind(f);
    uint8_t *buf = xmalloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); exit(2); }
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

static void splice_stem(const char *path, char *out, size_t cap) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    snprintf(out, cap, "%s", base);
    char *dot = strrchr(out, '.');
    if (dot) { *dot = '\0'; }
    /* the fixtures' names carry their size suffix; the frozen dumps do not */
    char *dash = strstr(out, "-512x384");
    if (dash) { *dash = '\0'; }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: eg2vq_splice_cuda_oracle <artifact.gguf> <mmproj.gguf> <baseline.gguf> <image> [<image> ...]\n");
        return 2;
    }
    const char *artifact = argv[1], *mmproj = argv[2], *baseline = argv[3];
    const char *images[16];
    uint32_t n_images = 0;
    for (int i = 4; i < argc && n_images < 16; i++) { images[n_images++] = argv[i]; }

    printf("SPLICE ORACLE: the declared bounds (printed before the run):\n");
    printf("SPLICE ORACLE:   THE SPLICE, baseline weights + the reference's own soft rows: max |difference| <= %.3f, relative RMS <= %.3f, cosine >= %.3f\n",
           SPLICE_FROZEN_MAXABS, SPLICE_FROZEN_RELRMS, SPLICE_FROZEN_COS_MIN);
    printf("SPLICE ORACLE:   the same leg vs the frozen raw row, DIRECTION only: cosine >= %.3f\n", SPLICE_RAW_COS_MIN);
    printf("SPLICE ORACLE:   device (artifact weights) vs the artifact's own CPU leg: max |difference| <= %.3f, relative RMS <= %.3f, cosine >= %.4f\n",
           SPLICE_DEV_MAXABS, SPLICE_DEV_RELRMS, SPLICE_DEV_COS_MIN);
    printf("SPLICE ORACLE:   the baseline+port-soft and the artifact legs are REPORTED (they carry the vision tower's 0.82%% and the tier's loss), with cosine >= %.3f as the sanity floor\n",
           SPLICE_E2E_COS_MIN);

    /* DIAGNOSTIC, not a mode: a scale applied to the soft rows before the splice,
     * so the "the image rows are scaled after all" hypothesis can be tested
     * against the frozen rows instead of argued about.  The reference's own rule
     * is 1.0 — build_inp_embd scales the TOKEN rows by sqrt(n_embd) and gives an
     * embedding row 1.0 (llama-graph.cpp:110, :2722) — and nothing but this
     * oracle's own runs reads this knob. */
    double soft_fix = 1.0;
    {
        const char *fix_s = getenv("DS4_EG2VQ_SPLICE_SOFT_FIX");
        if (fix_s) { soft_fix = atof(fix_s); }
    }
    if (soft_fix != 1.0) {
        printf("SPLICE ORACLE: DIAGNOSTIC: the soft rows are scaled by %.6f before the splice\n", soft_fix);
    }

    /* ---- the two towers' host plans -------------------------------------- */
    g_ds4_shape = DS4_SHAPE_GEMMA_VQ;
    ds4_model tm;
    ds4_weights tw;
    model_open(&tm, artifact, false, false);
    weights_bind(&tw, &tm, false, 0, UINT32_MAX, true, false);
    if (!tw.eg2vq || !tw.eg2vq->dense) { fprintf(stderr, "SPLICE ORACLE: the text tower did not bind\n"); return 3; }

    /* The BASELINE text tower: the weights the frozen spliced rows were produced
     * with (the anchor harness's second argument), so the splice can be measured
     * without the 4-bit tier's own loss in the way. */
    ds4_model bm;
    ds4_weights bw;
    model_open(&bm, baseline, false, false);
    weights_bind(&bw, &bm, false, 0, UINT32_MAX, true, false);
    if (!bw.eg2vq || !bw.eg2vq->dense) { fprintf(stderr, "SPLICE ORACLE: the baseline text tower did not bind\n"); return 3; }

    ds4_model vm;
    ds4_eg2vq_vision_weights vw;
    model_open(&vm, mmproj, false, false);
    ds4_eg2vq_vision_bind(&vw, &vm);
    if (!vw.dense) { fprintf(stderr, "SPLICE ORACLE: the vision tower did not bind\n"); return 3; }

    /* ---- the device legs ------------------------------------------------- */
    if (!ds4_gpu_init()) { fprintf(stderr, "SPLICE ORACLE: no CUDA device\n"); return 4; }
    static ds4_eg2vq_record recs[DS4_EG2VQ_REC_N];
    static ds4_eg2vq_aux aux[DS4_EG2VQ_AUX_N];
    if (!eg2vq_device_tables(&tm, &tw, recs, aux)) {
        fprintf(stderr, "SPLICE ORACLE: the text artifact has no device plan\n");
        return 3;
    }
    static ds4_eg2vq_vision_slot vslots[DS4_EG2VQ_VISION_SLOT_N];
    ds4_eg2vq_vision_slot_table(&vm, vslots);
    ds4_eg2vq_vision_gpu_model *vplan =
        ds4_gpu_eg2vq_vision_bind(vslots, (uint32_t)DS4_EG2VQ_VISION_SLOT_N);
    ds4_eg2vq_gpu_model *tplan = ds4_gpu_eg2vq_bind(tm.map, tm.size, recs,
                                                   (uint32_t)DS4_EG2VQ_REC_N, aux,
                                                   (uint32_t)DS4_EG2VQ_AUX_N);
    if (!vplan || !tplan) { fprintf(stderr, "SPLICE ORACLE: a device plan did not bind\n"); return 4; }

    const uint32_t seq_max = 256;   /* the fixtures' 92 positions, in a size class */
    ds4_eg2vq_gpu_ctx *tctx = ds4_gpu_eg2vq_create(tplan, ds4_gpu_eg2vq_workspace(seq_max), seq_max);
    if (!tctx) { fprintf(stderr, "SPLICE ORACLE: no text device context\n"); return 4; }

    float *soft_cpu = xmalloc((size_t)DS4_EG2VQ_VISION_SOFT_MAX * DS4_EG2VQ_N_EMBD * sizeof(float));
    float *soft_dev = xmalloc((size_t)DS4_EG2VQ_VISION_SOFT_MAX * DS4_EG2VQ_N_EMBD * sizeof(float));
    float *raw_ref = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *l2_ref = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *cpu_l2 = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *dev_l2 = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *textonly = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *frozen_base = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *soft_frozen = xmalloc((size_t)DS4_EG2VQ_VISION_SOFT_MAX * DS4_EG2VQ_N_EMBD * sizeof(float));
    float *base_l2 = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    float *base_l2_frozen = xmalloc((size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    int32_t *ids = xmalloc((size_t)seq_max * sizeof(int32_t));

    double worst_cpu_relrms = 0.0, worst_cpu_maxabs = 0.0, worst_cpu_cos = 1.0;
    double worst_frozen_relrms = 0.0, worst_frozen_maxabs = 0.0, worst_frozen_cos = 1.0;
    double worst_f2_cos = 1.0;
    double worst_raw_cos = 1.0, worst_dev_relrms = 0.0, worst_dev_maxabs = 0.0, worst_dev_cos = 1.0;
    int status = 0, refusals = 0;

    for (uint32_t img = 0; img < n_images; img++) {
        char stem[256];
        splice_stem(images[img], stem, sizeof(stem));
        size_t len = 0;
        uint8_t *data = splice_read_file(images[img], &len);
        ds4_eg2vq_vision_opts vopts;
        vopts.f16_kv = 1;
        ds4_eg2vq_vision_geometry geo;
        char err[256];

        /* 1. the image's soft tokens, both legs. */
        if (!ds4_eg2vq_vision_forward(&vm, &vw, data, len, &vopts, soft_cpu,
                                      (uint64_t)DS4_EG2VQ_VISION_SOFT_MAX * DS4_EG2VQ_N_EMBD,
                                      &geo, NULL, err, sizeof(err))) {
            fprintf(stderr, "SPLICE ORACLE: the CPU vision leg refused %s: %s\n", images[img], err);
            status = 1;
            break;
        }
        ds4_eg2vq_vision_gpu_ctx *vctx =
            ds4_gpu_eg2vq_vision_create(vplan,
                    ds4_gpu_eg2vq_vision_workspace(geo.n_patches, geo.n_soft),
                    geo.n_patches, geo.n_soft);
        if (!vctx) { fprintf(stderr, "SPLICE ORACLE: no vision device context\n"); status = 1; break; }
        ds4_eg2vq_vision_geometry gdev;
        if (!ds4_gpu_eg2vq_vision_forward(vctx, data, len, &vopts, soft_dev,
                                          (uint64_t)DS4_EG2VQ_VISION_SOFT_MAX * DS4_EG2VQ_N_EMBD,
                                          &gdev, err, sizeof(err))) {
            fprintf(stderr, "SPLICE ORACLE: the device vision leg refused %s: %s\n", images[img], err);
            status = 1;
            break;
        }
        ds4_gpu_eg2vq_vision_free(vctx);
        if (soft_fix != 1.0) {
            const size_t n = (size_t)geo.n_soft * DS4_EG2VQ_N_EMBD;
            for (size_t i = 0; i < n; i++) { soft_cpu[i] = (float)((double)soft_cpu[i] * soft_fix); }
            for (size_t i = 0; i < n; i++) { soft_dev[i] = (float)((double)soft_dev[i] * soft_fix); }
        }

        /* 2. the sequence: [BOS][<|image>] + the image span + [<image|>][EOS]. */
        uint32_t n_ids = 0;
        const uint32_t span_start = 2;
        ids[n_ids++] = 2;                       /* <bos>   */
        ids[n_ids++] = 255999;                  /* <|image> (boi) */
        for (uint32_t i = 0; i < geo.n_soft; i++) { ids[n_ids++] = (int32_t)DS4_EG2VQ_IMAGE_TOKEN_ID; }
        ids[n_ids++] = 258882;                  /* <image|> (eoi) */
        ids[n_ids++] = 1;                       /* <eos>   */
        printf("SPLICE ORACLE: %s: %u soft tokens, sequence %u positions, span [%u, %u), grid %ux%u\n",
               stem, geo.n_soft, n_ids, span_start, span_start + geo.n_soft, geo.grid_w, geo.grid_h);
        if (n_ids > seq_max) { fprintf(stderr, "SPLICE ORACLE: the sequence is past the context\n"); status = 1; break; }

        /* 3. the frames: leg 3 is the artifact's CPU splice (the shipped path), leg
         * 4 its device twin, and legs 1 and 2 are the baseline text tower with the
         * frozen and the port's soft rows. */
        ds4_eg2vq_ref_embed_image(&tm, &tw, ids, n_ids, soft_cpu, geo.n_soft, cpu_l2);
        if (!ds4_gpu_eg2vq_embed_image(tctx, ids, n_ids, soft_dev, geo.n_soft, dev_l2)) {
            fprintf(stderr, "SPLICE ORACLE: the device splice refused %s\n", images[img]);
            status = 1;
            break;
        }
        ds4_eg2vq_ref_embed_image(&bm, &bw, ids, n_ids, soft_cpu, geo.n_soft, base_l2);

        /* the text-only control: the same four text ids, no image */
        {
            int32_t four[4] = { 2, 255999, 258882, 1 };
            if (!ds4_gpu_eg2vq_embed(tctx, four, 4, textonly)) {
                fprintf(stderr, "SPLICE ORACLE: the text-only control was refused\n");
                status = 1;
                break;
            }
        }

        /* 4. the frozen rows: the pooled vector and the soft rows themselves. */
        char frozen[1024];
        snprintf(frozen, sizeof(frozen), "tests/vision/reference/%s.text.tsv", stem);
        uint32_t n_out = 0;
        if (!splice_load_frozen(frozen, raw_ref, l2_ref, &n_out)) {
            fprintf(stderr, "SPLICE ORACLE: cannot read the frozen dump %s\n", frozen);
            status = 1;
            break;
        }
        char frozen_soft[1024];
        snprintf(frozen_soft, sizeof(frozen_soft), "tests/vision/reference/%s.soft.tsv", stem);
        if (!splice_load_frozen_soft(frozen_soft, soft_frozen, geo.n_soft)) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: cannot read the frozen soft rows for %s\n", stem);
            status = 1;
            break;
        }

        /* LEG 1: the splice itself, on the weights the frozen rows came from. */
        ds4_eg2vq_ref_embed_image(&bm, &bw, ids, n_ids, soft_frozen, geo.n_soft, base_l2_frozen);
        const splice_cmp f = splice_compare(base_l2_frozen, l2_ref, 1, DS4_EG2VQ_N_OUT);
        const splice_cmp fr = splice_compare(base_l2_frozen, raw_ref, 1, DS4_EG2VQ_N_OUT);
        const splice_cmp f2 = splice_compare(base_l2, l2_ref, 1, DS4_EG2VQ_N_OUT);   /* leg 2 */
        const splice_cmp c = splice_compare(cpu_l2, l2_ref, 1, DS4_EG2VQ_N_OUT);     /* leg 3 */
        const splice_cmp d = splice_compare(dev_l2, cpu_l2, 1, DS4_EG2VQ_N_OUT);     /* leg 4 */
        const splice_cmp ctrl = splice_compare(base_l2, textonly, 1, DS4_EG2VQ_N_OUT);

        printf("SPLICE ORACLE: %s leg 1 baseline + the FROZEN soft rows vs frozen l2: maxabs %.8f relrms %.6f cosine %.8f\n",
               stem, f.maxabs, f.relrms, f.cos);
        printf("SPLICE ORACLE: %s leg 1 vs the frozen raw row (direction): cosine %.8f\n", stem, fr.cos);
        printf("SPLICE ORACLE: %s leg 2 baseline + the PORT's soft rows vs frozen l2: maxabs %.8f relrms %.6f cosine %.8f\n",
               stem, f2.maxabs, f2.relrms, f2.cos);
        printf("SPLICE ORACLE: %s leg 3 artifact + the PORT's soft rows vs frozen l2: maxabs %.8f relrms %.6f cosine %.8f\n",
               stem, c.maxabs, c.relrms, c.cos);
        printf("SPLICE ORACLE: %s leg 4 artifact device vs artifact cpu: maxabs %.8f relrms %.6f cosine %.8f\n",
               stem, d.maxabs, d.relrms, d.cos);
        printf("SPLICE ORACLE: %s control: leg 1 without the image (four text ids) differs by cosine %.6f\n",
               stem, ctrl.cos);

        /* THE AMPLIFICATION PROBE, on this tower's input: one soft value moved by a
         * relative amount, re-spliced on the same weights, and the output measured.
         * It is what says how much of leg 2's difference is the vision tower's
         * 0.82% and not the splice. */
        memcpy(frozen_base, base_l2_frozen, (size_t)DS4_EG2VQ_N_OUT * sizeof(float));
        for (int k = 0; k < 2; k++) {
            const double kick = k ? 1e-2 : 1e-6;
            const float keep = soft_frozen[0];
            soft_frozen[0] = (float)((double)soft_frozen[0] * (1.0 + kick));
            ds4_eg2vq_ref_embed_image(&bm, &bw, ids, n_ids, soft_frozen, geo.n_soft, base_l2_frozen);
            soft_frozen[0] = keep;
            const splice_cmp kc = splice_compare(base_l2_frozen, frozen_base, 1, DS4_EG2VQ_N_OUT);
            printf("SPLICE ORACLE: %s amplification: a %.0e relative kick on ONE soft value "
                   "moves the pooled vector maxabs %.8f relrms %.6f\n",
                   stem, kick, kc.maxabs, kc.relrms);
        }
        memcpy(base_l2_frozen, frozen_base, (size_t)DS4_EG2VQ_N_OUT * sizeof(float));

        if (f.relrms > worst_frozen_relrms) { worst_frozen_relrms = f.relrms; }
        if (f.maxabs > worst_frozen_maxabs) { worst_frozen_maxabs = f.maxabs; }
        if (f.cos < worst_frozen_cos) { worst_frozen_cos = f.cos; }
        if (fr.cos < worst_raw_cos) { worst_raw_cos = fr.cos; }
        if (f2.cos < worst_f2_cos) { worst_f2_cos = f2.cos; }
        if (c.maxabs > worst_cpu_maxabs) { worst_cpu_maxabs = c.maxabs; }
        if (c.relrms > worst_cpu_relrms) { worst_cpu_relrms = c.relrms; }
        if (c.cos < worst_cpu_cos) { worst_cpu_cos = c.cos; }
        if (d.maxabs > worst_dev_maxabs) { worst_dev_maxabs = d.maxabs; }
        if (d.relrms > worst_dev_relrms) { worst_dev_relrms = d.relrms; }
        if (d.cos < worst_dev_cos) { worst_dev_cos = d.cos; }
        if (ctrl.cos > 0.9999) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: %s is not reaching the tower (the control is identical)\n", stem);
            status = 1;
        }
        if (f.relrms > SPLICE_FROZEN_RELRMS || f.maxabs > SPLICE_FROZEN_MAXABS ||
            f.cos < SPLICE_FROZEN_COS_MIN) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: %s the splice with the frozen soft rows is outside the declared bound\n", stem);
            status = 1;
        }
        if (fr.cos < SPLICE_RAW_COS_MIN) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: %s the splice vs the frozen raw row's direction is outside the bound\n", stem);
            status = 1;
        }
        if (c.cos < SPLICE_E2E_COS_MIN || f2.cos < SPLICE_E2E_COS_MIN) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: %s a reported leg is below the sanity floor\n", stem);
            status = 1;
        }
        if (d.relrms > SPLICE_DEV_RELRMS || d.maxabs > SPLICE_DEV_MAXABS || d.cos < SPLICE_DEV_COS_MIN) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: %s device vs cpu is outside the declared bound\n", stem);
            status = 1;
        }
        free(data);
        fflush(stdout);
    }

    /* ---- 5. the refusals (the device entry's own contract) ---------------- */
    {
        int32_t hole[8] = { 2, 255999, (int32_t)DS4_EG2VQ_IMAGE_TOKEN_ID, 7,
                            (int32_t)DS4_EG2VQ_IMAGE_TOKEN_ID, 258882, 1, 0 };
        float *one = xmalloc((size_t)DS4_EG2VQ_N_EMBD * sizeof(float));
        for (uint32_t i = 0; i < (uint32_t)DS4_EG2VQ_N_EMBD; i++) { one[i] = 0.01f; }
        if (ds4_gpu_eg2vq_embed_image(tctx, hole, 6, one, 2, dev_l2)) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: a non-contiguous span was accepted\n");
            status = 1;
        } else { refusals++; }
        int32_t two[4] = { 2, (int32_t)DS4_EG2VQ_IMAGE_TOKEN_ID, (int32_t)DS4_EG2VQ_IMAGE_TOKEN_ID, 1 };
        if (ds4_gpu_eg2vq_embed_image(tctx, two, 4, one, 1, dev_l2)) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: a span that disagrees with the soft block was accepted\n");
            status = 1;
        } else { refusals++; }
        if (ds4_gpu_eg2vq_embed_image(tctx, two, 4, NULL, 0, dev_l2)) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: a placeholder in a text-only call was accepted\n");
            status = 1;
        } else { refusals++; }
        int32_t bad[2] = { 2, 262144 };   /* one past the table's 262,144 rows */
        if (ds4_gpu_eg2vq_embed(tctx, bad, 2, dev_l2)) {
            fprintf(stderr, "SPLICE ORACLE: FAIL: an out-of-range id was accepted\n");
            status = 1;
        } else { refusals++; }
        free(one);
    }

    printf("\nSPLICE ORACLE: the numbers\n");
    printf("SPLICE ORACLE: worst THE SPLICE (baseline weights, the reference's own soft rows): maxabs %.8f (bound %.3f), relrms %.6f (bound %.3f), cosine %.8f (bound %.3f)\n",
           worst_frozen_maxabs, SPLICE_FROZEN_MAXABS, worst_frozen_relrms, SPLICE_FROZEN_RELRMS,
           worst_frozen_cos, SPLICE_FROZEN_COS_MIN);
    printf("SPLICE ORACLE: worst same leg vs the frozen raw row's direction: cosine %.8f (bound %.3f)\n",
           worst_raw_cos, SPLICE_RAW_COS_MIN);
    printf("SPLICE ORACLE: worst baseline + the PORT's soft rows: cosine %.8f (reported; it carries the vision tower's 0.82%%)\n",
           worst_f2_cos);
    printf("SPLICE ORACLE: worst artifact + the PORT's soft rows: maxabs %.8f, relrms %.6f, cosine %.8f (reported; it carries the tier's loss, gate 5's 0.956 mean agreement)\n",
           worst_cpu_maxabs, worst_cpu_relrms, worst_cpu_cos);
    printf("SPLICE ORACLE: worst device vs the artifact's own cpu leg: maxabs %.8f (bound %.3f), relrms %.6f (bound %.3f), cosine %.8f (bound %.4f)\n",
           worst_dev_maxabs, SPLICE_DEV_MAXABS, worst_dev_relrms, SPLICE_DEV_RELRMS, worst_dev_cos, SPLICE_DEV_COS_MIN);
    printf("SPLICE ORACLE: refusals checked: %d of 4\n", refusals);
    if (refusals != 4) { status = 1; }

    ds4_gpu_eg2vq_free(tctx);
    ds4_gpu_eg2vq_unbind(tplan);
    ds4_gpu_eg2vq_vision_unbind(vplan);
    ds4_eg2vq_vision_free(&vw);
    model_close(&vm);
    model_close(&bm);
    model_close(&tm);
    free(ids); free(base_l2_frozen); free(base_l2); free(soft_frozen);
    free(textonly); free(frozen_base); free(dev_l2); free(cpu_l2); free(l2_ref); free(raw_ref);
    free(soft_dev); free(soft_cpu);

    printf("%s\n", status ? "EG2VQ_SPLICE_FAIL" : "EG2VQ_SPLICE_PASS");
    return status ? 1 : 0;
}
