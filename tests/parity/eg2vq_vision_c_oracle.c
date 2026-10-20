/* eg2vq_vision_c_oracle.c — the port's own C oracle for EmbeddingGemma-2's
 * IMAGE path (phase 8, unit 23).  It includes ds4.c with DS4_NO_GPU, exactly
 * like eg2vq_embd_c_oracle.c, so the vision forward in
 * ds4_eg2vq_vision_graph.inc runs with no CUDA in the path: the port links
 * nothing of the pin, and the numbers it prints are its own.
 *
 * Two modes, both native:
 *
 *   prep <image> [f32kv]
 *     the preprocessing alone, with NO weights: the geometry the size solver
 *     and the PAD_CEIL composite produced, the two position index vectors, and
 *     two run-length summaries that show the padding structurally (per column
 *     and per row, how many channel values are not the pad value).  This is
 *     what gate 23 checks BEFORE any matmul.
 *
 *   soft <mmproj.gguf> <image> [f32kv]
 *     the whole tower: the mmproj opened as its own model (the family's
 *     companion-GGUF pattern), bound by name, and the soft tokens printed in
 *     the frozen dumps' format (tests/vision/reference, eg2-vision-*.soft.tsv):
 *         <row> <TAB> v0 <TAB> v1 ... <TAB> v511     (%.7f each)
 *     plus, on stdout, the geometry and the sum of squares the anchor's own
 *     log prints for the same fixture.
 *
 * `f32kv` is the ATTRIBUTION switch, not a mode: it runs the graph's attention
 * in f32 instead of mirroring the reference's executed flash-attention f16
 * rounding of K and V (clip.cpp:787-790).  Gate 23 prints both.
 */

#include "../../ds4.c"

static void die_usage(void) {
    fprintf(stderr,
            "usage: eg2vq_vision_c_oracle prep <image> [f32kv]\n"
            "       eg2vq_vision_c_oracle soft <mmproj.gguf> <image> [f32kv]\n");
    exit(2);
}

/* A whole file, or a hard stop: the decode below is the reference's own
 * stbi_load_from_memory over the exact bytes on disk. */
static uint8_t *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eg2vq_vision: cannot open %s\n", path); exit(2); }
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "eg2vq_vision: cannot seek %s\n", path); exit(2); }
    const long n = ftell(f);
    if (n <= 0) { fprintf(stderr, "eg2vq_vision: %s is empty\n", path); exit(2); }
    rewind(f);
    uint8_t *buf = xmalloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "eg2vq_vision: short read on %s\n", path);
        exit(2);
    }
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

/* `label` <TAB> v*count <TAB> v*count ... over `n` values. */
static void print_rle(const char *label, const uint64_t *v, uint32_t n) {
    printf("%s", label);
    uint32_t i = 0;
    while (i < n) {
        uint32_t j = i + 1;
        while (j < n && v[j] == v[i]) { j++; }
        printf(" %llu*%u", (unsigned long long)v[i], j - i);
        i = j;
    }
    printf("\n");
}

/* The preprocessing-only mode: geometry, structural padding, positions. */
static int run_prep(const char *image, int f16_kv) {
    size_t len = 0;
    uint8_t *data = read_file(image, &len);
    uint32_t *pos_x = xmalloc((size_t)DS4_EG2VQ_VISION_MAX_PATCHES * sizeof(uint32_t));
    uint32_t *pos_y = xmalloc((size_t)DS4_EG2VQ_VISION_MAX_PATCHES * sizeof(uint32_t));
    float *canvas = xmalloc((size_t)DS4_EG2VQ_VISION_MAX_PIXELS * 3u * sizeof(float));
    ds4_eg2vq_vision_geometry g;
    char err[256];
    const double t0 = now_sec();
    if (!ds4_eg2vq_vision_prepare(data, len, &g, pos_x, pos_y, canvas,
                                 (uint64_t)DS4_EG2VQ_VISION_MAX_PIXELS * 3u,
                                 err, sizeof(err))) {
        fprintf(stderr, "eg2vq_vision: the preprocessing refused the image: %s\n", err);
        return 1;
    }
    printf("PREP canvas=%ux%u region=%ux%u offset=%d,%d grid=%ux%u patches=%u soft=%u\n",
           g.canvas_w, g.canvas_h, g.region_w, g.region_h,
           g.offset_x, g.offset_y, g.grid_w, g.grid_h, g.n_patches, g.n_soft);

    /* The padding, structurally and exactly: per column and per row, whether
     * EVERY channel value in it is the pad value (-1.0f).  A real image cannot
     * fill a whole 384-pixel column with exactly the pad, so 1 means "outside
     * the composited region" without any tolerance.  For the frozen fixtures
     * the columns are 1*8 0*512 1*8 and every row is 0. */
    uint64_t *cols = xmalloc((size_t)g.canvas_w * sizeof(uint64_t));
    uint64_t *rows = xmalloc((size_t)g.canvas_h * sizeof(uint64_t));
    for (uint32_t x = 0; x < g.canvas_w; x++) { cols[x] = 1; }
    for (uint32_t y = 0; y < g.canvas_h; y++) { rows[y] = 1; }
    for (uint32_t y = 0; y < g.canvas_h; y++) {
        for (uint32_t x = 0; x < g.canvas_w; x++) {
            for (uint32_t c = 0; c < 3u; c++) {
                const float v = canvas[(uint64_t)x + (uint64_t)g.canvas_w *
                        ((uint64_t)y + (uint64_t)g.canvas_h * c)];
                if (v != -1.0f) { cols[x] = 0; rows[y] = 0; }
            }
        }
    }
    print_rle("COLS", cols, g.canvas_w);
    print_rle("ROWS", rows, g.canvas_h);

    printf("POS_X");
    for (uint32_t i = 0; i < g.n_patches; i++) { printf(" %u", pos_x[i]); }
    printf("\nPOS_Y");
    for (uint32_t i = 0; i < g.n_patches; i++) { printf(" %u", pos_y[i]); }
    printf("\nPREPSECS %.3f f16kv=%d\n", now_sec() - t0, f16_kv);

    free(canvas); free(pos_y); free(pos_x); free(data);
    return 0;
}

static int run_soft(const char *mmproj, const char *image, int f16_kv) {
    /* The shape the artifact's own family declares: the vision binder names
     * every tensor explicitly, but the catalog and the dispatch read the same
     * table the text oracle sets. */
    g_ds4_shape = DS4_SHAPE_GEMMA_VQ;

    size_t len = 0;
    uint8_t *data = read_file(image, &len);

    const double t0 = now_sec();
    ds4_model mm;
    model_open(&mm, mmproj, false, false);
    ds4_eg2vq_vision_weights vw;
    ds4_eg2vq_vision_bind(&vw, &mm);
    const double t_bind = now_sec();

    float *soft = xmalloc((size_t)DS4_EG2VQ_VISION_SOFT_MAX
                        * DS4_EG2VQ_VISION_N_OUT * sizeof(float));
    ds4_eg2vq_vision_opts opts;
    opts.f16_kv = f16_kv;
    ds4_eg2vq_vision_geometry g;
    char err[256];
    if (!ds4_eg2vq_vision_forward(&mm, &vw, data, len, &opts, soft,
                                 (uint64_t)DS4_EG2VQ_VISION_SOFT_MAX
                                     * DS4_EG2VQ_VISION_N_OUT,
                                 &g, NULL, err, sizeof(err))) {
        fprintf(stderr, "eg2vq_vision: the vision forward refused the image: %s\n", err);
        return 1;
    }

    /* The frozen format, byte for byte: the index, then TAB %.7f per value. */
    printf("# %s: ds4-rs ds4_eg2vq_vision_forward, %u soft tokens x %u dims, 7 decimals\n",
           image, g.n_soft, (unsigned)DS4_EG2VQ_VISION_N_OUT);
    double sumsq = 0.0;
    for (uint32_t t = 0; t < g.n_soft; t++) {
        printf("%u", t);
        for (uint32_t d = 0; d < (uint32_t)DS4_EG2VQ_VISION_N_OUT; d++) {
            const float v = soft[(uint64_t)t * DS4_EG2VQ_VISION_N_OUT + d];
            printf("\t%.7f", (double)v);
            sumsq += (double)v * (double)v;
        }
        printf("\n");
    }
    printf("SOFT canvas=%ux%u region=%ux%u offset=%d,%d grid=%ux%u patches=%u soft=%u sumsq=%.6f\n",
           g.canvas_w, g.canvas_h, g.region_w, g.region_h, g.offset_x, g.offset_y,
           g.grid_w, g.grid_h, g.n_patches, g.n_soft, sumsq);
    fprintf(stderr, "eg2vq_vision: bind %.2fs tower %.2fs dense_mb=%.1f f16kv=%d\n",
            t_bind - t0, now_sec() - t_bind,
            (double)vw.dense_bytes / 1048576.0, f16_kv);

    ds4_eg2vq_vision_free(&vw);
    model_close(&mm);
    free(soft); free(data);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { die_usage(); }
    /* `f32kv` is the last argument in either mode, and the two modes have
     * different arities: prep <image> [f32kv] and soft <mmproj> <image> [f32kv]. */
    const int f32kv = (strcmp(argv[argc - 1], "f32kv") == 0);
    if (strcmp(argv[1], "prep") == 0) {
        if (argc != 3 && argc != 4) { die_usage(); }
        return run_prep(argv[2], !f32kv);
    }
    if (strcmp(argv[1], "soft") == 0) {
        if (argc != 4 && argc != 5) { die_usage(); }
        return run_soft(argv[2], argv[3], !f32kv);
    }
    die_usage();
    return 2;
}
