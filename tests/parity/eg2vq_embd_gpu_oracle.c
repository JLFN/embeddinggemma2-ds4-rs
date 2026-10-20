/* P3.3 step 4 -- the gemma_vq text tower end to end on this card, against the
 * CPU reference in the same process.
 *
 * The plan of record (embeddinggemma2-handoff.md PHASE 7, docs/eg2vq-device-gate7.txt
 * section 8) asks for three things from this gate, and this file is all three:
 *
 *   1. parity: the device tower's 768-dim embedding equals the CPU reference
 *      path's on the same item ids.  The CPU leg is the tree's own -O0 oracle
 *      entry (`ds4_eg2vq_ref_embed`, built here with the same flags as
 *      tests/parity/eg2vq_embd_c_oracle), so the comparison is device vs the
 *      arithmetic every earlier gate was measured against, not device vs a
 *      re-derived one.  With --ref the same vectors are also checked against a
 *      committed raw from an earlier CPU run, item by item.
 *   2. tokens/s: the wall time of each leg over the same ids, printed per item
 *      and as a total, so the "it should be much faster" premise is a measured
 *      number rather than an expectation.
 *   3. VRAM: the device footprint of the plan and the workspace, from
 *      cudaMemGetInfo around the bind and the context create.
 *
 * The parity bound is declared here, before any run, and is not tuned after one:
 * a unit-norm 768-vector pair at relative RMS 1e-3 has cosine 0.9999995, so the
 * two bounds below are the same statement twice, at the strength the phase-4
 * gate already reached between two independent implementations (768 mean cosine
 * 0.999996809 against llama.cpp's anchor).
 *
 *   usage: eg2vq_embd_gpu_oracle <artifact.gguf> <ids.tsv>
 *              [--out <device-raw.tsv>] [--ref <cpu-raw.tsv>]
 *              [--items N] [--max-tokens N] [--batch] [--batch-synth N]
 *
 * <ids.tsv> is the Rust host's own per-item token dump
 * (crates/ds4-core/examples/gemma_vq_tok.rs): <item> <TAB> <prompt> <TAB> <ntok>
 * <TAB> <id,id,...>, already carrying BOS and EOS.  --out writes the device leg
 * in the oracle TSV shape (<item> <TAB> <ntok> <TAB> <768 values>), so
 * tools/eg2_retrieval_score.sh can score it exactly as it scores the CPU legs.
 *
 * One device plan and one workspace serve every item: the payloads and the
 * weights cross PCIe once, at bind, and the per-item traffic is the ids up and
 * the 768-vector down.
 *
 * UNIT 27 adds the batch legs: --batch runs the same items as ONE packed call
 * (the item spans from the file's own order) and compares every element against
 * the per-item vectors above, printing one call's wall time against the per-item
 * sum; --batch-synth N runs one synthetic call of exactly N rows so the row
 * budget's boundary (8192 rows accepted, 8193 refused) is a measured line rather
 * than an arithmetic claim.  A refused synthetic batch exits 9.  --batch-bad runs
 * the malformed span tables (an empty item, a table that does not cover the rows,
 * zero items, an item past the token ceiling) and exits 9 if any was accepted;
 * --no-cpu skips the per-item CPU leg, which the batch legs do not need.
 */
#include "../../ds4.c"

/* This build carries the accelerator path (ds4.c includes ds4_gpu.h itself), so
 * the harness drives the device adapter through the tree's own entry points:
 * ds4_gpu_init to bring the backend up, and cuda_runtime.h for the VRAM probe.
 * Nothing in ds4.c reaches the GPU until the calls below. */
#include <cuda_runtime.h>

/* Declared before the first run, not tuned after one. */
static const double EG2VQ_GPU_REL_RMS_MAX = 1e-3;
static const double EG2VQ_GPU_COS_MIN = 0.999999;

typedef struct {
    char *id;
    unsigned long ntok;
    float v[DS4_EG2VQ_N_OUT];
} eg2vq_ref_item;

/* Unit 27: one item kept for the batch leg -- its ids, and the vector the
 * per-item device call produced for it (the thing the batch must reproduce). */
typedef struct {
    int32_t *ids;
    unsigned long ntok;
    float v[DS4_EG2VQ_N_OUT];
} eg2vq_kept_item;

typedef struct {
    unsigned long items;
    unsigned long rows;
    unsigned long run;      /* the run of items with one length (for --batch-synth) */
    double batch_sec;
    int ok;                 /* the device batch entry's own verdict */
    int identical;          /* every element of every item byte-identical to its per-item run */
    double max_delta;       /* the largest per-element difference seen */
} eg2vq_batch_leg;

static void usage(void) {
    fprintf(stderr, "usage: eg2vq_embd_gpu_oracle <artifact.gguf> <ids.tsv> "
                    "[--out <device-raw.tsv>] [--out-cpu <cpu-raw.tsv>] [--ref <cpu-raw.tsv>] "
                    "[--items N] [--skip N] [--max-tokens N] [--batch] [--batch-synth N] "
                    "[--batch-bad] [--no-cpu]\n"
                    "  --batch           also run the same items as ONE packed call and "
                    "compare it against the per-item vectors\n"
                    "  --batch-synth N   run one synthetic call of exactly N rows (the row "
                    "budget's boundary); exits 9 if it is refused\n"
                    "  --batch-bad       run the malformed span tables; exits 9 if any is accepted\n"
                    "  --no-cpu          skip the per-item CPU leg (the batch legs need none)\n");
    exit(2);
}

static char *eg2vq_dup(const char *s) {
    const size_t n = strlen(s) + 1;
    char *p = xmalloc(n);
    memcpy(p, s, n);
    return p;
}

/* Comma-separated ids, exactly as the CPU oracle parses them (the two harnesses
 * must read one ids file identically or the comparison would compare items). */
static int parse_ids(const char *text, int32_t *ids, unsigned long ntok, long n_vocab) {
    unsigned long count = 0;
    const char *p = text;
    while (*p) {
        char *end = NULL;
        const long id = strtol(p, &end, 10);
        if (end == p || id < 0 || id > INT32_MAX || id >= n_vocab) { return -1; }
        if (count >= ntok) { return -1; }
        ids[count++] = (int32_t)id;
        p = end;
        if (*p == ',') { p++; }
        else if (*p != '\0') { return -1; }
    }
    return count == ntok ? 0 : -1;
}

/* A raw TSV from the CPU oracle: <id> <TAB> <ntok> <TAB> <768 values>. */
static size_t load_ref(const char *path, eg2vq_ref_item **out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "eg2vq_embd_gpu: cannot open %s\n", path); return 0; }
    size_t n = 0, cap = 0;
    eg2vq_ref_item *items = NULL;
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t len;
    while ((len = getline(&line, &line_cap, fp)) >= 0) {
        char *item = line;
        char *s2 = strchr(item, '\t');
        if (!s2) { continue; }
        *s2++ = '\0';
        char *s3 = strchr(s2, '\t');
        if (!s3) { continue; }
        *s3++ = '\0';
        if (n == cap) {
            cap = cap ? cap * 2 : 256;
            items = xrealloc(items, cap * sizeof(*items));
        }
        eg2vq_ref_item *it = &items[n];
        it->id = eg2vq_dup(item);
        it->ntok = strtoul(s2, NULL, 10);
        size_t got = 0;
        const char *p = s3;
        while (*p && got < (size_t)DS4_EG2VQ_N_OUT) {
            char *end = NULL;
            const double v = strtod(p, &end);
            if (end == p) { break; }
            it->v[got++] = (float)v;
            p = end;
        }
        if (got != (size_t)DS4_EG2VQ_N_OUT) {
            /* A raw that was stopped mid-write ends in a partial line; the items
             * before it are complete and still compare. */
            fprintf(stderr, "eg2vq_embd_gpu: %s line %zu has %zu of %d values; the %zu items before it are compared\n",
                    path, n + 1, got, (int)DS4_EG2VQ_N_OUT, n);
            free(it->id);
            break;
        }
        n++;
    }
    free(line);
    fclose(fp);
    *out = items;
    return n;
}

static void compare(const float *a, const float *b, double *cos, double *rel) {
    double dot = 0.0, na = 0.0, nb = 0.0, d2 = 0.0;
    for (uint32_t i = 0; i < (uint32_t)DS4_EG2VQ_N_OUT; i++) {
        const double x = (double)a[i], y = (double)b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        d2 += (x - y) * (x - y);
    }
    *cos = (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 0.0;
    *rel = na > 0.0 ? sqrt(d2 / na) : 0.0;
}

/* Unit 27: one batch call over `n_items` items packed contiguously, against the
 * per-item vectors the caller already has.  `off` is the item spans, `out_i`
 * item i's per-item vector, `ids` the packed ids.  Returns what the call did;
 * the workspace is sized ONCE for the whole block, which is the point. */
static int eg2vq_batch_run(ds4_eg2vq_gpu_model *plan, const int32_t *ids,
                           const uint32_t *off, uint32_t n_items, uint32_t n_rows,
                           const eg2vq_kept_item *items, eg2vq_batch_leg *leg) {
    const uint32_t want = (n_rows + 255u) & ~255u;
    const uint64_t scratch = ds4_gpu_eg2vq_workspace(want);
    ds4_eg2vq_gpu_ctx *bctx = ds4_gpu_eg2vq_create(plan, scratch, want);
    leg->items = n_items;
    leg->rows = n_rows;
    leg->run = 0;
    leg->batch_sec = 0.0;
    leg->ok = 0;
    leg->identical = 0;
    leg->max_delta = 0.0;
    if (!bctx) { return 0; }                      /* the create refusal is its own report */
    float *out = xmalloc((size_t)n_items * (size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    const double t0 = now_sec();
    const int ok = ds4_gpu_eg2vq_embed_batch(bctx, ids, off, n_items, n_rows, out);
    const double secs = now_sec() - t0;
    if (!ok) { free(out); ds4_gpu_eg2vq_free(bctx); return 0; }
    double max_delta = 0.0;
    int identical = 1;
    for (uint32_t i = 0; i < n_items; i++) {
        for (uint32_t o = 0; o < (uint32_t)DS4_EG2VQ_N_OUT; o++) {
            const float a = out[(size_t)i * DS4_EG2VQ_N_OUT + o];
            const float b = items[i].v[o];
            const double d = fabs((double)a - (double)b);
            if (d > max_delta) { max_delta = d; }
            if (memcmp(&a, &b, sizeof a) != 0) { identical = 0; }
        }
    }
    free(out);
    ds4_gpu_eg2vq_free(bctx);
    leg->ok = 1;
    leg->identical = identical;
    leg->max_delta = max_delta;
    leg->batch_sec = secs;
    return 1;
}

int main(int argc, char **argv) {
    const char *gguf = NULL, *ids_path = NULL, *out_path = NULL, *out_cpu_path = NULL, *ref_path = NULL;
    unsigned long items_max = 0, skip = 0, max_tokens = 2048;
    int batch = 0, batch_bad = 0, no_cpu = 0;
    unsigned long batch_synth = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) { out_path = argv[++i]; }
        else if (!strcmp(argv[i], "--out-cpu") && i + 1 < argc) { out_cpu_path = argv[++i]; }
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) { ref_path = argv[++i]; }
        else if (!strcmp(argv[i], "--items") && i + 1 < argc) { items_max = strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) { skip = strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) { max_tokens = strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--batch")) { batch = 1; }
        else if (!strcmp(argv[i], "--batch-synth") && i + 1 < argc) { batch_synth = strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--batch-bad")) { batch_bad = 1; }
        else if (!strcmp(argv[i], "--no-cpu")) { no_cpu = 1; }
        else if (!gguf) { gguf = argv[i]; }
        else if (!ids_path) { ids_path = argv[i]; }
        else { usage(); }
    }
    if (!gguf || !ids_path) { usage(); }

    eg2vq_ref_item *ref = NULL;
    size_t n_ref = 0;
    if (ref_path) { n_ref = load_ref(ref_path, &ref); }
    FILE *out_fp = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out_fp) { fprintf(stderr, "eg2vq_embd_gpu: cannot write %s\n", out_path); return 2; }
    FILE *out_cpu_fp = out_cpu_path ? fopen(out_cpu_path, "w") : NULL;
    if (out_cpu_path && !out_cpu_fp) { fprintf(stderr, "eg2vq_embd_gpu: cannot write %s\n", out_cpu_path); return 2; }

    g_ds4_shape = DS4_SHAPE_GEMMA_VQ;
    ds4_model model;
    ds4_weights weights;
    const double t_open = now_sec();
    model_open(&model, gguf, false, false);
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    const double open_sec = now_sec() - t_open;
    if (!weights.eg2vq || !weights.eg2vq->dense) {
        fprintf(stderr, "eg2vq_embd_gpu: the text tower did not bind\n");
        return 3;
    }

    static ds4_eg2vq_record records[DS4_EG2VQ_REC_N];
    static ds4_eg2vq_aux aux[DS4_EG2VQ_AUX_N];
    if (!eg2vq_device_tables(&model, &weights, records, aux)) {
        fprintf(stderr, "eg2vq_embd_gpu: this file has no device plan (the device path is the VQ artifact's)\n");
        return 3;
    }

    if (!ds4_gpu_init()) { fprintf(stderr, "eg2vq_embd_gpu: no CUDA device\n"); return 4; }
    size_t free0 = 0, total0 = 0;
    if (cudaMemGetInfo(&free0, &total0) != cudaSuccess) { free0 = 0; }
    const double t_bind = now_sec();
    ds4_eg2vq_gpu_model *plan = ds4_gpu_eg2vq_bind(model.map, model.size, records,
                                                  (uint32_t)DS4_EG2VQ_REC_N, aux,
                                                  (uint32_t)DS4_EG2VQ_AUX_N);
    if (!plan) { fprintf(stderr, "eg2vq_embd_gpu: the device plan did not bind\n"); return 4; }
    const double bind_sec = now_sec() - t_bind;
    const uint64_t want = ds4_gpu_eg2vq_workspace((uint32_t)max_tokens);
    const double t_ctx = now_sec();
    ds4_eg2vq_gpu_ctx *ctx = ds4_gpu_eg2vq_create(plan, want, (uint32_t)max_tokens);
    if (!ctx) { fprintf(stderr, "eg2vq_embd_gpu: no device context\n"); ds4_gpu_eg2vq_unbind(plan); return 4; }
    const double ctx_sec = now_sec() - t_ctx;
    size_t free1 = 0, total1 = 0;
    if (cudaMemGetInfo(&free1, &total1) != cudaSuccess) { free1 = free0; }

    printf("== the two legs ==");
    printf("\nartifact:   %s\n", gguf);
    printf("ids:        %s\n", ids_path);
    printf("device:     max_tokens %lu, workspace %.1f MiB, plan bind %.2f s, context %.2f s, host open+bind %.2f s\n",
           max_tokens, (double)want / 1048576.0, bind_sec, ctx_sec, open_sec);
    printf("vram:       cudaMemGetInfo free before %.1f MiB, after %.1f MiB, of %.1f MiB total (delta %.1f MiB)\n",
           (double)free0 / 1048576.0, (double)free1 / 1048576.0, (double)total0 / 1048576.0,
           (double)(free0 > free1 ? free0 - free1 : 0) / 1048576.0);
    printf("cpu dense:  %.1f MiB decoded once by the CPU leg (the cache the device leg never builds)\n",
           (double)weights.eg2vq->dense_bytes / 1048576.0);
    if (ref_path) { printf("reference:  %s (%zu items)\n", ref_path, n_ref); }
    printf("\n%-28s %6s %12s %11s %9s %9s", "item", "ntok", "cos(dev,cpu)", "rel_rms", "cpu_ms", "gpu_ms");
    if (ref_path) { printf(" %12s %12s", "cos(cpu,ref)", "cos(dev,ref)"); }
    printf("\n");

    FILE *fp = fopen(ids_path, "r");
    if (!fp) { fprintf(stderr, "eg2vq_embd_gpu: cannot open %s\n", ids_path); return 2; }
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t len;
    uint64_t items = 0, tokens = 0, seen = 0, first_tokens = 0;
    double cpu_secs = 0.0, gpu_secs = 0.0, first_gpu_secs = 0.0;
    double min_cos = 1.0, max_rel = 0.0, min_cos_cpu_ref = 1.0, min_cos_dev_ref = 1.0;
    int status = 0, ref_mismatch = 0;
    float cpu_v[DS4_EG2VQ_N_OUT], dev_v[DS4_EG2VQ_N_OUT];
    /* unit 27: the items kept for the batch leg (--batch / --batch-synth) */
    eg2vq_kept_item *kept = NULL;
    size_t n_kept = 0, cap_kept = 0;

    while ((len = getline(&line, &line_cap, fp)) >= 0) {
        char *s = line;
        size_t n = (size_t)len;
        while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) { s[--n] = '\0'; }
        if (n == 0 || strncmp(s, "TOKENS ", 7) == 0) { continue; }
        if (skip && seen < skip) { seen++; continue; }
        if (items_max && items >= items_max) { break; }

        char *item = s;
        char *prompt = strchr(item, '\t'); if (!prompt) { status = 1; break; } *prompt++ = '\0';
        char *ntok_s = strchr(prompt, '\t'); if (!ntok_s) { status = 1; break; } *ntok_s++ = '\0';
        char *ids_s  = strchr(ntok_s, '\t'); if (!ids_s) { status = 1; break; } *ids_s++ = '\0';
        char *end = NULL;
        const unsigned long ntok = strtoul(ntok_s, &end, 10);
        if (end == ntok_s || *end != '\0' || ntok == 0) { status = 1; break; }
        if (ntok > max_tokens) {
            fprintf(stderr, "eg2vq_embd_gpu: %s has %lu tokens, past the max_tokens %lu the context holds\n",
                    item, ntok, max_tokens);
            status = 1;
            break;
        }
        int32_t *ids = xmalloc((size_t)ntok * sizeof(int32_t));
        if (parse_ids(ids_s, ids, ntok, (long)weights.eg2vq->token_embd->dim[1]) != 0) {
            free(ids);
            status = 1;
            break;
        }

        const double c0 = now_sec();
        if (!no_cpu) {
            ds4_eg2vq_ref_embed(&model, &weights, ids, (uint32_t)ntok, cpu_v);
        }
        const double cpu_sec = now_sec() - c0;

        const double g0 = now_sec();
        const int ok = ds4_gpu_eg2vq_embed(ctx, ids, (uint32_t)ntok, dev_v);
        const double gpu_sec = now_sec() - g0;
        if (!ok) {
            fprintf(stderr, "eg2vq_embd_gpu: the device tower refused %s (%lu tokens)\n", item, ntok);
            free(ids);
            status = 1;
            break;
        }
        /* unit 27: keep the ids and the per-item vector for the batch leg, which
         * compares the one call's vectors against exactly these. */
        if (batch || batch_synth || batch_bad) {
            if (n_kept == cap_kept) {
                cap_kept = cap_kept ? cap_kept * 2u : 64u;
                kept = xrealloc(kept, cap_kept * sizeof(*kept));
            }
            kept[n_kept].ids = ids;
            kept[n_kept].ntok = ntok;
            memcpy(kept[n_kept].v, dev_v, sizeof dev_v);
            n_kept++;
            ids = NULL;
        }
        free(ids);

        double cos = 0.0, rel = 0.0;
        if (!no_cpu) {
            compare(cpu_v, dev_v, &cos, &rel);
            if (cos < min_cos) { min_cos = cos; }
            if (rel > max_rel) { max_rel = rel; }
        }
        if (items == 0) { first_gpu_secs = gpu_sec; first_tokens = ntok; }

        if (no_cpu) {
            printf("%-28.28s %6lu %12s %11s %9s %9.2f", item, ntok, "-", "-", "-",
                   gpu_sec * 1000.0);
        } else {
            printf("%-28.28s %6lu %12.9f %11.3e %9.2f %9.2f",
                   item, ntok, cos, rel, cpu_sec * 1000.0, gpu_sec * 1000.0);
        }
        if (ref_path) {
            if (seen < n_ref && ref[seen].ntok == ntok && !strcmp(ref[seen].id, item)) {
                double rc = 0.0, rr = 0.0, dc = 0.0, dr = 0.0;
                if (!no_cpu) { compare(cpu_v, ref[seen].v, &rc, &rr); }
                compare(dev_v, ref[seen].v, &dc, &dr);
                printf(" %12.9f %12.9f", no_cpu ? 0.0 : rc, dc);
                if (!no_cpu && rc < min_cos_cpu_ref) { min_cos_cpu_ref = rc; }
                if (dc < min_cos_dev_ref) { min_cos_dev_ref = dc; }
            } else {
                printf(" %12s %12s", "-", "-");
                ref_mismatch++;
            }
        }
        printf("\n");

        if (out_fp) {
            fprintf(out_fp, "%s\t%lu\t", item, ntok);
            for (uint32_t o = 0; o < (uint32_t)DS4_EG2VQ_N_OUT; o++) {
                fprintf(out_fp, o ? " %1.7f" : "%1.7f", (double)dev_v[o]);
            }
            fprintf(out_fp, "\n");
        }
        if (out_cpu_fp) {
            fprintf(out_cpu_fp, "%s\t%lu\t", item, ntok);
            for (uint32_t o = 0; o < (uint32_t)DS4_EG2VQ_N_OUT; o++) {
                fprintf(out_cpu_fp, o ? " %1.7f" : "%1.7f", (double)cpu_v[o]);
            }
            fprintf(out_cpu_fp, "\n");
        }
        /* Line-buffered per item: a run that is stopped or killed mid-set keeps
         * every completed item's line and vector, which is what let the first
         * attempt at this gate lose its last fifteen items to a block buffer. */
        fflush(stdout);

        cpu_secs += cpu_sec;
        gpu_secs += gpu_sec;
        items++;
        seen++;
        tokens += ntok;
    }
    free(line);
    fclose(fp);
    if (out_fp) { fclose(out_fp); }
    if (out_cpu_fp) { fclose(out_cpu_fp); }

    /* ---- unit 27: the batch --------------------------------------------------
     * The same items in ONE call, against the per-item vectors above: this is the
     * batch's whole claim (`a batched item's vector EQUALS its single-item
     * vector`), measured element by element and byte by byte, plus what one call
     * costs against the sum of the per-item calls.  Nothing here is a tolerance
     * the run may drift inside: the two legs compute the same items with the same
     * plan, and the item-relative kernels make the batched arithmetic identical,
     * so the comparison is `identical`.  --batch-synth N then builds a batch of
     * exactly N rows out of the loaded items, to walk the row budget's boundary.
     */
    if ((batch || batch_synth || batch_bad) && !status && items) {
        int bad_accepted = 0;
        /* ---- the span table is the block's own description -------------------
         * A table that does not describe the rows must be REFUSED, never
         * reinterpreted: an empty item (a span that does not rise), a table that
         * does not end at n_rows, zero items, and an item past the token ceiling
         * (which the kernel's key plane cannot hold -- unit 27 moved that check
         * out of the launcher and into the core, where the lengths are known).
         * Each case is one call on a workspace sized for the rows it claims. */
        if (batch_bad && n_kept >= 2) {
            const uint32_t a = (uint32_t)kept[0].ntok, b = (uint32_t)kept[1].ntok;
            const uint32_t rows = a + b;
            int32_t *packed = xmalloc((size_t)rows * sizeof(int32_t));
            memcpy(packed, kept[0].ids, (size_t)a * sizeof(int32_t));
            memcpy(packed + a, kept[1].ids, (size_t)b * sizeof(int32_t));
            float *out = xmalloc((size_t)2 * DS4_EG2VQ_N_OUT * sizeof(float));
            ds4_eg2vq_gpu_ctx *c = ds4_gpu_eg2vq_create(
                plan, ds4_gpu_eg2vq_workspace(rows), rows);
            const uint32_t good[3] = { 0, a, rows };
            const uint32_t empty_item[3] = { 0, 0, rows };
            const uint32_t short_table[3] = { 0, a, rows - 1u };
            const uint32_t flat_table[3] = { 0, a, a };
            struct { const char *name; const uint32_t *off; uint32_t n_items; } cases[] = {
                { "a span that does not rise (empty item)", empty_item, 2 },
                { "a table that does not end at n_ids", short_table, 2 },
                { "a table that does not cover (flat end)", flat_table, 2 },
                { "zero items", good, 0 },
            };
            printf("\n== the batch's refusals (unit 27) ==\n");
            if (!c) {
                printf("FAIL: no workspace for the refusal cases\n");
                bad_accepted = 1;
            } else {
                for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
                    /* each refusal must leave the CALLER's buffer alone: fill it
                     * with a sentinel first and check every float after */
                    for (size_t k = 0; k < (size_t)2 * DS4_EG2VQ_N_OUT; k++) { out[k] = 5.0f; }
                    const int ok = ds4_gpu_eg2vq_embed_batch(c, packed, cases[i].off,
                                                             cases[i].n_items, rows, out);
                    int untouched = 1;
                    for (size_t k = 0; k < (size_t)2 * DS4_EG2VQ_N_OUT; k++) {
                        if (out[k] != 5.0f) { untouched = 0; }
                    }
                    printf("%-44s %s\n", cases[i].name,
                           ok ? "ACCEPTED (BUG)"
                              : (untouched ? "refused, output untouched" : "refused but WROTE"));
                    if (ok || !untouched) { bad_accepted = 1; }
                }
                /* and the item ceiling: 2049 rows in one item, on a block of 2049
                 * rows built by cycling an item that is long enough */
                const uint32_t long_rows = (uint32_t)DS4_EG2VQ_MAX_TOKENS + 1u;
                int32_t *longpacked = xmalloc((size_t)long_rows * sizeof(int32_t));
                for (uint32_t t = 0; t < long_rows; t++) {
                    longpacked[t] = kept[0].ids[t % (uint32_t)kept[0].ntok];
                }
                ds4_eg2vq_gpu_ctx *lc = ds4_gpu_eg2vq_create(
                    plan, ds4_gpu_eg2vq_workspace(long_rows), long_rows);
                if (!lc) {
                    printf("%-44s %s\n", "a 2049-token item (past the key plane)",
                           "not testable: no workspace");
                    bad_accepted = 1;
                } else {
                    const uint32_t one[2] = { 0, long_rows };
                    for (size_t k = 0; k < (size_t)2 * DS4_EG2VQ_N_OUT; k++) { out[k] = 5.0f; }
                    const int ok = ds4_gpu_eg2vq_embed_batch(lc, longpacked, one, 1, long_rows, out);
                    int untouched = 1;
                    for (size_t k = 0; k < (size_t)2 * DS4_EG2VQ_N_OUT; k++) {
                        if (out[k] != 5.0f) { untouched = 0; }
                    }
                    printf("%-44s %s\n", "a 2049-token item (past the key plane)",
                           ok ? "ACCEPTED (BUG)"
                              : (untouched ? "refused, output untouched" : "refused but WROTE"));
                    if (ok || !untouched) { bad_accepted = 1; }
                    ds4_gpu_eg2vq_free(lc);
                }
                free(longpacked);
                ds4_gpu_eg2vq_free(c);
            }
            free(out);
            free(packed);
        }
        if (!batch && !batch_synth) {
            for (size_t i = 0; i < n_kept; i++) { free(kept[i].ids); }
            free(kept);
            ds4_gpu_eg2vq_free(ctx);
            ds4_gpu_eg2vq_unbind(plan);
            model_close(&model);
            return bad_accepted ? 9 : 0;
        }
        /* the refusal leg must leave no CUDA error behind for the legs after it:
         * a stale error would surface at the next leg's own error check */
        {
            const cudaError_t leftover = cudaGetLastError();
            if (leftover != cudaSuccess) {
                printf("FAIL: the CUDA context carried %s before the batch leg\n",
                       cudaGetErrorString(leftover));
                bad_accepted = 1;
            }
        }
        eg2vq_batch_leg leg = {0};
        eg2vq_kept_item *use = kept;
        size_t n_use = n_kept;
        int synth = 0;
        if (batch_synth) {
            /* Items of the ceiling's own length (2048) so the row count is exact:
             * N / 2048 items of 2048 rows and one more with the remainder, their
             * ids cycled from the first loaded item (the tower does not care that
             * the text is nonsense -- the row budget is what is under test), and
             * no per-item run to compare against. */
            synth = 1;
            const unsigned long synth_items = (unsigned long)(batch_synth / DS4_EG2VQ_MAX_TOKENS) +
                                              ((batch_synth % DS4_EG2VQ_MAX_TOKENS) ? 1u : 0u);
            use = xmalloc((size_t)synth_items * sizeof(*use));
            unsigned long at = 0;
            for (unsigned long i = 0; i < synth_items; i++) {
                const unsigned long len = (batch_synth - at > DS4_EG2VQ_MAX_TOKENS)
                                              ? DS4_EG2VQ_MAX_TOKENS
                                              : (batch_synth - at);
                use[i].ids = xmalloc((size_t)len * sizeof(int32_t));
                for (unsigned long t = 0; t < len; t++) {
                    use[i].ids[t] = kept[0].ids[t % kept[0].ntok];
                }
                use[i].ntok = len;
                memset(use[i].v, 0, sizeof use[i].v);
                at += len;
            }
            n_use = (size_t)synth_items;
        }
        unsigned long rows = 0;
        for (size_t i = 0; i < n_use; i++) { rows += use[i].ntok; }
        int32_t *packed = xmalloc((size_t)rows * sizeof(int32_t));
        uint32_t *off = xmalloc((n_use + 1u) * sizeof(uint32_t));
        off[0] = 0;
        for (size_t i = 0; i < n_use; i++) {
            for (unsigned long t = 0; t < use[i].ntok; t++) {
                packed[off[i] + t] = use[i].ids[t];
            }
            off[i + 1] = off[i] + (uint32_t)use[i].ntok;
        }
        const unsigned long want_rows = ((uint32_t)rows + 255u) & ~(unsigned long)255u;
        int ran = eg2vq_batch_run(plan, packed, off, (uint32_t)n_use, (uint32_t)rows, use, &leg);
        const double batch_first_sec = leg.batch_sec;
        /* A second call on a FRESH context (the same workspace, its pages now
         * touched): the first call above pays the allocation's own first-touch
         * cost, which the per-item loop did not, so the comparison is reported
         * both ways rather than against whichever one flatters the batch. */
        eg2vq_batch_leg warm_leg = {0};
        if (ran && !synth) {
            (void)eg2vq_batch_run(plan, packed, off, (uint32_t)n_use, (uint32_t)rows, use, &warm_leg);
        }
        printf("\n== the batch (unit 27) ==\n");
        printf("call:  %zu items, %lu rows, one workspace of %.2f MiB, %.1f ms\n",
               n_use, rows,
               (double)ds4_gpu_eg2vq_workspace((uint32_t)want_rows) / 1048576.0,
               leg.batch_sec * 1000.0);
        int refused = 0;
        if (!ran) {
            printf("refused: the device batch entry took none of this call "
                   "(the %u-row budget or the %u-token item ceiling)\n",
                   (unsigned)DS4_EG2VQ_MAX_ROWS, (unsigned)DS4_EG2VQ_MAX_TOKENS);
            refused = 1;
            if (!synth) { status = 1; }              /* the fixture itself must batch */
        } else if (synth) {
            printf("accepted: %lu rows in one call (a synthetic batch has no per-item "
                   "vectors to compare against)\n", rows);
        } else {
            const double warm_sec = gpu_secs - first_gpu_secs;
            const unsigned long warm_rows = tokens - first_tokens;
            printf("equality: %s, max per-element delta %.3e over %zu items x %d values\n",
                   leg.identical ? "every element byte-identical to the per-item run" : "DIFFERENT",
                   leg.max_delta, n_use, (int)DS4_EG2VQ_N_OUT);
            printf("per-item: %.1f ms for %lu rows (%.0f rows/s), of which the first item's "
                   "%.1f ms/%lu rows carries this process's first wave\n",
                   gpu_secs * 1000.0, rows, gpu_secs > 0.0 ? (double)rows / gpu_secs : 0.0,
                   first_gpu_secs * 1000.0, (unsigned long)first_tokens);
            printf("per-item, warm (first excluded): %.1f ms for %lu rows (%.0f rows/s)\n",
                   warm_sec * 1000.0, warm_rows,
                   warm_sec > 0.0 ? (double)warm_rows / warm_sec : 0.0);
            printf("one call: %.1f ms for %lu rows (%.0f rows/s) -- ONE launch wave against %lu\n",
                   leg.batch_sec * 1000.0, rows,
                   leg.batch_sec > 0.0 ? (double)rows / leg.batch_sec : 0.0,
                   (unsigned long)items);
            printf("one call, warm context: %.1f ms for %lu rows (%.0f rows/s)\n",
                   warm_leg.batch_sec * 1000.0, rows,
                   warm_leg.batch_sec > 0.0 ? (double)rows / warm_leg.batch_sec : 0.0);
            printf("ratio: one warm call is %.2fx the per-item total, %.2fx its warm rate\n",
                   warm_leg.batch_sec > 0.0 ? warm_leg.batch_sec / gpu_secs : 0.0,
                   gpu_secs > 0.0 ? (gpu_secs / warm_leg.batch_sec) : 0.0);
            if (!leg.identical || !warm_leg.identical) { status = 1; }
            (void)batch_first_sec;
        }
        if (synth) {
            for (size_t i = 0; i < n_use; i++) { free(use[i].ids); }
            free(use);
        }
        free(packed);
        free(off);
        for (size_t i = 0; i < n_kept; i++) { free(kept[i].ids); }
        free(kept);
        ds4_gpu_eg2vq_free(ctx);
        ds4_gpu_eg2vq_unbind(plan);
        model_close(&model);
        /* A refused synthetic batch is the boundary leg's expected OUTCOME, so it
         * gets its own exit code (9) instead of being read as a pass. */
        if (refused || bad_accepted) { return 9; }
        return status ? 1 : 0;
    }
    for (size_t i = 0; i < n_kept; i++) { free(kept[i].ids); }
    free(kept);

    printf("\n== the numbers ==\n");
    printf("items %llu tokens %llu\n", (unsigned long long)items, (unsigned long long)tokens);
    printf("parity: min cos(device, cpu) %.9f  max rel_rms %.3e  (declared: cos >= %.6f, rel_rms <= %.0e)\n",
           min_cos, max_rel, EG2VQ_GPU_COS_MIN, EG2VQ_GPU_REL_RMS_MAX);
    if (no_cpu) { printf("parity: NOT MEASURED in this run (--no-cpu: the batch legs need no per-item CPU leg)\n"); }
    if (ref_path) {
        printf("against the committed cpu raw: min cos(cpu, ref) %.9f  min cos(device, ref) %.9f",
               min_cos_cpu_ref, min_cos_dev_ref);
        if (ref_mismatch) { printf("  (%d items not comparable: id/ntok or length mismatch)", ref_mismatch); }
        printf("\n");
    }
    if (!no_cpu) {
        printf("cpu:    %8.2f s total, %7.3f s/item, %7.1f tokens/s (this harness's -O0 reference entry)\n",
               cpu_secs, items ? cpu_secs / (double)items : 0.0, cpu_secs > 0.0 ? (double)tokens / cpu_secs : 0.0);
    }
    printf("device: %8.2f s total, %7.3f s/item, %7.1f tokens/s (ids upload + 24 layers + readback)\n",
           gpu_secs, items ? gpu_secs / (double)items : 0.0, gpu_secs > 0.0 ? (double)tokens / gpu_secs : 0.0);
    if (items > 1) {
        const double warm = gpu_secs - first_gpu_secs;
        printf("device steady state (first item excluded): %7.3f s/item, %7.1f tokens/s\n",
               warm / (double)(items - 1), warm > 0.0 ? (double)(tokens - first_tokens) / warm : 0.0);
    }
    if (!no_cpu) {
        printf("speedup: %.1fx per item over the same ids in this process\n",
               gpu_secs > 0.0 ? cpu_secs / gpu_secs : 0.0);
    }

    ds4_gpu_eg2vq_free(ctx);
    ds4_gpu_eg2vq_unbind(plan);
    model_close(&model);

    const int pass = !status && (no_cpu || (min_cos >= EG2VQ_GPU_COS_MIN && max_rel <= EG2VQ_GPU_REL_RMS_MAX));
    if (status) { printf("eg2vq_embd_gpu: the ids file stopped the run\n"); }
    printf("%s\n", pass ? "EG2VQ_GPU_PASS" : "EG2VQ_GPU_FAIL");
    return pass ? 0 : 1;
}
