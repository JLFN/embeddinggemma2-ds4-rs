/* C embedding oracle for the gemma_vq text tower (EmbeddingGemma-2).
 *
 * Includes ds4.c (DS4_NO_GPU) so the reference forward in
 * ds4_eg2vq_graph.inc runs without linking CUDA, exactly like
 * tokenizer_c_oracle.  g_ds4_shape is assigned from the family constant
 * before model_open, so the shape macros (DS4_N_*, the catalog and the
 * bind dispatch) see the same table DS4_SHAPE_GEMMA_VQ carries.
 *
 *   usage: eg2vq_embd_c_oracle <artifact.gguf> <ids.tsv>
 *
 * <ids.tsv> is the Rust host's own per-item token dump
 * (crates/ds4-core/examples/gemma_vq_tok.rs), TAB-separated:
 *   <item> <TAB> <prompt_name> <TAB> <ntok> <TAB> <id,id,...>
 * It already carries BOS and EOS.  For every non-empty line the oracle prints
 * exactly one output line:
 *   <item> <TAB> <ntok> <TAB> <768 values as %1.7f, space separated>
 * and one summary line to stderr:
 *   eg2vq_embd: items=N tokens=M dense_mb=X secs=Y
 * An empty or malformed line, a wrong field count, or an ntok that does not
 * equal the id list exits non-zero.
 */

#include "../../ds4.c"

static void die_usage(void) {
    fprintf(stderr, "usage: eg2vq_embd_c_oracle <artifact.gguf> <ids.tsv>\n");
    exit(2);
}

/* Parse ids[0..ntok) from a comma-separated list; 0 on success, -1 malformed.
 * `n_vocab` rejects a token id the embedding table cannot address. */
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

int main(int argc, char **argv) {
    if (argc != 3) { die_usage(); }

    g_ds4_shape = DS4_SHAPE_GEMMA_VQ;

    ds4_model model;
    ds4_weights weights;
    const double t0 = now_sec();

    model_open(&model, argv[1], false, false);
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    if (!weights.eg2vq || !weights.eg2vq->dense) {
        fprintf(stderr, "eg2vq_embd: the text tower did not bind\n");
        return 3;
    }

    FILE *fp = fopen(argv[2], "r");
    if (!fp) {
        fprintf(stderr, "eg2vq_embd: cannot open %s\n", argv[2]);
        return 2;
    }

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    uint64_t items = 0, tokens = 0;
    int status = 0;
    float out[DS4_EG2VQ_N_OUT];

    while ((len = getline(&line, &cap, fp)) >= 0) {
        char *s = line;
        size_t n = (size_t)len;
        while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) { s[--n] = '\0'; }
        if (n == 0) { continue; }
        /* The host driver's own trailing summary, so this oracle can be pointed
         * at its output unchanged.  Everything else must be a four-field item
         * line. */
        if (strncmp(s, "TOKENS ", 7) == 0) { continue; }

        char *item = s;
        char *prompt = strchr(item, '\t'); if (!prompt) { status = 1; break; } *prompt++ = '\0';
        char *ntok_s = strchr(prompt, '\t'); if (!ntok_s) { status = 1; break; } *ntok_s++ = '\0';
        char *ids_s  = strchr(ntok_s, '\t'); if (!ids_s) { status = 1; break; } *ids_s++ = '\0';
        if (item[0] == '\0' || prompt[0] == '\0' || ntok_s[0] == '\0') { status = 1; break; }

        char *end = NULL;
        const unsigned long ntok = strtoul(ntok_s, &end, 10);
        if (end == ntok_s || *end != '\0' || ntok == 0 || ntok > 1000000UL) { status = 1; break; }

        int32_t *ids = xmalloc((size_t)ntok * sizeof(int32_t));
        const long n_vocab = (long)weights.eg2vq->token_embd->dim[1];
        if (parse_ids(ids_s, ids, ntok, n_vocab) != 0) { free(ids); status = 1; break; }

        ds4_eg2vq_ref_embed(&model, &weights, ids, (uint32_t)ntok, out);
        free(ids);

        printf("%s\t%lu\t", item, ntok);
        for (uint32_t o = 0; o < (uint32_t)DS4_EG2VQ_N_OUT; o++) {
            printf(o ? " %1.7f" : "%1.7f", (double)out[o]);
        }
        printf("\n");

        items++;
        tokens += ntok;
    }

    free(line);
    fclose(fp);

    if (status) {
        fprintf(stderr, "eg2vq_embd: malformed ids line\n");
        return 1;
    }

    fprintf(stderr, "eg2vq_embd: items=%llu tokens=%llu dense_mb=%.1f secs=%.2f\n",
            (unsigned long long)items, (unsigned long long)tokens,
            (double)weights.eg2vq->dense_bytes / 1048576.0, now_sec() - t0);
    model_close(&model);
    return 0;
}
