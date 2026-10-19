/* ds4_eg2vq_graph.inc — the EmbeddingGemma-2 text tower as this project's CPU
 * reference forward.  Included from ds4.c beside ds4_eg2vq_bind.inc.
 *
 * Authority, first-hand: /data/llama.cpp/src/models/gemma-embedding2.cpp (the
 * whole graph, 232 lines) with the cited ggml helpers.  Every mechanism below
 * carries its file:line; nothing is approximated (global rule 0.2).  The
 * project's committed port map is docs/ds4-gemma-port-reference.md section 4.
 *
 * Reference skeleton (gemma-embedding2.cpp:78-212):
 *   inpL = embed_row(token) * sqrt(n_embd)                    :86
 *   inp_per_layer = PLE(inpL)                                 :97, :215-232
 *   per layer il:
 *     x  = RMS(inpL, attn_norm)                                :109
 *     q,k,v = Wq/Wk/Wv @ x                                     :113-115
 *     q,k = weighted RMS; v = weightless RMS                   :125-127
 *     RoPE on q,k (per-layer base, n_rot = head_dim)           :133-136
 *     attention, scale 1.0, no cache, non-causal, SWA window   :139-141
 *     h1 = inpL + RMS(attn, attn_post_norm)                    :149-152
 *     f  = RMS(h1, ffn_norm)                                   :155
 *     ffn = Wdown @ (gelu(Wgate @ f) * (Wup @ f))              :159-165
 *     h2 = h1 + RMS(ffn, ffn_post_norm)                        :168-170
 *     h3 = h2 + RMS(Wple_proj @ (gelu(Wple_gate @ h2) * ple[il]), ple_post_norm) :172-191
 *     h  = h3 * layer_scalar                                   :194-195
 *   e = RMS(h, output_norm); out768 = Wout @ e                 :206-211
 *   mean pool                                                 src/llama-graph.cpp:4049-4053
 *   L2 normalize                                              common/common.cpp:1923-1941
 *
 * The control-vector hook (gemma-embedding2.cpp:199) is a no-op without a cvec
 * file, so there is deliberately no equivalent here (handoff PHASE 4).
 */

/* tanh GELU, exactly ggml's (ggml/src/ggml-cpu/vec.h:963-969): the FFN's
 * LLM_FFN_GELU + LLM_FFN_PAR is ggml_geglu_split, i.e. gelu(gate) * up
 * (src/llama-graph.cpp:1995-2000), and the per-layer input gate uses the same
 * plain ggml_gelu with no 1/sqrt(2) factor. */
#define EG2VQ_GELU_COEF_A    0.044715f
#define EG2VQ_SQRT_2_OVER_PI 0.79788456080286535587989211986876f

static inline float eg2vq_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(EG2VQ_SQRT_2_OVER_PI * x * (1.0f + EG2VQ_GELU_COEF_A * x * x)));
}

/* RMS over `n` with eps inside the square root and an optional weight.
 * Same arithmetic as ggml's rms_norm (ggml/src/ggml-cpu/ops.cpp:4224-4231):
 * mean = (double)sum/n cast to float, scale = 1/sqrtf(mean + eps),
 * y = x*scale*w.  A NULL weight is the weightless V norm
 * (gemma-embedding2.cpp:127, ggml_rms_norm). */
static void eg2vq_rms(float *out, const float *x, const float *w, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

/* One rank-1 bf16 vector (a norm weight or layer_scalar) as f32. */
static void eg2vq_read_f32(const ds4_model *m, const ds4_tensor *t, float *out) {
    ds4_ref_row(m, t, 0, out);
}

/* NeoX RoPE over the whole head vector.  ggml_cpu: theta_scale =
 * powf(freq_base, -2.0f/n_dims) (ops.cpp:6455); the per-pair angle starts at
 * the position and multiplies by theta_scale each step (ops.cpp:6289-6301,
 * with ext_factor 0 and freq_scale 1); rotate_pairs pairs (i, i + n_dims/2)
 * (ops.cpp:6375-6391).  n_dims = n_rot = head_dim (rope.dimension_count is
 * 512 full / 256 swa, out/artifact-kv.txt), so every head dimension rotates. */
static void eg2vq_rope(float *v, uint32_t n_rot, float pos, float theta_scale) {
    const uint32_t half = n_rot / 2u;
    float theta = pos;
    for (uint32_t i = 0; i < half; i++) {
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = v[i], x1 = v[i + half];
        v[i]        = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/* Batched dense matvec: out[t][o] = sum_i dense[o][i] * x[t][i].  `x_stride`
 * is the input row stride (the projection's in_features); the output rows are
 * packed at the matrix's out_features.  The parallel unit is one output row of
 * one token, so a single short item still fills all workers: ds4_parallel_for
 * gets n_tokens * out_features rows, exactly the row loop ds4_ref_matvec_rows
 * runs over one input. */
typedef struct {
    const ds4_eg2vq_matrix *w;
    const float            *x;
    float                  *out;
    uint64_t                x_stride;
    uint64_t                rows;
} eg2vq_matvec_ctx;

static void eg2vq_matvec_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const eg2vq_matvec_ctx *c = vctx;
    for (uint64_t i = r0; i < r1; i++) {
        const uint64_t t = i / c->rows;
        const uint32_t o = (uint32_t)(i % c->rows);
        const float *x = c->x + t * c->x_stride;
        c->out[t * c->rows + o] = ref_row_dot(c->w->dense + (uint64_t)o * c->w->cols, x, c->w->cols);
    }
}

static void eg2vq_matvec(const ds4_eg2vq_matrix *w, const float *x, uint64_t x_stride,
                         float *out, uint32_t n_tokens) {
    eg2vq_matvec_ctx ctx = { w, x, out, x_stride, w->rows };
    ds4_parallel_for((uint64_t)n_tokens * w->rows, eg2vq_matvec_worker, &ctx);
}

/* Attention over a whole item, no KV cache, no causal mask, scale 1.0
 * (gemma-embedding2.cpp:7 causal_attn = false, :8 f_attention_scale = 1.0,
 * :139-141).  The only mask is the symmetric sliding window of the SWA layers:
 * -INFINITY where |query - key| > n_swa/2 = 512 (src/llama-hparams.h:524-533;
 * n_swa is the file's 1024, out/artifact-kv.txt).  Query head h reads KV head
 * ik2 = iq2 / rk2 with rk2 = n_head / n_head_kv
 * (ggml/src/ggml-cpu/ops.cpp:8981, :9044; flash_attn_ext falls back to this on
 * any mask, and the non-flash path is build_attn_mha's kq = k^T q with the same
 * repeat). */
typedef struct {
    const float *q, *k, *v;
    float       *out;
    uint32_t     n_tokens, n_head, n_head_kv, head_dim, group;
    int32_t      half_swa;  /* < 0 disables the sliding mask (full layers) */
} eg2vq_attn_ctx;

static void eg2vq_attn_worker(void *vctx, uint64_t r0, uint64_t r1) {
    const eg2vq_attn_ctx *c = vctx;
    float *scores = xmalloc((size_t)c->n_tokens * sizeof(float));
    const uint64_t q_stride = (uint64_t)c->n_head * c->head_dim;
    const uint64_t kv_stride = (uint64_t)c->n_head_kv * c->head_dim;
    for (uint64_t qp = r0; qp < r1; qp++) {
        for (uint32_t h = 0; h < c->n_head; h++) {
            const uint32_t kvh = h / c->group;
            const float *qh = c->q + qp * q_stride + (uint64_t)h * c->head_dim;
            float *oh = c->out + qp * q_stride + (uint64_t)h * c->head_dim;
            float max_score = -INFINITY;
            for (uint32_t kp = 0; kp < c->n_tokens; kp++) {
                if (c->half_swa >= 0) {
                    const int64_t diff = (int64_t)qp - (int64_t)kp;
                    if (diff < -(int64_t)c->half_swa || diff > (int64_t)c->half_swa) {
                        scores[kp] = -INFINITY;
                        continue;
                    }
                }
                const float *kh = c->k + (uint64_t)kp * kv_stride + (uint64_t)kvh * c->head_dim;
                double acc = 0.0;
                for (uint32_t d = 0; d < c->head_dim; d++) acc += (double)qh[d] * (double)kh[d];
                scores[kp] = (float)acc;
                if ((float)acc > max_score) max_score = (float)acc;
            }
            double denom = 0.0;
            for (uint32_t kp = 0; kp < c->n_tokens; kp++) {
                const float e = expf(scores[kp] - max_score);
                scores[kp] = e;
                denom += e;
            }
            const float inv = denom > 0.0 ? (float)(1.0 / denom) : 0.0f;
            for (uint32_t d = 0; d < c->head_dim; d++) oh[d] = 0.0f;
            for (uint32_t kp = 0; kp < c->n_tokens; kp++) {
                const float p = scores[kp];
                if (p == 0.0f) continue;
                const float *vh = c->v + (uint64_t)kp * kv_stride + (uint64_t)kvh * c->head_dim;
                for (uint32_t d = 0; d < c->head_dim; d++) oh[d] += p * vh[d];
            }
            for (uint32_t d = 0; d < c->head_dim; d++) oh[d] *= inv;
        }
    }
    free(scores);
}

void ds4_eg2vq_ref_embed(const ds4_model *m, const ds4_weights *w,
                         const int32_t *ids, uint32_t n_ids, float *out768) {
    const ds4_eg2vq_weights *g = w->eg2vq;
    if (!g || !g->dense) { ds4_die("gemma_vq: the text tower weights are not bound"); }
    if (n_ids == 0) { ds4_die("gemma_vq: the item has no tokens"); }

    const uint32_t E  = (uint32_t)DS4_EG2VQ_N_EMBD;
    const uint32_t PL = (uint32_t)DS4_EG2VQ_PER_LAYER;
    const uint32_t NF = (uint32_t)DS4_EG2VQ_N_FF;
    const uint32_t qw_max = (uint32_t)DS4_EG2VQ_N_HEAD * (uint32_t)DS4_EG2VQ_HEAD_DIM_FULL;
    const uint32_t kvw_max = (uint32_t)DS4_EG2VQ_HEAD_KV_SWA * (uint32_t)DS4_EG2VQ_HEAD_DIM_SWA;
    const float eps = DS4_EG2VQ_RMS_EPS;

    float *h        = xmalloc((size_t)n_ids * E * sizeof(float));
    float *x        = xmalloc((size_t)n_ids * E * sizeof(float));
    float *f        = xmalloc((size_t)n_ids * E * sizeof(float));
    float *a        = xmalloc((size_t)n_ids * E * sizeof(float));
    float *q        = xmalloc((size_t)n_ids * qw_max * sizeof(float));
    float *k        = xmalloc((size_t)n_ids * kvw_max * sizeof(float));
    float *v        = xmalloc((size_t)n_ids * kvw_max * sizeof(float));
    float *attn     = xmalloc((size_t)n_ids * qw_max * sizeof(float));
    float *ffn_gate = xmalloc((size_t)n_ids * NF * sizeof(float));
    float *ffn_up   = xmalloc((size_t)n_ids * NF * sizeof(float));
    float *ffn_out  = xmalloc((size_t)n_ids * NF * sizeof(float));
    float *pg       = xmalloc((size_t)n_ids * PL * sizeof(float));
    float *pp       = xmalloc((size_t)n_ids * PL * sizeof(float));
    float *ple      = xmalloc((size_t)n_ids * PL * (size_t)DS4_EG2VQ_N_LAYER * sizeof(float));

    /* 1. token embeddings scaled by sqrt(n_embd) (gemma-embedding2.cpp:86;
     * f_embedding_scale stays 0 for this arch, so this is the only scale). */
    const float emb_scale = sqrtf((float)E);
    const uint64_t n_vocab = g->token_embd->dim[1];
    for (uint32_t t = 0; t < n_ids; t++) {
        if ((uint32_t)ids[t] >= n_vocab) {
            ds4_die("gemma_vq: token id is outside the embedding table");
        }
        float *ht = h + (size_t)t * E;
        ds4_ref_row(m, g->token_embd, (uint64_t)(uint32_t)ids[t], ht);
        for (uint32_t i = 0; i < E; i++) ht[i] *= emb_scale;
    }

    /* 2. per-layer input, once from the embedding (gemma-embedding2.cpp:215-232):
     * P @ inpL * 1/sqrt(n_embd), reshaped to [per_layer][layer], RMS with
     * per_layer_proj_norm.  The result is per (token, layer): the layer's
     * 512-wide slice is ple[t][il*512 + j]. */
    {
        eg2vq_matvec(&g->per_layer_model_proj, h, E, ple, n_ids);
        float pln[DS4_EG2VQ_PER_LAYER];
        eg2vq_read_f32(m, g->per_layer_proj_norm, pln);
        const float inv = 1.0f / sqrtf((float)E);
        for (uint32_t t = 0; t < n_ids; t++) {
            float *pt = ple + (size_t)t * PL * (size_t)DS4_EG2VQ_N_LAYER;
            for (uint32_t r = 0; r < PL * (uint32_t)DS4_EG2VQ_N_LAYER; r++) pt[r] *= inv;
            for (uint32_t il = 0; il < DS4_EG2VQ_N_LAYER; il++) {
                float *slice = pt + (size_t)il * PL;
                eg2vq_rms(slice, slice, pln, PL, eps);
            }
        }
    }

    for (uint32_t il = 0; il < DS4_EG2VQ_N_LAYER; il++) {
        const ds4_eg2vq_layer *l = &g->layer[il];
        const uint32_t hd = ds4_eg2vq_head_dim(il);
        const uint32_t nkv = ds4_eg2vq_head_kv(il);
        const uint32_t qw = (uint32_t)DS4_EG2VQ_N_HEAD * hd;
        const uint32_t kw = nkv * hd;
        const int sliding = ds4_eg2vq_layer_is_swa(il);
        const int32_t half_swa = sliding ? (int32_t)((uint32_t)DS4_EG2VQ_SWA / 2u) : -1;

        float ln_in[DS4_EG2VQ_N_EMBD], ln_ap[DS4_EG2VQ_N_EMBD], ln_fn[DS4_EG2VQ_N_EMBD];
        float ln_fp[DS4_EG2VQ_N_EMBD], ln_pp[DS4_EG2VQ_N_EMBD];
        float qn[DS4_EG2VQ_HEAD_DIM_FULL], kn[DS4_EG2VQ_HEAD_DIM_FULL], scalar[1];
        eg2vq_read_f32(m, l->input_norm, ln_in);
        eg2vq_read_f32(m, l->q_norm, qn);
        eg2vq_read_f32(m, l->k_norm, kn);
        eg2vq_read_f32(m, l->attn_post_norm, ln_ap);
        eg2vq_read_f32(m, l->ffn_norm, ln_fn);
        eg2vq_read_f32(m, l->ffn_post_norm, ln_fp);
        eg2vq_read_f32(m, l->ple_post_norm, ln_pp);
        eg2vq_read_f32(m, l->layer_scalar, scalar);

        /* a. attn_norm */
        for (uint32_t t = 0; t < n_ids; t++) {
            eg2vq_rms(x + (size_t)t * E, h + (size_t)t * E, ln_in, E, eps);
        }

        /* b. q/k/v projections */
        eg2vq_matvec(&l->q, x, E, q, n_ids);
        eg2vq_matvec(&l->k, x, E, k, n_ids);
        eg2vq_matvec(&l->v, x, E, v, n_ids);

        /* c. per-head RMS (weight on Q and K, weightless on V) and d. RoPE */
        const float base = sliding ? (float)DS4_EG2VQ_ROPE_BASE_SWA : (float)DS4_EG2VQ_ROPE_BASE;
        const float theta_scale = powf(base, -2.0f / (float)hd);
        for (uint32_t t = 0; t < n_ids; t++) {
            const float pos = (float)t;
            float *qt = q + (size_t)t * qw;
            float *kt = k + (size_t)t * kw;
            float *vt = v + (size_t)t * kw;
            for (uint32_t h = 0; h < DS4_EG2VQ_N_HEAD; h++) {
                eg2vq_rms(qt + (size_t)h * hd, qt + (size_t)h * hd, qn, hd, eps);
                eg2vq_rope(qt + (size_t)h * hd, hd, pos, theta_scale);
            }
            for (uint32_t j = 0; j < nkv; j++) {
                eg2vq_rms(kt + (size_t)j * hd, kt + (size_t)j * hd, kn, hd, eps);
                eg2vq_rope(kt + (size_t)j * hd, hd, pos, theta_scale);
                eg2vq_rms(vt + (size_t)j * hd, vt + (size_t)j * hd, NULL, hd, eps);
            }
        }

        /* e. attention */
        eg2vq_attn_ctx actx;
        actx.q = q; actx.k = k; actx.v = v; actx.out = attn;
        actx.n_tokens = n_ids;
        actx.n_head = (uint32_t)DS4_EG2VQ_N_HEAD;
        actx.n_head_kv = nkv;
        actx.head_dim = hd;
        actx.group = (uint32_t)DS4_EG2VQ_N_HEAD / nkv;
        actx.half_swa = half_swa;
        ds4_parallel_for(n_ids, eg2vq_attn_worker, &actx);

        /* f. o_proj */
        eg2vq_matvec(&l->o, attn, qw, a, n_ids);

        /* g. h += attn_post_norm(o) */
        for (uint32_t t = 0; t < n_ids; t++) {
            float *at = a + (size_t)t * E;
            eg2vq_rms(at, at, ln_ap, E, eps);
            float *ht = h + (size_t)t * E;
            for (uint32_t i = 0; i < E; i++) ht[i] += at[i];
        }

        /* h. FFN: down(gelu(gate(f)) * up(f)) */
        for (uint32_t t = 0; t < n_ids; t++) {
            eg2vq_rms(f + (size_t)t * E, h + (size_t)t * E, ln_fn, E, eps);
        }
        eg2vq_matvec(&l->ffn_gate, f, E, ffn_gate, n_ids);
        eg2vq_matvec(&l->ffn_up, f, E, ffn_up, n_ids);
        for (size_t i = 0; i < (size_t)n_ids * NF; i++) {
            ffn_gate[i] = eg2vq_gelu(ffn_gate[i]) * ffn_up[i];
        }
        eg2vq_matvec(&l->ffn_down, ffn_gate, NF, ffn_out, n_ids);

        /* i. h += ffn_post_norm(ffn) */
        for (uint32_t t = 0; t < n_ids; t++) {
            float *ft = ffn_out + (size_t)t * E;
            eg2vq_rms(ft, ft, ln_fp, E, eps);
            float *ht = h + (size_t)t * E;
            for (uint32_t i = 0; i < E; i++) ht[i] += ft[i];
        }

        /* j. per-layer input block */
        eg2vq_matvec(&l->ple_gate, h, E, pg, n_ids);
        for (uint32_t t = 0; t < n_ids; t++) {
            float *pgt = pg + (size_t)t * PL;
            const float *pt = ple + ((size_t)t * DS4_EG2VQ_N_LAYER + il) * PL;
            for (uint32_t i = 0; i < PL; i++) pgt[i] = eg2vq_gelu(pgt[i]) * pt[i];
        }
        eg2vq_matvec(&l->ple_proj, pg, PL, pp, n_ids);
        for (uint32_t t = 0; t < n_ids; t++) {
            float *ppt = pp + (size_t)t * E;
            eg2vq_rms(ppt, ppt, ln_pp, E, eps);
            float *ht = h + (size_t)t * E;
            for (uint32_t i = 0; i < E; i++) ht[i] += ppt[i];
        }

        /* k. layer_scalar */
        for (size_t i = 0; i < (size_t)n_ids * E; i++) h[i] *= scalar[0];
    }

    /* 4. output_norm and the 768 projection (gemma-embedding2.cpp:206-211). */
    float on[DS4_EG2VQ_N_EMBD];
    eg2vq_read_f32(m, g->output_norm, on);
    float *ow = xmalloc((size_t)DS4_EG2VQ_N_OUT * E * sizeof(float));
    for (uint32_t o = 0; o < DS4_EG2VQ_N_OUT; o++) {
        ds4_ref_row(m, g->output, o, ow + (size_t)o * E);
    }

    float *e = xmalloc((size_t)E * sizeof(float));
    memset(out768, 0, (size_t)DS4_EG2VQ_N_OUT * sizeof(float));
    for (uint32_t t = 0; t < n_ids; t++) {
        eg2vq_rms(e, h + (size_t)t * E, on, E, eps);
        for (uint32_t o = 0; o < DS4_EG2VQ_N_OUT; o++) {
            out768[o] += ref_row_dot(ow + (size_t)o * E, e, E);
        }
    }

    /* 5. mean pool over the token axis (src/llama-graph.cpp:4049-4053,
     * src/llama-graph.cpp:304-347: every token weight 1/n_tokens), then L2
     * normalize with a zero-norm guard (common/common.cpp:1923-1941). */
    const float mean = 1.0f / (float)n_ids;
    for (uint32_t o = 0; o < DS4_EG2VQ_N_OUT; o++) out768[o] *= mean;
    double ss = 0.0;
    for (uint32_t o = 0; o < DS4_EG2VQ_N_OUT; o++) ss += (double)out768[o] * (double)out768[o];
    const double norm = sqrt(ss);
    const float inv = norm > 0.0 ? (float)(1.0 / norm) : 0.0f;
    for (uint32_t o = 0; o < DS4_EG2VQ_N_OUT; o++) out768[o] *= inv;

    free(e); free(ow); free(ple); free(pp); free(pg);
    free(ffn_out); free(ffn_up); free(ffn_gate); free(attn);
    free(v); free(k); free(q); free(a); free(f); free(x); free(h);
}
