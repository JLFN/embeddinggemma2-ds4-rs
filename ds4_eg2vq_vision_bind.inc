/* ds4_eg2vq_vision_bind.inc — the vision tower's weight adapter (unit 23).
 *
 * The companion mmproj is opened as its OWN model (the family pattern:
 * ds4.c:72566 opens opt->vision_path the same way for GLM-5.3), so this file
 * only names tensors and checks geometry.  Every name below is one the
 * reference's own loader read back from the same file
 * (out/eg2vq-vision-anchor.log, 211 tensors), and the mmproj's 19 metadata
 * keys are what gate 22 freezes as the cross-check that the names mean what
 * they say.
 *
 * Layout, read from the reference graph rather than assumed:
 *  - v.patch_embd.weight is one 16x16 stride-16 convolution with NO bias
 *    (gemma4v.cpp:12-16).  Its ggml shape is [16,16,3,768], so its own memory
 *    order is exactly kw + 16*kh + 256*ic + 768*oc (ggml: ne0 fastest) and a
 *    contiguous widening needs no permutation at all.  (Unit 22's writer
 *    produced this layout from the checkpoint's [768][768] linear with
 *    conversion/gemma.py:1046-1050's permute; the port reads what the file
 *    carries, so nothing is undone here.)
 *  - v.position_embd.weight is f32 [768, 10240, 2]: slice 0 is the X table and
 *    slice 1 the Y table, each addressed as (882 patch row) * 768 + dim
 *    (gemma4v.cpp:26-34, whose tbl_y offset is pos_size * nb1).
 *  - mm.input_projection.weight is bf16 [768, 512], i.e. 512 rows of 768.
 *  - the four per-layer norms and the two head norms are f32; the matrices are
 *    bf16.  The norms stay tensors (read on demand, they are 64..768 wide);
 *    the matrices are widened ONCE into v->dense, the same decode-once idiom
 *    the text tower uses (ds4_eg2vq_bind.inc:295-313).
 *
 * Reused from ds4_eg2vq_bind.inc (same translation unit, included above):
 * eg2vq_refuse, eg2vq_plain_row, eg2vq_bind_plain_matrix.
 */

static void eg2vq_vision_refuse(const ds4_tensor *t, const char *reason) {
    char msg[256];
    snprintf(msg, sizeof(msg), "gemma_vq vision: tensor %.*s %s",
             (int)t->name.len, t->name.ptr, reason);
    ds4_die(msg);
}

/* One block's norm vector: f32, rank 1, exactly `d0` wide. */
static ds4_tensor *eg2vq_vision_norm(const ds4_model *m, const char *fmt,
                                     uint32_t il, uint32_t d0) {
    char name[192];
    snprintf(name, sizeof(name), fmt, il);
    ds4_tensor *t = required_tensor(m, name);
    tensor_expect_layout(t, DS4_TENSOR_F32, 1, d0, 0, 0);
    return t;
}

/* One weight matrix, widened into the caller's cache slot. */
static void eg2vq_vision_matrix(const ds4_model *m, const char *fmt, uint32_t il,
                                float *dst, uint32_t rows, uint32_t cols) {
    char name[192];
    snprintf(name, sizeof(name), fmt, il);
    ds4_eg2vq_matrix mx;
    eg2vq_bind_plain_matrix(m, &mx, name, rows, cols);
    for (uint32_t o = 0; o < rows; o++) {
        eg2vq_plain_row(m, mx.tensor, o, dst + (uint64_t)o * cols);
    }
}

void ds4_eg2vq_vision_bind(ds4_eg2vq_vision_weights *v, const ds4_model *m) {
    memset(v, 0, sizeof(*v));

    /* 1. the patch convolution and the two position tables. */
    {
        ds4_tensor *t = required_tensor(m, "v.patch_embd.weight");
        tensor_expect_layout4(t, DS4_TENSOR_BF16, 4,
                              (uint64_t)DS4_EG2VQ_VISION_PATCH,
                              (uint64_t)DS4_EG2VQ_VISION_PATCH, 3,
                              (uint64_t)DS4_EG2VQ_VISION_N_EMBD);
        const uint64_t n_el = (uint64_t)DS4_EG2VQ_VISION_N_EMBD
                            * DS4_EG2VQ_VISION_PATCH * DS4_EG2VQ_VISION_PATCH * 3u;
        float *dense = xmalloc((size_t)n_el * sizeof(float));
        /* dim[0] = 16, so the row walk is over the rank-4 tensor's own 36864
         * rows of 16 and lands in exactly the conv's memory order. */
        for (uint64_t r = 0; r < n_el / DS4_EG2VQ_VISION_PATCH; r++) {
            ds4_ref_row(m, t, r, dense + r * DS4_EG2VQ_VISION_PATCH);
        }
        v->patch_embd = dense;
    }
    v->position_embd = required_tensor(m, "v.position_embd.weight");
    tensor_expect_layout(v->position_embd, DS4_TENSOR_F32, 3,
                         (uint64_t)DS4_EG2VQ_VISION_N_EMBD,
                         (uint64_t)DS4_EG2VQ_VISION_POS_SIZE, 2);

    /* 2. the tie-in projection, the tower's only output. */
    {
        ds4_tensor *t = required_tensor(m, "mm.input_projection.weight");
        tensor_expect_layout(t, DS4_TENSOR_BF16, 2,
                             (uint64_t)DS4_EG2VQ_VISION_N_IN,
                             (uint64_t)DS4_EG2VQ_VISION_N_OUT, 0);
        const uint64_t n_el = (uint64_t)DS4_EG2VQ_VISION_N_OUT * DS4_EG2VQ_VISION_N_IN;
        float *dense = xmalloc((size_t)n_el * sizeof(float));
        for (uint32_t o = 0; o < (uint32_t)DS4_EG2VQ_VISION_N_OUT; o++) {
            eg2vq_plain_row(m, t, o, dense + (uint64_t)o * DS4_EG2VQ_VISION_N_IN);
        }
        v->input_projection = dense;
    }

    /* 3. one cache for the 16 * 7 matrices, sized from the architecture. */
    const uint64_t per_layer = 4ull * DS4_EG2VQ_VISION_N_EMBD * DS4_EG2VQ_VISION_N_EMBD
                             + 2ull * DS4_EG2VQ_VISION_N_FF * DS4_EG2VQ_VISION_N_EMBD
                             + 1ull * DS4_EG2VQ_VISION_N_EMBD * DS4_EG2VQ_VISION_N_FF;
    const uint64_t total = per_layer * DS4_EG2VQ_VISION_N_LAYER;
    v->dense = xmalloc((size_t)total * sizeof(float));
    v->dense_bytes = total * sizeof(float);

    float *cursor = v->dense;
    const uint32_t E = DS4_EG2VQ_VISION_N_EMBD;
    const uint32_t NF = DS4_EG2VQ_VISION_N_FF;
    for (uint32_t il = 0; il < (uint32_t)DS4_EG2VQ_VISION_N_LAYER; il++) {
        ds4_eg2vq_vision_layer *l = &v->layer[il];
        l->ln1            = eg2vq_vision_norm(m, "v.blk.%u.ln1.weight", il, E);
        l->attn_post_norm = eg2vq_vision_norm(m, "v.blk.%u.attn_post_norm.weight", il, E);
        l->ln2            = eg2vq_vision_norm(m, "v.blk.%u.ln2.weight", il, E);
        l->ffn_post_norm  = eg2vq_vision_norm(m, "v.blk.%u.ffn_post_norm.weight", il, E);
        /* The checkpoint's head norms are [64] == head_dim, which is what makes
         * clip.cpp:440-465 take the norm_per_head path (RMS inside each 64-wide
         * head, not over the whole 768-wide projection). */
        l->q_norm = eg2vq_vision_norm(m, "v.blk.%u.attn_q_norm.weight", il,
                                      DS4_EG2VQ_VISION_HEAD_DIM);
        l->k_norm = eg2vq_vision_norm(m, "v.blk.%u.attn_k_norm.weight", il,
                                      DS4_EG2VQ_VISION_HEAD_DIM);
        l->q = cursor; cursor += (uint64_t)E * E;
        eg2vq_vision_matrix(m, "v.blk.%u.attn_q.weight", il, l->q, E, E);
        l->k = cursor; cursor += (uint64_t)E * E;
        eg2vq_vision_matrix(m, "v.blk.%u.attn_k.weight", il, l->k, E, E);
        l->v = cursor; cursor += (uint64_t)E * E;
        eg2vq_vision_matrix(m, "v.blk.%u.attn_v.weight", il, l->v, E, E);
        l->o = cursor; cursor += (uint64_t)E * E;
        eg2vq_vision_matrix(m, "v.blk.%u.attn_out.weight", il, l->o, E, E);
        l->ffn_gate = cursor; cursor += (uint64_t)NF * E;
        eg2vq_vision_matrix(m, "v.blk.%u.ffn_gate.weight", il, l->ffn_gate, NF, E);
        l->ffn_up = cursor; cursor += (uint64_t)NF * E;
        eg2vq_vision_matrix(m, "v.blk.%u.ffn_up.weight", il, l->ffn_up, NF, E);
        l->ffn_down = cursor; cursor += (uint64_t)E * NF;
        eg2vq_vision_matrix(m, "v.blk.%u.ffn_down.weight", il, l->ffn_down, E, NF);
    }
    if ((uint64_t)(cursor - v->dense) != total) {
        eg2vq_vision_refuse(v->position_embd, "took a cache the architecture does not account for");
    }
}

void ds4_eg2vq_vision_free(ds4_eg2vq_vision_weights *v) {
    if (!v) { return; }
    free(v->dense);
    free(v->patch_embd);
    free(v->input_projection);
    memset(v, 0, sizeof(*v));
}
