/* ds4_eg2vq_vision.h — the C side of the gemma_vq family's VISION tower
 * (EmbeddingGemma-2, embeddinggemma2-handoff.md PHASE 8, unit 23).
 *
 * Authority, first-hand: /data/llama.cpp @ 31288c9c, the projector the
 * checkpoint's own config.json maps to (tools/mtmd/models/gemma4v.cpp, 133
 * lines) with the builders it calls in tools/mtmd/clip.cpp.  Every constant
 * below carries its file:line there; nothing is approximated (rule 0.2).  The
 * frozen numbers this tower is measured against are the dumps in
 * tests/vision/reference (unit 22's anchor), and the mmproj the weights come
 * from is bound by NAME (ds4_eg2vq_vision_bind.inc), exactly as the GLM-5.3
 * companion GGUF is (ds4.c:72689 model_open of opt->vision_path).
 *
 * Two facts about the reference that a reader of the graph alone would get
 * WRONG, both measured in this session (2026-10-20) from the anchor's own run
 * log, out/eg2vq-vision-anchor.log:
 *
 *  1. The FFN activation is GELU_QUICK, not the tanh GELU.  clip.cpp:1414-1429
 *     reads clip.use_gelu / clip.use_silu and, when NEITHER is set, assigns
 *     FFN_GELU_QUICK — the clip-model.h:91 default (FFN_GELU) never survives.
 *     Our mmproj writes neither key, and the reference printed
 *     "load_hparams: ffn_op: gelu_quick".  ggml's gelu_quick is
 *     x * sigmoid(1.702 x) (ggml/src/ggml-cpu/vec.h:964, :1031).
 *  2. The encode ran with FLASH ATTENTION ENABLED: clip.cpp:3928-3961 resolves
 *     CLIP_FLASH_ATTN_TYPE_AUTO to ENABLED when the backend supports the op
 *     (the CPU backend does), and the reference printed
 *     "warmup: flash attention is enabled".  The flash branch of build_attn
 *     (clip.cpp:787-790) casts K and V to f16 before the kernel, so the frozen
 *     numbers carry that rounding.  ds4_eg2vq_vision_opts therefore has an
 *     explicit `f16_kv` switch: 1 mirrors the reference's executed path (the
 *     default), 0 runs the graph in f32 and exists only so the gate can
 *     ATTRIBUTE the divergence it measures instead of guessing at it.
 */
#ifndef DS4_EG2VQ_VISION_H
#define DS4_EG2VQ_VISION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_model;
struct ds4_tensor;

/* The tower's shape, all from out/eg2vq-vision-anchor.log's own hparams dump
 * (clip.cpp:2202-2230) and the checkpoint's vision_config: hidden 768, 16
 * layers, 12 heads / 12 KV, head_dim 64, ffn 3072, patch 16, pooling 3,
 * position table 10240 rows per axis, rope theta 100, rms eps 1e-6. */
enum {
    DS4_EG2VQ_VISION_N_LAYER   = 16,
    DS4_EG2VQ_VISION_N_EMBD    = 768,
    DS4_EG2VQ_VISION_N_HEAD    = 12,
    DS4_EG2VQ_VISION_N_HEAD_KV = 12,
    DS4_EG2VQ_VISION_HEAD_DIM  = 64,
    DS4_EG2VQ_VISION_N_FF      = 3072,
    DS4_EG2VQ_VISION_PATCH     = 16,
    DS4_EG2VQ_VISION_MERGE     = 3,      /* clip.cpp:1726 n_merge (pooling kernel) */
    DS4_EG2VQ_VISION_POS_SIZE  = 10240,  /* v.position_embd.weight ne[1] */
    DS4_EG2VQ_VISION_N_IN      = 768,    /* the tie-in projection's input */
    DS4_EG2VQ_VISION_N_OUT     = 512,    /* mm.input_projection.weight ne[1] */
    /* The image budget: hparams.set_limit_image_tokens(70, 1120)
     * (clip.cpp:1734) over patch_area = patch*patch*merge*merge = 2304
     * (clip-model.h:207-212), i.e. 70..1120 soft tokens and
     * 161280..2580480 pixels.  The size solver aligns to patch*merge
     * (mtmd-image.cpp:775-782). */
    DS4_EG2VQ_VISION_ALIGN      = 48,       /* DS4_EG2VQ_VISION_PATCH * MERGE */
    DS4_EG2VQ_VISION_PATCH_AREA = 2304,
    DS4_EG2VQ_VISION_MIN_PIXELS = 161280,
    DS4_EG2VQ_VISION_MAX_PIXELS = 2580480,
    DS4_EG2VQ_VISION_MIN_TOKENS = 70,
    DS4_EG2VQ_VISION_MAX_TOKENS = 1120,
    /* One soft token is one row of the projection, so this is the largest
     * buffer the caller ever has to hold. */
    DS4_EG2VQ_VISION_SOFT_MAX   = 1120,
    DS4_EG2VQ_VISION_MAX_PATCHES = 2580480 / (16 * 16),
};

#define DS4_EG2VQ_VISION_ROPE_THETA 100.0f   /* clip.cpp:1725 */
#define DS4_EG2VQ_VISION_RMS_EPS    1e-6f    /* clip-model.h:95 */
/* The pooler scales by sqrt(n_embd) (gemma4v.cpp:96): computed, not a rounded
 * literal, so the port applies the reference's own expression. */

/* The preprocessing's own result: the canvas the reference resizes to, the
 * region it composites into it (PAD_CEIL, centered with integer division —
 * mtmd-image.cpp:57-96), and the patch grid that follows from it.  For the
 * frozen 512x384 fixtures this is canvas 528x384, region 512x384, offset 8,0,
 * grid 33x24, 792 patches, 88 soft tokens; the reference's own log prints the
 * canvas ("clip_encode: copying image 1/1 to input buffer (nx=528, ny=384)")
 * and the token count (image_tokens->nx = 88). */
typedef struct {
    uint32_t canvas_w, canvas_h;
    uint32_t region_w, region_h;
    int32_t  offset_x, offset_y;
    uint32_t grid_w, grid_h;
    uint32_t n_patches;
    uint32_t n_soft;
} ds4_eg2vq_vision_geometry;

/* One block's matrices, in the artifact's own names.  The dense pointer is
 * the widened f32 operand (the mmproj carries the matrices bf16), row-major
 * [rows][cols] with rows = out features — the same memory order as
 * ds4_eg2vq_matrix, whose batched matvec this tower reuses. */
typedef struct {
    struct ds4_tensor *ln1;             /* v.blk.N.ln1.weight            [768] f32 */
    struct ds4_tensor *attn_post_norm;  /* v.blk.N.attn_post_norm.weight [768] f32 */
    struct ds4_tensor *ln2;             /* v.blk.N.ln2.weight            [768] f32 */
    struct ds4_tensor *ffn_post_norm;   /* v.blk.N.ffn_post_norm.weight  [768] f32 */
    struct ds4_tensor *q_norm;          /* v.blk.N.attn_q_norm.weight    [64]  f32 */
    struct ds4_tensor *k_norm;          /* v.blk.N.attn_k_norm.weight    [64]  f32 */
    float *q, *k, *v, *o;               /* [768][768]  */
    float *ffn_gate, *ffn_up;           /* [3072][768] */
    float *ffn_down;                    /* [768][3072] */
} ds4_eg2vq_vision_layer;

typedef struct {
    struct ds4_tensor *position_embd;   /* v.position_embd.weight [768,10240,2] f32 */
    float *patch_embd;                  /* [16,16,3,768] widened: kw0 + 16*kh + 256*ic + 768*oc */
    float *input_projection;            /* mm.input_projection.weight [512][768] widened */
    ds4_eg2vq_vision_layer layer[DS4_EG2VQ_VISION_N_LAYER];
    float *dense;                       /* the one decode cache, all matrices above */
    uint64_t dense_bytes;
} ds4_eg2vq_vision_weights;

/* The one place the port can depart from the graph as written, named rather
 * than hidden: the reference ROUNDED K AND V TO F16 (clip.cpp:787-790, the
 * flash attention branch its CPU backend selected).  f16_kv = 1 mirrors that
 * and is what the frozen dumps carry; f16_kv = 0 is the graph in f32 and is
 * not a supported default. */
typedef struct {
    int f16_kv;
} ds4_eg2vq_vision_opts;

/* Bind every tensor by name, widen the matrices once into `v->dense`, and
 * read the norms on demand.  `m` is the COMPANION mmproj opened as its own
 * model (the family pattern), never the text artifact.  A missing, mis-typed
 * or mis-shaped tensor stops the load by name. */
void ds4_eg2vq_vision_bind(ds4_eg2vq_vision_weights *v, const struct ds4_model *m);
void ds4_eg2vq_vision_free(ds4_eg2vq_vision_weights *v);

/* The preprocessing alone, image bytes -> the geometry, the graph's own first
 * tensor and the two position index vectors.  `canvas` receives
 * geo->canvas_w * geo->canvas_h * 3 floats in the graph's inp_raw layout
 * (x fastest, then y, then the 3 channels), each value already
 * (pixel/255) * 2 - 1 (mtmd-image.cpp:7-14 for the v/255, gemma4v.cpp:9 for
 * the 2x-1); `canvas_cap` is its capacity in floats and a canvas that does not
 * fit is refused, never truncated.  `pos_x`/`pos_y` each need
 * DS4_EG2VQ_VISION_MAX_PATCHES entries.  `canvas` may be NULL, which is the
 * gate's preprocessing-only mode.  Returns 0 on any refusal, with `err` set. */
int ds4_eg2vq_vision_prepare(const uint8_t *data, size_t len,
                             ds4_eg2vq_vision_geometry *geo,
                             uint32_t *pos_x, uint32_t *pos_y,
                             float *canvas, uint64_t canvas_cap,
                             char *err, size_t errlen);

/* The whole image path on the CPU: image bytes -> `geo->n_soft` soft tokens of
 * 512 floats each, in the frozen dumps' own row order.  `soft` receives
 * n_soft * 512 floats.  Returns 0 (and writes nothing) on any refusal. */
int ds4_eg2vq_vision_forward(const struct ds4_model *m,
                             const ds4_eg2vq_vision_weights *v,
                             const uint8_t *data, size_t len,
                             const ds4_eg2vq_vision_opts *opts,
                             float *soft, uint64_t soft_cap,
                             ds4_eg2vq_vision_geometry *geo,
                             char *err, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif
