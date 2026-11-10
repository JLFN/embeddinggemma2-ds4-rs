# The VQ payload GEMM on tensor cores

embeddinggemma2 performance unit, 2026-11-10 (engine `feat/eg2vq-attn-perf`).
The kernel is `eg2vq_vq_gemm_kernel` in `cuda/eg2vq_vq.cuh`; this unit adds
`eg2vq_vq_gemm_mma_kernel` beside it and makes the tensor-core path the public
entry. The arithmetic is the same GEMM: `out[t][r] = gain[r] * sum_i
codebook[idx[r][i/4]][i%4] * x[t][i]`, with the same payload bytes and the same
output layout. What changes is how the multiply is done: 16-bit weights and
activations go to `mma.sync.aligned.m16n8k16` with f32 accumulation instead of
scalar FFMA.

## 1. Attribution and the baseline in one window

The attention unit left the GEMM at 64.3% of a batched forward (126.9 ms of
197.5 ms, `nsys` 8 items / 2451 tokens). The bench baseline was frozen here
before the change, in this session, on this card:

    tests/bench_eg2vq_vq 7    scalar: 5.3-7.1 TFLOP/s at the large shapes

Raw: `mma-bench-5way.txt` (the same run after the change, five paths).

## 2. The authority, read first-hand, and one correction

The other gemma4's tensor-core VQ prefill is **not** in the mainline checkout:
`cuda/gemma_vq_prefill.cuh` there is the scalar-FMA form. The tensor-core form
is on the local branch `feat/gemma4-tensor-core-fp8`, tip `757ed27a`, which is
PR 143 (`https://github.com/Baekpica/ds4-dfm-rs/pull/143`). Read with
`git show feat/gemma4-tensor-core-fp8:cuda/gemma_vq_prefill.cuh`:

- `:54-66` the tile and `G4_VQP_PAD = 8`, the 16-byte stride wmma needs;
- `:121-166` the restored weight tile (E4M3 + GVR8 residual -> bf16 in shared);
- `:233-304` the `USE_MMA` arm: eight warps, 16 rows each, two n16 token tiles,
  `wmma::load_matrix_sync` / `mma_sync` / `store_matrix_sync`, and
  `DS4_GEMMA4_VQ_PREFILL_SCALAR=1` as the fallback knob.

The sibling VQ engines were read as the arithmetic precedent:

- `cuda/ds41_vq_prefill_mma.cuh:259`: "the codebook lookup ... that IS the mma
  A fragment: no A/B tile writes, no ldmatrix", with the gather at `:411-421`;
- `cuda/ds41_vq_prefill_mma.cuh:24`: "v2's f16 codebook would lose 3 mantissa
  bits to bf16, so v2 stays on the fused kernels";
- `cuda/glm_vq_mma.cuh:190-210`: the same code-word gather and a direct uint2
  B read for a 12-bit stream;
- `cuda/gemma4_attn_mma.cuh:76-91`: the f16 m16n8k16 helper and the f16
  rounding rationale.

This unit implements **both** forms and measures them, because the branch's
choice is forced by its codebook, not by the GPU:

- The branch's codebook is E4M3, so a bf16 tile is exact.
- This artifact's codebook is f16 (dim 4 x f16, `ds4_eg2vq.h:34-40`). A bf16
  tile would round every weight from 11 to 8 significand bits; the branch's own
  f16 arm (`gemma_vq_prefill.cuh:154-165`) does exactly that rounding, which is
  why it is not used here.

## 3. The arithmetic, and the one rounding

- weights: the codebook entry is copied as its own two 32-bit f16 words into
  the A fragment. Exact, no conversion.
- activations: rounded once to f16 (`__float2half_rn`), 11 significand bits.
- accumulation: f32, inside the tensor core.
- gain: multiplied after the sum, per row, in f32. It is never rounded into a
  weight.
- The only error the kernel adds is the f16 rounding of x and the tensor core's
  own k16 summation order. Bit identity with the scalar path is not claimed.

## 4. The kernel gate

`tests/test_eg2vq_vq.cu` now runs five paths on every case: the scalar kernel,
both register-direct tiles (64 and 128 tokens), and both WMMA tiles (32 and 64).
Each is compared with the same FP64 dot of the payload's IEEE decode. Declared
bounds: scalar `rel <= 1e-5, cos >= 1 - 1e-6` (unchanged); every tensor-core
path `rel <= 3e-4, cos >= 1 - 1e-7`.

Measured, all six tower cases (`make test-eg2vq-vq` -> `EG2VQ_VQ_PASS`):

    scalar        rel 4.01e-07 - 8.22e-07   min cos 1.0000000000
    mma-f16-64    rel 2.04e-04 - 2.09e-04   min cos 0.9999999680 - 0.9999999793
    mma-f16-128   identical to mma-f16-64 (same arithmetic, different tiling)
    wmma-f16-32   identical to mma-f16-64
    wmma-f16-64   identical to mma-f16-64

The print carries the measured value on every run, so a drift is visible even
while the declared bound still passes. The earlier claim in the file header
that the scalar kernel's only error is summation order still holds for the
scalar path; the tensor-core paths are a new declared bound, not a relaxed one.

## 5. The kernel A/B: five paths, one window

`tests/bench_eg2vq_vq 7`, medians of 7 samples per row per path, the tower's
eleven matrices and six token counts. ms, best path = the register-direct tile
the dispatcher selects. Full table in `mma-bench-5way.txt`; the large shapes:

| shape | tokens | scalar | mma64 | mma128 | wmma32 | wmma64 | speedup |
|---|---|---|---|---|---|---|---|
| ple_model_proj 12288x512 | 4520 | 8.030 | 1.364 | **1.124** | 3.973 | 2.218 | 7.14x |
| q_proj_f 2048x512 | 4520 | 1.340 | 0.232 | **0.195** | 0.644 | 0.385 | 6.89x |
| ffn_gate_s 2048x512 | 4520 | 1.407 | 0.242 | **0.201** | 0.678 | 0.405 | 7.01x |
| ffn_down_s 512x2048 | 4520 | 1.354 | 0.268 | **0.234** | 0.700 | 0.419 | 5.78x |
| k/v_proj_s 512x512 | 1414 | 0.156 | **0.029** | 0.033 | 0.083 | 0.045 | 5.40x |
| ffn_gate_s 2048x512 | 115 | 0.0727 | **0.0195** | 0.0297 | 0.0236 | 0.0307 | 3.61x |
| ple_model_proj 12288x512 | 1 | 0.0564 | **0.0215** | 0.0358 | 0.0327 | 0.0369 | 2.62x |

Peak measured: 50.6 TFLOP/s on the 12288x512 x 4520 shape, against ~140
TFLOP/s of dense f16 tensor throughput on this part.

What the table decides, in order:

1. **The register-direct form beats the staged WMMA form everywhere**, by 1.5x
   to 3.0x on the large shapes. The staged tile writes 128x72 f16 per slab,
   reads it back through `load_matrix_sync`, and needs two extra barriers; the
   register-direct form gathers four words per lane straight from the codebook
   and never materializes the tile. The branch's form is kept in the header as
   the measured alternative (the same way the reference keeps `vqm` and `vqs`
   side by side), not as the shipped path.
2. **The 64-token tile wins below ~512 tokens; the 128-token tile wins above
   it** when the grid fills a wave. This is the dispatch rule below.
3. The scalar kernel is never the best path at a shape the MMA tiles can take,
   including one token (1.4x-2.6x there). It stays as the fallback for a shape
   whose rows are not a multiple of 128 or whose columns are not a multiple of
   64.

## 6. The dispatch rule

`eg2vq_vq_gemm` calls `eg2vq_vq_gemm_mma` when the shape fits the tiles, and
falls back to `eg2vq_vq_gemm_scalar` otherwise. Inside the MMA entry:

    use BN=128  when n_tok >= 512 and (rows/128) * ceil(n_tok/128) >= 128
    use BN=64   otherwise

The two conditions are the measured crossover: the 128-token tile halves the
block count and the index traffic per token, which pays from about one full
wave of blocks up; below that the call cannot fill the card and the larger tile
only lengthens the wave (1.5x for the 64-token tile at 115-306 tokens). The
wave is the card's SM count, not a tuned constant.

## 7. The end-to-end gate and A/B

Gate 7 (`tests/parity/eg2vq_embd_gpu_oracle`), run A, 300 items / 8,059 tokens,
both legs in one process, against the committed CPU raw:

    min cos(device, cpu)   0.999999781   declared >= 0.999999
    max rel_rms            6.619e-04     declared <= 1e-03
    min cos(cpu, ref)      1.000000000
    min cos(device, ref)   0.999999781   (45 items not comparable, id/ntok)
    EG2VQ_GPU_PASS

Raw: `gate7-mma-vq.txt`.

The device-leg A/B, two interleaved rounds against the pre-change build
(`7cafa2a2`) on the same ids and card, `--no-cpu`:

| round | old (scalar GEMM) | new (tensor core) | ratio |
|---|---|---|---|
| 1 | 3.15 s, 0.011 s/item, 2556.9 tok/s | 1.92 s, 0.006 s/item, 4193.9 tok/s | 1.64x |
| 2 | 3.13 s, 0.010 s/item, 2578.7 tok/s | 1.92 s, 0.006 s/item, 4193.4 tok/s | 1.63x |

Both builds print `EG2VQ_GPU_PASS` on their own gate. Raw:
`gate7-device-ab.txt`. The 1.63x is the whole tower per call; the kernel itself
is 5.7x-7.1x on the large matrices, and the residual is the attention kernel and
the small ops, which this unit does not touch.

The corpus sweep (143 pages / 80,747 tokens, one fresh server per run,
interleaved old/new, two rounds; raw CSVs and health readouts beside this file,
summary in `corpus-ab.txt`):

| round | row | old | new | ratio |
|---|---|---|---|---|
| 1 | batch 1 | 8.257 s, 9,779 tok/s | 5.306 s, 15,217 tok/s | 1.56x |
| 1 | batch 4 | 6.280 s, 12,858 tok/s | 3.565 s, 22,650 tok/s | 1.76x |
| 1 | batch 8 | 6.049 s, 13,348 tok/s | 3.365 s, 23,998 tok/s | 1.80x |
| 2 | batch 1 | 8.265 s, 9,770 tok/s | 5.249 s, 15,383 tok/s | 1.57x |
| 2 | batch 4 | 6.254 s, 12,912 tok/s | 3.553 s, 22,728 tok/s | 1.76x |
| 2 | batch 8 | 6.077 s, 13,287 tok/s | 3.363 s, 24,007 tok/s | 1.81x |

The batch-14 row is partial in every run by the model's own 8192-token request
budget and is not comparable. The single-item p50 is dominated by the server's
per-request cost and the two rounds disagree (24.85 -> 24.59 ms, then
24.87 -> 15.94 ms); the corpus rows are the end-to-end number claimed here.

## 8. Resources

    eg2vq_vq_gemm_mma_kernel<64>    REG 118  STACK 0   LOCAL 0   SHARED 13312
    eg2vq_vq_gemm_mma_kernel<128>   REG 128  STACK 16  LOCAL 0   SHARED 21504
    eg2vq_vq_gemm_scalar            REG 128  STACK 0   LOCAL 0   SHARED 41472

`cuobjdump -sass` shows the expected `HMMA.16816.F32` instructions: 32 in the
64-token instantiation, 64 in the 128-token one. The 128-token tile's 16-byte
stack frame (32 B spill stores, 16 B spill loads) is the cost of holding 64
accumulator floats under the 128-register cap that keeps two blocks resident;
its measured time already includes it. The scalar kernel's 32 KB codebook pane
is gone from the tensor-core path: the codebook is read from the read-only cache
(the 32 KB table stays resident in L1), which is why the MMA kernel uses 13-21
KB of shared memory instead of 41 KB.

## 9. What this unit does not prove

- It does not re-run the retrieval floor (gate 5). That gate exercises the CPU
  oracle and is untouched by a device-kernel change; the device's fidelity is
  gate 7's cosine and rel_rms, both inside their declared bounds.
- It does not touch attention. `eg2vq_attn_kernel` is still the scalar-FMA
  grouped kernel; the tensor-core attention is the next unit (the gemma4 form,
  `cuda/gemma4_attn_mma.cuh`, adapted to EG2's 4 query heads, 2/1 KV heads and
  symmetric +/-512 window).
- It does not re-measure the retrieval floor on the device: gate 5 scores the
  CPU oracle, and the device's fidelity is gate 7's cosine and rel_rms.
- The corpus sweep ran on this card in this window; it is a wall-clock server
  measurement, so it moves with card state, which is why both rounds and the
  old/new interleave are recorded rather than one number.

## 10. Commands

    # the kernel gate, five paths against FP64
    (cd ds4-rs && make test-eg2vq-vq)                     -> EG2VQ_VQ_PASS

    # the five-path bench (rebuild after any header edit: the bench compiles the
    # header in, a stale binary reports the previous policy)
    (cd ds4-rs && make tests/bench_eg2vq_vq && ./tests/bench_eg2vq_vq 7)

    # resources and the tensor-core instructions
    /usr/local/cuda/bin/cuobjdump -res-usage ds4-rs/tests/test_eg2vq_vq
    /usr/local/cuda/bin/cuobjdump -sass ds4-rs/tests/test_eg2vq_vq | grep HMMA

    # gate 7 run A (about 5.4 min: the CPU leg is the -O0 reference)
    ./ds4-rs/tests/parity/eg2vq_embd_gpu_oracle \
        out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
        out/phase5-ids.tsv --items 300 \
        --ref out/phase5-artifact-raw-stopped-1628Z.txt \
        --out out/phase7-gpu-raw.txt --out-cpu out/phase7-cpu-raw.txt

    # the pre-change build, for the A/B
    git worktree add --detach /tmp/ab-vq-old <pre-change-commit>
    cp .ds4-cuda-config.mk /tmp/ab-vq-old/ && (cd /tmp/ab-vq-old && make \
        tests/parity/eg2vq_embd_gpu_oracle)
