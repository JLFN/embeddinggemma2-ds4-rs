# The attention on f16 tensor cores (performance unit 5, 2026-11-10)

The VQ payload GEMM moved to f16 tensor cores in the previous unit (7.1x on the
large matrices, 1.80x end to end). This unit does the same for the text tower's
attention, which that change left as the largest single item: 43.9 ms of 87.6 ms
of kernel time, 50.1%, in one nsys profile of the same 8-item / 2,451-token
forward the last three units profiled (`nsys-kern-sum-b8-mma-vq.txt`).

Measured, one window, both round pairings: the kernel is up to **10.03x** faster
on a batch-8 block of 565-token items and **9.26x** on one 1,414-token item, and
the whole tower gains **1.35x / 1.72x / 1.86x** on the 143-page corpus at
batches 1 / 4 / 8, on top of the GEMM's change. Gate 7 is unchanged and passes
with `min cos 0.999999768, max rel_rms 6.813e-04` against its declared
`cos >= 0.999999, rel_rms <= 1e-03`.

## 1. Attribution: what was left after the GEMM

One `nsys` profile of one real batched forward through `ds4-server` (8 items,
2,451 tokens — the identical request the attention and VQ units profiled, i.e.
pages 001-008 of the corpus, 42+162+11+104+624+481+424+603 tokens):

| kernel | this profile | previous unit's profile |
|---|---|---|
| `eg2vq_attn_kernel` (both head dims) | 43.9 ms (50.1%): `<(int)4>` 31.5 ms / 20 launches, `<(int)8>` 12.4 ms / 4 launches | 46.0 ms (23.3%) |
| `eg2vq_vq_gemm_mma_kernel` (64 + 128 tiles) | 19.4 ms (22.1%): 10.8 ms / 144 launches, 8.6 ms / 73 | 126.9 ms (64.3%) |
| the small ops (`rms`, `rms_head`, `proj768`, `add`, `gelu`, `scale`, `rope`) | 24.3 ms (27.7%) | 24.5 ms (12.4%) |
| total kernel time | 87.6 ms | 197.5 ms |

Raw: `nsys-kern-sum-b8-mma-vq.txt` (`nsys stats --report cuda_gpu_kern_sum`).
The GEMM row is a 6.5x drop with the same code as the previous unit's build, so
it is the machine's state and the split that this table is good for, not the
cross-window delta — the same caution every record in this directory carries.

## 2. The references, read on their own branches

Two, and the dense one is the closer model because this tower is dense:

- **PR 142, `feat/spark2_5-family`** (JLFN/ds4-dfm-rs), `cuda/spark2_5.cuh`
  `spark_attn_mma_fp8_kernel` (:887-1044). One CTA per (KV head, row tile),
  8 warps, 32-key tiles, Q in registers, K and V both staged per tile, the
  swizzle at :835-837, `ldmatrix` at :839-848, `m16n8k16` at :855-860, f32 online
  softmax and P rounded to f16. It serves that dense model's 27 sliding **and** 9
  full layers with ONE kernel (`window = 0` means full) because both of its
  geometries are 256 wide. This unit's sliding kernel is that walk.
- **PR 143, `feat/gemma4-tensor-core-fp8`**, `cuda/gemma4_attn_mma.cuh:344-583`.
  The D=512 form: two 256-wide halves, one per warp of a pair, the pair
  exchanging its partial 16x32 scores in shared at :450-468. This unit's full
  kernel is that split, because this tower's full layers are 512 wide and 4 query
  heads share one KV head — a shape neither reference has.

What is taken from both, unchanged: the 16-byte-chunk swizzle, `ldmatrix.x4`
(non-transposed for QK^T, transposed for P\*V), `m16n8k16` with f32 accumulators,
the [row g, row g+8] x [col, col+8] A-fragment order, the per-row mask and online
rescale, and the un-normalised accumulator with one `1/l` at the write.

## 3. What had to change, and why

1. **The mask is symmetric.** A CTA's key loop runs from the minimum begin over
   its rows to the maximum last, rounded to 32-key tiles, and each row applies its
   own `|pos - key_pos| <= 512` inside the tile — the scalar kernel's
   union-plus-per-row-bound shape. The mask-edge case (distance 512 sees, 513
   blind) is therefore the same gate on both paths, and on the tensor-core path a
   fully masked tile must leave the accumulator at exactly zero, which it does.
2. **The key range is item-relative, and a CTA is one item.** Unit 27 packs
   several items into one row block, so a row tile that straddled an item would
   read another item's keys through a shared K tile. The tile ordinal is mapped
   to `(item, tile)` by walking the item lengths (`len[t]`, the table the scalar
   kernel already reads), instead of dividing the row index, and the launcher
   sizes the grid from the caller's exact span table.
3. **K and V are f32 in this port's cache**, so staging converts to f16 on the
   way into shared (`__halves2half2` per pair) rather than `cp.async`-ing a bf16
   tile. f16 rather than bf16 for the same reason the VQ unit chose it: this
   tower's tensors are f16 and bf16 would drop three significand bits
   (`ds41_vq_prefill_mma.cuh:24`).
4. **Q is f32 on entry** and is rounded to f16 in the fragment load. That is one
   real new rounding of the query per layer — the D=512 branch's Q is already
   bf16 and its conversion is 0 ULP; here it is not.

## 4. Geometry

| | sliding | full |
|---|---|---|
| layers | 20 | 4 (5, 11, 17, 23) |
| D, KV heads, group | 256, 2, 2 | 512, 1, 4 |
| warp owns | 8 tokens x 2 heads = 16 M rows | 4 tokens x 4 heads = 16 M rows, one 256-dim half |
| RTILE | 64 tokens | 16 tokens |
| shared | K 16 KiB + V 16 KiB | K/V tile 32 KiB + score exchange 16 KiB |
| per 32-key tile | 64 QK mma + 64 PV mma | 64 + 64 per warp-half, plus the exchange |

The full geometry's pair exchange is `4 mgroups x 2 halves x 16 x 32 floats` =
16 KiB; K is dead after the score barrier so V stages over it, which is exactly
the D=512 reference's flow.

## 5. The gate

`make test-eg2vq-attn` -> `EG2VQ_ATTN_PASS`. Both paths run on the same inputs
against the same f64 CPU oracle, per tower shape, plus the mask edge and the
packed block:

| leg | bound | measured |
|---|---|---|
| scalar (unchanged) | rel <= 1e-5, cos >= 1-1e-6 | rel 3.31e-07 to 9.73e-07, cos 1.000000000 |
| tensor cores, sliding | rel <= 2e-3, cos >= 1-2e-4 | rel 1.06e-03 to 1.28e-03, cos 0.999942612 to 0.999961871 |
| tensor cores, full | same | rel 1.45e-03 to 1.50e-03, cos 0.999869621 to 0.999877946 |
| mask edge, both paths | 512 sees / 513 blind, blind EXACTLY zero | PASS |
| two items packed (40 + 1200 rows), both paths | rel <= 1e-5 (scalar) / <= 2e-3 (mma) | scalar 6.52e-07; mma 1.25e-03, cos 0.999944735 |
| dispatch | the tensor-core path's output, bit for bit | PASS on every covered shape |

Those tensor-core bounds are 200x the scalar path's, and the reason is this
model's own arithmetic, not slack: the attention scale is exactly 1.0
(gemma-embedding2.cpp:8), so scores run to ~16 sigma over 256 or 512 dims and the
softmax is sharp; a ~6e-3 absolute score error (what f16 rounding of the
operands produces) moves the weights by ~1%. What makes it acceptable is the
TOWER: the per-token error averages out in the pooled vector, and gate 7
measures 6.813e-04 against the previous unit's 6.619e-04 with its declaration
unchanged. A three-pass split-QK form would recover ~1e-5 at twice the mma work;
it is not in this unit because the tower gate does not need it, and that is
recorded here as the option if a future tier needs tighter parity.

## 6. The kernel A/B, one window, both paths interleaved

`./tests/bench_eg2vq_attn 20` (`attn-mma-ab-bench.txt`), median of 20 launches
per row per path, the two paths run back to back in one process:

| shape | rows | keys | scalar ms | mma ms | speedup | mma TFLOP/s |
|---|---|---|---|---|---|---|
| single27_s | 27 | 729 | 0.0109 | 0.0113 | 0.97 | 0.3 |
| cli115_s | 115 | 13,225 | 0.0317 | 0.0313 | 1.01 | 1.7 |
| single184_s | 184 | 33,856 | 0.0769 | 0.0408 | 1.88 | 3.4 |
| page294_s | 294 | 86,436 | 0.1352 | 0.0597 | 2.26 | 5.9 |
| page565_s | 565 | 316,469 | 0.4700 | 0.0991 | 4.74 | 13.1 |
| max1414_s | 1,414 | 1,186,694 | 1.6301 | 0.1761 | 9.26 | 27.6 |
| single27_f | 27 | 729 | 0.0184 | 0.0186 | 0.99 | 0.3 |
| page294_f | 294 | 86,436 | 0.2632 | 0.0765 | 3.44 | 9.3 |
| max1414_f | 1,414 | 1,999,396 | 5.2372 | 0.6203 | 8.44 | 26.4 |
| batch8x306_s | 2,448 | 749,088 | 1.0209 | 0.1208 | 8.45 | 25.4 |
| batch8x115_s | 920 | 105,800 | 0.1533 | 0.0360 | 4.26 | 12.0 |
| batch8x565_s | 4,520 | 2,531,752 | 3.2879 | 0.3279 | **10.03** | 31.6 |
| batch8x565_f | 4,520 | 2,553,800 | 7.2622 | 0.7667 | 9.47 | 27.3 |
| batch64x20_s | 1,280 | 25,600 | 0.0591 | 0.0379 | 1.56 | 2.8 |

A first run of the same binary (`attn-mma-ab-bench-run1.txt`) read 7.56x for
`batch8x306_s` and 10.89x for `batch8x565_s`, i.e. the top rows move ~12% between
runs of one build in one window; the table above is the second run and both are
kept.

**Why there is no row threshold in the dispatch.** The measurements above say a
call below ~184 rows gains nothing (0.97x-1.01x). A threshold on that would be
wrong anyway: unit 27's contract is that a batched item's vector is its
single-item vector BYTE FOR BYTE, and a call-sized threshold picks the arithmetic
from the whole call's length, so a 20-token item would take the scalar path alone
and the tensor-core path inside a 64-item batch. One arithmetic per shape,
call-independent, is worth more than the 3% the smallest single item would save —
and 3% of the attention kernel is a fraction of a percent of that forward, as the
device A/B below shows.

## 7. End to end

**Gate 7** (`EG2VQ_GPU_PASS`), 300 items / 8,059 tokens, artifact
`...-vq4x4096-q80token-gemma_vq.gguf`:

| | previous unit (scalar attention) | this unit |
|---|---|---|
| min cos(device, cpu) | 0.999999781 | 0.999999768 |
| max rel_rms | 6.619e-04 | 6.813e-04 |
| declared | cos >= 0.999999, rel_rms <= 1e-03 | unchanged |

**Device A/B, the 300 one-item calls** (`--no-cpu`, interleaved new/old/new/old,
`ab-{new,old}-dev-r{1,2}.txt`): new 4,135.6 / 4,182.6 tokens/s against old
4,210.7 / 4,197.9, i.e. **0.98x and 1.00x**. These items average 27 tokens, where
the tensor-core CTA's fixed cost is all the work there is, and attention is ~4%
of such a forward: the cost is at most 2% of one call and inside the run-to-run
spread in the second pairing. The 3,710 tokens/s in the gate-7 log above is a
different window (that run included the CPU leg) and is not comparable.

**Corpus A/B, the 143-page / 80,747-token corpus, both round pairings**
(`corpus-attn-ab-{new,old}-{1,2}.csv`), old build = df0ee621 (tensor-core GEMM,
scalar attention):

| batch | new r1 | old r1 | ratio | new r2 | old r2 | ratio |
|---|---|---|---|---|---|---|
| 1 | 20,808.6 | 15,385.2 | 1.352x | 20,680.4 | 15,337.1 | 1.348x |
| 4 | 39,458.6 | 22,949.1 | 1.719x | 39,539.5 | 22,798.5 | 1.734x |
| 8 | 45,013.6 | 24,227.3 | 1.858x | 44,853.6 | 24,035.5 | 1.866x |

**Gate 27** (`EMBED BATCH GATE 27: PASS`): the 64-small-item leg (1,280 rows) is
byte-identical to the 64 per-item runs, max per-element delta 0.000e+00, and that
is the leg that proves the item-aligned CTA mapping: 64 items share one call and
each reads only its own keys. One call is 17.13x the warm per-item rate on that
set (23.4 ms against 395.2 ms), against unit 27's recorded 8.65x — the tensor
cores now apply to the short items too.

## 8. Resources, and what the SASS contains

`cuobjdump -res-usage tests/test_eg2vq_attn`:

| kernel | REG | STACK | LOCAL | shared (dynamic) |
|---|---|---|---|---|
| `eg2vq_attn_mma_slide_kernel` | 254 | 0 | 0 | 32,768 B |
| `eg2vq_attn_mma_full_kernel` | 255 | 48 | 0 | 49,152 B |

`cuobjdump -sass tests/test_eg2vq_attn | grep HMMA.16816.F32` -> 256 instances.
The full kernel's 48 B of stack is a real spill (its 128-float accumulator plus a
64-register Q fragment at 16 k16 steps); it costs 4% of the per-flop rate against
the sliding kernel (26.4 against 27.6 TFLOP/s at the 1,414-token shape) and is
left in rather than trimmed by re-splitting the tile, which would need a second
shared exchange.

## 9. What this unit does not claim

- It does not re-run the deep path, gate 5, gate 20's route gate or the loader
  digest; the change is one kernel header, two harnesses and documentation.
- It does not measure the 8192-row workspace case; gate 27 exercises the row
  budget's boundary and its byte identity, not its throughput.
- The corpus numbers are wall-clock server rows, so they move with the card's
  state; only the interleaved pairs above are evidence on their own.
- It does not make the kernel-level parity tighter than 1.1e-3-1.5e-3. The tower
  gate is what proves the precision is enough, and the split-QK option is named
  above if it ever is not.

## 10. Commands

    # the kernel gate (both paths, the mask edge, the packed block, the dispatch)
    (cd ds4-rs && make test-eg2vq-attn)                    -> EG2VQ_ATTN_PASS

    # the interleaved A/B table
    (cd ds4-rs && make tests/bench_eg2vq_attn && ./tests/bench_eg2vq_attn 20)

    # the attribution profile (8 pages = 2,451 tokens, one batch request)
    /usr/local/cuda/bin/nsys launch --session-new=eg2attn -t cuda --cuda-graph-trace=node \
        ./ds4-server -m <artifact> --port 18109
    /usr/local/cuda/bin/nsys start --session=eg2attn -o /tmp/eg2-attn-prof
    curl -s -X POST localhost:18109/v1/embeddings -H 'Content-Type: application/json' -d @batch8.json
    /usr/local/cuda/bin/nsys stop --session=eg2attn
    /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum /tmp/eg2-attn-prof.nsys-rep

    # gate 7 (5.5 min: the CPU leg is the -O0 reference)
    ./ds4-rs/tests/parity/eg2vq_embd_gpu_oracle <artifact> out/phase5-ids.tsv --items 300 \
        --ref out/phase5-artifact-raw-stopped-1628Z.txt --out out/gate7-attn-dev.txt

    # the corpus A/B (one server per run, both build)
    EG2_EMBED_BATCHES="1 4 8" EG2_EMBED_REPEATS=3 EG2_EMBED_CSV=... \
        EG2_EMBED_SERVER=<old|new>/ds4-server sh speed-bench/eg2_embed_sweep.sh <artifact> \
        out/live-demo/pages 8098

    # the batch contract
    (cd .. && sh tools/eg2_embed_batch_gate27.sh)
