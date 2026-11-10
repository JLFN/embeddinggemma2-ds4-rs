# The two norms on one warp per row (performance unit 6, 2026-11-10)

The VQ payload GEMM moved to f16 tensor cores in unit 4 and the attention in unit
5; together they took the tower from 197.5 ms of kernel time to 49.2 ms on the
same 8-item / 2,451-token forward. What was left in that 49.2 ms was the small
ops -- 24.3 ms, 49.2% -- and inside them two kernels carrying 14.1 ms:

| kernel | launches | ms | ms per launch |
|---|---|---|---|
| `eg2vq_ops_rms_kernel` | 121 | 8.64 | 71.4 |
| `eg2vq_ops_rms_head_kernel` | 72 | 5.44 | 75.6 |

An elementwise `add` over the same 5 MB of a 2,451-token batch took **11 us** per
launch. The norms, which read and write the same bytes, took 6x that. That is a
structural defect, not arithmetic: this unit replaces the reduction and is the
first of the small-ops units.

Measured, one window, both forms interleaved in one process: **3.07x** on the
121-launch row norm (0.0622 -> 0.0203 ms per launch), **2.63x** on the q norm and
**2.52x** on the k/v norms, i.e. about **8.5 ms** off the 49.2 ms forward, and the
whole tower gains **1.19x / 1.21x** on the 143-page corpus at batches 4 / 8. Gate
7 is unchanged in its declaration and passes; gate 27 stays byte-identical; and
the two reductions agree **bit for bit** at every width the tower runs, so no
other leg of the port can have moved.

## 1. The defect, read from the kernel

The old form is one 256-thread block per row (`eg2vq_ops_rms_kernel`): a
`__shared__ double ssum[256]`, a load loop, then a nine-step tree in shared
memory with a `__syncthreads()` between every step, then a second loop that reads
the row again to scale it. For a 512-wide row each thread owns exactly two
values, so the load is two instructions against a 2 KB shared round-trip and nine
barriers. `rms_head` is the same shape with 128 threads over a 256- or 512-wide
head vector.

Both kernels are memory-bound in principle (a row is a sum of squares and a
scale) and latency-bound in fact: 121 launches of 71 us where the bytes say 20 us
at this card's ~500 GB/s.

## 2. What changed

`cuda/eg2vq_ops.cuh`, one reduction used by both primitives:

- **One warp per row.** Each of the 32 lanes reads its own strided slice
  (`lane, lane+32, ...`) into registers, so the row is read **once** instead of
  twice; eight warps make a block, i.e. eight independent rows per block and no
  block-wide barrier.
- **`__shfl_xor_sync` butterfly** for the sum of squares: five shuffle-adds,
  no shared memory, and every lane ends with the total, so the scale needs no
  broadcast.
- **The register slice is reused for the write**, which is what removes the
  second read of the row.
- `VM` (the per-lane value count) is a **template** parameter so `#pragma
  unroll` makes the slice a register array instead of a dynamic local array;
  five instantiations cover 2/4/8/16/32 values per lane, i.e. widths up to 1,024.
  Wider rows fall back to the old kernel, kept as `eg2vq_ops_rms_block` /
  `eg2vq_ops_rms_head_block`.

The arithmetic is the reference's and does not change: the sum of squares
accumulates in double, the mean is divided in double, the cast to float happens
before the eps is added, and the output is `x * scale * w`
(`ds4_eg2vq_graph.inc:44-53`; `ggml/src/ggml-cpu/ops.cpp:4224-4231`). Only the
summation **tree** differs, and the two orders differ by ~1e-16 relative in
double.

**That the difference never reaches the output is measured, not assumed.** The
f64 sum is then rounded to f32 in the scale; a ~1e-16 relative difference in a
double cannot move a f32 rounding except within a whisker of a tie, and the gate
checks the whole surface: `make test-eg2vq-ops` runs both forms on the same input
at every width (64, 96, 256, 512, 768, 1,024 wide, 1 and 73 rows, weighted and
weightless) and compares the f32 bit patterns element by element -- **0 of
283,648 values differ, max 0 ULP**, plus the three per-head widths (64, 256, 512):
**0 of 242,944 differ, max 0 ULP**. The vision tower calls the same two
primitives (768 wide, head 64), so its arithmetic is bit-unchanged as well, which
is why this unit does not re-run the vision gates and says so in section 6.

## 3. The kernel A/B

`./tests/bench_eg2vq_ops 200` (`norm-warp-ab-bench.txt`), median of 200 launches
per row per form, the two forms run back to back in one process on the same
input, after a bit-identity check of the pair:

| shape | rows | block ms | warp ms | speedup | block GB/s | warp GB/s |
|---|---|---|---|---|---|---|
| rms 512 x 2451 (8 pages) | 2,451 | 0.0622 | 0.0203 | **3.07** | 161.4 | 495.6 |
| rms 512 x 58824 (PLE) | 58,824 | 1.4179 | 0.5577 | 2.54 | 169.9 | 432.0 |
| rms 512 x 294 (1 page) | 294 | 0.0092 | 0.0058 | 1.59 | 130.7 | 207.9 |
| rms 512 x 27 (short item) | 27 | 0.0041 | 0.0055 | 0.75 | 27.0 | 20.2 |
| rms 768 x 1120 (vision) | 1,120 | 0.0287 | 0.0160 | 1.80 | 240.0 | 430.9 |
| rms_head 256 x 4 (q, swa) | 9,804 | 0.1044 | 0.0397 | **2.63** | 192.2 | 506.0 |
| rms_head 256 x 2 (k,v swa) | 4,902 | 0.0532 | 0.0212 | 2.52 | 188.5 | 474.6 |
| rms_head 512 x 4 (q, full) | 9,804 | 0.1231 | 0.0584 | 2.11 | 326.3 | 688.0 |
| rms_head 512 x 1 (k,v full) | 2,451 | 0.0328 | 0.0181 | 1.81 | 306.4 | 555.3 |
| rms_head 256 x 4 (q, 1 page) | 1,176 | 0.0151 | 0.0076 | 1.98 | 159.8 | 316.2 |
| rms_head 64 x 12 (vision) | 13,440 | 0.1229 | 0.0344 | 3.57 | 56.0 | 200.0 |

A second run of the same binary (`norm-warp-ab-bench-run1.txt`) reads 3.05x,
2.29x, 2.63x, 2.51x, 2.12x for the same rows: the top rows move ~10% between
runs of one build in one window, and both files are kept.

Read the GB/s columns as a common denominator, not as each kernel's literal
traffic: the old kernel read the row twice and this one reads it once, and at the
widths where the buffer fits L2 (the 5 MB hidden block, the 20 MB q block on this
card's 48 MB) the repeats run warm, so 688 GB/s on the widest per-head row is
L2-assisted and is not a claim about HBM.

**The 27-row row is slower on the warp form (0.75x), and is left that way.** With
27 rows the call is 4 blocks of 8 warps, which cannot fill the card; the block
form launches 27 blocks and gets its parallelism from more, smaller blocks. There
is no row-count threshold for the same reason unit 5 has none: a threshold would
pick the arithmetic from the call's size, and although these two forms are
bit-identical *today* (measured above), a dispatch that depends on how many items
share a call is a contract waiting to break. The cost of the decision is 1.4 us
per launch on a 27-row call -- 193 launches of a short single item, ~0.25 ms on a
~20 ms forward, below the run-to-run spread of the device A/B.

## 4. What the forward should gain, and what it measured

Per launch, from the table and the tower's own call list
(`ds4_eg2vq_gpu.cuh:576-638`), at the 2,451-token batch:

```
 rms      120 x (0.0622 - 0.0203) = 5.03 ms   (5 per layer x 24)
 rms PLE    1 x (1.4179 - 0.5577) = 0.86 ms
 rms_head  20 x (0.1044 - 0.0397) = 1.29 ms   q, the 20 sliding layers
           40 x (0.0532 - 0.0212) = 1.28 ms   k and v, the same layers
            4 x (0.1231 - 0.0584) = 0.26 ms   q, the 4 full layers
            8 x (0.0328 - 0.0181) = 0.12 ms   k and v, the full layers
                                  ------
                                   8.84 ms   of the 49.2 ms forward
```

Section 6 carries the measured end-to-end number against that prediction.

## 5. Resources

`cuobjdump -res-usage ds4-server`:

| kernel | REG | STACK | shared |
|---|---|---|---|
| `eg2vq_ops_rms_warp_kernel<2>` | 29 | 0 | 0 |
| `eg2vq_ops_rms_warp_kernel<8>` | 38 | 0 | 0 |
| `eg2vq_ops_rms_warp_kernel<16>` | 52 | 0 | 0 |
| `eg2vq_ops_rms_warp_kernel<32>` | 84 | 0 | 0 |
| `eg2vq_ops_rms_block_kernel` (fallback) | 25 | 0 | 2,048 B |
| `eg2vq_ops_rms_head_block_kernel` (fallback) | 27 | 0 | 1,024 B |

No spill, no shared memory on the shipped path, and even the widest
instantiation at 84 registers runs 256 threads per block without an occupancy
penalty.

## 6. End to end

**Gate 7** (`EG2VQ_GPU_PASS`, `gate7-norm-warp.txt`), the frozen 300-item /
8,059-token query partition, artifact
`...-vq4x4096-q80token-gemma_vq.gguf`:

| | previous unit | this unit |
|---|---|---|
| min cos(device, cpu) | 0.999999768 | 0.999999768 |
| max rel_rms | 6.813e-04 | 6.813e-04 |
| declared | cos >= 0.999999, rel_rms <= 1e-03 | unchanged |

The two rows are the same digits, which is what section 2's bit-identity says
they must be: the device leg of this model did not move.

**Corpus A/B, the 143-page / 80,747-token corpus, both round pairings**
(`corpus-norm-ab-{new,old}-{1,2}.csv`), old build = `c207e97f` (unit 5's binary,
kept at `out/ds4-server-c207e97f`), one fresh server per run, the CLI first:

| batch | new r1 | old r1 | ratio | new r2 | old r2 | ratio |
|---|---|---|---|---|---|---|
| 1 | 21,246.3 | 20,725.7 | 1.025x | 21,233.2 | 20,852.5 | 1.018x |
| 4 | 45,012.0 | 39,222.0 | **1.148x** | 44,973.5 | 39,396.0 | **1.142x** |
| 8 | 52,393.3 | 44,759.4 | **1.171x** | 51,948.4 | 44,744.5 | **1.161x** |

Against the 8.84 ms per 2,451-token forward that section 4 computes: an 8-item
corpus call averages 4,520 tokens (the corpus's 565 tokens a page), so the same
arithmetic scales to ~16.3 ms a call, and 18 calls of the 143 pages are
1.8040 - 1.5412 = **0.2628 s**, i.e. 14.6 ms a call — 90% of the prediction, the
shortfall being the per-launch fixed cost that does not scale with rows. Batch 4
is 36 calls of ~2,260 tokens: 0.2648 s, 7.4 ms a call against 8.2 predicted.
Batch 1 is 143 calls of ~565 tokens where one request's own cost dominates, and
its 0.67 ms a call matches the 565-row arithmetic (~0.5 ms).

**The single-item row of the sweep is NOT this unit's number.** Its p50 read
24.46 ms on the old build and 14.70 ms on the new in round 1, and the same pair
again in round 2, which would be a 1.66x gain on a 115-token item — 20x more than
the norms can explain at 115 rows. A controlled re-measurement of the two builds
(12 requests each, two rounds, one warmup, `/tmp/eg2-item400.txt` = the sweep's
own 400-byte fixture) reads **new 15.4/14.3 ms against old 15.5/14.3 ms**, i.e. no
difference, which is what the arithmetic says. The sweep's p50 row is the state of
the machine right after a boot; it is reported in both csvs and is not used here.

**The project's whole fast path** (`tests/run.sh`, exit 0): the fourteen digests,
the DQVD container, the retrieval scorer's self-test, loader gate 3, gate 22
(`EG2VQ_VISION_MMPROJ_PASS`), **gate 23** (`EG2VQ_VISION_GATE23_PASS`),
**gate 24, the vision tower on the device** (`EG2VQ_VISION_GATE24_PASS`), gate 25
(`EG2VQ_SPLICE_GATE25_PASS`), gate 26 (`EG2VQ_EMBED_MEDIA_GATE26_PASS`), gate 27
(`EMBED BATCH GATE 27: PASS`) and the rule 19 QA gate (`ALL PASS`, no new runtime,
route or public-API surface). The vision legs are the point of running them: the
vision tower calls the same two primitives, at 768 and 64 wide, and its numbers
are the **recorded ones to the last printed digit** -- gate 23 pattern `maxabs
0.10275560, relrms 0.008235, min row cosine 0.99978457`, gradient `0.06645030,
0.005285, 0.99988406` (`docs/eg2vq-vision-tower-gate23.txt`), and gate 24's
device leg at `maxabs 0.15841668, relrms 0.009006, mincos 0.99933243` against
`docs/eg2vq-vision-tower-gate24.txt`, with all four attribution flips still
further from the reference than the port is.

**Gate 27** (`EMBED BATCH GATE 27: PASS`): the 64-small-item leg is byte-identical
to the 64 per-item runs, max per-element delta 0.000e+00, and one call is
**21.56x** the warm per-item rate on that set (19.2 ms against 407.8 ms), against
the 17.13x unit 5 recorded and unit 27's original 8.65x: the 1,280-row call gains
from the warp norm while the 19-row per-item calls stay in the shape where the
block form was faster.

**Pre-commit legs**, run in this session: P-1's five checks all zero (no
interpreter, no pyo3/torch/candle, `ldd ds4-server` free of python and torch, no
`.py` in the runtime, no onnx); OSV `osv_map_dependencies` on this root -- 140
packages, 0 vulnerable; `make test-eg2vq-ops` -> `EG2VQ_OPS_PASS`,
`make test-eg2vq-attn` -> `EG2VQ_ATTN_PASS`, `make test-eg2vq-vq` ->
`EG2VQ_VQ_PASS`.

## 7. What this unit does not claim

- The vision tower is not re-measured for speed: `tests/run.sh` re-runs its gates
  (23, 24, 25) because the primitives are shared, and their numbers are the
  recorded ones to the last digit, but no vision timing claim is made here.
- No claim that the norms are now at the roofline for every shape: the 27-row
  call is slower on the warp form and that is stated, not dispatched around.
- The deep path (`tests/run.sh --deep`), gate 20's route gate, the loader digest
  and gate 5 are not re-run: nothing here touches the file, the loader, the
  route or the retrieval set.
- Nothing about the VQ GEMM, the attention, or the op order: no expression, no
  constant and no call site moved.

## 8. Commands

    # the arithmetic gate: both forms against the f64 reference, and against each other
    (cd ds4-rs && make test-eg2vq-ops)                     -> EG2VQ_OPS_PASS

    # the interleaved A/B table
    (cd ds4-rs && make tests/bench_eg2vq_ops && ./tests/bench_eg2vq_ops 200)

    # gate 7
    ./ds4-rs/tests/parity/eg2vq_embd_gpu_oracle <artifact> out/phase5-ids.tsv --items 300 \
        --ref out/phase5-artifact-raw-stopped-1628Z.txt --out <out>

    # the corpus A/B (one server per run, both builds)
    EG2_EMBED_BATCHES="1 4 8" EG2_EMBED_REPEATS=3 EG2_EMBED_CSV=... \
        EG2_EMBED_SERVER=<old|new>/ds4-server sh speed-bench/eg2_embed_sweep.sh <artifact> \
        out/live-demo/pages 8098

    # the batch contract
    (cd .. && sh tools/eg2_embed_batch_gate27.sh)

## 9. The split after this unit -- the next unit's baseline

One nsys profile of the identical 8-item / 2,451-token request on the build that
carries this unit (`nsys-kern-sum-b8-norm-warp.txt`, request in
`norm-batch8-request.json`):

| kernel | before this unit | after |
|---|---|---|
| VQ GEMM (`mma<64>` 144 + `mma<128>` 73) | 19.4 ms (39.4%) | 19.39 ms (48.5%) |
| `eg2vq_ops_proj768_kernel` (**one** launch) | 7.13 ms (14.5%) | 7.13 ms (17.8%) |
| `eg2vq_attn_mma_slide_kernel` (20) | 3.85 ms | 3.83 ms |
| `eg2vq_attn_mma_full_kernel` (4) | 1.59 ms | 1.58 ms |
| `eg2vq_ops_rms_kernel` (block, 121 launches, 512 wide) | 8.64 ms (17.6%) | replaced |
| `eg2vq_ops_rms_head_kernel` (block, 72 launches) | 5.44 ms (11.1%) | replaced |
| `eg2vq_ops_rms_warp_kernel<16>` (133: the 121 row norms + the 12 full-layer head norms) | -- | 3.18 ms |
| `eg2vq_ops_rms_warp_kernel<8>` (60: the sliding layers' q, k and v head norms) | -- | 1.72 ms |
| the rest of the small ops | 3.06 ms | 2.79 ms |
| total kernel time | 49.2 ms | 40.0 ms |

The two norm rows together are 4.90 ms against 14.08 ms, a 2.87x drop, i.e. **9.18
ms off the forward** -- the arithmetic of section 4 (8.84 ms) within 4%.

What that leaves, in order: the VQ GEMM at 19.4 ms and 217 launches, `proj768` at
7.13 ms in ONE launch (17.8% of the kernel time and the largest single kernel left
that is not a tensor-core GEMM), the attention at 5.41 ms, and ~2.8 ms of
elementwise work. `proj768` is the next unit: one launch, 2,451 rows, 512 -> 768,
a bf16 weight read per row-block -- a tensor-core GEMM in the shape the last three
units have been building, with the output_norm RMS as its prologue.
