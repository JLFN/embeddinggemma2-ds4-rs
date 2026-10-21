# The attention kernel: whose K row each lane should read

embeddinggemma2 performance unit, 2026-10-21 (engine changes on top of
`1c9938ba`). The kernel is `eg2vq_attn_kernel` in `cuda/eg2vq_attn.cuh`. The
change is the *assignment* in pass 1: instead of one lane owning one whole K row,
`LAB = head_dim/64` lanes cooperate on one key, each owning 16 float4 chunks of
it, so a group's lanes read consecutive addresses, one warp load covers whole
128 B lines, and each lane keeps its own 64-float slice of the query row in
registers instead of re-reading the whole row once per key. The lane-per-key loop
is kept verbatim: it is the fallback for every shape the grouped form does not
cover and it finishes each window's remainder. No other pass, no launcher guard,
no shared-memory size, no gate bound changes.

## 1. What the ablation named, before anything was changed

`ncu` refuses on this machine (`ERR_NVGPUCTRPERM`), so the instrument is an
ablation: the shipped kernel copied into a `template<int V>` with exactly one
component changed per variant and the register footprint held constant
(`STACK:0 LOCAL:0` on every instantiation, `cuobjdump -res-usage`), the numbers
wrong for every `V != 0` by construction, only the time read.
`probes/probe-attn.cu`, raw output `attn-ablation.txt`, 20 repeats, medians,
`xV0` = the mean of the two `V0` rounds in the same run (one before the ladder,
one after):

| variant (one component changed) | page294_s | page565_s | max1414_s | batch8x565_s | batch8x565_f |
|---|---|---|---|---|---|
| V1 pass 1's K address block-uniform — 1 line instead of 32, same instruction count | 2.08x | 2.07x | 2.05x | 2.25x | 1.86x |
| V2 pass 1's K operand from a register (no K traffic, same FMA count) | 2.64x | 2.66x | 2.39x | 2.42x | 2.25x |
| V3 pass 2 without `expf` | 1.01x | 0.95x | 0.96x | 1.05x | 0.99x |
| V4 pass 3's V operand from a register | 1.31x | 1.43x | 1.29x | 1.40x | 1.48x |
| V5 pass 3 reads no score from the shared plane | 1.01x | 1.07x | 1.00x | 1.08x | 1.01x |
| V6 pass 1 stores nothing into the shared plane | 1.01x | 1.07x | 1.01x | 1.01x | 1.02x |
| V7 pass 1 + the row max only | 1.45x | 1.67x | 1.37x | 1.50x | 1.48x |
| V8 passes 1 + 2 only | 1.40x | 1.48x | 1.35x | 1.55x | 1.49x |
| V9 every component above removed at once | 14.81x | 19.50x | 17.60x | 19.98x | 21.89x |

Read it this way. The exponentials are free (V3), the score plane's write side
is free (V6) and its read side is free (V5), and pass 2 as a whole is free (V7
vs V8) — the whole softmax structure costs nothing. Pass 1 plus the row max is
60-73% of the kernel (V7/V8). Inside pass 1 the cost is one thing: the K load's
address pattern. Making that address block-uniform — same instruction count,
same FMA count, one 128 B line per warp load instead of 32 — is 1.86x-2.25x
(V1); removing the K operand from memory entirely is 2.25x-2.66x (V2), i.e. once
the K access is one line the remaining K cost is a fraction of the whole. V9 is
an upper bound and is labelled as one: removing every component also removes the
32 KB static shared allocation, which lifts the three-blocks-per-SM occupancy
limit, so its 15x-22x is not a load-removal number and nothing here rests on it.

The kernel's own history had already made the same observation in the other
direction: the previous unit vectorized these loads (float4) and recorded that
"the strided scalar loads were [the lever]", not the plane's size
(`attention-occupancy.md`). Vectorizing a 32-line-per-instruction access makes
each line fetch 16 B instead of 4 B and does not change the line count.

## 2. The candidate, measured as a probe before it was landed

`probes/probe-attn-grp.cu` (`attn-grouped-probe.txt`) is the shipped kernel's
body with pass 1 offered in two forms behind a template parameter: `ON=0` is the
shipped lane-per-key loop, `ON=1` is the grouped form. Its `ON=0` instantiation
is checked against the shipped launcher byte for byte on every shape (`V0 vs
shipped`), and every grouped run prints its cosine and maximum element-wise
relative difference against the shipped output, so no throughput number can come
from a form that computes something else.

Parameter pairs, with `LAB` lanes per key and `GS` float4 chunks per lane:
`LAB*4*GS == head_dim`, so `(8,8)` and `(4,16)` cover head_dim 256 and `(16,8)`
and `(8,16)` cover 512. Measured, 20 repeats, `xS` against the shipped form
measured twice in the same run:

| shape | (8,8) | (4,16) | | shape | (16,8) | (8,16) |
|---|---|---|---|---|---|---|
| page294_s (hd 256) | 1.69x | **2.27x** | | max1414_f (hd 512) | 1.80x | **2.53x** |
| page565_s | 1.75x | **2.39x** | | batch8x565_f | 1.50x | **2.01x** |
| max1414_s | 1.66x | **2.31x** | | | | |
| batch8x565_s | 1.37x | **2.06x** | | | | |

Every row agrees to cosine 1.000000000 with a maximum element-wise relative
difference of 2.3e-06 to 6.5e-06. The wider pairings (`GS = 16`, i.e. LAB=4 at
256 and LAB=8 at 512) win on both head dims: more of the key per lane means more
keys in flight per chunk (KC = 32/LAB is 8 and 4, against 4 and 2), a shorter
butterfly per key, and one predicated store per chunk instead of one per key.
The narrow pairings are recorded here because they were measured: 1.37x-1.80x.
The `xS` column divides each variant by the run's FIRST shipped measurement
(`S-a`), not by the mean of the two; the mean-based value differs by 1-4% on the
slower variants (max1414_s's `(8,8)` reads 1.66x this way and 1.62x against the
mean).

One instrument trap is worth keeping. The first version of this probe strided a
lane's chunks by `4*GS` instead of `4*LAB`, which happens to be right only when
`LAB == GS`; it built, ran, and produced plausible-looking speedups for the wrong
pairings -- (LAB=4, GS=16) at 2.41x-2.60x and (LAB=8, GS=16) at 2.12x-2.59x,
while the (LAB=16, GS=8) rows read only 1.43x-1.90x -- with cosine 0.02-0.24
against the shipped output. `attn-grouped-probe-spanbug.txt` is that run, kept
for the same reason
the GEMM unit kept its spilling probe: a wrong instrument that looks fast is the
failure mode, and the cosine column is what caught it. The probe then got the
`static_assert`s on the header's geometry that `probe-vq.cu` taught the last
unit to carry.

## 3. What shipped

`cuda/eg2vq_attn.cuh`:

- `eg2vq_attn_kernel` becomes `template <int LAB>`; `LAB > 0` selects the
  grouped pass 1, `LAB == 0` is the lane-per-key loop alone.
- The grouped pass 1: lane `l` owns chunk `msub = l & (LAB-1)` of the head and
  the key `gsub = l / LAB` inside the chunk; the lane's 64-float query slice
  (`EG2VQ_ATTN_GRP_GS = 16` float4, stride `4*LAB` elements) is loaded once per
  row into registers; per chunk each lane accumulates four chains (the
  `i % 4`-equivalent mapping the shipped `dot4` uses, so the four-component
  chain split is unchanged), the group's `LAB` partial sums are reduced by
  `log2(LAB)` `__shfl_xor_sync` rounds, and lane `msub == 0` stores the score and
  updates its own maximum. The row maximum, the exponentials, the plane and pass
  3 are untouched.
- Windows are consumed in chunks of `KC = 32/LAB`; fewer than `KC` leftover keys
  fall through to the shipped lane-per-key loop, which is also what every other
  shape and every misaligned or non-vectorizable case uses.
- The launcher dispatches on `head_dim == 64*LAB` for `LAB` in `{2,4,8,16}`
  (head_dim 128/256/512/1024) and to `LAB = 0` otherwise. The tower's two head
  dims are 256 (LAB 4) and 512 (LAB 8).
- `cuobjdump -res-usage`: `LAB=0` REG:48 (unchanged from the shipped kernel),
  `LAB=2/4/8/16` REG:98/100/98/98, every instantiation `STACK:0 LOCAL:0`, shared
  32,768 B and the caps unchanged. The grouped instantiations' register growth
  does not move occupancy: the 32 KB static plane caps a block at three per SM,
  and three 128-thread blocks at REG:100 use 38,400 of the 65,536 registers.

The grouped form is not bit-identical to the lane-per-key form: it sums a head's
terms as `LAB` partial sums reduced by a butterfly, where the shipped form uses
its four chains alone. The measured difference is cosine 1.000000000 with a
maximum element-wise relative difference of 2.3e-06-6.5e-06 (probe), and the gate
itself reads `rel` 3.31e-07 to 9.73e-07 with `mincos 1.000000000` on all nine
cases.

`tests/test_eg2vq_attn.cu` gains the two cases the dispatcher made necessary:
head_dim 384 (`LAB = 0`, vector on) and head_dim 320 (`LAB = 0`, vector off, and
the only case in the gate that executes the scalar bodies at all). Without them
the instantiation every other shape takes would be reachable from the launcher
and executed by no gate.

## 4. Measurement

### 4.1 The kernel, interleaved in one window

`tests/bench_eg2vq_attn.cu`, 30 repeats per row, `ms_med`. Old = the binary built
from `attn-before.cuh.txt` (the pre-change header, engine `1c9938ba`), new = the
current tree; both rebuilt for this run, registers confirmed to differ before
measuring (1 kernel instantiation against 5). Round 1 is old-then-new, round 2
old-then-new; both pairings are published because a sweep drifts.

| shape | old r1 | new r1 | x r1 | old r2 | new r2 | x r2 |
|---|---|---|---|---|---|---|
| cli115_s | 0.0623 | 0.0288 | 2.16x | 0.0557 | 0.0287 | 1.94x |
| page294_s | 0.3021 | 0.1208 | 2.50x | 0.2714 | 0.1208 | 2.25x |
| page565_s | 1.0977 | 0.4198 | 2.61x | 0.9861 | 0.4209 | 2.34x |
| max1414_s | 3.6485 | 1.4680 | 2.49x | 3.3966 | 1.4643 | 2.32x |
| page294_f | 0.5489 | 0.2366 | 2.32x | 0.5499 | 0.2365 | 2.33x |
| max1414_f | 12.4180 | 4.8749 | 2.55x | 12.0986 | 4.7165 | 2.57x |
| batch8x306_s | 2.0500 | 0.9196 | 2.23x | 2.0477 | 0.9196 | 2.23x |
| batch8x565_s | 7.6600 | 3.4356 | 2.23x | 7.2294 | 3.4578 | 2.09x |
| batch8x565_f | 15.3088 | 7.8653 | 1.95x | 15.2719 | 7.2868 | 2.10x |

The conservative headline is 1.94x-2.61x (round 1: 1.95x-2.61x; round 2:
1.94x-2.57x), the floor being `cli115_s`, the smallest shape, where one launch is
a larger share of the row. Effective rate, recomputed from the two raw runs'
own `gflop/s` column (the table above carries milliseconds): 869-1,498 GFLOP/s
before (the maxima are 1,496.7 in round 1 and 1,498.4 in round 2), 1,881-3,473
after.

### 4.2 The corpus, six sweeps interleaved

`speed-bench/eg2_embed_sweep.sh`, 143 pages / 80,747 tokens, one fresh server per
sweep, `wall` clock as the sweep documents. Old r1 and r2 were measured first,
then new r1, old r3, new r2, old r4 — so each new run sits between old runs.
`batch 14` is the sweep's own partial row (14 items, 5,091 tokens), not the
corpus, and is marked as such.

| row | old r1 | old r2 | old r3 | old r4 | new r1 | new r2 | speedup |
|---|---|---|---|---|---|---|---|
| batch 1 | 11.2924 | 10.9453 | 11.1772 | 11.2254 | 8.5732 | 8.7262 | 1.29x |
| batch 2 | 9.6088 | 9.3714 | 9.5059 | 9.6180 | 7.1761 | 7.2315 | 1.32x |
| batch 4 | 8.8601 | 8.5800 | 8.7378 | 8.7341 | 6.7148 | 6.5862 | 1.31x |
| batch 8 | 8.5083 | 8.2726 | 8.4570 | 8.4492 | 6.4559 | 6.3480 | 1.32x |
| batch 14 (partial) | 0.5302 | 0.4842 | 0.4916 | 0.4934 | 0.3859 | 0.3683 | 1.33x |

Tokens/s over the corpus: 7,150-7,377 (batch 1) to 9,490-9,761 (batch 8) before,
9,253-9,418 to 12,507-12,720 after. The speedup column above is the ratio of the
four-old-runs mean to the two-new-runs mean, rounded to two decimals (batch 1
1.29023 -> 1.29x, batch 2 1.32237 -> 1.32x, batch 4 1.31238 -> 1.31x, batch 8
1.31551 -> 1.32x, batch 14 1.32562 -> 1.33x). Pairing each new run against the
old runs measured immediately around it:
mean(old r2, old r3) / new r1 = 8.365/6.456 = 1.30x,
mean(old r3, old r4) / new r2 = 8.453/6.348 = 1.33x.

Cross-window, and labelled as such: the previous unit's record carries the
pre-GEMM batch-8 figure of 18.7998 s (4,295 tokens/s) from the attention unit's
own corpus sweep. Against that, today's 6.40 s / 12,507-12,720 tokens/s is 2.9x.
The three units' published factors do not reconstruct that number exactly, and
none of them was measured in one window with the others: the attention unit's
own factor is 1.61x, recomputed here from the committed CSVs rather than quoted
(`speed-bench/eg2_embed_vizzio.csv` 18.799776866 s at batch 8 against 11.654060797
/ 11.717910747 / 11.658361879 s in its three `eg2_embed_vizzio_attn-vec4` runs,
i.e. 1.6131 / 1.6044 / 1.6126); the GEMM's is 1.20x-1.25x pooled, 1.25x at the top
of that range; this one is 1.32x. 1.61 x 1.25 x 1.32 = 2.66x, against the 2.94x
the two endpoints give and the 2.7x the previous record published. Every one of
those products is a cross-window figure and is labelled as one here: the machine's
own state moved between the windows (today's old build measures 8.42 s at batch 8
where the previous unit's record carries 8.89 s for the same code), so the only
number in this section that is evidence on its own is this unit's interleaved
1.32x.

### 4.4 Where a whole forward's time goes now

One `nsys` profile of the same batched forward the previous unit profiled (8
items, 2451 tokens, one request through `ds4-server`), taken after the change:
`nsys-kern-sum-b8-attn-grouped.txt`.

| kernel | before (previous unit's profile) | after (this one) |
|---|---|---|
| `eg2vq_vq_gemm_kernel` | 139.2 ms (48.4%), 434 launches | 126.9 ms (64.3%), 434 launches |
| `eg2vq_attn_kernel` (both head dims) | 116.7 ms (40.5%) | 46.0 ms (23.3%): `<(int)4>` 31.9 ms / 40 launches, `<(int)8>` 14.1 ms / 8 launches |
| the small ops (`rms`, `rms_head`, `proj768`, `add`, `gelu`, `scale`, `rope`, ...) | ~31.9 ms (11.1%) | 24.5 ms (12.4%) |
| total kernel time | 287.8 ms | 197.5 ms |

The GEMM row moved too (139.2 -> 126.9 ms) with its code unchanged, which is the
same caution the previous record carries: a cross-window profile delta is not
evidence. The attention row is not used as the unit's measurement either — the
interleaved bench (4.1) and the interleaved corpus (4.2) are. What the profile
is good for is the split, and the split now says the payload GEMM is 64% of a
forward's kernel time and attention 23%, against 48% and 40% before.

### 4.5 Gates

- `make test-eg2vq-attn` -> `EG2VQ_ATTN_PASS`, 12 lines, all nine arithmetic cases
  `ok` with `mincos 1.000000000` and `rel` 3.31e-07-9.73e-07 against the f64 CPU
  reference, plus the mask-edge, packed-batch, refusal and untouched-output
  cases. The gate's bound is `rel <= 1e-5` and `mincos >= 1 - 1e-6`.
- `compute-sanitizer --tool memcheck --error-exitcode 7 ./tests/test_eg2vq_attn`
  -> `ERROR SUMMARY: 0 errors`.
- gate 7, `tests/parity/eg2vq_embd_gpu_oracle` on the artifact and
  `phase7-ids300.tsv` -> `EG2VQ_GPU_PASS`, `min cos(device, cpu) 0.999999950`,
  `max rel_rms 3.149e-04` against the declared `cos >= 0.999999`,
  `rel_rms <= 1e-03`; device 2,230.5 tokens/s (steady state 2,232.8), 91.8x the
  in-process `-O0` CPU reference. The parity figures are the same to three
  significant figures as the previous unit's record carries (this run
  0.999999950 / 3.149e-04 against that record's 0.999999950 / 3.150e-04), i.e.
  the attention change moved the tower's output by less than the fifth decimal
  of that budget; the tokens/s beside them move with the window (this run's own
  committed output reads 2,230.5 / 2,232.8, a re-run during the QA pass read
  2,144.7 / 2,146.5 for the same binary).

## 5. What this does NOT prove

- No performance counter was read (`ncu` remains unavailable), so "the K line
  pattern costs the time" rests on the ablation's ratios, not on a metric. The
  ablation's variants are wrong by construction; only their times were read.
- The corpus rows are wall clock around `curl`, as the sweep documents; the
  server route carries no per-request elapsed field. The kernel rows are the
  kernel alone, one launch each.
- The forward-level split in 4.4 is one profile per build, taken in different
  windows, so its delta is not evidence of this change: the interleaved bench and
  corpus rows are. The profile's own attention names are per instantiation
  (`<(int)4>`, `<(int)8>`), which is a direct read of the dispatcher in a live
  forward.
- The fast path's gate 25 failed on its first run in this session and passed on
  the second with the same binaries, because a 12-hour-old `ds4-server` from an
  earlier session was still holding `/tmp/ds4.lock` and the gate's engine-entry
  leg could not start its own process. `tests/run.sh` exits 0 on the re-run
  (gates 23-27 and the QA gate); the first log is not evidence of anything about
  this unit and is not kept.
- `V9`'s 15x-22x is an upper bound on removing everything at once, not a
  prediction of what any single change can buy, and it is not used as one.
- Pass 3's V loads still cost 1.29x-1.48x (V4). Nothing in this unit touched
  them; reusing a V row across query rows needs more than one score plane per
  warp and is left to a future unit with its own measurement.
- The grouped form's improvement is measured on the tower's own shapes; the
  `LAB = 0` fallback is gated but its performance was not compared (it is the
  pre-change body, byte for byte).

## 6. Rejected, with the measurement that rejected it

- `(LAB=8, GS=8)` for head_dim 256 and `(LAB=16, GS=8)` for 512: 1.37x-1.80x,
  against 2.01x-2.53x for the pairings that shipped (`attn-grouped-probe.txt`).
- The plane-sized-to-the-item policy stays rejected from the previous unit: the
  ablation re-confirms the plane is free at its ceiling size (V5, V6: 1.00x-1.08x
  both ways), and the plane's size is what bounds occupancy, so the ceiling costs
  nothing measurable and needs no plumbing.
- Pass 2 (the exponentials) was a candidate limiter in the previous unit's plan
  and is now measured out: 0.95x-1.05x, i.e. no change at all.

## 7. Reproduce

```sh
# --- from the ENGINE ROOT (ds4-rs/) -----------------------------------------
# the gate (now 9 arithmetic cases + the edge, packed and refusal cases)
make test-eg2vq-attn                             # EG2VQ_ATTN_PASS
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 7 ./tests/test_eg2vq_attn

# the tower gate (~7 min)
make tests/parity/eg2vq_embd_gpu_oracle
DS4_LOCK_FILE=/tmp/ds4.lock.attn ./tests/parity/eg2vq_embd_gpu_oracle \
    /data/embeddinggemma2/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
    /data/embeddinggemma2/out/phase7-ids300.tsv

# the interleaved kernel A/B: the old binary comes from the pre-change header
mkdir -p /tmp/ab-before/cuda /tmp/ab-before/tests
cp docs/benchmarks/eg2-embedding-2026-10-20/attn-before.cuh.txt /tmp/ab-before/cuda/eg2vq_attn.cuh
cp tests/bench_eg2vq_attn.cu /tmp/ab-before/tests/
nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_89 \
     -Xcompiler -march=native -Xcompiler -pthread -I. \
     -o /tmp/bench-attn-old /tmp/ab-before/tests/bench_eg2vq_attn.cu
make tests/bench_eg2vq_attn
/tmp/bench-attn-old 30 > attn-ab-old-r1.txt ; ./tests/bench_eg2vq_attn 30 > attn-ab-new-r1.txt
/tmp/bench-attn-old 30 > attn-ab-old-r2.txt ; ./tests/bench_eg2vq_attn 30 > attn-ab-new-r2.txt

# --- from the BENCHMARK DIRECTORY (ds4-rs/docs/benchmarks/eg2-embedding-2026-10-20/) ---
# the ablation and the candidate probe, AS MEASURED: both anchor their baseline
# to the header on the include path, so they must be built against the PRE-change
# header -- the same one the A/B stages -- or their own "V0 is not the shipped
# kernel" check fires and the numbers are of a different kernel.  This is the
# same requirement vq-gemm-rows.md records for probe-r8.cu and probe-big.cu.
mkdir -p /tmp/ab-attn/cuda && cp attn-before.cuh.txt /tmp/ab-attn/cuda/eg2vq_attn.cuh
nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_89 \
     -Xcompiler -march=native -Xcompiler -pthread -I/tmp/ab-attn \
     -o /tmp/probe-attn probes/probe-attn.cu && /tmp/probe-attn 20
nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_89 \
     -Xcompiler -march=native -Xcompiler -pthread -I/tmp/ab-attn \
     -o /tmp/probe-attn-grp probes/probe-attn-grp.cu && /tmp/probe-attn-grp 20

# the interleaved corpus A/B (one fresh server per sweep, ports must be FREE:
# the sweep's own health probe cannot tell another listener from ds4-server)
EG2_EMBED_SERVER=/tmp/ab-before/bin/ds4-server EG2_EMBED_CLI=/tmp/ab-before/bin/ds4 \
  EG2_EMBED_CSV=/tmp/attn-ab-old.csv sh ../../speed-bench/eg2_embed_sweep.sh \
  <artifact.gguf> /data/embeddinggemma2/out/live-demo/pages 8121
EG2_EMBED_SERVER=$PWD/../../ds4-server EG2_EMBED_CLI=$PWD/../../ds4 \
  EG2_EMBED_CSV=/tmp/attn-ab-new.csv sh ../../speed-bench/eg2_embed_sweep.sh \
  <artifact.gguf> /data/embeddinggemma2/out/live-demo/pages 8120
```

Building either probe against the CURRENT header instead compiles and runs but
fails its own anchor (`V0 IS NOT THE SHIPPED KERNEL`, five shapes, exit 1): the
shipped launcher runs the grouped form for head_dim 256/512 now, so a probe whose
baseline is the lane-per-key body can only be the pre-change instrument its
committed output says it is.

Files: `attn-ablation.txt` (the ablation), `attn-grouped-probe.txt` (the
candidate) and `attn-grouped-probe-spanbug.txt` (the first version of it, kept as
the wrong-instrument example), `probes/probe-attn.cu`, `probes/probe-attn-grp.cu`,
`attn-ab-{old,new}-r{1,2}.txt` (the kernel bench raw), `attn-ab-sweep.txt` (the
joined corpus table), `speed-bench/eg2_embed_vizzio_attn-ab-{old,new}-r{1,2}.csv`
plus the `-r{3,4}` old runs and their `_health.json` (all six health readouts are
the same caps payload, byte for byte — they identify the port and the caps, not
the build, which is what the sweep writes), `gate7-embd-attn-grouped.txt`,
`nsys-kern-sum-b8-attn-grouped.txt` (`nsys stats --report cuda_gpu_kern_sum` of one
batch-8 forward), `attn-before.cuh.txt` (the pre-change header the old binary and
both probes are built from).

Two traps this unit paid for, both worth keeping:

- **A "healthy" port is not necessarily your server.** The first corpus A/B
  attempt measured nothing: an unrelated `python3` listener already held
  127.0.0.1:8099 and :8101, so the sweep's `curl /health` succeeded against it,
  and every request came back as a 5-7 ms non-JSON body that `jq` refused. The
  failure looked like the new kernel producing bad JSON. Check the port is free
  (or read the sweep log for the server's own `listening on` line) before
  trusting a row; the third attempt failed differently and just as silently — a
  leftover debug server of my own held the `DS4_LOCK_FILE`, so no server could
  start and the log said `another ds4 process is already running`.
- **An ablation probe's variant can look fast and be wrong.** The `4*GS` stride
  bug above produced 2.41x-2.60x on the `(LAB=4, GS=16)` pairing and 2.12x-2.59x
  on `(LAB=8, GS=16)`, at cosine 0.02-0.24. Every probe
  here prints its agreement against the shipped kernel for exactly this reason,
  and the grouped probe's `ON=0` is checked byte for byte against the shipped
  launcher.
- **A probe anchored to the live header stops meaning what its committed output
  says the moment that header's behaviour changes.** Both probes here anchor
  their baseline to whatever `cuda/eg2vq_attn.cuh` says at compile time, which
  is what makes them trustworthy before a change and what makes them fail
  afterwards: after this unit they must be compiled against the PRE-change
  header (`attn-before.cuh.txt`, staged into the include path ahead of the
  repo), exactly as `vq-gemm-rows.md` requires of `probe-r8.cu` and
  `probe-big.cu`. The QA pass caught this by running section 7's commands as
  written and getting `PROBE_ATTN_FAIL` / `PROBE_ATTN_GRP_FAIL` with
  "V0 IS NOT THE SHIPPED KERNEL" on every shape.

## 8. Rule-19 QA pass

See `qa-evidence/qa-report.md` and `qa-evidence/qa-findings-attn-grouped.md` in
the outer repository, written at this boundary.
