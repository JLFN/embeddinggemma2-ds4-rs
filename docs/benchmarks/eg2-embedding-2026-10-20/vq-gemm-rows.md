# The VQ payload GEMM: how many rows one thread should own

embeddinggemma2 performance unit, 2026-10-21. The kernel is
`eg2vq_vq_gemm_kernel` in `cuda/eg2vq_vq.cuh` (the dim-4 DQVQ payload GEMM, the
gemma_vq device path's weight consumer). The change is one number: **output rows
per thread, 4 -> 8**, plus the block shape that follows from it. Everything else
in the kernel — the tile, the shared-memory budget, the launcher and its guard,
the accumulation order — is unchanged.

## 1. Attribution: the largest single item left, and what the ablation said about it

`nsys profile` of one real batched forward through `ds4-server` (8 items, 2451
tokens, the same request the attention unit profiled), before and after:

| kernel | 2026-10-20 19:40 (before) | 2026-10-21 07:34 (after) |
|---|---|---|
| `eg2vq_vq_gemm_kernel` | 212.6 ms of 401.2 ms (53.0%), 434 launches, 489.9 us avg | 139.2 ms of 287.8 ms (48.4%), 434 launches, 320.7 us avg |
| `eg2vq_attn_kernel` | 159.7 ms (39.8%) | 116.7 ms (40.6%) |

The attention row is the caution in this table: `cuda/eg2vq_attn.cuh` is
byte-identical between the two builds, and its average per launch still reads
3.33 ms in one profile and 2.43 ms in the other. A single-run profile moves with
the GPU's state, not only with the code, so nothing in this document rests on a
cross-window profile delta; they are here for the split, and the split says the
GEMM is still the largest single item at 48.4%.

Raw: `nsys-kern-sum-b8-before-vq-r8.txt`, `nsys-kern-sum-b8-vq-r8.txt`.

## 2. The instrument: an ablation, because there are no counters

`ncu` refuses on this machine (`ERR_NVGPUCTRPERM`: performance counters need a
driver parameter, i.e. root on a shared machine), so the honest question "which
part of this kernel costs the time" was answered by building the kernel with one
component removed at a time and timing it. `probes/probe-vq.cu` is the shipped
body copied into a `template<int V>` (so every variant gets its own clean
codegen) with every variant register-neutral by construction; `cuobjdump
-res-usage` confirms `STACK:0 LOCAL:0` on all of them. The numbers are wrong for
every V != 0 by construction; only the time is read. 15 repeats, medians, one
sitting.

| variant | k/v_proj_s 306 | k/v_proj_s 4520 | ffn_gate_s 306 | ffn_gate_s 4520 | ple_model_proj 306 | ple_model_proj 4520 |
|---|---|---|---|---|---|---|
| V1 one x load per j, reused by four tokens | 2.00x | 2.47x | 1.99x | 2.59x | 2.45x | 2.63x |
| V2 no codebook gather, no index read | 1.08x | 1.09x | 1.08x | 1.08x | 1.09x | 1.02x |
| V4 all loads kept, outer product replaced by FADDs | 1.02x | 1.01x | 1.01x | 1.00x | 1.02x | 0.94x |
| V5 four x loads, block-uniform address | 2.36x | 3.47x | 2.88x | 3.64x | 3.49x | 3.66x |

(faster than baseline = larger; V3, "FMA only", is in the raw file and is an
upper bound only — with identical operands ptxas CSEs the chain, which is why it
reports 27-67 TFLOP/s, above this part's ~35.5 TFLOP/s of fp32.)

What that says, in order of how hard it is to argue with:

- **The arithmetic is not the limit.** Replace the whole 16-FFMA outer product
  with a few FADDs and keep every load: 1.00x. The multiply-add is entirely
  hidden behind the loads.
- **The codebook gather is not the limit either.** Cut the shared-memory gather
  and the index read to a constant: 1.02-1.09x.
- **The activation load is the limit.** The two variants that touch it are the
  only ones that move the kernel: cutting the x-load instructions to a quarter
  (V1: 2.0-2.6x) or keeping the instructions and making the addresses
  block-uniform (V5: 1 sector per instruction instead of sixteen: 2.4-3.7x).

The model behind the fix follows from the same geometry. Per thread per j the
kernel loads a `float4` per token — 16 bytes — and spends `T*R` FMAs on it, so
it pays **16/R bytes of activation per FMA**: R=4 pays 4 B/FMA, R=8 pays 2
B/FMA. The load side is expensive per instruction and not per byte: a warp's
sixteen `tx` lanes read the same column chunk of sixteen different token rows
`t_stride` apart, i.e. sixteen 32-byte sectors per instruction, sixteen times
the FMA path's cost per j. More rows per thread is the only register-blocking
move that reduces it without changing what is loaded.

## 3. What shipped

| | before | after |
|---|---|---|
| block | 16 x 16 threads | 16 x 8 threads |
| per thread | 4 tokens x 4 rows = 16 accumulators | 4 tokens x 8 rows = 32 accumulators |
| tile | 64 tokens x 64 rows | unchanged |
| index slab | 64 x 68 uint16 (8.5 KB), one unpack pass | same slab, two unpack passes (128 threads cover 32 rows per pass) |
| shared per block | 41,472 B, static | 41,472 B, static |
| registers | REG:80, no spill | REG:128, STACK:0 LOCAL:0 |
| launcher, guard, grid math | `rows % 64`, `cols % 256`, `grid(rows/64, ceil(tok/64))` | unchanged |

The accumulation order per output element is untouched — same j order, same
order inside a codeword — so the new kernel is **bit-identical**, and that was
checked before it was wired in rather than after: `probes/probe-r8.cu` runs the
shipped kernel and the R=8 variant over the same icon and compares bytes
(`IDENTITY_OK`: 0 of 256, 0 of 17,920, 0 of 65,792 floats differing at 1, 70 and
257 tokens on 256 x 512, and 0 of 32,896 on 128 x 1024 — the ragged tails
included). Gate 7's own numbers (`min cos 0.999999950`, `max rel_rms 3.150e-04`)
are identical to the previous unit's record, which is what bit-identity predicts.

## 4. Measurement

**Kernel, same window, interleaved.** The pre-change and post-change benches
(`bench_eg2vq_vq`, 20 repeats) were run alternately — old, new, old, new — inside
one sitting at 07:32, so both saw the same GPU state. All 66 rows are in
`ab-table.txt`; the shape:

| shape | 1 tok | 115 | 306 | 565 | 1414 | 4520 |
|---|---|---|---|---|---|---|
| k/v_proj_s 512x512 | 1.37x | 1.57x | 1.51x | 1.76x | 1.76x | 1.77x |
| q_proj_s 1024x512 | 1.39x | 1.59x | 1.77x | 1.68x | 1.75x | 1.79x |
| ffn_gate_s 2048x512 | 1.27x | 1.62x | 1.49x | 1.56x | 1.56x | 1.61x |
| o_proj_f 512x2048 | 1.34x | 1.51x | 1.50x | 1.67x | 1.67x | 1.67x |
| ple_model_proj 12288x512 | 1.27x | 1.57x | 1.60x | 1.60x | 1.63x | 1.76x |

Range 1.26x-1.79x, mostly about 1.6x; the second round agrees with the first to
within 1% on every row. Achieved throughput at 4520 tokens rises from
3,674-4,337 to 6,519-7,002 GFLOP/s.

**End to end, same window, interleaved.** The same trick with real binaries: the
pre-change tree was built into `/tmp/eg2-prof/old` (verified `REG:80` vs the new
`REG:128`), and `speed-bench/eg2_embed_sweep.sh` was run against each in turn —
old, new, old, new — over the 143-page corpus (80,747 tokens):

| row | old r1 / r2 | new r1 / r2 | |
|---|---|---|---|
| batch 1 | 14.618 / 14.186 s | 11.710 / 11.828 s | 1.22x |
| batch 2 | 12.321 / 12.212 s | 10.039 / 10.021 s | 1.22x |
| batch 4 | 11.383 / 11.368 s | 9.158 / 9.135 s | 1.24x |
| batch 8 | 11.153 / 11.146 s | 8.893 / 8.903 s | 1.25x |

`ab-sweep.txt`; the four CSVs are `speed-bench/eg2_embed_vizzio_vq-ab-{old,new}-r{1,2}.csv`
with their health payloads. At batch 8 that is 7,240/7,244 -> 9,080/9,070
tokens/s.

**Device tower, gate 7.** `tests/parity/eg2vq_embd_gpu_oracle` on the 300-item /
8,059-token id set: device steady state 1,499.6 -> 2,048.6 tokens/s (1.37x),
same-process speedup over the CPU leg 84.5x. This row is cross-window and is
quoted as corroboration, not as the claim.

**Cumulative.** With the attention unit's committed CSV as the dev baseline, the
corpus at batch 8 reads 18.7998 s (2026-10-20) -> 8.903 s (2026-10-21): 2.11x
over the two units, 4,295 -> 9,070 tokens/s.

## 5. What this does NOT prove

- **No hardware counters.** No stall is attributed to L1 sectors versus L2
  latency; the ablation brackets the cost, it does not name the mechanism.
- **Single-run profiles move with the GPU, not only with the code.** Evidence in
  section 1: a byte-identical attention kernel reads 3.33 ms/launch in one
  profile and 2.43 ms in another. Anything quoted across windows here is marked.
- **V3 is an upper bound, not a measurement** (ptxas CSE).
- **The bench's error bar is about 10% on any single row**, and corpus sweeps
  drift by several percent between windows; the claim is the ratio measured
  inside one window, not a millisecond.
- **The small-shape regime is untouched.** At one token a 512-row matrix is 8
  blocks on a 56-SM part and the gain is only 1.27x. That is a grid-size problem,
  not a rows-per-thread problem, and it is not addressed here.
- **The GEMM is still the largest item** (48.4% of a batch forward's kernel
  time). This unit moved it; it did not finish it.

## 6. Rejected, with the measurement that rejected it

- **BN=128 rows, 256 threads, 50,176 B of dynamic shared** — the same R=8 with
  twice the rows per block, built and measured (`probes/probe-big.cu`,
  `probe-bn128.txt`): 0.83x at 512x512 x 306 tokens (half the blocks of the
  shipped tile), 1.23x-1.75x elsewhere, and it needs the dynamic-shared opt-in
  that this file deliberately avoids. Not carried.
- **T=2, R=8, 32-token blocks** (`B<2,8,8>`, `probe-r8-rows.txt`): 1.28x-1.38x,
  less than the shipped <4,8,8>, because halving the token tile doubles the
  index-unpack and codebook-prologue work per FMA.
- **The first probe's V1 hoisting a whole 256-column activation slab into
  registers**: it spilled and ran 5-7x slower; kept in the raw file only as the
  reason every variant is register-neutral now.

## 7. Reproduce

```sh
# the kernel gate and the tower gate
make test-eg2vq-vq                                  # EG2VQ_VQ_PASS
./tests/parity/eg2vq_embd_gpu_oracle <artifact.gguf> /data/embeddinggemma2/out/phase7-ids300.tsv

# the ablation (throwaway: build, measure, discard)
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_89 -I/data/embeddinggemma2/ds4-rs \
     -o /tmp/probe-vq probes/probe-vq.cu && /tmp/probe-vq 15

# the interleaved bench A/B
make tests/bench_eg2vq_vq && ./tests/bench_eg2vq_vq 20

# the interleaved corpus A/B
EG2_EMBED_SERVER=<old build>/ds4-server EG2_EMBED_CLI=<old build>/ds4 \
  EG2_EMBED_CSV=<out>.csv bash speed-bench/eg2_embed_sweep.sh <artifact.gguf> <pages> 8098
```

One caveat the probes share with the tree: **rebuild the binary after touching
a header**. `tests/bench_eg2vq_vq` compiles `cuda/eg2vq_vq.cuh` in, and a stale
binary reports the previous policy's numbers silently — this unit produced
exactly that mistake once (two "different" binaries measuring identically) before
rebuilding.

Files: `ab-bench-{old,new}-r{1,2}.txt` (raw bench output), `ab-table.txt` (the
joined 66-row table), `ab-sweep.txt` and the four `speed-bench` CSVs (end to
end), `probe-ablation.txt`, `probe-r8-rows.txt`, `probe-bn128.txt` (probe
output), `probes/*.cu` (the three probe sources), `nsys-kern-sum-b8-*.txt` (the
profiles' kernel summaries), `bench-vq-before.txt` (the 2026-10-20 baseline this
unit argued against).
