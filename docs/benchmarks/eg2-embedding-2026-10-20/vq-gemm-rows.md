# The VQ payload GEMM: how many rows one thread should own

embeddinggemma2 performance unit, 2026-10-21 (engine commit 8d618ecb). The kernel
is `eg2vq_vq_gemm_kernel` in `cuda/eg2vq_vq.cuh` (the dim-4 DQVQ payload GEMM, the
gemma_vq device path's weight consumer). The change is one number: **output rows
per thread, 4 -> 8**, plus the block shape that follows from it. The tile, the
shared-memory budget, the launcher's guard and grid math, and the accumulation
order are unchanged.

## 1. Attribution: the largest single item left, and the ablation that named its cost

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

Raw: `nsys-kern-sum-b8-before-vq-r8.txt`, `nsys-kern-sum-b8-vq-r8.txt` (both are
the `cuda_gpu_kern_sum` report of the two profiles, with the per-kernel totals
summed to 401.2 and 287.8 ms).

## 2. The instrument: an ablation, because there are no counters

`ncu` refuses on this machine (`ERR_NVGPUCTRPERM`: performance counters need a
driver parameter, i.e. root on a shared machine), so "which part of this kernel
costs the time" was answered by building the kernel with one component removed at
a time and timing it. `probes/probe-vq.cu` is the pre-change body in a
`template<int V>` with every variant register-neutral except the one that exists
to be an example of spilling; `cuobjdump -res-usage` shows `STACK:0 LOCAL:0` on
six of the seven and `STACK:4096` on the seventh. The numbers are wrong for every
V != 0 by construction; only the time is read. 15 repeats, medians, one sitting
(`probe-ablation.txt`).

| variant | k/v 306 | k/v 4520 | gate 306 | gate 4520 | ple 306 | ple 4520 |
|---|---|---|---|---|---|---|
| V1 one x load per K step, reused by four tokens | 2.03x | 2.47x | 1.98x | 2.59x | 2.46x | 3.05x |
| V2 no codebook gather, no index read | 1.09x | 1.09x | 1.09x | 1.08x | 1.09x | 1.18x |
| V4 all loads kept, the outer product replaced by FADDs | 1.02x | 1.01x | 1.12x | 1.00x | 1.02x | 1.03x |
| V5 four x loads, block-uniform address | 2.41x | 3.45x | 3.17x | 3.65x | 3.48x | 4.23x |
| V6 the whole 256-column slab per token in registers | 0.52x | 0.17x | 0.19x | 0.14x | 0.14x | 0.15x |

(faster than baseline = larger. V3, "FMA only", is in the raw file and is an
upper bound only: with identical operands ptxas CSEs the chain, which is why it
reports 26.5-67.2 TFLOP/s, above this part's ~35.5 TFLOP/s of fp32.)

What that says, in order of how hard it is to argue with:

- **The arithmetic is not the limit.** Replace the whole outer product with a few
  FADDs and keep every load: 1.00x-1.12x. The multiply-add is hidden behind the
  loads.
- **The codebook gather is not the limit either.** Cut the shared-memory gather
  and the index read to a constant: 1.08x-1.18x.
- **The activation load is the limit.** The two variants that touch it are the
  only ones that move the kernel: cutting its instructions to a quarter (V1:
  2.0x-3.1x) or keeping the instructions and making the addresses block-uniform
  (V5: one 32-byte sector per instruction instead of sixteen: 2.4x-4.2x).

The model behind the fix follows from the same geometry. Per thread per K step
the kernel loads a `float4` per token — 16 bytes — and spends `T*R*4` scalar
multiply-adds on it, so it pays **4/R bytes of activation per multiply-add**:
R=4 pays 4 B, R=8 pays 2 B. The load side is expensive per instruction and not
per byte: a warp's sixteen `tx` lanes read the same column chunk of sixteen
different token rows `t_stride` apart, i.e. sixteen 32-byte sectors per
instruction, and that path costs several times what the FMA path costs per step.
More rows per thread is the only register-blocking move that reduces it without
changing what is loaded.

## 3. What shipped

| | before | after |
|---|---|---|
| block | 16 x 16 threads | 16 x 8 threads |
| per thread | 4 tokens x 4 rows = 16 accumulators | 4 tokens x 8 rows = 32 accumulators |
| tile | 64 tokens x 64 rows | unchanged |
| index slab | 64 x 68 uint16 (8.5 KB), one unpack pass | same slab, two unpack passes (128 threads cover 32 rows per pass) |
| shared per block | 41,472 B, static | 41,472 B, static |
| registers | REG:80, no spill | REG:128, STACK:0 LOCAL:0 |
| launcher's guard and grid math | `rows % 64`, `cols % 256`, `t_stride >= cols`, `grid(rows/64, ceil(tok/64))` | unchanged (the block dimension is the only launcher change) |

The accumulation order per output element is untouched — same K order, same order
inside a codeword — so the new kernel is **bit-identical**. That is a claim about
bytes, and it is backed by a driver that feeds the same payload and activation
bytes to the OLD kernel (built from the parent commit's header, renamed so both
can link) and the SHIPPED one in one process and compares raw float bytes:
`qa-identity/` (committed with this unit, `qa-identity/output.txt`), 15 cases —
256x512 at 1/64/70/257/512 tokens, 12288x2048 at the same token counts, rows
exactly 64, `t_stride > cols`, ragged tails — **12,492,416 floats compared, 0
differing**, and both kernels refusing 9 of 9 invalid shapes. `compute-sanitizer
--tool memcheck` on the same driver, with the activation allocated at exactly
`n_tok` rows (the strongest out-of-bounds probe for the removed fast path),
reports 0 errors.

`probes/probe-r8.cu` is the instrument this change was developed with; it
compares a templated R=8 kernel against whatever the header says at compile time,
which on 2026-10-21 07:19 meant the pre-change R=4 kernel (`probe-r8-rows.txt`,
`IDENTITY_OK`) and today means the shipped R=8 kernel, so re-running it now
checks the transcription, not the change. Section 7 says how to compile it
against the pre-change header if the historical comparison is wanted.

## 4. Measurement

**Kernel, same window, interleaved.** The pre-change and post-change benches
(`bench_eg2vq_vq`, 20 repeats) were run alternately — old, new, old, new — inside
one sitting at 07:32, so both saw the same GPU state. All 66 rows are in
`ab-table.txt` (`ab-bench-{old,new}-r{1,2}.txt` are the raw outputs; the file's
old column is round 1). The shape, round-paired so no row mixes windows:

| shape | 1 tok | 115 | 306 | 565 | 1414 | 4520 |
|---|---|---|---|---|---|---|
| k/v_proj_s 512x512 | 1.37x | 1.57x | 1.51x | 1.76x | 1.76x | 1.77x |
| q_proj_s 1024x512 | 1.39x | 1.59x | 1.77x | 1.68x | 1.75x | 1.79x |
| ffn_gate_s 2048x512 | 1.27x | 1.62x | 1.49x | 1.56x | 1.56x | 1.61x |
| o_proj_f 512x2048 | 1.34x | 1.51x | 1.50x | 1.67x | 1.67x | 1.67x |
| ple_model_proj 12288x512 | 1.27x | 1.57x | 1.60x | 1.60x | 1.63x | 1.76x |

Round 1 pairs (old-r1/new-r1) span 1.26x-1.79x; round 2 pairs span 1.23x-1.75x.
The spread between the two pairings is not the new kernel: the OLD binary's first
round is up to 10.5% slower than its own second round on the first rows of the
sitting (k/v_proj_s 4520: 0.6451 then 0.5796 ms), a warm-up effect on whichever
binary runs first, while the new binary's two rounds agree to 0.3-2.7%. The
conservative reading is therefore **1.23x-1.75x**, and the honest headline is
"about 1.6x on most rows". Achieved throughput at 4520 tokens rises from
3,674-4,337 to 6,519-7,002 GFLOP/s.

**End to end, same window, interleaved.** The same trick with real binaries: the
pre-change tree was built into `/tmp/eg2-prof/old` (verified `REG:80` against the
new `REG:128`), and `speed-bench/eg2_embed_sweep.sh` was run against each in turn
— old, new, old, new — over the 143-page corpus (80,747 tokens):

| row | old r1 / r2 | new r1 / r2 | speedup |
|---|---|---|---|
| batch 1 | 14.618 / 14.186 s | 11.710 / 11.828 s | 1.22x |
| batch 2 | 12.321 / 12.212 s | 10.039 / 10.021 s | 1.22x |
| batch 4 | 11.383 / 11.368 s | 9.158 / 9.135 s | 1.24x |
| batch 8 | 11.153 / 11.146 s | 8.893 / 8.903 s | 1.25x |

(`ab-sweep.txt`; the four CSVs are `speed-bench/eg2_embed_vizzio_vq-ab-{old,new}-r{1,2}.csv`
with their health payloads, and every ratio here is round-paired and agrees with
its other pairing to within 1%.) At batch 8 that is 7,240/7,244 -> 9,080/9,070
tokens/s.

**Device tower, gate 7.** `tests/parity/eg2vq_embd_gpu_oracle` on the 300-item /
8,059-token id set: `EG2VQ_GPU_PASS`, `min cos(device, cpu) 0.999999950` and
`max rel_rms 3.150e-04` — numerically identical to the previous unit's record,
which is what bit-identity predicts — device steady state 2,060.9 tokens/s against
the previous unit's recorded 1,499.6 (attention-occupancy.md:147), same process,
85.0x over the CPU leg. Three runs of the shipped kernel inside an hour read
2,048.6 / 2,008.9 / 2,060.9 tokens/s, so the row is quoted as "about 2,000, up
from about 1,500", and both its sides are cross-window. Raw:
`gate7-embd-gpu-oracle.txt` (the third run).

**Cumulative.** With the attention unit's committed CSV as the dev baseline, the
corpus at batch 8 reads 18.7998 s (2026-10-20, `speed-bench/eg2_embed_vizzio.csv`)
-> 8.903 s (2026-10-21): 2.11x over the two units, 4,295 -> 9,070 tokens/s.

## 5. What this does NOT prove

- **No hardware counters.** No stall is attributed to L1 sectors versus L2
  latency; the ablation brackets the cost, it does not name the mechanism.
- **Single-run profiles move with the GPU, not only with the code.** Evidence in
  section 1: a byte-identical attention kernel reads 3.33 ms/launch in one
  profile and 2.43 ms in another. Anything quoted across windows here is marked.
- **V3 is an upper bound, not a measurement** (ptxas CSE).
- **The bench's error bar is about 10% on any single row**, and corpus sweeps
  drift by several percent between windows; the claim is the ratio measured
  inside one window, not a millisecond. The two round pairings above are the
  honest spread of that same measurement.
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
  index-unpack and codebook-prologue work per multiply-add.
- **Holding the activation slab in registers** (V6 above, and the mistake the
  first version of the ablation probe made): the variant spills
  (`STACK:4096`) and runs at 0.14x-0.19x of baseline on the large shapes and
  0.52x on the smallest — i.e. 2x to 7x slower. That is why every other variant
  is register-neutral by construction.

## 7. Reproduce

```sh
# the kernel gate and the tower gate (~7 min for the second one)
make test-eg2vq-vq                                  # EG2VQ_VQ_PASS
./tests/parity/eg2vq_embd_gpu_oracle \
    /data/embeddinggemma2/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
    /data/embeddinggemma2/out/phase7-ids300.tsv

# the bit-identity driver (needs the pre-change header beside it; see build.sh)
cd docs/benchmarks/eg2-embedding-2026-10-20/qa-identity && bash build.sh && ./qa-identity

# the ablation, as committed (the probe pins its own R=4 tile; see its PINNED note)
nvcc -O3 -g -lineinfo --use_fast_math -std=c++17 -arch=sm_89 \
     -Xcompiler -march=native -Xcompiler -pthread -I/data/embeddinggemma2/ds4-rs \
     -o /tmp/probe-vq probes/probe-vq.cu && /tmp/probe-vq 15

# the interleaved bench A/B
make tests/bench_eg2vq_vq && ./tests/bench_eg2vq_vq 20

# the interleaved corpus A/B
EG2_EMBED_SERVER=<old build>/ds4-server EG2_EMBED_CLI=<old build>/ds4 \
  EG2_EMBED_CSV=<out>.csv bash speed-bench/eg2_embed_sweep.sh <artifact.gguf> <pages> 8098
```

Two traps this unit paid for, both worth keeping:

- **Rebuild every binary after touching a header.** `tests/bench_eg2vq_vq`
  compiles `cuda/eg2vq_vq.cuh` in, and a stale binary reports the previous
  policy's numbers silently. This unit produced exactly that mistake once (two
  "different" binaries measuring identically) before rebuilding and confirming
  with `cuobjdump` that the registers differ (80 vs 128).
- **The probes are pinned to the header they were measured against.**
  `probe-vq.cu` carries its own `PVQ_T`/`PVQ_R`; `probe-r8.cu` and
  `probe-big.cu` compare against the shipped kernel and must be compiled against
  the pre-change header (`git show 31285531:cuda/eg2vq_vq.cuh`) to mean what
  their committed outputs show. Rebuilding `probe-vq.cu` against the current
  header without the pin crashed with an illegal memory access — a rebuilt
  instrument that cannot say what the record says is worse than no instrument.

Files: `ab-bench-{old,new}-r{1,2}.txt` (raw bench output), `ab-table.txt` (the
joined 66-row table), `ab-sweep.txt` and the four `speed-bench` CSVs (end to
end), `probe-ablation.txt`, `probe-r8-rows.txt`, `probe-bn128.txt` (probe
output), `probes/*.cu` (the three probe sources), `qa-identity/` (the bit-identity
driver, its runner for the old kernel and its output), `gate7-embd-gpu-oracle.txt`
(the tower gate's own output), `nsys-kern-sum-b8-*.txt` (the profiles' kernel
summaries), `bench-vq-before.txt` (the 2026-10-20 baseline this unit argued
against).

## 8. Rule-19 QA pass

The project's recorded QA model (`deepseek-v4.1-CC-flash`) was run over the unit
at the boundary. Its verdict on the first submission was FAIL, and no code defect
was among the findings: bit-identity was confirmed independently (12,492,416
floats, 0 differing, both kernels in one process), the memory-safety run was
clean, all three gates passed on rebuilt binaries, and the edge and refusal paths
were intact. What it rejected was the record and the process around it, and each
finding was fixed rather than argued with:

| finding | disposition |
|---|---|
| F1, F13 the committed identity/ablation probes can no longer demonstrate what the record says (one now self-compares, the other crashes when rebuilt from the documented command) | `probe-vq.cu` pins its own R=4 tile (`PVQ_T`/`PVQ_R`) and its V0 reproduces; the record names each probe's scope and says how to compile the others against the pre-change header; the identity claim now rests on the committed driver, not on the probe |
| F2, F3, F15 no code defect: identity at byte level, sanitizer clean, edge and refusal paths intact | kept as the unit's evidence (the driver is committed) |
| F4, F11 the artifact's build command was undocumented and one quoted figure had no artifact | the exact command (`-Xcompiler -march=native` included) is in section 7; the register-hoisting figure now comes from the measured V6 row |
| F8 "round 2 agrees within 1% on every row" was false (three rows at 1.4-2.7%) | replaced by the measured round-pairing spread in section 4 |
| F9 the published range used only the more favourable old round | both round pairings are published, the conservative one is the headline, and the reason (the old binary's warm-up) is stated |
| F10 "27-67 TFLOP/s" did not trace | replaced by the committed artifact's 26.5-67.2 TFLOP/s |
| F12 the gate-7 corroboration pair had no committed raw output | `gate7-embd-gpu-oracle.txt` is committed and the "before" value is cited to the file that recorded it |
| F14 two wording imprecisions (the launcher did change; "bytes per FMA" counted a 4-wide group) | fixed in section 3 and section 2 |
| F16 the rule-19 gate had no fresh report for this unit | `qa-evidence/qa-report.md` and `qa-evidence/qa-findings-vq-r8.md` in the outer repository, written at this boundary, `bash tests/qa-gate.sh` exit 0 |
