# The tower's attention kernel: where a forward's time went, what one change bought, and one idea that did not survive measurement (2026-10-20)

This is the optimization half of the owner's performance request ("lets start to
optimize so we can have fast prefill decode etc"). It is written the way the
baseline record beside it is: every number has a command behind it, what is not
measured is labelled as such, and one of the two changes this unit originally
shipped is recorded here because it was measured, rejected and reverted.

The work was driven by attribution, not by a guess. The baseline record
(`README.md`, section "Where the cost most plausibly is") listed three hypotheses
and said plainly that none was measured. The first profile of a real forward
settled which one it was — and it was none of the three.

## 1. Attribution first: nsys on real forwards

`nsys profile` over the CLI embedding path and over a server batch request, one
item and one packed 8-item / 2451-token block:

| kernel | single 294-token item | packed batch (2451 tokens, 8 items) |
|---|---|---|
| `eg2vq_attn_kernel` | 31.5 ms, 24 launches, **47.9%** of kernel time | 337.2 ms, 48 launches (incl. warmup), **58.9%** |
| `eg2vq_vq_gemm_kernel` | 30.8 ms, 217 launches, 46.9% | 208.8 ms, 434 launches, 36.5% |
| all other kernels (rms, rope, gelu, pool, embed, proj768) | 3.4 ms, 3.8% | 27 ms, 4.6% |

Two facts make this decisive:

- The attention kernel was doing 1.2% of the forward's arithmetic and paying 48%
  of its kernel time. At 294 tokens the layers' attention is ~10 GFLOP against
  the GEMM's ~67 GFLOP.
- Both kernels were running at a few percent of the part's ~35.5 TFLOP/s of FP32:
  attention at 0.29-0.46 TFLOP/s, the GEMM at 2.2-2.9 TFLOP/s.

The kernel-level cause is visible in the code: pass 1 read the K rows one `float`
at a time while the 32 lanes of a warp walked 32 different rows (`a` broadcasts
from q, `b` is the strided K row), so every load instruction touched 32 separate
32-byte sectors to deliver 128 B; pass 3 wrote one float per lane. `cuobjdump
-res-usage` reports `REG:40 SHARED:32768` for a 128-thread block, and sm_89 gives
an SM 100 KB of shared memory — three blocks, 12 warps of 64 — so the kernel was
also occupancy-limited, but section 3 shows that this was NOT the lever.

## 2. What changed

One change shipped: **both ends of the kernel are vectorized** where the shape
allows it. With `head_dim % 128 == 0` and 16-byte-aligned `q`/`k`/`v`/`out`, pass
1 reads the K rows as `float4` and pass 3 reads V and writes the output as
`float4`; otherwise the original scalar loops run unchanged. The accumulation
ORDER IS UNCHANGED element for element — the scalar loop fed accumulator chain
`i%4` from element `i`, and both vector loops keep that mapping — so the vector
path is bit-identical to the scalar one. The kernel also gained a guard: a row
needing more keys than the plane holds returns without writing, rather than
walking past the block's shared window.

One change was tried and REVERTED: sizing the score plane per call. The idea was
that the plane was allocated for the worst case the tower allows (a 2048-token
item) while serving items of a few hundred tokens, so a per-call plane should buy
occupancy. It never did. On the shape bench it looked like a large win (the item
plane measured 2.5-3.7x over the scalar kernel), but driven end to end over the
143-page corpus — the workload this exists for — the item-sized plane was SLOWER
than the fixed ceiling plane on every row, and a direct sweep with plane size as
the only variable shows a non-monotonic curve that occupancy cannot explain:
| shape | plane = longest item | plane = 1025 (window) | plane = 2048 (ceiling) |
|---|---|---|---|
| `page294_s` | 0.258 ms | 0.419 ms | 0.302 ms |
| `page565_s` | 1.378 ms | 1.232 ms | 0.977 ms |
| `max1414_s` | 3.160 ms | 4.345 ms | 3.325 ms |
| `batch8x306_s` | 1.890 ms | 2.444 ms | 2.032 ms |
| `batch8x565_s` | 10.748 ms | 9.144 ms | 6.903 ms |

(probe: the kernel launched directly with `plane_keys` as the only variable, 25
interleaved repetitions, the vector path held fixed). Smaller is not better, the
ordering is not monotonic, and the settings that win are not the ones the
occupancy model predicts. The corpus is unambiguous, at least: the item-sized
plane measured 13.09 s at batch 8 against 11.05 and 10.60 s for the fixed ceiling
plane, run back to back in one sitting (18:07-18:17 by the session log; the CSVs
themselves carry no timestamps — see section 5 on why the window matters). So the
plane is the item ceiling — the size the kernel has always used, no plumbing
through the caller, no tuning constant — and the vectorization is the whole win.

A second variant was also tried and measured as a wash: declaring the plane
dynamically instead of statically at the same 32,768 B. It is not in the diff.

## 3. The measurement

`tests/bench_eg2vq_attn` (new, `make bench-eg2vq-attn`) times the kernel alone on
the tower's own shapes, 20-30 repeats, reporting the exact FLOPs each shape's
window implies. Before is the kernel as it stands on `dev` (10e73aa0); after is
this branch.

| shape | rows | keys | ms before | ms after | speedup | GFLOP/s before | GFLOP/s after |
|---|---|---|---|---|---|---|---|
| `cli115_s` (the CLI's item) | 115 | 13,225 | 0.1860 | 0.0625 | 2.98x | 291 | 867 |
| `page294_s` (4 PDF pages) | 294 | 86,436 | 0.9646 | 0.3021 | 3.19x | 367 | 1,172 |
| `page565_s` (the corpus mean) | 565 | 316,469 | 3.8973 | 1.0966 | 3.55x | 333 | 1,182 |
| `max1414_s` (longest frozen item) | 1,414 | 1,186,694 | 12.0750 | 3.6536 | 3.30x | 403 | 1,330 |
| `page294_f` (head_dim 512, no mask) | 294 | 86,436 | 1.6865 | 0.5499 | 3.07x | 420 | 1,288 |
| `max1414_f` | 1,414 | 1,999,396 | 42.8206 | 11.0434 | 3.88x | 383 | 1,483 |
| `batch8x306_s` (packed 8 items) | 2,448 | 749,088 | 6.6560 | 2.0500 | 3.25x | 461 | 1,497 |
| `batch8x565_s` | 4,520 | 2,531,752 | 22.8270 | 7.0246 | 3.25x | 454 | 1,476 |
| `batch8x565_f` | 4,520 | 2,553,800 | 44.9843 | 14.3275 | 3.14x | 465 | 1,460 |

Every shape is 3.0x to 3.9x faster and every shape is still far from the part's
ceiling (~4% of FP32 peak). The `after` column is the committed
`bench-attn-after.txt` verbatim; the `before` column is `bench-attn-before.txt`.
A single shape's median moves a few percent between bench runs (the long 1414_s
case moved 3.27-3.65 ms across runs in this session), so treat any one row as
±10% and the row-to-row consistency as the finding.

End to end, on the server route. Every row is a committed CSV, and **read the
window caveat before comparing across rows**: corpus-level numbers taken more than
a few minutes apart on this machine are not comparable (section 5). The order the
rows are in is the order they were taken, from this session's command log — the
CSVs carry no timestamp of their own.

| | build | corpus artifact (`speed-bench/`) | batch 8 (80,747 tokens) |
|---|---|---|---|
| baseline policy | `dev` (scalar, static plane) | `eg2_embed_vizzio.csv` | 18.7998 s, 4,295 tok/s |
| item-sized plane | vectorized, dynamic | `..._attn-itemplane.csv` | 13.0896 s, 6,168.8 tok/s |
| fixed ceiling plane | vectorized, dynamic | `..._attn-plane2048{,_r2}.csv` | 11.0454 / 10.5965 s, 7,310.5 / 7,620.2 tok/s |
| fixed 1536 plane | vectorized, dynamic | `..._attn-plane1536{,_r2}.csv` | 10.5544 / 10.8699 s, 7,650.5 / 7,428.5 tok/s |
| **this branch** | vectorized, static plane | `..._attn-vec4{,_r2,_r3}.csv` | 11.6541 / 11.7179 / 11.6584 s, 6,928.7 / 6,890.9 / 6,926.1 tok/s |

Two runs of each policy inside the 18:07-18:17 window could not separate 1536 from
2048 (the sign flips per row, all differences inside the 2-4% run-to-run spread),
which is why the shipped constant is the ceiling rather than a tuned value.

The same request profiled by nsys, before and after the vectorization (8 PDF
pages, 2,451 tokens):

| | before | after | change |
|---|---|---|---|
| attention in the profile | 337.2 ms | 159.7 ms | 2.11x |
| whole kernel time in the profile | 572.8 ms | 401.1 ms | 1.43x |
| VQ GEMM in the profile | 208.8 ms | 212.6 ms | unchanged (not touched) |

(That pair is from the item-sized-plane build, which is strictly worse than the
shipped one end to end; the nsys report was taken before the plane was reverted.)

## 4. Correctness, on the gates that already existed

- `make test-eg2vq-attn` — `EG2VQ_ATTN_PASS`, every case: min cosine 1.000000000
  against the CPU reference for 26 to 1414 tokens on both layer geometries, the
  mask edge (512 sees / 513 blind), two items packed into one block each reading
  only its own item, and the 5-head refusal leaving the output untouched. The
  `maxabs` and `rel` numbers are identical to the pre-change kernel's, which is
  the expected result of a bit-identical accumulation order.
- `tests/parity/eg2vq_embd_gpu_oracle <artifact> out/phase7-ids300.tsv` — the
  end-to-end device tower against the CPU tower: `EG2VQ_GPU_PASS`, min
  cos(device, cpu) 0.999999950 and max rel_rms 3.150e-04 inside its declared
  cos >= 0.999999 / rel_rms <= 1e-03.
- Nothing in the wire, the caps or the workspace changed: `DS4_EG2VQ_MAX_TOKENS`
  is still the ceiling, `/health`'s embed caps are byte-identical, and the
  launcher still refuses any head count but the tower's 4.

## 5. What this does NOT prove

- **No hardware counters.** `ncu` refuses on this machine
  (`ERR_NVGPUCTRPERM`: performance counters need a driver parameter change, i.e.
  root on a shared machine). Everything here is wall/kernel duration plus
  `cuobjdump` resource usage — enough to A/B a change, not enough to attribute a
  stall to L1 sectors versus L2 latency. That is exactly why the plane-size curve
  above is reported as measured and unexplained rather than explained.
- **Corpus sweeps drift between windows, and the bench is not as steady as a first
  look suggests.** Three sweeps of ONE build at 18:21-18:25 read 11.654, 11.718 and
  11.658 s at batch 8; four sweeps of two other builds at 18:07-18:17 read
  10.596-11.045 s; and rebuilding the 18:07-18:17 configuration exactly gave 11.717 s
  in the later window. The isolated kernel bench is the instrument of record here and
  it did not move between the two windows by more than a few percent — but a QA
  re-measurement of the SAME static binary minutes apart saw `page294_s` at 0.3021
  vs 0.2724 ms and `batch8x565_s` at 7.6520 vs 6.9530 ms, so the bench's practical
  error bar is about 10% on any single row, not 1%. The before/after ratios (3.0x to
  3.9x) are far above that, and the QA reproduced 9/9 rows of the committed after-file
  to within 0.3%, which is why the ratio — not any individual millisecond — is the
  claim. Compare builds only inside one window, or interleaved.
- **3-4x is not the end.** The kernel now runs at 0.9-1.6 TFLOP/s, a few percent
  of an FP32 peak of ~35.5 TFLOP/s. The structural limit left is that one block is
  one query ROW, so each of an item's rows re-reads its item's K and V from L2:
  an item of L tokens with 4 heads reads 4*L*L*head_dim*2 floats for the layer.
  Sharing K/V across a tile of query rows (or a flash-style single pass, as
  `cuda/eg2vq_vis.cuh` already does for the vision tower) is the next lever — a
  rewrite rather than a tuning change.
- **The bottleneck has moved.** With attention at ~160 ms the VQ GEMM is now the
  largest single item left in a batch profile (212.6 ms of 401.1 ms before the
  plane revert, 53%), at the same ~2.9 TFLOP/s. Its own limiter is not
  established here and must be measured before it is changed; the instrument for
  that (`tests/bench_eg2vq_vq.cu`, the tower's matrices across token counts) is
  in the tree.

## 6. Reproduce

The raw rows are committed beside this record, one file per policy, each holding
the 143-page corpus sweep it is named for (the `srv_batch_8` row identifies each
one): `speed-bench/eg2_embed_vizzio_attn-itemplane.csv` (the rejected item-sized
plane, 13.0896 s / 6,168.8 tok/s), `-plane2048.csv` and `-plane2048_r2.csv` (the
fixed ceiling plane, 11.0454 / 10.5965 s), `-plane1536.csv` and `-plane1536_r2.csv`
(the fixed 1536 plane, 10.5544 / 10.8699 s), and `-vec4.csv` with `_r2.csv` and
`_r3.csv` (the shipped build, 11.6541 / 11.7179 / 11.6584 s), under the single
`speed-bench/eg2_embed_vizzio_attn_health.json` the shipped build served. The
harness's own output either side of the change is `bench-attn-before.txt` and
`bench-attn-after.txt`. The before column was taken from the `dev` tip 10e73aa0
with the bench harness built against that kernel, and the QA pass rebuilt the
`dev` kernel from its own sources and matched it within 7.4%.

A caveat on provenance, because the QA pass checked it and it matters: these CSVs
carry no timestamp, so the ordering the tables rely on (which sweep ran in which
window) comes from this session's command log, not from the artifacts. Read the
numbers as the record of measurements that were taken, and read section 5 before
comparing any two rows taken apart in time.

```sh
cd ds4-rs
make ds4 ds4-server tests/bench_eg2vq_attn tests/test_eg2vq_attn
make test-eg2vq-attn                       # EG2VQ_ATTN_PASS
make bench-eg2vq-attn                      # the per-shape table above
./tests/parity/eg2vq_embd_gpu_oracle \
  /data/embeddinggemma2/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
  /data/embeddinggemma2/out/phase7-ids300.tsv
# the attribution and the end-to-end rows: nsys profile -o /tmp/x ./ds4-server ... ,
# then nsys stats --report cuda_gpu_kern_sum /tmp/x.nsys-rep
sh speed-bench/eg2_embed_sweep.sh \
  /data/embeddinggemma2/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
  /data/embeddinggemma2/out/live-demo/pages 8098
```
