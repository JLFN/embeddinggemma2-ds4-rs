# The 768 projection on bf16 tensor cores (performance unit 7, 2026-11-10)

The output projection was the largest kernel left in the text tower that is not a
VQ payload GEMM: one launch, 2,451 rows x 512 x 768 = 1.93 GFLOP, 7.13 ms of a
40.0 ms forward. It now runs on the tensor cores in 0.18 ms on the same request.
This record states how the operand problem was measured before the kernel was
written, what the two forms cost, and what the unit does not claim.

## 1. What it was

`cuda/eg2vq_ops.cuh`'s `eg2vq_ops_proj768_kernel` — now
`eg2vq_ops_proj768_block_kernel` — is one block of 256 threads per ROW. The
prologue is the row's `output_norm` RMS into a dynamic-shared `e_s` of `n_embd`
floats (double sum, the shared tree the norms used before unit 6); then each
thread owns a few of the 768 outputs and walks all 512 weights of that output's
bf16 row, widening each with `u << 16` and accumulating in f32. Every block reads
the whole 768 x 512 bf16 matrix (786 KB, L2-resident) and the inner loop is
scalar. Measured on one nsys profile of the 8-item / 2,451-token request:
7,127,434 ns in ONE launch, 17.8% of the forward's 40.0 ms of kernel time, i.e.
271 GFLOP/s, one hundredth of what this card does on the payload GEMM.

The arithmetic is the graph's: `graph:339-356` builds `e` with the row's RMS and
the `output_norm` weight, dots it against each of the 768 bf16 rows, and writes
the result to that row's own plane.

## 2. The operand problem, measured before anything was written

An mma operand cannot be f32, and the projection's A is the f32 `e` row. The
kernel gate has declared `proj768 + mean pool + L2` at `rel_rms <= 1e-5` since
phase 7, so the operand format is a decision that has to be measured, not
assumed. A one-off probe (`/tmp/prec_proj.cu`, not committed: it carries no
project code, it is the door that answered this question) ran the test's own
shape (349 rows x 512, 768 outputs, bf16 weights at 0.02 sigma) in f64
accumulation with A rounded to each candidate format, then pooled and L2
normalized exactly as the gate does:

    A operand            pooled rel_rms     cosine
    bf16                 1.651e-03          0.999998636
    f16                  1.996e-04          0.999999980
    tf32                 1.995e-04          0.999999980
    bf16 hi + bf16 lo    2.495e-06          1.000000000

The bound declared before the run: the tensor-core form had to stay inside the
gate's existing `1e-5` and inside gate 7's declared `cos >= 0.999999` and
`rel_rms <= 1e-03`. One bf16 and one f16/tf32 operand are 20x to 200x outside
`1e-5`; the split pair is 4x inside it. So A is staged as `hi = bf16(e)` and
`lo = bf16(e - hi)`, and BOTH products are accumulated into the same f32
accumulators. B is the weight matrix exactly as the file stores it — bf16 — so
its widening is exact and it needs no conversion at all.

## 3. What the kernel does

`eg2vq_ops_proj768_mma_kernel` (sm_80+; the bf16 sibling of the instruction
`cuda/eg2vq_vq.cuh:350-359` and `cuda/eg2vq_attn.cuh:474-477` already use):

- a block owns TM = 64 tokens x TN = 64 outputs and runs K in 64-wide chunks;
- 8 warps are 4 row groups x 2 column groups, each warp 16 rows x 32 outputs
  (four n8 tiles), `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`;
- the row's `output_norm` RMS is the prologue, four threads per row, double sum,
  the block form's own constants;
- shared tiles are `[64][64 + 8]` bf16 (the +8 makes each row start four banks
  later, so the fragment reads hit distinct banks);
- the fragment orders are the ones the attention kernel already builds and this
  project's own harnesses already gate: A as `[row g, row g + 8] x [col, col + 8]`
  (`cuda/eg2vq_attn.cuh:554-575`), C as `c0,c1 = row g` and `c2,c3 = row g + 8`
  at columns `2q, 2q + 1` (`cuda/eg2vq_attn.cuh:611-633`);
- the epilogue writes each row to ITS OWN plane (`acc + row * n_out`), which is
  unit 27's contract.

`eg2vq_ops_proj768_pool_batch` keeps its signature and dispatches: the
tensor-core form when `n_embd % 64 == 0 && n_out % 8 == 0` (every text-tower
shape), the block form otherwise. `eg2vq_ops_proj768_pool_form(..., form)` exists
so the gate and the bench can run either one; `form = 0` is the block form,
`1` the tensor-core form, `-1` the shipped dispatch.

## 4. The A/B

One process, both forms interleaved (`proj768-ab-bench.txt`, `-run2.txt`):

    shape                              rows   ms_block   ms_mma  speedup
    proj768 512x768, 2451 rows         2451     6.5311   0.3018   21.64x
    proj768 512x768, 294 rows           294     0.8953   0.0706   12.68x
    proj768 512x768, 27 rows             27     0.1543   0.0262    5.89x

Round 2 reproduces every row to the printed digit (21.63x / 12.67x / 5.87x). The
bench's block column reads 6.53 ms where the profile reads 7.13 ms: same kernel,
different measurement (the bench times a synthetic f32 `h` with its own events,
the profile times the real tower), and the bench columns are the comparable pair.

In the real forward (`nsys-kern-sum-b8-proj-mma.txt`, same 8-item /
2,451-token request as the previous unit's profile):

    kernel                                    before     after
    proj768 (ONE launch)                      7.13 ms    0.178 ms   (40.0x)
    total kernel time of the request         40.0 ms    33.0 ms

0.178 ms for 1.93 GFLOP is 10.8 TFLOP/s of useful work (the mma does 2x that,
because of the hi/lo split).

End to end, one fresh server per run, both round pairings, 143 pages / 80,747
tokens (`corpus-proj-ab-{old,new}-{1,2}.csv`):

    round 1                      old               new       ratio
    batch 1   3.841 s  21,021.9 tok/s   3.677 s  21,959.3 tok/s   1.04x
    batch 4   1.794 s  45,010.1 tok/s   1.569 s  51,450.0 tok/s   1.14x
    batch 8   1.531 s  52,725.6 tok/s   1.371 s  58,917.2 tok/s   1.12x

    round 2                      old               new       ratio
    batch 1   3.799 s  21,254.0 tok/s   3.664 s  22,038.4 tok/s   1.04x
    batch 4   1.799 s  44,876.5 tok/s   1.568 s  51,508.2 tok/s   1.15x
    batch 8   1.535 s  52,594.5 tok/s   1.368 s  59,010.6 tok/s   1.12x

The single-item p50 row is NOT this unit's number: round 1 reads 16.0 -> 15.4 ms
and round 2 reads 24.2 -> 14.6 ms, the same per-window disagreement the previous
unit recorded on a one-item request; the corpus rows are the claim.

## 5. The gates

- `make test-eg2vq-ops` -> `EG2VQ_OPS_PASS`. `proj768 + mean pool + L2` (the
  shipped tensor-core form) reads rel RMS 2.764e-06, max abs 3.371e-07 against
  the unchanged 1e-5 bound; the block fallback reads 4.121e-07; the two forms
  differ by 2.75e-06; the three-item packed case reads 2.47e-06 (tensor-core) and
  3.65e-07 (block), each item against its own reference.
- `make test-eg2vq-attn` -> `EG2VQ_ATTN_PASS`, `make test-eg2vq-vq` ->
  `EG2VQ_VQ_PASS`: neither kernel family was touched and both still pass.
- Gate 7 run A, 300 items: `cos(device,cpu) 0.999999768`, `max rel_rms 6.814e-04`
  (declared `>= 0.999999` / `<= 1e-03`). Run A's worst rel_rms is the recorded
  6.813e-04 moved by one digit.
- Gate 7 run B, the twelve longest documents: `cos 0.999999744`,
  `max rel_rms 7.156e-04`, inside the same bounds. The old form was not re-run on
  run B at this boundary, so this record does not claim a before/after for it.
- Gate 27: `PASS` — a batched item's vector is its single-item vector byte for
  byte on both legs (CUDA and CPU), the 8192-row budget binds where advertised,
  and every malformed block is refused by name.
- `tests/run.sh` fast path: exit 0 through gates 22-27 and the rule 19 gate.
- P-1's five checks return zero; `sha256sum -c SHA256SUMS` passes 14/14.

One harness defect was found and fixed on the way: `tests/test_eg2vq_ops.cu`'s
`report()` computed `bad = rel > bound`, which is FALSE for a NaN, so a NaN
output printed "ok". It is now `bad = !(rel <= bound)`.

## 6. What this unit does not claim

- Not the fastest this GEMM can be. 10.8 TFLOP/s of useful work against this
  card's bf16 tensor peak leaves room in the tile (64 x 64, K chunk 64) and in
  the 2x mma work the split costs; no bound is claimed.
- The 27-row call is 5.89x and still 26.2 us: at that size the pool launch and
  the kernel's own fixed cost are a larger share than the GEMM, and no
  call-sized dispatch was added (the reason is unit 6's: a dispatch that picks
  arithmetic from a call's size is a contract waiting to break).
- Gate 7 run B has no before/after here (section 5).
- Nothing about the expression: the norm, the per-row plane and the pool stay
  where the graph puts them, and no model, artifact, loader or route changed.
- The vision tower does not call this kernel.
- Not re-run: the deep path (`tests/run.sh --deep`), gate 20's route gate, the
  loader digest and gate 5. The unit is one kernel family, its gate and its
  bench.

## 7. Commands

    (cd ds4-rs && make test-eg2vq-ops)                    -> EG2VQ_OPS_PASS
    (cd ds4-rs && make test-eg2vq-attn test-eg2vq-vq)     -> the two untouched gates
    (cd ds4-rs && make tests/bench_eg2vq_ops && ./tests/bench_eg2vq_ops 200)

    # gate 7, both legs
    ./ds4-rs/tests/parity/eg2vq_embd_gpu_oracle <artifact> out/phase5-ids.tsv \
        --items 300 --ref out/phase5-artifact-raw-stopped-1628Z.txt --out <out>
    ./ds4-rs/tests/parity/eg2vq_embd_gpu_oracle <artifact> out/phase7-long-ids.tsv \
        --out <out> --out-cpu <out-cpu>

    # the corpus A/B (one server per run, both builds)
    EG2_EMBED_BATCHES="1 4 8" EG2_EMBED_REPEATS=3 EG2_EMBED_CSV=... \
        EG2_EMBED_SERVER=<old|new>/ds4-server sh speed-bench/eg2_embed_sweep.sh \
        <artifact> out/live-demo/pages 8098

    # the batch contract
    (cd .. && sh tools/eg2_embed_batch_gate27.sh)
