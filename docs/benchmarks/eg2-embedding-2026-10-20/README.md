# EmbeddingGemma-2 embedding path: baseline speed record (2026-10-20)

The embedding path had no benchmark before this record. It is the first
measurement of it, taken so that any later optimization is argued against a
number instead of a feeling.

## What was measured, on what

| | |
|---|---|
| Artifact | `EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf`, sha256 `5bcf2216…`, 417.2 MiB, 1376 tensors |
| Family | `gemma_vq` (embedding; no language-model head) |
| Backends | `cuda` (RTX 4070 SUPER, 12 GB, CUDA 13.3) and `cpu` (28 threads) |
| Host | `vizzio`, 31 GiB RAM |
| Workload | the 143 pages of *Riksavtalet 2023-2025* (Kollektivavtal2023_230607_0.pdf), 36,385 words, 80,747 tokens through the tower as Swedish legal text |
| Route | `POST /v1/embeddings`, task `Retrieval-document` for the corpus, `Retrieval-query` for single items |
| Harness | `speed-bench/eg2_embed_sweep.sh` (this repo's speed-bench CSV standard) |
| Raw result | `speed-bench/eg2_embed_vizzio.csv` and its `_health.json` |

Every server row is **wall clock around the request**, because the route returns
no elapsed field and `GET /v1/stats` counts only the chat/completion/responses
surfaces — its `routes` object has no embed row at all, so the server neither
counts nor times an embedding request. The CLI rows carry the port's own number
too: the `secs` field in the JSON the CLI prints.

## The numbers

| Row | Items | Tokens | Wall (s) | Tokens/s |
|---|---|---|---|---|
| `cli_cold` (process + load + 1 item) | 1 | 115 | 0.681 | 169 |
| `srv_single_p50` | 1 | 115 | 0.0449 | 2,561 |
| `srv_single_p95` | 1 | 115 | 0.0544 | 2,113 |
| `srv_batch_1` | 143 | 80,747 | 24.42 | 3,307 |
| `srv_batch_2` | 143 | 80,747 | 21.06 | 3,834 |
| `srv_batch_4` | 143 | 80,747 | 19.51 | 4,138 |
| `srv_batch_8` | 143 | 80,747 | 18.80 | 4,295 |
| `srv_batch_14` (PARTIAL) | 14 | 5,091 | 1.050 | 4,851 |

Two consecutive runs agree to within 1.5% (batch 8: 4,294 and 4,295 tokens/s;
batch 1: 3,223 and 3,307), so the curve is a measurement, not a sample of noise.

The same single item through the CLI on each backend, from the port's own `secs`:

| Backend | Tower (s) | Tokens/s | Wall (s) |
|---|---|---|---|
| `cuda` | 0.0504 | 2,282 | 0.684 |
| `cpu` | 0.7160 | 161 | 1.286 |

The device tower is **14.2x** the CPU tower on identical input.

## What the numbers say

1. **The CLI's cost is process, not compute.** For one 115-token item the tower
   takes 0.049 s of a 0.681 s wall — 7%. The other 0.63 s is process spawn, the
   417 MiB read, the mapping and CUDA init, and it is paid again on every call.
   The `ds4` binary also refuses a second process on the machine
   ("another ds4 process is already running (pid N); refusing to start", exit 2),
   so a CLI-driven job cannot overlap with itself either. Anything throughput-
   shaped belongs on the route, and the route is what the PDF workflow uses.

2. **Batching buys about 30%, and the 8192-token cap then binds.** 3,307 tokens/s
   at one item per request rises to 4,295 at eight, and the next step up is
   refused by name rather than silently split. A single-item request therefore
   leaves roughly a quarter of the achievable rate unused, which is the whole
   argument for the array form of `input`.

3. **The path is nowhere near the memory roofline.** 4,295 tokens/s is about
   760 GFLOP/s at roughly 176 MFLOP per token, against an FP32 SIMT peak on the
   order of tens of TFLOP/s on this part. One pass over the 417 MiB of weights
   costs about 0.87 ms at this card's bandwidth, which at 8,192 tokens per
   forward would be millions of tokens per second. Neither the weights read nor
   the arithmetic is the wall.

4. **The `aligned tier` hint in the load log does not apply here** — a negative
   result worth recording so nobody chases it. The CLI prints
   `aligned artifacts: none — raw-layout dispatch tier (discrete GPU (use
   ds4_weight_server --repack-*-aligned for the aligned tier))`, but
   `ds4_weight_server --help` shows those repacks cover IQ2_XXS and Q2_K
   routed-expert stacks and large base Q8_0 dense tensors. This artifact's
   payload is neither: it is dense 4-bit `DQVQ` and a q8_0 token table.

## Where the cost most plausibly is, and what would settle it

Stated as hypotheses, not findings — none is measured yet:

- The VQ dequant itself (a per-element gather, `value[row][col] =
  codebook[idx][col % dim] * gain[row]`, `ds41_vq_fmt.h:7`) sits on the critical
  path of every matmul, and is the one structure here that ordinary quantized
  GEMMs do not pay.
- Small per-request shapes: an item's ~565 rows is a narrow GEMM relative to the
  8,192-row ceiling, so kernel launch and weight re-read overhead amortize
  poorly at batch 1.
- The per-request path may touch weights per call rather than once per server.

Settling which one it is needs phase attribution, which is exactly what
`ds4-perf` exists for — but `ds4-perf` is not built in this checkout and has no
embedding workload (`crates/ds4-perf`'s only "embed" hits are unrelated strings
in `serving/review.rs:73` and `csv.rs:4`), and `ds4-bench` has no embed verb
either. So the honest next step is instrumentation first: either an embed row in
`/v1/stats` with the tower's own elapsed time, or an embedding workload in
`ds4-perf`, before any claim about where the time goes.

## Reproduce

```sh
cd ds4-rs
sh speed-bench/eg2_embed_sweep.sh \
  /data/embeddinggemma2/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf \
  <dir of pNNN.txt pages> 8098
```

Boot mode measures the CLI row first (no other `ds4` process may be alive), then
boots its own server, warms up, and runs the latency and batch rows. Attach mode
(`EG2_EMBED_URL=http://127.0.0.1:PORT`) measures a server already listening and
records the CLI row as skipped rather than faking it.
