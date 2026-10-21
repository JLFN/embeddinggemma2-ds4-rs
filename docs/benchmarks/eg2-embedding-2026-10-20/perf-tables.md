# EmbeddingGemma-2 embedding path: before and after, generated

Generated from the committed raw files by `plot-eg2-embed-perf.py` (2026-10-20).

## The corpus, end to end (143 pages / 80,747 tokens, tokens/s)

| build | runs | batch 1 | batch 2 | batch 4 | batch 8 |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 3,307 | 3,834 | 4,138 | 4,295 |
| attention vectorized | 3 | 5,455 | 6,430 | 6,753 | 6,915 |
| payload GEMM rows 8 | 2 | 6,861 | 8,051 | 8,828 | 9,075 |
| attention grouped | 2 | 9,336 | 11,209 | 12,143 | 12,614 |

Speedup against the baseline, batch 8: 2.94x (4,295 to 12,614 tokens/s).

## The attention kernel alone (median ms per shape)

| shape | rows | keys | original | vectorized | grouped | original/grouped |
|---|---:|---:|---:|---:|---:|---:|
| `cli115_s` | 115 | 13,225 | 0.1860 | 0.0625 | 0.0287 | 6.47x |
| `page294_s` | 294 | 86,436 | 0.9646 | 0.3021 | 0.1208 | 7.99x |
| `page565_s` | 565 | 316,469 | 3.8973 | 1.0966 | 0.4204 | 9.27x |
| `max1414_s` | 1,414 | 1,186,694 | 12.0750 | 3.6536 | 1.4661 | 8.24x |
| `page294_f` | 294 | 86,436 | 1.6865 | 0.5499 | 0.2365 | 7.13x |
| `max1414_f` | 1,414 | 1,999,396 | 42.8206 | 11.0434 | 4.7957 | 8.93x |
| `batch8x306_s` | 2,448 | 749,088 | 6.6560 | 2.0500 | 0.9196 | 7.24x |
| `batch8x565_s` | 4,520 | 2,531,752 | 22.8270 | 7.0246 | 3.4467 | 6.62x |
| `batch8x565_f` | 4,520 | 2,553,800 | 44.9843 | 14.3275 | 7.5761 | 5.94x |

## One forward's kernel split (nsys, 8 items / 2,451 tokens)

| build | payload GEMM | attention | small ops | total |
|---|---:|---:|---:|---:|
| before: payload GEMM rows 8 | 139.2 ms | 116.7 ms | 32.0 ms | 287.8 ms |
| after: + attention grouped | 126.9 ms | 46.0 ms | 24.5 ms | 197.5 ms |

The two profiles are not window-comparable: the payload GEMM row moves (139.2 to 126.9 ms) with its code unchanged between them, and the third unit's record names an older committed report (401.2 ms, attention 159.7 ms) whose score plane that record calls superseded. The split, not the delta, is what these reports are used for.

