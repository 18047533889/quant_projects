# Real COS F61: explicit four-worker prefetch

Compare `real_cos_f61_prefetch4_pearson_ab_20260930.json` with the preceding
`real_cos_f61_consuming_source_pearson_ab_20260930.json` (two workers).
Both use the same manifest, factor IDs, 2586 dates × 5461 stocks × 61 factors,
float64 values, four Pearson metrics, source cap 12288 MiB and tile cap 16.
Four-worker prefetch additionally reserves 1024 MiB; default settings are unchanged.

| Setting | CPU seconds | CUDA seconds | auto seconds |
| --- | ---: | ---: | ---: |
| 2 workers | 134.4443 | 90.4782 | 89.8809 |
| 4 workers | 96.1931 | 60.6541 | 56.4398 |

Four-worker source-read times were 51.0144 / 57.7930 / 54.6353 seconds for
CPU / CUDA / auto, versus approximately 88 seconds with two workers.
CPU/CUDA and auto parity passed; all metric value byte hashes match the
corresponding backend in the two-worker report. Masks and observation counts match.
Cumulative process peak RSS was 13,528,644 KiB (~12.90 GiB), not isolated backend peaks.
The logical source estimate was 11,467,641,280 bytes under the explicit 12 GiB cap.

The first experiment temporarily configured the existing source constructor
inside the benchmark process (restored afterwards). The harness now exposes
equivalent reproducible options: `--source-adapter cos --cos-prefetch-workers 4
--max-prefetch-memory-mib 1024 --max-source-memory-mib 12288`.
Use factors 61, tile 16, total bound 6144 MiB, the existing F61 axis index,
and metrics `pearson_ic,pearson_ic_series,pearson_ic_std,pearson_ic_ir`.

The two-worker run used CPU→CUDA→auto; four workers used CUDA→CPU→auto.
These are encouraging opposite-order experiments, not randomized repeated
concurrency trials or universal optimality proof. Reverse-order repetition remains required.
