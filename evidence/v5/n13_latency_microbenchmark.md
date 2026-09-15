# N13 latency microbenchmark

> Synthetic bounded profile only. This is neither a production SLA nor 100k batch-throughput evidence.

Created: `2026-09-07T16:51:10.335002+00:00` on `qs-compute-gpu-hk-01`.

Profile: one factor, 40 assets, RankIC-series H10; 8 warmups and 80 measured runs. The single-factor case evaluates 12 bars through public QE. The single-bar case measures one typed bar through public maturity enqueue and drain using a temporary SQLite durable store.

| Path | p50 ms | p95 ms | p99 ms | mean ms | min–max ms |
|---|---:|---:|---:|---:|---:|
| Single-factor public QE evaluate | 3.888 | 3.992 | 5.724 | 3.959 | 3.802–6.668 |
| Single-bar maturity enqueue + drain | 7.156 | 9.774 | 10.581 | 7.641 | 6.551–10.877 |

The JSON companion contains the exact environment, thread settings, profile, percentiles, and content digest.
