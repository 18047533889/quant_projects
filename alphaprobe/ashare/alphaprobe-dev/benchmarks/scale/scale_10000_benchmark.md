# AlphaPROBE Search-Scale Benchmark (T23) — scale=10000

- timestamp: 2026-09-05T03:03:43Z
- git HEAD: `c5bfbf03105e26228218993e021d639047d37d10`
- host cores: 32 / python 3.12.3
- faiss available: False

| bench | p50 ms | p95 ms | p99 ms | mean ms | peak RSS MB | notes |
|---|---|---|---|---|---|---|
| identity_lookup | 677.1274 | 688.5435 | 723.6177 | 680.2446 | 21.2 |  |
| seed_selection | 25.3642 | 25.3642 | 25.3642 | 25.3642 | 21.2 | touch=1792/10000 (0.1792, sublinear=True) seeds=100 |
| ann_nearest | 2.9765 | 3.4075 | 3.448 | 3.0463 | 82.9 | index=10000 rebuilds=1 |
| cluster_context | 2.3216 | 2.3989 | 2.4016 | 2.3186 | 82.9 |  |
| retriever_scoring_100k | 109.0772 | 110.878 | 110.9868 | 109.3774 | 82.9 |  |
| memory_packet_generation | 573.9167 | 590.4279 | 591.262 | 575.427 | 82.9 |  |
| sqlite_memory_event_writes | 1.4916 | 1.4916 | 1.4916 | 1.4916 | 82.9 | 10000 events, 670.4 w/s, total=14915.853ms |
| resume_checkpoint_latency | 19.2708 | 31.0719 | 32.8181 | 22.6115 | 82.9 | save+load round-trip |