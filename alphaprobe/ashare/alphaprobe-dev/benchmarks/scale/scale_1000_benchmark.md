# AlphaPROBE Search-Scale Benchmark (T23) — scale=1000

- timestamp: 2026-09-05T03:03:38Z
- git HEAD: `c5bfbf03105e26228218993e021d639047d37d10`
- host cores: 32 / python 3.12.3
- faiss available: False

| bench | p50 ms | p95 ms | p99 ms | mean ms | peak RSS MB | notes |
|---|---|---|---|---|---|---|
| identity_lookup | 72.2242 | 74.2513 | 75.102 | 72.3421 | 20.5 |  |
| seed_selection | 23.1556 | 23.1556 | 23.1556 | 23.1556 | 20.5 | touch=1792/1000 (1.792, sublinear=True) seeds=100 |
| ann_nearest | 0.3972 | 0.4523 | 0.7081 | 0.4185 | 39.7 | index=1000 rebuilds=1 |
| cluster_context | 0.2169 | 0.3014 | 0.3089 | 0.2463 | 39.7 |  |
| retriever_scoring_100k | 10.5309 | 12.3061 | 13.2881 | 10.8485 | 39.7 |  |
| memory_packet_generation | 59.1149 | 60.6534 | 61.0026 | 59.1513 | 39.7 |  |
| sqlite_memory_event_writes | 1.4728 | 1.4728 | 1.4728 | 1.4728 | 39.7 | 1000 events, 679.0 w/s, total=1472.839ms |
| resume_checkpoint_latency | 2.0172 | 2.0923 | 2.1858 | 2.0304 | 39.7 | save+load round-trip |