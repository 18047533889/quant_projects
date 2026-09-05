# AlphaPROBE Search-Scale Benchmark (T23) — scale=100000

- timestamp: 2026-09-05T03:04:26Z
- git HEAD: `c5bfbf03105e26228218993e021d639047d37d10`
- host cores: 32 / python 3.12.3
- faiss available: False

| bench | p50 ms | p95 ms | p99 ms | mean ms | peak RSS MB | notes |
|---|---|---|---|---|---|---|
| identity_lookup | 6921.1363 | 7094.778 | 7125.577 | 6929.1674 | 28.0 |  |
| seed_selection | 27.7194 | 27.7194 | 27.7194 | 27.7194 | 28.0 | touch=1792/100000 (0.01792, sublinear=True) seeds=100 |
| ann_nearest | 21.2166 | 21.3923 | 21.4079 | 21.2745 | 502.8 | index=100000 rebuilds=1 |
| cluster_context | 21.3798 | 21.6237 | 21.6454 | 21.4487 | 502.8 |  |
| retriever_scoring_100k | 1074.6026 | 1074.6026 | 1074.6026 | 1074.6026 | 502.8 |  |
| memory_packet_generation | 5428.7291 | 5636.3214 | 5654.7741 | 5498.2563 | 502.8 |  |
| sqlite_memory_event_writes | 1.5174 | 1.5174 | 1.5174 | 1.5174 | 502.8 | 100000 events, 659.0 w/s, total=151740.8489ms |
| resume_checkpoint_latency | 86.3946 | 87.6042 | 87.7118 | 86.7644 | 502.8 | save+load round-trip |