# AlphaPROBE Search-Scale Benchmark (T23) — scale=100000

- timestamp: 2026-09-05T06:42:33Z
- git HEAD: `3fe105b037aaca3ce8445b865ebe037b5470ff7f`
- host cores: 32 / python 3.12.3
- faiss available: False

| bench | p50 ms | p95 ms | p99 ms | mean ms | peak RSS MB | notes |
|---|---|---|---|---|---|---|
| identity_lookup | 7053.6171 | 7077.994 | 7082.7257 | 7052.7945 | 41.8 |  |
| seed_selection | 25.7995 | 25.7995 | 25.7995 | 25.7995 | 41.8 | touch=1792/100000 (0.01792, sublinear=True) seeds=100 |
| ann_nearest | 20.6698 | 21.0745 | 21.1105 | 20.8082 | 504.4 | index=100000 rebuilds=1 |
| cluster_context | 22.0307 | 22.2204 | 22.2373 | 22.095 | 504.4 |  |
| retriever_scoring_100k | 1066.0563 | 1066.0563 | 1066.0563 | 1066.0563 | 504.4 |  |
| memory_packet_generation | 5491.5162 | 5508.9305 | 5510.4784 | 5477.9156 | 504.4 |  |
| sqlite_memory_event_writes | 0.0356 | 0.0356 | 0.0356 | 0.0356 | 504.4 | 100000 events, 28094.1 w/s, total=3559.4682ms |
| resume_checkpoint_latency | 85.6317 | 101.5055 | 102.9165 | 91.497 | 504.4 | save+load round-trip |