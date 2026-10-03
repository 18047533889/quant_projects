# Chunked member scan ABBA benchmark, 2026-10-04

Receipt: [`chunked_member_scan_abba_oct04.json`](chunked_member_scan_abba_oct04.json).

This is a descriptive, synthetic-only comparison of the bounded chunked member
scan with the benchmark's legacy behavior. Each measurement ran in a fresh
process. The fixed ABBA order was legacy, bounded, bounded, legacy; each mode
was measured twice on the same seeded input. Synthetic input construction was
outside the timed interval. Timing covers member preparation plus exact winner
queries. Peak RSS includes imports and synthetic input objects, not just scan
scratch.

| Sample | Mode | Time (s) | Process peak RSS (bytes) |
| --- | --- | ---: | ---: |
| 1 | Legacy | 0.777025 | 484,012,032 |
| 2 | Bounded | 0.676903 | 307,851,264 |
| 3 | Bounded | 0.731923 | 307,834,880 |
| 4 | Legacy | 0.962708 | 484,442,112 |
| **Median** | **Legacy** | **0.869866** | **484,227,072** |
| **Median** | **Bounded** | **0.704413** | **307,843,072** |

On this run, bounded median time was 19.0206% lower than legacy (time ratio
0.809794), and median process peak RSS was 36.4259% lower. The legacy cache
estimate was 35,904,256 bytes, above the 32 MiB admission threshold, so this
comparison fairly models legacy full-matrix preparation once per query rather
than granting it the reusable member cache. There were four queries.

Both modes used the same synthetic input fingerprint
`10d215c2fcf901c764afd6a2e7777b22984c01563f69ec7fee3e0a1232577a16` (17,000
members × 256 dimensions; input estimate 243,771,392 bytes). This is below the
256 MiB input cap (268,435,456 bytes). The four winner identities matched:
`M02056`, `M06527`, `M01068`, and `M08721`. Similarity scores agreed within
the configured absolute epsilon of `2e-12`; this is not a bitwise-equality
promise.

Runtime: `qs-compute-gpu-hk-01`, x86_64, 32 reported CPUs, Python 3.12.3,
NumPy 2.2.6. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS`
were each set to `1`. The JSON receipt records identical declared source hashes
before and after the run for:

| Source | SHA-256 |
| --- | --- |
| `factor_assets/clustering/incremental.py` | `6d81abd393774c5f80bd56c22c40b8b50952b829685a0cd94af44b40cb72db28` |
| `factor_assets/clustering/incremental_recall.py` | `49a90ef906d1b5bd499ce8102e7135530e1b7f058fe87d67f4f5ca6ecf0747bc` |
| `factor_assets/contracts/fingerprint.py` | `3510218720c5c17bb5db0011a5308cb85c6d21c2ca1af638c13029a4a51cb302` |
| `factor_assets/scripts/benchmark_chunked_member_scan_oct04.py` | `010a8dff62fbb1c464ec3aac6ad860b52fc862e4fc1c0830d7101a1c6504e374` |
| `factor_assets/similarity/unit_vectors.py` | `4e5234254e812dcd5b557ae2280a9427c790457eab5c96b93b4454839f1158af` |

These declared hashes are not a complete transitive runtime closure. The
measurements use synthetic embeddings and one host, and do not establish that
the bounded method is universally faster or predict production-data
performance.
