# Benchmark B01  (B01 10y daily panel: full + pruned)

- scale: `small`  verdict: **PASS**
- commit: a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc  version: 0.10.2+build.a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc
- platform: Linux-5.15.0-171-generic-x86_64-with-glibc2.35  cpu: INTEL(R) XEON(R) PLATINUM 8576C
- ram_gb: 30.5  python: 3.10.12

| metric | value |
|---|---|
| ttfc_ms | 262.7298939914908 |
| ttdc_ms | 301.8299239920452 |
| wall_ms | 805.4898559930734 |
| rows_per_sec | 6000629.1 |
| gb_per_sec | 0.427 |
| physical_object_count | 2 |
| physical_scan_count | 1 |
| bytes_scanned | 16685925 |
| bytes_returned | 16685925 |
| resolution_ms | 97.98 |
| snapshot_ms | n/a |
| schema_ms | n/a |
| calendar_ms | n/a |
| remote_list_ms | n/a |
| remote_head_ms | n/a |
| governor_wait_ms | n/a |
| duckdb_wait_ms | n/a |
| duckdb_execute_ms | 39.1 |
| polars_execute_ms | n/a |
| cache_hit | n/a |
| session_reuse | n/a |
| peak_rss_mb | 313.59 |
| spill_bytes | n/a |
| status | n/a |
| direct_backend_ms | 28.77 |
| direct_backend_rows_per_sec | 8153909.7 |
| column_count | 9 |
| rows | 234600 |
| rows_read_full | 234600 |
| cols_read_full | 9 |
| warm_throughput | 9633241.6 |
| warm_direct_ratio | 1.1814 |
| pruned_object_count | 1 |
| pruned_ms | 16.0 |
| pruned_rows | 52400 |
| pruned_bytes | 3726950 |

## Gates

- **PASS**  warm_throughput: warm throughput >= 90% of direct backend (delta=0.1814)
