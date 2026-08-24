# Benchmark B03  (B03 daily + fundamental PIT join + universe)

- scale: `small`  verdict: **PASS**
- commit: a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc  version: 0.10.2+build.a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc
- platform: Linux-5.15.0-171-generic-x86_64-with-glibc2.35  cpu: INTEL(R) XEON(R) PLATINUM 8576C
- ram_gb: 30.5  python: 3.10.12

| metric | value |
|---|---|
| ttfc_ms | 11.982467985944822 |
| ttdc_ms | 95.7511919841636 |
| wall_ms | 173.23451998527162 |
| rows_per_sec | 465591.8 |
| gb_per_sec | 0.033 |
| physical_object_count | n/a |
| physical_scan_count | 1 |
| bytes_scanned | 2773875 |
| bytes_returned | 2773875 |
| resolution_ms | n/a |
| snapshot_ms | n/a |
| schema_ms | n/a |
| calendar_ms | n/a |
| remote_list_ms | n/a |
| remote_head_ms | n/a |
| governor_wait_ms | n/a |
| duckdb_wait_ms | n/a |
| duckdb_execute_ms | 83.76 |
| polars_execute_ms | n/a |
| cache_hit | n/a |
| session_reuse | n/a |
| peak_rss_mb | 313.59 |
| spill_bytes | n/a |
| status | n/a |
| rows | 39000 |
| join_rows | 39000 |
| join_columns | 9 |
| time_range | 2021-01-01..2023-01-01 |
| output_symbols | 150 |
| universe_applied | True |
| no_universe_ms | 52.25 |
| no_universe_rows | 78000 |
