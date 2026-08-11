# Benchmark B02  (B02 minute read (single-day / month / year) + minute→daily aggregate)

- scale: `small`  verdict: **PASS**
- commit: a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc  version: 0.10.2+build.a2dc3a1a7c3696f4196a125d7d5da1a6b6930ebc
- platform: Linux-5.15.0-171-generic-x86_64-with-glibc2.35  cpu: INTEL(R) XEON(R) PLATINUM 8576C
- ram_gb: 30.5  python: 3.10.12

| metric | value |
|---|---|
| ttfc_ms | 61.29519600654021 |
| ttdc_ms | 79.63302300777286 |
| wall_ms | 143.75238300999627 |
| rows_per_sec | 288038.7 |
| gb_per_sec | 0.019 |
| physical_object_count | n/a |
| physical_scan_count | 2 |
| bytes_scanned | 354079 |
| bytes_returned | 353760 |
| resolution_ms | n/a |
| snapshot_ms | n/a |
| schema_ms | n/a |
| calendar_ms | n/a |
| remote_list_ms | n/a |
| remote_head_ms | n/a |
| governor_wait_ms | n/a |
| duckdb_wait_ms | n/a |
| duckdb_execute_ms | 18.33 |
| polars_execute_ms | n/a |
| cache_hit | n/a |
| session_reuse | n/a |
| peak_rss_mb | 313.59 |
| spill_bytes | n/a |
| status | n/a |
| rows | 5288 |
| window | single_day:2024-12-02 |
| day_rows | 5280 |
| month_ms | 24.13 |
| month_rows | 105600 |
| year_ms | 26.88 |
| year_rows | 105600 |
| agg_ms | 12.65 |
| agg_rows | 1 |
| agg_sample | [{'ts': datetime.date(2024, 12, 2), 'inst': '600000', 'value': Decimal('6707759')}] |
