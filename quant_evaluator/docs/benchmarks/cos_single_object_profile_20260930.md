# Single-object COS source profile (2026-09-30)

This evidence profiles one object from the verified F61 benchmark sample through the existing QE → FO read_bound_factor → DataAccess research-object path. The profiler is isolated under quant_evaluator/scripts; it wraps existing calls at runtime and does not alter DataAccess behavior.

## Bound and identity

- Manifest SHA-256: b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864
- Selection: middle record by (bytes, factor_id) after the existing F61 selector returned 61 verified records.
- Factor: weekly_3b94de18d97efb6b; object size 103,359,672 bytes (about 98.6 MiB).
- The profile ran twice in sequence, with the same bound manifest, URI, ETag, object SHA-256, byte count and axis hash. Each call retained the existing before/after HEAD, SHA/ETag, URI/manifest checks.
- DataAccess’s private temporary object was removed after each call. No persistent factor copy or cache was created. The script enforces a 128 MiB per-object cap and checks at least 320 MiB free in the configured cache filesystem before reading.

## Measurements

Seconds; the two HEAD calls are aggregated. store.read includes the Parquet scan and produced an already materialized Arrow-backed handle; to_arrow() itself was effectively a no-op for this run. Unattributed time is the remainder within read_bound_factor after the non-overlapping timed stages.

| Stage | First read | Sequential repeat |
|---|---:|---:|
| Full read_bound_factor | 3.414 | 2.212 |
| HEAD × 2 | 1.157 | 0.241 |
| COS CLI download | 0.906 | 0.741 |
| Local hash file reads | 0.0186 | 0.0180 |
| MD5 + SHA-256 updates | 0.1916 | 0.1916 |
| store.read / Parquet scan | 1.082 | 1.007 |
| to_arrow() materialization | 0.000003 | 0.000003 |
| Arrow to pandas | 0.0330 | 0.0322 |
| Axis normalization/filter/sort | 0.0450 | 0.0456 |
| Axis hash | 0.0015 | 0.0015 |
| Other time inside bound read | 0.0583 | 0.0140 |
| Manifest bind/read (outside factor timer) | 0.428 | 0.408 |

The object contained 2,588 rows and 5,461 factor columns, close to the reported Pearson source panel shape. The two HEAD calls, one download, one store.read, and one Arrow materialization hook were observed on both passes. The first pass had slower HEAD latency; the repeat was faster, but this is only a sequential second remote read. OS, gateway, and COS cache state was not controlled, so it is not evidence of a true cold/warm cache comparison.

## Readout and next step

For this sample, the dominant measured stages are the Parquet scan and the two metadata requests; pandas conversion and axis processing together were under 0.08 seconds. The full factor read took 2.2–3.4 seconds. This is consistent with a meaningful COS/DataAccess contribution to an 85-second, 61-factor pass, but this single-object sample cannot allocate the reported pass total or justify extrapolating linearly.

Next, run a bounded 8–16-object sample through the existing two-worker source with the same phase hooks, recording overlap and per-object bytes. That will show whether two-way prefetch hides transfer/query latency and whether the full-run gap is per-object service time or source assembly. Keep the current URI, content digest, ETag, manifest checks and per-read temporary cleanup in that measurement. Do not add a cache based on this evidence alone.
