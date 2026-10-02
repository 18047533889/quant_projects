# Snapshot replay A/B, 2026-10-02

Receipt: `snapshot_replay_ab_20261002.json`.
Source revision: `8f7818c2121d833a8a28ec3eefaf364acb40c5d3`.

The benchmark compares the one-pass replay with a benchmark-local copy of
the former per-asset sort-and-scan algorithm. It uses synthetic lifecycle-only
histories, 12 events per asset, and two measured repetitions per route in
opposite orders after warmup. It compares ordered assets, milestone timestamps,
and query content; it excludes the generated snapshot creation timestamp.

| Assets / events | Legacy median | One-pass median | Ratio |
| --- | --- | --- | --- |
| 100 / 1,200 | 0.137095 s | 0.003319 s | 41.30x |
| 1,000 / 12,000 | 14.625159 s | 0.034805 s | 420.20x |

Both repetitions matched output at both scales. The harness verified unchanged
HEAD and five source hashes before and after timing. Peak process RSS was
30,932,992 bytes. The JSON records raw timings, host, resource admission and
source hashes. This measures lifecycle replay on one host, not mixed-health
performance or COS ingestion. Two repetitions do not establish a universal
speedup. The separate FactorAssets suite passed 1,594 tests after the historical
health fix; its API contract requires complete event history.
