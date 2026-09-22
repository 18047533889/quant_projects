# `trailing_sma` bounded performance check (2026-09-22)

The implementation changed from a Python loop over assets to pandas grouped
`shift` plus grouped `rolling`. A temporary positional index preserves input
order and duplicate-index behavior; the public contract remains a strictly
lagged, per-asset moving average.

Environment: project `.venv`, pandas 2.3.3, `OPENBLAS_NUM_THREADS=1`.

Workload: 300 timestamps x 256 interleaved assets (76,800 rows), float64
values with 3% missing observations, `window=20`, `min_periods=10`. Ten
timed repetitions followed one warm-up call per implementation. Invocation
order alternated between the scalar and vectorized paths on each repetition.

| implementation | median seconds | best seconds |
| --- | ---: | ---: |
| scalar per-asset loop | 0.05877 | 0.05802 |
| grouped vectorized path | 0.02551 | 0.02505 |

The measured median speedup was 2.30x for this bounded workload. This is not a
whole-pipeline or optimizer claim. Exact-output A/B checks covered interleaved
assets, duplicate input indices, tied timestamps, missing values, missing
asset identifiers, unused categorical levels, float32/float64 inputs, several
window/minimum-period combinations, current-value exclusion, and prefix
invariance. The host had a low load average (1.70) during measurement; one
unrelated short single-test pytest process was also active, so these timings
should be read as bounded comparative evidence rather than isolated hardware
benchmarks.
