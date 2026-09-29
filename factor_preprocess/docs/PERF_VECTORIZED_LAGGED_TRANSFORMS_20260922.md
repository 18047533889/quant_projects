# Vectorized lagged per-asset transforms performance audit (2026-09-22)

## Scope

This audit covers the lagged, strictly-causal, per-asset rolling/EWMA kernels in
`factor_preprocess.transforms.rolling` and `factor_preprocess.transforms.smoothing`:

- `rolling_mean`, `rolling_std`, `rolling_zscore`, `ewma` (`transforms/rolling.py`)
- `trailing_median`, `robust_ewma` (`transforms/smoothing.py`)

These previously scanned each asset in a Python loop (`groupby(...).indices`
+ per-asset `iloc` slice + per-asset `shift(1)`/`rolling`/`ewm`). The
vectorized paths use pandas `groupby` + `shift(1)` + `groupby` + `rolling` /
`groupby` + `ewm`, scattered back to the original row order by positional index
(the same technique already applied to `trailing_sma` in the
`TRAILING_SMA_PERFORMANCE_20260922` audit). The recursive filters in this module
(`kama`, `one_sided_iir_lowpass`, `kalman_local_level`) were deliberately left
as forward-only per-asset recursions: their state semantics must not be rewritten
as commutative windows.

The contract is unchanged: strict `shift(1)` lag (no current/future use), per-asset
isolation, NaN warmup propagation, duplicate-input-index tolerance, and output
alignment to the input index.

## Correctness evidence

The original scalar loops are retained verbatim as private oracles
(`_rolling_mean_reference`, `_rolling_std_reference`, `_rolling_zscore_reference`,
`_ewma_reference`, `_trailing_median_reference`, `_robust_ewma_reference`). The
new path is validated against them in `tests/test_perf_ab_equivalence.py`:

- Equivalence (value **and** NaN mask) on 3 seeds × float32/float64 × duplicate
  index. The two paths are **bit-for-bit identical** on finite values (asserted
  with an `np.array_equal` byte check; `rtol=1e-8, atol=1e-10, equal_nan=True`
  is kept only as a documented ceiling).
- Causal guards: no-future-leakage (perturbing the last row of an asset leaves
  every earlier output unchanged), prefix invariance, and real-time-append
  equivalence (appending rows to each asset reproduces earlier outputs when
  aligned by `(asset_id, date)`).
- Degenerate inputs: single asset, all-NaN, all-constant, window larger than the
  series, duplicate index, and that unsorted input raises `ValueError`.
- Output hash stability: identical input yields byte-identical output across
  calls (determinism).

All 78 new tests pass; the existing 488 tests are unchanged (no regression).

## Bounded A/B timings

The per-kernel timings below used `OPENBLAS_NUM_THREADS=1`, one warm-up call per
implementation, and the mean of five timed calls on a deterministic 256 asset × 500
date panel (3% missing, float64) with `window=20, min_periods=10`
(`halflife=10` for the EWMA variants). The reference is the original scalar loop.
The implementation order was fixed (new path first, reference second), so these
numbers may include order, cache, allocator, or thermal bias; the calls were not
interleaved. Treat the timings as indicative local measurements, not a controlled
benchmark. `tests/test_perf_ab_equivalence.py` uses the same fixed order and
reports a five-call arithmetic mean; its loose 1.5× guard is a regression check,
not a precise speed estimate.

| Kernel | Reference (scalar loop) | New (vectorized) | Speedup |
| ---: | ---: | ---: | ---: |
| `rolling_mean` | 0.0778 s | 0.0375 s | 2.07x |
| `rolling_std` | 0.0802 s | 0.0387 s | 2.07x |
| `rolling_zscore` | 0.1596 s | 0.0651 s | 2.45x |
| `ewma` | 0.0745 s | 0.0350 s | 2.13x |
| `trailing_median` | 0.1235 s | 0.0671 s | 1.84x |
| `robust_ewma` | 0.3038 s | 0.0922 s | 3.30x |

These timings are local kernel measurements on the stated hardware and inputs.
They are not a claim that the complete optimizer or any downstream pipeline is
Nx faster: candidate construction, other transforms, data access, and
evaluator metrics remain outside this measurement.

The measurements cover only 256 × 500 (128,000 rows). They do not establish
runtime or peak-memory behavior for a broad market panel. `scripts/measure_lagged_rss.py`
provides a bounded, process-isolated RSS sampling run at 256 × 500 and 1,024 × 500
(up to 512,000 rows); its results are below. RSS is sampled during each transform
and can miss very short peaks. These two sizes are still not a full-market
benchmark and must not be extrapolated linearly to multi-year, thousands-of-assets
panels.

### Bounded RSS sample

Command: `python3 scripts/measure_lagged_rss.py`. Each operation and shape ran
once in a fresh subprocess using deterministic float64 values (3% missing),
int32 asset IDs, and 500 weekly dates. The child was limited to 3 GiB virtual
address space and 30 CPU seconds; the parent enforced a 35-second per-child and
240-second overall wall budget. RSS was sampled every 5 ms after the input panel
was built and immediately before the transform. “Increment” is sampled peak RSS
minus that baseline; input-frame memory is therefore excluded.

| Transform | 256 × 500 increment (MiB) | 1,024 × 500 increment (MiB) |
| --- | ---: | ---: |
| rolling_mean | 7.9 | 27.9 |
| rolling_std | 7.9 | 28.0 |
| rolling_zscore | 8.7 | 32.1 |
| ewma | 8.0 | 30.2 |
| trailing_median | 8.6 | 28.3 |
| robust_ewma | 20.8 | 77.1 |

These measurements show the largest observed increment was 77.1 MiB for
`robust_ewma` at 512,000 rows in this environment. RSS polling can miss short
allocation peaks, and process allocator behavior is environment-dependent. The
sample is useful for bounded size-growth comparison only; it is not evidence of
safe peak memory or throughput at 5,000 assets × multiple years.

## Other modules reviewed (not changed)

- `transforms/volatility.py` and `transforms/missingness.py` already use
  `groupby(...).transform(lambda: series.shift(1).rolling(...))`. They are ~2x
  faster than the scalar loops above and were left unchanged; vectorizing them
  further is a possible follow-up but carries contract nuance (zero-vol → NaN
  mask in `volatility_scale`; fraction semantics in `missing_fraction`).
- `neutralization/ols.py` (`ols_neutralize`, `compute_exposures`) loops per date
  with `np.linalg.lstsq`. At 256×500×5 this is ~0.27 s — acceptable, and
  batching the per-date least squares is risky because of per-date pairwise
  NaN deletion, rank cutoffs, and the saturation-to-zero rule. Left unchanged;
  noted as a future candidate.
- Recursive/sequential filters and decompositions (`kama`, `one_sided_iir_lowpass`,
  `kalman_local_level`, `decomposition/wavelet.py`, `decomposition/seasonal.py`)
  are intrinsically forward-only and must stay per-asset; no change.
