# H18–H33 focused regression evidence (2026-09-10)

## Modified paths

- `factor_engine/cleaned_operators/intraday/{sufficient_stats.py,_core.py,sufficient_stats_ops.py,realized_beta.py}`
- `factor_engine/cleaned_operators/ts_model/{dynamic_regression.py,sequence_anomaly.py,volatility.py,_rolling_core.py}`
- `factor_engine/cleaned_operators/{candle_state_space.py,regression_models.py}`
- `factor_engine/tests/operators/test_intraday_sufficient_stats.py`
- `factor_engine/tests/operators/{test_model_semantics_fixes.py,test_advanced_intraday.py,test_v9_m01_shared_huber.py,test_v9_m01_cs_huber_shared.py}`
- `tests/operators/{test_model_semantics_fixes.py,test_advanced_intraday.py}` (tracked root mirrors)
- `tests/operators/test_regression_models.py` and its `factor_engine/tests/operators/` mirror
- `factor_engine/tests/intraday/test_h27_h33_intraday_regressions.py`
- `factor_engine/tests/ts_model/test_h18_h31_model_regressions.py`

## Measured invariants

- H19 stability performs 30 fits on the 30-row fixture. The former nested-refit implementation would perform 170 fits for the same cutoffs (30 primary fits plus 140 repeated historical fits).
- H29 session-grid allocation measured 192, 384, and 768 bytes for 4, 8, and 16 days respectively (3 slots, 2 instruments, float64), confirming linear growth in day count.
- Cache admission is broker-controlled. Denied admission returns read-only views of the compute workspace without a second full-bundle copy and retains no cache entry.
- Admitted cache leases remain live while any extracted ndarray, ndarray view, `np.asarray` result, or `Index.to_numpy(copy=False)` backing array is live after LRU eviction.
- A forced `MemoryError` after broker admission releases the lease immediately. A zero-physical-owner bundle also releases immediately and is not retained.

## Test commands and results

```text
.venv/bin/python -m pytest -q factor_engine/tests/intraday/test_h27_h33_intraday_regressions.py
13 passed in 0.74s

.venv/bin/python -m pytest -q factor_engine/tests/intraday/test_h27_h33_intraday_regressions.py factor_engine/tests/operators/test_intraday_sufficient_stats.py factor_engine/tests/ts_model/test_h18_h31_model_regressions.py
48 passed, 2 warnings in 33.46s
```

Earlier focused joint regression (intraday sufficient statistics, H18–H33 tests, targeted AR/Huber semantics, vector equivalence, and runtime compute-many) completed with `100 passed, 105 warnings`; the final cache-only additions are independently covered by the 13-test run above.

```text
.venv/bin/python -m pytest -q factor_engine/tests/operators/test_model_semantics_fixes.py factor_engine/tests/operators/test_model_semantics_fixes_round2.py factor_engine/tests/operators/test_advanced_intraday.py factor_engine/tests/intraday/test_h27_h33_intraday_regressions.py factor_engine/tests/ts_model/test_h18_h31_model_regressions.py
105 passed, 2 warnings in 35.72s
```

The corresponding tracked root mirrors in `tests/operators/` received the same contract corrections. A larger root-mirror plus package-H/v9 run reached `132 passed` before one unrelated batch-planner resource failure: the isolated bridge test received a dynamic read-wave budget of 1 byte for an estimated 8 MB atomic scan. This is not counted as Huber numerical evidence.

## Broad isolated-suite resolution

- Wasserstein pair tests now supply both required inputs and separately assert that missing `y` is rejected. NaN, Inf, empty, and single-column coverage remains.
- GARCH expected state is independently replayed from `var(seg[:-1])`; perturbing only the current return verifies that `h_t` is unchanged.
- Huber exact-majority handling now reports the observable reason `exact_consensus` only after strict-majority, support, full-rank, stable-refit, original-unit zero-residual, finite-prediction, and bounded-subgradient stationarity certificates pass. Rank-deficient consensus, one-iteration non-convergence, and high-leverage minority non-certification remain fail-closed controls.
- The AR poison test now follows the published max-lookback contract: `window=W` means exactly W source rows and W-lag regression pairs. It checks `t-W` is outside, `t-W+1` is inside, and compares each coefficient with an independent intercept-inclusive `numpy.linalg.lstsq` reference. Both tracked mirrors passed independently.

## Uncovered production boundaries

- No production factor was published or deployed.
- No live COS write/read or production-scale memory-pressure run was performed.
- Broker behavior was validated with deterministic test doubles, including admission denial and post-admission allocation failure; integration with a live broker under eviction pressure remains outside this focused test scope.
- The cross-sectional Huber batch-bridge test keeps two lanes: an unmodified default-broker integration lane and a bounded-fixture mathematical lane. Both require finite outputs, exact index alignment with direct `_cs_robust_resid`, value agreement, and translation error at most `3e-7`. Before the separate broker-authority fix, the bounded lane passed while the real default lane correctly exposed `read_wave_bytes=0` becoming a 1-byte planning budget; this evidence does not claim the default resource path was fixed by H18-H33 changes.
- The distinct `factor_engine.cleaned_operators.intraday.state_space` module remains intentionally excluded from `load_all`; its isolated legacy suite passed separately (28 tests). `candle_state_space.py` was modified for H21 and is not the excluded module.
