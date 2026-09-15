# V5 public CPU/GPU numerical capability evidence

Run date: 2026-09-08. Hardware: NVIDIA L20, 46,068 MiB, driver 580.126.20.

This is numerical execution evidence, not a registry-only capability claim. The new matrix test used public `evaluate`, T=40, N=100, F=3 synthetic panels, CPU and `cuda_strict`, and compared every returned typed artifact with `allclose(equal_nan=True)`. Runtime metadata confirmed `backend_used=cuda`, GPU device 0, positive peak VRAM, and at least one factor tile.

## Actually executed supported matrix (32/32)

- Factor/label metrics (18): coverage, daily_quantile_monotonicity_rate, daily_quantile_monotonicity_series, factor_turnover_rate, ic_ir, ic_median, ic_std, pearson_ic, pearson_ic_ir, pearson_ic_series, pearson_ic_std, quantile_monotonicity, quantile_returns_daily, quantile_returns_full, quantile_spread, rank_ic, rank_ic_series, turnover.
- Portfolio metrics using an independent `HoldingReturnPanel` and `PortfolioSpec` (5): sharpe_ratio, sortino_ratio, win_rate, max_drawdown, calmar_ratio.
- Exposure metrics using an axis-bound `ExposurePanel` (9): industry_exposure, size_exposure, beta_exposure, liquidity_exposure, volatility_exposure, momentum_exposure, max_absolute_style_exposure, exposure_drift, purity_ratio.

The combined focused run also exercised real tiling, forced OOM retiling, shape kernels, exposure kernels, calendar reductions, validity masks, ICIR invalid-parameter parity, and low-level GPU parity: 69 passed in 4.93 seconds. Log: `evidence/v5/gpu_public_numeric_matrix_tests.log`.

## Strictly unsupported V5 LO/cost metrics

With correctly tagged execution-trajectory legs supplied, `cuda_strict` rejected each metric before `DeviceEvaluationSession._open`: turnover_cost, tracking_error, information_ratio, relative_max_drawdown, mean_investment_fraction. This proves no silent CPU fallback and does not claim GPU implementation for them.

## Evidence boundaries and closure notes

1. `docs/V5_DELTA_ACCEPTANCE_20260907.md` is stale: it still marks T01/T02, T07-T14, T29-T32 and all four closures partial despite new named tests, streaming/GC tests, durable SQLite/PostgreSQL fencing, and end-to-end tests now present. The ledger should be regenerated from current files rather than edited as a completion claim.
2. T06 now includes N=100/Q20/min_bucket=10 coverage and the exact 99-ties-plus-singleton Q20/min_bucket=5 public counterexample.
3. The 100,000-factor staircase used the deliberately bounded T=4/N=20 profile. Full production-profile certification is an external capacity boundary, not a failure of the profile tested in this V5 run.
4. The focused V5 trajectory suite passes 19 tests. The broader optional `vectorbt_qs/tests` collection remains locally constrained by unavailable optional dependencies, so no broader packaging claim is made.

The initial matrix audit found CPU/GPU observation-count differences for quantile spread/monotonicity and a missing ordinary-portfolio probe tile counter. The implementation was corrected before this final run: counts and sample units now match the informative daily-spread/profile-pair semantics, and `probe_trajectory_factor_tiles` is public for the ordinary portfolio path.

No production publish, deployment, or full-profile capacity claim was made.
