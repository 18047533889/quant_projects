# R47 focused repairs — 2026-09-18

Direct server-c main working tree, no branch/copy/deployment.

## Implemented
- Six intraday profile kernels retain missing physical day/slot history instead of compressing gaps; unavailable estimates keep original date/symbol axes.
- Intraday beta-family and jump-clustering results preserve the daily input grid even with no valid estimate.
- Polars ts_jump_bipower now implements (pi/2) times adjacent absolute return products, not differences of returns.
- Polars ts_lag1_autocorr uses window aligned pairs over window+1 physical bars, scale-safe Pearson, no missing-row reconnection. Legacy n_bins is non-searchable compatibility-only.
- Execution history and forward impact: lag1 window, bipower window-1. Tests compare exact-overlap chunk outputs with full-series outputs and prove one fewer history row is insufficient.
- Semantic versions bumped for six profiles and two rolling estimators to invalidate old identities. No historical production outputs deleted or published.

## Root verification
- evidence/r47-profile-beta-cluster-root.log: 22 passed (independent profile/grid, semibeta/kurtosis and jump-gap formulas).
- evidence/r47-bipower-lag-history-root.log: 14 passed (numerics, scale, missingness, future perturbation, history/chunk boundaries).
- Bounds: sampled subprocess-family RSS 1536 MiB, timeout 180s. Sampled guard is not a hard memory cap.
- Existing next-stage parity test strengthened to compare full date/symbol axes and NaN masks; full suite still pending separate limit/path repairs, so that test file is not included in this commit.

These are focused regression results, not certification of all 1756 operators or 110k factor execution. CSV unchanged.
