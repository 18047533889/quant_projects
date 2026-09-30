# Trailing SMA: shared FactorEngine implementation

Optimizer and preprocess now delegate to `factor_engine.backend.long_smoothing.lagged_mean`.
The FE expression is `ts_mean(ts_delay(value, n=1), window=w, min_periods=m)`:
for each asset, average valid values in the previous w observed rows, excluding the current row.
Sparse dates do not create synthetic observations; equal-date rows retain input order.
Null asset rows remain missing; infinities follow the existing pandas missing-value semantics.
Optimizer requires a complete window; preprocess supports its existing minimum-valid-count option.
Production preprocess routing fails closed when FE is unavailable; native fallback requires explicit research mode.

The companion `TRAILING_SMA_THREE_ROUTE_200K_20260930.json` records 200,000 generated rows,
windows 3/10/30, minimum counts 0/1/window, and five alternating warm rounds.
All nine FP/FE cases and three complete-window FO cases have maximum absolute difference zero.
Cold FE setup plus first evaluation is reported separately from warm timings (12.4744 seconds in this run).
These are in-memory primitive measurements, not end-to-end optimizer, real COS, or production certification.
