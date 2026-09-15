# Rolling-statistics numerical audit

For exactly representable `x=1e12+[1,2,4,7,11]` and
`y=1e12+[3,1,5,2,9]`, pre-fix selected pandas results were covariance `0`
versus centered reference `9.5`, variance `16.49993896484375` versus `16.5`,
and standard deviation `4.062011689402647` versus `4.06201920231798`.
With `FACTOR_ENGINE_USE_NUMBA=1`, standard deviation collapsed to `0`.

The pandas authority now uses finite, window-local anchor centering and scale
normalization. Covariance preserves pairwise finite masking, `ddof`, current-row
semantics, and the public exact-axis rejection. The inadmissible raw-update
standard-deviation fast path is not selected. Polars-long uses native O(window)
list windows; DuckDB uses centered list expressions. Constant kurtosis remains
undefined (NaN), including constants `0`, `1`, and `1e308`.

The DataAccess SQL allowlist initially rejected DuckDB's `list_count` macro as
a table function; the retained exception is in `evidence/r4/duckdb-list-macro-before.txt`.
New SQL uses scalar `array_length`, list indexing, and `list_aggregate` and is
verified through real DataAccess DuckDB execution, not fallback.

Bounded final gate: 83 passed, 22 expected skips, peak sampled family RSS
785,113,088 bytes, 60.10 seconds, no watchdog stop. Real three-backend
large-offset covariance/variance/std and constant-kurtosis cases passed.
Evidence: `evidence/r4/rolling-stats-final-regression.log` and watchdog JSON.

Measured on 500 rows x 4 columns, window 20: stable/native seconds were
0.052/0.0019 covariance, 0.028/0.00021 variance, and 0.031/0.00026 standard
deviation. This roughly 28x-132x pandas slowdown is material; correctness is
preferred, and no fastest-path claim is made.

Verified domains are bounded synthetic finite scales (`1e-100` through
`1e100`), translation `1e12`, pairwise NaN/Inf, constants, short windows,
`ddof` defaults, two axes, exact-axis rejection, and extreme pandas centering.
DuckDB/Polars opposite-sign nonconstant extremes near `1e308`, uncommon
`ddof>1`, exhaustive windows, and large-panel throughput remain unverified and
must not be described as certified.
