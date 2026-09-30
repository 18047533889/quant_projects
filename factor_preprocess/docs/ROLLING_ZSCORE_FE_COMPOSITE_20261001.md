# Rolling z-score FE composite recipe (v1)

The registered recipe identity is `FE_COMPOSITE:long_smoothing.lagged_zscore:v1`.
It is a semantic contract for the causal paired rolling mean/std computation,
not a package-release number or a hash of implementation contents.

For row *t* within an asset, the result is
`(value[t] - mean(past_window)) / std(past_window, ddof)`. The current row is
excluded from both statistics. `window` counts preceding observed rows, not
calendar slots; `min_periods` is the minimum number of valid observations.
Output remains positional to the input rows, including duplicate index labels,
and is isolated by asset. Rolling statistics follow the FP oracle's treatment
of non-finite history; division retains IEEE results (`0/0` is NaN and a
nonzero numerator divided by zero is signed infinity).

The FE implementation requires both:

- `factor_engine/backend/long_smoothing.py::lagged_zscore`, which exposes the
  paired recipe and its public parameter/row-alignment contract.
- `factor_engine/backend/native_long_rolling_moments.py::collect_lagged_moments`,
  which supplies the key-preserving native Polars rolling moments used by that
  recipe (one lazy collection; no key join or map-groups fallback).

The FP consumer binds the exact identity in its adapter and transform registry.
Its production path must fail closed when the FE import, executor, or recipe
identity is unavailable. The retained FP-native implementation is available
only through explicit research mode; it is not an implicit production fallback.

## Cross-repository synchronization

An FP checkout that selects this composite must use an FE dependency/source state
that contains both FE entry points above and supports the v1 identity. In
particular, the separate HKUST FP consumer currently paired with an older FE
must be upgraded/synchronized before enabling this route. Do not guess a release
number: synchronize the corresponding FE and FP source changes in the consuming
deployment, then verify `get_fe_composite_executor("rolling_zscore",
"FE_COMPOSITE:long_smoothing.lagged_zscore:v1")` returns an executor and run
the paired parity/routing tests in that environment. If the identity is not
available, production execution should raise the existing governance error;
do not hide incompatibility with an FP fallback.

This document describes source-level recipe compatibility. It is not a claim
that a particular deployment or external consumer has already been upgraded.

## Verification and performance

The full `factor_preprocess` suite completed with **732 passed, 1 xfailed, and
30 warnings in 44.82 seconds**. This run preceded the addition of the
near-overflow (`±1e308`) and tiny-magnitude (`~1e-300`, whose squared variance
can underflow) cases to the numerical stress
test; after those cases were added, that stress test alone passed (**1 passed in
0.57 seconds**). The final dedicated z-score suite on the updated main tree
completed with **37 passed, 1 warning in 0.79 seconds** (session 57595). The
paired tests compare NaN, positive-infinity, and negative-infinity
classifications exactly, and compare finite values within their stated
numerical tolerance. They do not establish correctness for every possible
float64 input.

A final three-pair alternating microbenchmark, after the constant-window
stabilization, measured these in-memory synthetic cases (FP and FE alternated
in the order FP→FE, FE→FP, FP→FE; medians are across the three measured calls):

| Case | FP median | FE median | FP/FE speed ratio | Maximum finite absolute difference |
|---|---:|---:|---:|---:|
| Balanced, 128,000 rows (256 × 500) | 64.007 ms | 27.923 ms | 2.292× | 2.71e-14 |
| Ragged/interleaved, 92,160 rows | 52.621 ms | 23.427 ms | 2.246× | 7.99e-15 |

The benchmark's tested IEEE classifications matched; its finite outputs were
compared with `rtol=3e-12` and `atol=3e-12`. Concurrent background GFN processes
were using approximately one CPU core each, so these timings are conditional on
that shared-server load, not a clean isolated benchmark. The native-path test
also instruments Polars and verifies exactly one `LazyFrame.collect()` with no
join or map-groups plan. These are primitive timings and numerical checks only;
they are not real full-factor workload measurements and do not establish an
end-to-end production performance improvement.
