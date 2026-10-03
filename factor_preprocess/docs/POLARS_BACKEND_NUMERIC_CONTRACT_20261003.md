# Polars cross-sectional numeric contract (2026-10-03)

The Polars rank and z-score helpers execute their numeric operations as Polars expressions. Pandas input is converted at the API boundary; no NumPy or pandas numeric kernel is used by these implementations.

## Rank

`cs_rank_polars` follows `factor_preprocess.transforms.cross_sectional.cs_rank`:

- The accepted rank methods are `average`, `min`, `max`, `dense`, and `ordinal`. Other method names raise `ValueError`.
- Rank eligibility uses the FP kernel's `np.isfinite` policy. NaN, null, positive infinity, and negative infinity produce NaN output and do not contribute to ranks or percentile denominators.
- Integer input remains in its original integer dtype through sorting/ranking. It is not cast to float, which preserves exact ordering and ties above `2**53`.
- Percentile ranks use `(rank - 1) / (n_finite - 1)` for groups with more than one finite value. A sole finite value receives `0.5`; invalid members in that group remain NaN.
- Tie behavior follows Polars' matching rank method and caller row order for `ordinal`. Input row order is preserved.

## Z-score

`cs_zscore_polars` follows the public FP kernel's `np.nanmean`/`np.nanstd` semantics:

- NaN and null values are excluded from mean and standard deviation; their outputs are NaN.
- Infinities are retained in the moment calculations. They are not converted to missing. If the resulting standard deviation is NaN, NumPy's `std > 0` condition is false and nonmissing outputs use `constant_value`. If standard deviation is positive infinity, it satisfies the comparison and ordinary IEEE arithmetic determines the result.
- Constant groups, singleton groups, and groups without valid observations use the same conditional behavior as the FP kernel. The caller-supplied `ddof` and `constant_value` are passed through without adding stricter parameter validation.

For pandas frames, the result is restored to the caller's original index after Polars computes values positionally. Duplicate index labels therefore remain duplicated and in caller order; they are never used to align computed values. For Polars frames, it returns a Polars Series in input row order. An all-null Polars value column (`pl.Null`) is accepted as an all-missing numeric input.

## Validation and performance

`tests/test_polars_backend_numeric_contract_oct03.py` compares pandas and Polars public inputs against the FP kernels over tie methods, percentiles, large integers, NaN/null/Inf, constants, singletons, all-missing groups, and a deterministic 1,000-row CPU case. This is semantic A/B validation; it makes no speedup claim. Performance measurements should report input conversion, Polars expression execution, and pandas output conversion separately.
