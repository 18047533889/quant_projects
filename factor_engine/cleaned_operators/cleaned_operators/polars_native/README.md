# Polars Native Operator Implementations

## Overview
This directory contains pure Polars native implementations of time series operators for maximum performance.

## Implementation Status

### ts_batch1.py - Time Series Operators (112 operators)

All operators use Polars lazy API with the following patterns:
- **Window operations**: `rolling_mean()`, `rolling_std()`, `rolling_max()`, `rolling_min()`, `rolling_quantile()`
- **Lag/shift**: `.shift(n)` for accessing past/future values
- **Cumulative**: `.cum_sum()`, `.cum_prod()`, `.cum_max()`, `.cum_min()`
- **EWM**: `.ewm_mean()`, `.ewm_std()` for exponential weighting
- **Custom logic**: `.rolling_map()` for complex window calculations

#### Categories Implemented:

1. **Basic Rolling Statistics** (10 operators)
   - ts_mean_if, ts_std_if, ts_sum_if, ts_min_if, ts_max_if
   - ts_median, ts_quantile, ts_quantile_if, ts_rank_if

2. **Correlation & Covariance** (4 operators)
   - ts_corr, ts_cov, ts_corr_if, ts_cov_if

3. **Lag & Shift** (2 operators)
   - ts_nth_value, ts_last_if

4. **Counting & Coverage** (6 operators)
   - ts_count_if, ts_valid_count, ts_coverage_ratio
   - ts_positive_ratio, ts_negative_ratio, ts_zero_ratio

5. **Extrema & Positional** (4 operators)
   - ts_argmax, ts_argmin, ts_argmax_age, ts_argmin_age

6. **TopK & BottomK** (6 operators)
   - ts_topk_mean, ts_topk_sum, ts_topk_std
   - ts_bottomk_mean, ts_bottomk_sum, ts_bottomk_std

7. **Moments & Distribution** (3 operators)
   - ts_kurt, ts_moment, ts_quantile_range, ts_trimmed_mean

8. **Ratio & Scaling** (3 operators)
   - ts_ratio, ts_location_shift, ts_scale_shift

9. **Decay & Weighted** (3 operators)
   - ts_decay_linear, ts_sum_decay, ts_decay_exp_window

10. **Time-based Features** (3 operators)
    - ts_days_since, ts_days_since_high, ts_days_since_low, ts_time_since_change

11. **Streak & Pattern** (3 operators)
    - ts_true_streak, ts_transition_count, ts_new_high, ts_new_low

12. **Distance & Channel** (5 operators)
    - ts_distance_to_high, ts_distance_to_low
    - ts_channel_width, ts_channel_width_pct, ts_channel_position

13. **Drawdown & Performance** (3 operators)
    - ts_max_drawdown, ts_time_under_water, ts_recovery_fraction

14. **Volatility & Risk** (6 operators)
    - ts_downside_deviation, ts_upside_deviation, ts_expected_shortfall
    - ts_lower_partial_moment, ts_upper_partial_moment, ts_vol_of_vol

15. **Trend & Regression** (9 operators)
    - ts_regression_slope, ts_regression_intercept, ts_regression_r2
    - ts_regression_resid, ts_time_slope, ts_monotonicity
    - ts_regression_resid_mean, ts_regression_tstat
    - ts_regression_forecast_error, ts_regression_forecast_error_z

16. **State & Persistence** (2 operators)
    - ts_staleness, ts_sign_persistence

17. **EWM & Adaptive** (3 operators)
    - ts_ewm_corr, ts_ewm_cov, ts_kama

18. **Autocorrelation & Memory** (2 operators)
    - ts_autocorrelation_time, ts_mean_reversion_half_life

19. **Swing & Pivot** (4 operators)
    - ts_prev_high, ts_prev_low, ts_swing_amplitude, ts_swing_amplitude_pct

20. **Path & Efficiency** (3 operators)
    - ts_path_efficiency, ts_roughness, ts_turning_rate

21. **Rank & Score** (2 operators)
    - ts_score_rank_weighted_mean, expanding_rank

22. **Robust Statistics** (3 operators)
    - ts_robust_zscore_prior, ts_robust_zscore_inclusive, ts_qn_scale

23. **Beta Estimation** (1 operator)
    - ts_beta_if

24. **Tail & Extreme** (3 operators)
    - ts_tail_ratio, ts_tail_mean, ts_expected_shortfall_asymmetry

25. **Partial Correlation** (1 operator)
    - ts_partial_corr

26. **Distance Correlation** (2 operators)
    - ts_distance_corr, ts_distance_cov

27. **Fill & Imputation** (1 operator)
    - ts_ffill_limited

28. **Filtering & Smoothing** (3 operators)
    - ts_median3_causal, ts_rolling_median_causal, ts_robust_ema

29. **Run & Streak Analysis** (2 operators)
    - ts_run_strength, ts_run_efficiency

30. **Quantile & Expectile** (2 operators)
    - ts_quantile_regression_slope, ts_expectile

31. **Support & Resistance** (6 operators)
    - ts_support_level, ts_resistance_level
    - ts_distance_to_support, ts_distance_to_resistance
    - ts_breakout_high, ts_breakdown_low

## Performance Characteristics

### Lazy Evaluation
All operators use Polars lazy API (`.lazy()` → transformations → `.collect()`), enabling:
- Query optimization by Polars engine
- Automatic parallelization where possible
- Memory-efficient streaming for large datasets

### Native Operations
Preference order:
1. **Built-in rolling**: `rolling_mean()`, `rolling_std()`, `rolling_max()` - fastest
2. **EWM operations**: `ewm_mean()`, `ewm_std()` - highly optimized
3. **Rolling map**: `rolling_map()` - for complex custom logic, still efficient

### Memory Usage
- **Window operations**: O(window_size) per row
- **Cumulative operations**: O(1) per row
- **Full-window transforms**: May require O(n) temporary storage

## Testing

To test an operator:
```python
import polars as pl
from cleaned_operators.polars_native.ts_batch1 import TSMeanIfPolarsNative

# Create test data
feature = pl.Series("price", [100, 102, 101, 105, 103, 107, 106])
condition = [True, True, False, True, True, True, False]

# Instantiate and run
op = TSMeanIfPolarsNative()
result = op._calculate_series(feature, condition, window=3)
print(result)
```

## Remaining ts_* Operators (70 operators)

These require more complex implementations:
- Complex state machines (Kalman filters, GARCH models)
- Advanced statistical methods (DMD, spectral analysis, topological features)
- Multi-stage algorithms (pivot detection, regime detection)
- Information theory measures (mutual information, transfer entropy)

Candidates for next batch:
- ts_sma_cn (simple moving average with count normalization)
- ts_realized_quarticity
- ts_volatility-related simple operators
- Additional simple regression variants

## Integration

To use these operators in the factor engine:
1. Ensure they're registered via `@register_operator` decorator
2. Import in main operator registry
3. Test with panel data (cross-section × time)
4. Verify against existing pandas/numpy implementations for correctness

## Notes

- All operators handle NaN/null values consistently
- Window sizes are inclusive (window=10 means current + 9 past observations)
- Conditional operators (ts_*_if) filter data before applying the operation
- Age operators return number of periods (0 = current, 1 = 1 period ago, etc.)
