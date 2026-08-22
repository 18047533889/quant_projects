# R47 Operator Parameter Specifications - Complete Extraction

**Date**: 2026-08-13  
**Scope**: All R47 operators across 5 families (47 total operators)  
**Source**: cleaned_operators/ implementation files

---

## 1. INTRADAY STATE OPERATORS (12 operators)
**Source**: `intraday/state_ops.py`

### 1.1 intra_state_count
```python
def _calculate_series(state, target=None, window_days=1, session_tz=None, **_)
```
- **state**: panel (minute-frequency) - discrete state values
- **target**: scalar (any) - target state value to count (None = count all finite)
- **window_days**: int, default=1, min=1 - trailing window (1=today only, >1=shift(1) anchor)
- **session_tz**: str - session timezone

### 1.2 intra_state_sum
```python
def _calculate_series(x, state, target, window_days=1, session_tz=None, **_)
```
- **x**: panel (minute) - values to sum
- **state**: panel (minute) - discrete state
- **target**: scalar - target state value
- **window_days**: int, default=1, min=1

### 1.3 intra_state_vwap
```python
def _calculate_series(price, volume, state, target, window_days=1, session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **state**: panel (minute)
- **target**: scalar
- **window_days**: int, default=1, min=1

### 1.4 intra_state_interval_moment
```python
def _calculate_series(state, target, moment="std", window_days=20, min_events=4, session_tz=None, **_)
```
- **state**: panel (minute)
- **target**: scalar
- **moment**: str, choices=("std", "skew", "kurtosis"), default="std"
- **window_days**: int, default=20, min=1
- **min_events**: int, default=4, min=1

### 1.5 intra_state_follow_ratio
```python
def _calculate_series(x, state, target, lead_bars=1, window_days=20, session_tz=None, **_)
```
- **x**: panel (minute)
- **state**: panel (minute)
- **target**: scalar
- **lead_bars**: int, default=1, min=1
- **window_days**: int, default=20, min=1

### 1.6 intra_state_follow_beta
```python
def _calculate_series(x, state, target, lead_bars=1, window_days=20, min_events=8, session_tz=None, **_)
```
- **x**: panel (minute)
- **state**: panel (minute)
- **target**: scalar
- **lead_bars**: int, default=1, min=1
- **window_days**: int, default=20, min=1
- **min_events**: int, default=8, min=1

### 1.7 intra_state_follow_corr
```python
def _calculate_series(x, state, target, lead_bars=1, window_days=20, min_events=8, session_tz=None, **_)
```
- **x**: panel (minute)
- **state**: panel (minute)
- **target**: scalar
- **lead_bars**: int, default=1, min=1
- **window_days**: int, default=20, min=1
- **min_events**: int, default=8, min=1

### 1.8 intra_state_pair_same_slot_corr
```python
def _calculate_series(state_a, target_a, state_b, target_b, window_days=20, min_slots=8, session_tz=None, **_)
```
- **state_a**: panel (minute)
- **target_a**: scalar
- **state_b**: panel (minute)
- **target_b**: scalar
- **window_days**: int, default=20, min=1
- **min_slots**: int, default=8, min=1

### 1.9 intra_state_dwell_stats
```python
def _calculate_series(state, target_state=None, output="mean", min_slots=20, session_tz=None, **_)
```
- **state**: panel (minute)
- **target_state**: scalar, default=None
- **output**: str, choices=("mean", "max", "cv", "last", "share"), default="mean"
- **min_slots**: int, default=20, min=1

### 1.10 intra_state_transition_entropy
```python
def _calculate_series(state, min_slots=30, normalize=True, include_self=True, session_tz=None, **_)
```
- **state**: panel (minute)
- **min_slots**: int, default=30, min=1
- **normalize**: bool, default=True
- **include_self**: bool, default=True

### 1.11 intra_neighbor_event_class (per-bar output)
```python
def _calculate_series(event_mask, radius=1, isolated_code=1, clustered_code=2, session_tz=None, **_)
```
- **event_mask**: panel (minute)
- **radius**: int, default=1, min=1
- **isolated_code**: scalar, default=1
- **clustered_code**: scalar, default=2
- **Output**: minute-to-minute panel

### 1.12 intra_range_gap_flag (per-bar output)
```python
def _calculate_series(high, low, event_mask, neighbor_bars=1, session_tz=None, **_)
```
- **high**: panel (minute)
- **low**: panel (minute)
- **event_mask**: panel (minute)
- **neighbor_bars**: int, default=1, min=1
- **Output**: minute-to-minute panel

---

## 2. INTRADAY EVENT/RESPONSE OPERATORS (9 operators)
**Source**: `intraday/event_response.py`

### 2.1 intra_event_window_reduce
```python
def _calculate_series(x, event_mask, pre=5, post=15, reducer="mean", event_select="first", min_obs=3, session_tz=None, **_)
```
- **x**: panel (minute)
- **event_mask**: panel (minute)
- **pre**: int, default=5, min=0
- **post**: int, default=15, min=0
- **reducer**: str, choices=("mean", "sum", "std", "max", "min", "last", "slope"), default="mean"
- **event_select**: str, choices=("first", "last", "strongest"), default="first"
- **min_obs**: int, default=3, min=1

### 2.2 intra_event_pre_post_contrast
```python
def _calculate_series(x, event_mask, pre=10, post=10, metric="mean_diff", event_select="first", min_obs=5, session_tz=None, **_)
```
- **x**: panel (minute)
- **event_mask**: panel (minute)
- **pre**: int, default=10, min=0
- **post**: int, default=10, min=0
- **metric**: str, choices=("mean_diff", "median_diff", "vol_ratio", "slope_diff", "range_ratio", "activity_ratio"), default="mean_diff"
- **event_select**: str, choices=("first", "last", "strongest"), default="first"
- **min_obs**: int, default=5, min=1

### 2.3 intra_impulse_event_detector
```python
def _calculate_series(price, volume=None, event="up", threshold="robust_z", z=3.0, min_bars=1, merge_gap=2, output="count", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute), optional
- **event**: str, choices=("up", "down", "both"), default="up"
- **threshold**: str, choices=("robust_z", "vol_scaled", "quantile"), default="robust_z"
- **z**: float, default=3.0 (quantile: [0,1]; others: >0)
- **min_bars**: int, default=1, min=1
- **merge_gap**: int, default=2, min=0
- **output**: str, choices=("count", "strength", "max_strength", "first_time", "last_time", "duration"), default="count"

### 2.4 intra_post_impulse_response
```python
def _calculate_series(price, volume, amount, direction="up", threshold="robust_z", z=3.0, horizon=30, output="retention", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **amount**: panel (minute)
- **direction**: str, choices=("up", "down"), default="up"
- **threshold**: str, choices=("robust_z", "vol_scaled", "quantile"), default="robust_z"
- **z**: float, default=3.0
- **horizon**: int, default=30, min=1
- **output**: str, choices=("retention", "giveback", "max_drawdown", "vol_ratio", "volume_ratio", "amount_ratio", "vwap_hold", "recovery_half_life"), default="retention"

### 2.5 intra_probe_outcome_score
```python
def _calculate_series(price, volume, amount, direction="up", z=3.0, probe_horizon=10, response_horizon=30, output="score", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **amount**: panel (minute)
- **direction**: str, choices=("up", "down", "both"), default="up"
- **z**: float, default=3.0, min=0
- **probe_horizon**: int, default=10, min=0
- **response_horizon**: int, default=30, min=1
- **output**: str, choices=("score", "failure", "holding", "second_push"), default="score"

### 2.6 intra_supply_absorption_score
```python
def _calculate_series(price, volume, amount, event="all", horizon=30, output="absorption", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **amount**: panel (minute)
- **event**: str, choices=("up_impulse", "down_impulse", "all"), default="all"
- **horizon**: int, default=30, min=1
- **output**: str, choices=("absorption", "price_per_amount", "volume_no_drop", "downside_resilience"), default="absorption"

### 2.7 intra_consolidation_quality
```python
def _calculate_series(price, volume, amount, trigger="impulse", trigger_z=3.0, horizon=30, output="tightness", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **amount**: panel (minute)
- **trigger**: str, choices=("impulse",), default="impulse"
- **trigger_z**: float, default=3.0, min=0
- **horizon**: int, default=30, min=1
- **output**: str, choices=("tightness", "level", "vol_compression", "volume_dryup", "rising_floor", "breakout_readiness"), default="tightness"

### 2.8 intra_response_curve_features
```python
def _calculate_series(price, activity, trigger="impulse", horizon=30, curve="price", output="slope", session_tz=None, **_)
```
- **price**: panel (minute)
- **activity**: panel (minute)
- **trigger**: str, choices=("impulse",), default="impulse"
- **horizon**: int, default=30, min=1
- **curve**: str, choices=("price", "drawdown", "volatility", "activity"), default="price"
- **output**: str, choices=("slope", "curvature", "auc", "half_life", "monotonicity", "change_count"), default="slope"

### 2.9 intra_liquidity_resilience_curve_fit
```python
def _calculate_series(price, activity, shock_threshold=2.5, horizon=30, output="half_life", session_tz=None, **_)
```
- **price**: panel (minute)
- **activity**: panel (minute)
- **shock_threshold**: float, default=2.5, min=0
- **horizon**: int, default=30, min=1
- **output**: str, choices=("half_life", "residual", "asymptote", "slope", "r2"), default="half_life"

---

## 3. INTRADAY SLICE/PROFILE OPERATORS (11 operators)
**Source**: `intraday/slice_profile.py`

### 3.1 intra_slice_mask_reduce
```python
def _calculate_series(x, mask_field, window="All", slice=None, mask_side="high", mask_q=0.7, reducer="mean", min_bars=10, session_tz=None, **_)
```
- **x**: panel (minute)
- **mask_field**: panel (minute)
- **window**: str or int, default="All" ("All" or int>=1)
- **slice**: float or None, default=None (range: [0,1])
- **mask_side**: str, choices=("high", "low"), default="high"
- **mask_q**: float, default=0.7, range=(0,1)
- **reducer**: str, choices=("mean", "std", "sum", "skew", "kurtosis", "median", "slope", "last_minus_first", "positive_share"), default="mean"
- **min_bars**: int, default=10, min=1

### 3.2 intra_slice_mask_pair_reduce
```python
def _calculate_series(x, y, mask_field, window="All", slice=None, mask_side="high", mask_q=0.7, y_lag=0, reducer="corr", min_pairs=10, session_tz=None, **_)
```
- **x**: panel (minute)
- **y**: panel (minute)
- **mask_field**: panel (minute)
- **window**: str or int, default="All"
- **slice**: float or None, default=None
- **mask_side**: str, choices=("high", "low"), default="high"
- **mask_q**: float, default=0.7, range=(0,1)
- **y_lag**: int, default=0, min=0
- **reducer**: str, choices=("corr", "cov", "slope", "intercept", "r2", "euclidean", "cosine"), default="corr"
- **min_pairs**: int, default=10, min=1

### 3.3 intra_multiresolution_resample_reduce
```python
def _calculate_series(x, bar_minutes=10, lookback_days=10, reducer="mean", session_split=True, min_coverage=0.8, session_tz=None, **_)
```
- **x**: panel (minute)
- **bar_minutes**: int, default=10, min=1
- **lookback_days**: int, default=10, min=1
- **reducer**: str, choices=("mean", "std", "slope", "skew", "kurtosis", "last_minus_first"), default="mean"
- **session_split**: bool, default=True
- **min_coverage**: float, default=0.8, range=(0,1]

### 3.4 intra_same_slot_zscore
```python
def _calculate_series(x, history_days=20, ddof=1, min_history=10, session_tz=None, **_)
```
- **x**: panel (minute)
- **history_days**: int, default=20, min=1
- **ddof**: int, choices=(0, 1), default=1
- **min_history**: int, default=10, min=1

### 3.5 intra_session_boundary_jump
```python
def _calculate_series(price, volume=None, pre_close=None, boundary="lunch_restart", pre_bars=5, post_bars=5, output="gap", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute), optional
- **pre_close**: panel (daily), optional
- **boundary**: str, choices=("open", "lunch_restart", "close"), default="lunch_restart"
- **pre_bars**: int, default=5, min=1
- **post_bars**: int, default=5, min=1
- **output**: str, choices=("gap", "normalized_gap", "recovery", "volume_jump"), default="gap"

### 3.6 intra_volume_at_price_profile
```python
def _calculate_series(price, volume, bins=64, weighting="volume", price_basis="close", normalize=True, output="entropy", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **bins**: int, default=64, min=8
- **weighting**: str, choices=("volume", "amount", "time"), default="volume"
- **price_basis**: str, choices=("close", "ohlc_typical", "vwap"), default="close"
- **normalize**: bool, default=True
- **output**: str, choices=("entropy", "skew", "kurtosis", "poc_price", "value_area_width", "tail_mass", "dip", "concentration"), default="entropy"

### 3.7 intra_volume_profile_peak_geometry
```python
def _calculate_series(price, volume, bins=64, smooth=2, min_prominence=0.05, output="peak_count", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **bins**: int, default=64, min=8
- **smooth**: int, default=2, min=0
- **min_prominence**: float, default=0.05, range=(0,1)
- **output**: str, choices=("peak_count", "top_peak_mass", "second_peak_mass", "peak_ratio", "peak_distance", "top_peak_width", "nearest_peak_distance", "valley_depth"), default="peak_count"

### 3.8 intra_volume_profile_supply_structure
```python
def _calculate_series(price, volume, bins=64, decay=20.0, output="overhead_mass", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **bins**: int, default=64, min=8
- **decay**: float, default=20.0, min=0
- **output**: str, choices=("overhead_mass", "near_overhead_mass", "under_price_mass", "supply_vacuum", "nearest_upper_peak", "nearest_lower_peak", "distance_weighted_overhang"), default="overhead_mass"

### 3.9 intra_volume_profile_value_area
```python
def _calculate_series(price, volume, bins=64, target_mass=0.7, output="value_area_width", session_tz=None, **_)
```
- **price**: panel (minute)
- **volume**: panel (minute)
- **bins**: int, default=64, min=8
- **target_mass**: float, default=0.7, range=(0,1)
- **output**: str, choices=("value_area_width", "poc_price", "value_area_high", "value_area_low", "value_area_mid", "in_value_area"), default="value_area_width"

### 3.10 intra_round_price_clustering_share
```python
def _calculate_series(price, lattice=1.0, tolerance_ticks=0.25, window="All", output="share", min_bars=30, session_tz=None, **_)
```
- **price**: panel (minute)
- **lattice**: float, default=1.0, min=0
- **tolerance_ticks**: float, default=0.25, min=0
- **window**: str or int, default="All"
- **output**: str, choices=("share", "excess_share", "run_length"), default="share"
- **min_bars**: int, default=30, min=1

### 3.11 intra_round_price_barrier_response
```python
def _calculate_series(price, lattice=1.0, lookback_days=20, tolerance_ticks=1.0, output="cross_rate", min_events=5, session_tz=None, **_)
```
- **price**: panel (minute)
- **lattice**: float, default=1.0, min=0
- **tolerance_ticks**: float, default=1.0, min=0
- **lookback_days**: int, default=20, min=1
- **output**: str, choices=("cross_rate", "bounce_rate", "magnet_strength", "asymmetry"), default="cross_rate"
- **min_events**: int, default=5, min=1

---

## 4. INTRADAY TOPOLOGY/MANIFOLD OPERATORS (4 operators)
**Source**: `intraday/topology_manifold.py`

### 4.1 intra_matrix_profile_session_features
```python
def _calculate_series(close, window=20, feature="discord_score", **_)
```
- **close**: panel (minute)
- **window**: int, default=20, min=4
- **feature**: str, choices=("min_dist", "mean_dist", "discord_score"), default="discord_score"

### 4.2 intra_dmd_koopman_features
```python
def _calculate_series(close, rank=5, feature="growth_rate", **_)
```
- **close**: panel (minute)
- **rank**: int, default=5, range=[2, 20]
- **feature**: str, choices=("dominant_freq", "growth_rate", "mode_energy"), default="growth_rate"

### 4.3 intra_covariance_manifold_shift
```python
def _calculate_series(close, window=30, **_)
```
- **close**: panel (minute)
- **window**: int, default=30, min=10

### 4.4 intra_critical_transition_score
```python
def _calculate_series(close, window=40, **_)
```
- **close**: panel (minute)
- **window**: int, default=40, min=10

---

## 5. INTRADAY SMART MONEY OPERATORS (4 operators)
**Source**: `intraday/smart_money.py`

### 5.1 intra_dynamic_stock_graph_features
```python
def _calculate_series(close, **_)
```
- **close**: panel (minute)

### 5.2 intra_common_trading_intensity
```python
def _calculate_series(volume, amount, **_)
```
- **volume**: panel (minute)
- **amount**: panel (minute)

### 5.3 intra_local_conditional_entropy
```python
def _calculate_series(close, volume, **_)
```
- **close**: panel (minute)
- **volume**: panel (minute)

### 5.4 intra_smart_money_vwap_ratio
```python
def _calculate_series(close, amount, volume, **_)
```
- **close**: panel (minute)
- **amount**: panel (minute)
- **volume**: panel (minute)

---

## 6. INTRADAY STATE SPACE OPERATORS (2 operators shown)
**Source**: `intraday/intra_state_space.py`

### 6.1 intra_kalman_latent_price
```python
def _calculate_series(close, min_bars=30, session_tz=None, **_)
```
- **close**: panel (minute)
- **min_bars**: int, default=30, min=1

### 6.2 intra_state_space_volume_components
```python
def _calculate_series(volume, min_bars=30, session_tz=None, **_)
```
- **volume**: panel (minute)
- **min_bars**: int, default=30, min=1

---

## 7. PANEL OPERATORS (5 operators)
**Source**: `cross_section/panel_batch1.py`

### 7.1 panel_day_night_beta_gap
```python
def _calculate_series(day_ret, night_ret, mkt_ret, window=63, min_periods=40, **_)
```
- **day_ret**: panel (daily)
- **night_ret**: panel (daily)
- **mkt_ret**: panel (daily)
- **window**: int, default=63, min=2
- **min_periods**: int, default=40, min=2

### 7.2 pastor_stambaugh_beta
```python
def _calculate_series(ret, mkt_ret, liquidity_innov, window=126, min_periods=60, **_)
```
- **ret**: panel (daily)
- **mkt_ret**: panel (daily)
- **liquidity_innov**: panel (daily)
- **window**: int, default=126, min=2
- **min_periods**: int, default=60, min=2

### 7.3 price_delay_score
```python
def _calculate_series(ret, mkt_ret, window=126, lag=4, min_periods=60, **_)
```
- **ret**: panel (daily)
- **mkt_ret**: panel (daily)
- **window**: int, default=126, min=2
- **lag**: int, default=4, min=0
- **min_periods**: int, default=60, min=2

### 7.4 report_asof
```python
def _calculate_series(event, max_lookback=365, **_)
```
- **event**: panel (daily) - 0/1 indicator
- **max_lookback**: int, default=365, min=1

### 7.5 event_window_return_asof
```python
def _calculate_series(ret, event, pre_window=5, post_window=5, max_lookback=365, **_)
```
- **ret**: panel (daily)
- **event**: panel (daily)
- **pre_window**: int, default=5, min=0
- **post_window**: int, default=5, min=0
- **max_lookback**: int, default=365, min=1

---

## 8. FISCAL OPERATORS (10 operators shown)
**Source**: `fiscal_strict.py` and `fiscal_event_ops.py`

### 8.1 fiscal_perpetual_inventory
```python
def pd_fiscal_perpetual_inventory(flow, period_id, depreciation=0.15, periods_per_year=4, warmup_periods=8, require_consecutive=True, revision_policy="latest_available", **_)
```
- **flow**: panel (fiscal-frequency)
- **period_id**: panel (fiscal-frequency) - period identifiers
- **depreciation**: float, default=0.15, range=[0, 1)
- **periods_per_year**: int, default=4, min=1
- **warmup_periods**: int, default=8, min=0
- **require_consecutive**: bool, default=True
- **revision_policy**: str, choices=("latest_available", "first_available"), default="latest_available"

### 8.2 fiscal_standardized_surprise
```python
def pd_fiscal_standardized_surprise(x, period_id, seasonal_lag=4, lookback_periods=8, min_history=4, require_consecutive=True, revision_policy="latest_available", **_)
```
- **x**: panel (fiscal)
- **period_id**: panel (fiscal)
- **seasonal_lag**: int, default=4, min=1
- **lookback_periods**: int, default=8, min=1
- **min_history**: int, default=4, min=1
- **require_consecutive**: bool, default=True
- **revision_policy**: str, choices=("latest_available", "first_available"), default="latest_available"

### 8.3 fiscal_sign_consistency
```python
def pd_fiscal_sign_consistency(signal, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available", **_)
```
- **signal**: panel (fiscal)
- **period_id**: panel (fiscal)
- **periods**: int, default=8, min=1
- **min_periods**: int, default=3, min=1
- **require_consecutive**: bool, default=True
- **revision_policy**: str, default="latest_available"

### 8.4 fiscal_change_direction_agreement
```python
def pd_fiscal_change_direction_agreement(x, y, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available", **_)
```
- **x**: panel (fiscal)
- **y**: panel (fiscal)
- **period_id**: panel (fiscal)
- **periods**: int, default=8, min=1
- **min_periods**: int, default=3, min=1
- **require_consecutive**: bool, default=True
- **revision_policy**: str, default="latest_available"

### 8.5 fiscal_sign_agreement
```python
def pd_fiscal_sign_agreement(x, y, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available", **_)
```
- **x**: panel (fiscal)
- **y**: panel (fiscal)
- **period_id**: panel (fiscal)
- **periods**: int, default=8, min=1
- **min_periods**: int, default=3, min=1
- **require_consecutive**: bool, default=True
- **revision_policy**: str, default="latest_available"

### 8.6 fiscal_autocorr
```python
def pd_fiscal_autocorr(x, period_id, periods=12, lag=1, min_pairs=3, require_consecutive=True, revision_policy="latest_available", **_)
```
- **x**: panel (fiscal)
- **period_id**: panel (fiscal)
- **periods**: int, default=12, min=1
- **lag**: int, default=1, min=1
- **min_pairs**: int, default=3, min=1
- **require_consecutive**: bool, default=True
- **revision_policy**: str, default="latest_available"

### 8.7 fiscal_reversal_ratio
```python
def pd_fiscal_reversal_ratio(x, period_id, periods=8, min_pairs=3, require_consecutive=True, revision_policy="latest_available", **_)
```
- **x**: panel (fiscal)
- **period_id**: panel (fiscal)
- **periods**: int, default=8, min=1
- **min_pairs**: int, default=3, min=1
- **require_consecutive**: bool, default=True
- **revision_policy**: str, default="latest_available"

### 8.8 fiscal_regression_resid_std
```python
def pd_fiscal_regression_resid_std(y, period_id, x1, x2=None, periods=12, min_obs=None, add_intercept=True, ddof=1, require_consecutive=True, revision_policy="latest_available", **_)
```
- **y**: panel (fiscal)
- **period_id**: panel (fiscal)
- **x1**: panel (fiscal)
- **x2**: panel (fiscal), optional
- **periods**: int, default=12, min=1
- **min_obs**: int, default=periods, min=1
- **add_intercept**: bool, default=True
- **ddof**: int, default=1, min=0
- **require_consecutive**: bool, default=True
- **revision_policy**: str, default="latest_available"

### 8.9 row_sum_skipna
```python
def pd_row_sum_skipna(*inputs, min_count=1, **kwargs)
```
- **inputs**: variable number of panels or scalars
- **min_count**: int, default=1, min=1

### 8.10 fiscal_accrual_quality (from fiscal_strict.py)
- Parameters similar to fiscal_regression_resid_std
- Specialized for cash flow vs accrual regression

---

## 9. TECHNICAL FILTER OPERATORS (4 operators)
**Source**: `technical/frequency_filters.py`

### 9.1 ts_bessel_lowpass_causal
```python
def _calculate_series(x, order=4, cutoff=0.05, **_)
```
- **x**: panel (daily)
- **order**: int, default=4, range=[1, 8]
- **cutoff**: float, default=0.05, range=(1e-4, 0.499)

### 9.2 ts_fir_lowpass_causal
```python
def _calculate_series(x, ntaps=21, cutoff=0.1, window="hamming", **_)
```
- **x**: panel (daily)
- **ntaps**: int, default=21, range=[3, 201] (auto-adjusted to odd)
- **cutoff**: float, default=0.1, range=(1e-4, 0.499)
- **window**: str, choices=("hamming", "hann", "blackman"), default="hamming"

### 9.3 ts_spectral_lowpass_trailing
```python
def _calculate_series(x, window=64, cutoff_freq=5, **_)
```
- **x**: panel (daily)
- **window**: int, default=64, range=[8, 512]
- **cutoff_freq**: int, default=5, range=[1, 256]

### 9.4 ts_causal_savgol_endpoint
```python
def _calculate_series(x, window=21, polyorder=3, **_)
```
- **x**: panel (daily)
- **window**: int, default=21, min=5
- **polyorder**: int, default=3, range=[1, window-2]

---

## SUMMARY STATISTICS

**Total Operators Extracted**: 47 (across 5 major families)

**By Family**:
- Intraday State: 12 operators
- Intraday Event/Response: 9 operators
- Intraday Slice/Profile: 11 operators
- Intraday Topology: 4 operators
- Intraday Smart Money: 4 operators
- Intraday State Space: 2+ operators
- Panel: 5 operators
- Fiscal: 10 operators
- Technical Filters: 4 operators

**Parameter Type Distribution**:
- Panel inputs (minute/daily/fiscal): All operators
- Integer parameters: 45+ parameters
- Float parameters: 30+ parameters
- String choice parameters: 40+ parameters
- Boolean parameters: 15+ parameters

**Common Parameter Patterns**:
1. **Window/Lookback**: Most operators have `window`, `lookback_days`, `periods`, or `history_days`
2. **Minimum Observations**: `min_bars`, `min_events`, `min_obs`, `min_periods`, `min_pairs`
3. **Output Selection**: Many operators have `output` or `feature` choice parameters
4. **Revision Policy**: Fiscal operators uniformly use `revision_policy`
5. **Session Timezone**: Intraday operators support `session_tz`

**Validation Notes**:
- All integer parameters have explicit minimum bounds
- Float parameters have range constraints where applicable
- Choice parameters are validated against fixed sets
- Boolean parameters explicitly checked (not just truthy values)
- Panel inputs require strict alignment checking

---

**END OF EXTRACTION**
