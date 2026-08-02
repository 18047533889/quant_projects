# FactorEngine Operator Expansion V2

> Scope: factor-shaped production targets only. ResearchTool remains reserved for diagnostics/statistical research utilities.

## Global parameter rules

1. Technical horizons are parameters; names such as `RSI14`/`MA20` are not new canonicals.
2. Rolling baselines that define a breakout/support reference exclude the current bar when the economic meaning requires a prior reference.
3. Confirmed pivots are emitted on the confirmation timestamp, never retroactively on the pivot timestamp.
4. Structural searches are bounded by explicit `history_window`/`window` arguments.
5. Recursive operators are stateful/checkpointed or full-replay; they are not treated as ordinary rolling transforms.
6. Minute→daily operators expose `bar_minutes`, `cutoff_time`, and `min_coverage`; non-A-share minute data require explicit market-session/source contracts.
7. Fundamental time operators use `period_id`/`target_period_id`; report periods are not interchangeable with trading-day rows.

## Price structure / pivots

```text
ts_prev_high(x, window)
ts_prev_low(x, window)
ts_distance_to_high(x, window)
ts_distance_to_low(x, window)
ts_breakout_high(x, window)
ts_breakdown_low(x, window)
ts_new_high(x, window)
ts_new_low(x, window)
ts_channel_position(x, window)
ts_days_since_high(x, window)
ts_days_since_low(x, window)
ts_range_expansion(high, low, window)

ts_confirmed_pivot_high(high, left_window, right_window)
ts_confirmed_pivot_low(low, left_window, right_window)
ts_last_pivot_high(high, left_window, right_window, history_window)
ts_last_pivot_low(low, left_window, right_window, history_window)
ts_nth_pivot_high(high, left_window, right_window, history_window, n)
ts_nth_pivot_low(low, left_window, right_window, history_window, n)
ts_nth_pivot_high_age(...)
ts_nth_pivot_low_age(...)
ts_pivot_high_count(...)
ts_pivot_low_count(...)
ts_pivot_high_spacing(...)
ts_pivot_low_spacing(...)
```

## Swing / support / resistance / geometry

```text
ts_swing_amplitude(...)
ts_swing_amplitude_pct(...)
ts_swing_amplitude_atr(..., atr_window)
ts_swing_duration(...)
ts_swing_velocity(...)

ts_resistance_level(..., history_window, points)
ts_support_level(..., history_window, points)
ts_resistance_slope(...)
ts_support_slope(...)
ts_resistance_fit_r2(...)
ts_support_fit_r2(...)
ts_distance_to_resistance(...)
ts_distance_to_support(...)
ts_resistance_break(...)
ts_support_break(...)

ts_channel_width(...)
ts_channel_width_pct(...)
ts_channel_width_atr(..., atr_window)
ts_channel_width_slope(..., window)
ts_line_convergence(...)
ts_line_parallelism(...)
ts_pattern_symmetry(...)
```

## Chart patterns

All pattern outputs are causal detections/scores, not discretionary trading recommendations.

```text
pattern_double_top(..., tolerance, min_depth, min_spacing, max_spacing)
pattern_double_bottom(...)
pattern_triple_top(...)
pattern_triple_bottom(...)
pattern_head_shoulders(..., shoulder_tolerance, head_min_prominence, max_neckline_slope)
pattern_inverse_head_shoulders(...)
pattern_123_bull(..., min_swing)
pattern_123_bear(..., min_swing)

pattern_sym_triangle(..., slope_threshold)
pattern_ascending_triangle(...)
pattern_descending_triangle(...)
pattern_rising_wedge(...)
pattern_falling_wedge(...)
pattern_rectangle(...)
pattern_rising_channel(..., parallel_tolerance)
pattern_falling_channel(...)
pattern_broadening(...)

pattern_bull_flag(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold)
pattern_bear_flag(...)
pattern_bull_pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold)
pattern_bear_pennant(...)

pattern_rounding_bottom(close, window, min_fit)
pattern_rounding_top(close, window, min_fit)
pattern_cup(close, window, min_depth, max_edge_diff, min_fit)
pattern_cup_handle(close, high, low, cup_window, handle_window, min_depth, max_edge_diff, min_fit, max_handle_retracement)

pattern_breakout_retest(close, window, max_wait, tolerance)
pattern_breakdown_retest(close, window, max_wait, tolerance)
```

## Price-volume / liquidity

```text
average_volume(volume, window)
average_turnover(turnover, window)
adv(close, volume, window)
abnormal_volume(volume, window)
abnormal_turnover(turnover, window)
volume_volatility(volume, window)
turnover_volatility(turnover, window)
volume_autocorr(volume, window, lag)
turnover_autocorr(turnover, window, lag)

amihud_illiquidity(ret, close, volume, window)
price_impact(ret, dollar_volume, window)
return_per_turnover(ret, turnover)
volume_shock(volume, window)
turnover_shock(turnover, window)
volume_acceleration(volume, short_window, long_window)
turnover_acceleration(turnover, short_window, long_window)

up_volume_ratio(ret, volume, window)
down_volume_ratio(ret, volume, window)
signed_volume_imbalance(ret, volume, window)
up_down_volume_ratio(ret, volume, window)
volume_weighted_return(ret, volume, window)
volume_weighted_momentum(close, volume, window)
price_volume_divergence(close, volume, price_window, volume_window)
price_turnover_divergence(close, turnover, price_window, turnover_window)
return_volume_beta(ret, volume, window)
return_turnover_beta(ret, turnover, window)

ADL(high, low, close, volume, window)
ChaikinOscillator(high, low, close, volume, fast_window, slow_window, adl_window)
ForceIndex(close, volume, window)
EaseOfMovement(high, low, volume, window, volume_scale)
bounded_nvi(close, volume, window)
bounded_pvi(close, volume, window)

zero_return_ratio(ret, window, threshold)
roll_spread_proxy(ret, window)
corwin_schultz_spread(high, low, window)
high_low_spread_proxy(high, low, window)
turnover_adjusted_volatility(ret, turnover, window)
volume_to_range(volume, high, low, window)
```

## Parameterized technical indicators

```text
DMI_plus(high, low, close, window)
DMI_minus(high, low, close, window)
DX(high, low, close, window)
NATR(high, low, close, window)
PPO(close, fast_window, slow_window)
PPO_signal(close, fast_window, slow_window, signal_window)
PPO_hist(...)
PVO(volume, fast_window, slow_window)
PVO_signal(...)
PVO_hist(...)
CMO(close, window)
VortexPlus(high, low, close, window)
VortexMinus(high, low, close, window)
KeltnerMid(close, ema_window)
KeltnerUpper(high, low, close, ema_window, atr_window, multiplier)
KeltnerLower(...)
KeltnerPosition(...)
TSI(close, long_window, short_window)
TSI_signal(close, long_window, short_window, signal_window)
UltimateOscillator(high, low, close, short_window, medium_window, long_window, short_weight, medium_weight, long_weight)
DEMA(x, window)
TEMA(x, window)
ichimoku_tenkan(high, low, tenkan_window)
ichimoku_kijun(high, low, kijun_window)
ichimoku_senkou_a(high, low, tenkan_window, kijun_window)
ichimoku_senkou_b(high, low, senkou_b_window)
ichimoku_cloud_width(...)
ichimoku_cloud_position(...)
KAMA(close, er_window, fast_window, slow_window)
Supertrend(high, low, close, atr_window, multiplier)
SupertrendDirection(...)
PSAR(high, low, acceleration, maximum)
```

`KAMA`, `Supertrend`, `SupertrendDirection`, and `PSAR` are stateful/full-replay production targets and must not be optimized as ordinary stateless rolling operators.

## Continuous candlestick geometry

```text
candle_body(open, close)
candle_abs_body(open, close)
candle_range(high, low)
candle_body_ratio(open, high, low, close)
candle_upper_shadow(...)
candle_lower_shadow(...)
candle_upper_shadow_ratio(...)
candle_lower_shadow_ratio(...)
candle_close_location(high, low, close)
candle_gap(open, close)
candle_gap_pct(open, close)
candle_direction(open, close)
candle_range_atr(..., atr_window)

candle_body_zscore(open, close, window)
candle_range_zscore(high, low, window)
candle_upper_shadow_zscore(..., window)
candle_lower_shadow_zscore(..., window)
candle_body_percentile(open, close, window)
candle_range_percentile(high, low, window)
candle_gap_atr(open, high, low, close, atr_window)
candle_body_position(...)
candle_overlap_ratio(high, low)
candle_inside_ratio(high, low)
candle_close_strength(high, low, close)
candle_rejection_upper(...)
candle_rejection_lower(...)
```

## Japanese candlesticks

Core dedicated names remain supported. Additional patterns are recipes over one adaptive semantic engine:

```text
candlestick_pattern(open, high, low, close, pattern, body_window, shadow_window, penetration)
```

The adaptive engine uses prior-window body/range statistics and keeps all lookbacks explicit. It is FactorEngine-defined semantics; TA-Lib may be used later as an optional golden oracle, not as an installation-dependent production backend.

Recipe aliases include Two Crows, Three Inside/Outside, Three-Line Strike, Three Stars in the South, Abandoned Baby, Advance Block, Belt Hold, Breakaway, Closing Marubozu, Counterattack, Doji Star, Evening Doji Star, High Wave, Homing Pigeon, Identical Three Crows, In/On Neck, Thrusting, Ladder Bottom, Long-Legged Doji, Long/Short Line, Matching Low, Mat Hold, Rickshaw Man, Rise/Fall Three Methods, Separating Lines, Stalled Pattern, Stick Sandwich, Takuri, Tasuki Gap, Tristar, Unique Three River, Upside Gap Two Crows, and X-Side Gap Three Methods.

## Fundamental period operators

All period-count parameters are explicit reporting periods, not trading days.

```text
fin_lag(x, period_id, periods)
fin_diff(x, period_id, periods)
fin_pct_change(x, period_id, periods)
fin_log_change(x, period_id, periods)
fin_qoq(x, period_id)
fin_yoy(x, period_id, periods_per_year)
fin_ttm(x, period_id, periods_per_year)
fin_average_balance(x, period_id, periods)

fin_growth(x, period_id, periods)
fin_cagr(x, period_id, periods, periods_per_year)
fin_growth_acceleration(x, period_id, short_periods, long_periods)
fin_growth_change(x, period_id, growth_periods, compare_periods)
fin_growth_volatility(x, period_id, growth_periods, window_periods)
fin_growth_stability(...)
fin_growth_persistence(...)

fin_std(x, period_id, window_periods)
fin_mad(...)
fin_cv(...)
fin_stability(...)
fin_range(...)
fin_zscore_history(...)
fin_percentile_history(...)
fin_trend_slope(...)
fin_trend_r2(...)
fin_trend_tstat(...)
fin_trend_acceleration(...)
fin_monotonicity(...)
fin_positive_streak(x, period_id, max_periods)
fin_negative_streak(...)
fin_sign_change_count(...)

fin_ratio(numerator, denominator)
fin_common_size(x, base)
fin_turnover(flow, balance, period_id, average_periods)
fin_divergence(x, y, period_id, periods)
fin_cash_earnings_gap(earnings, cashflow, scale_base)
fin_accrual_ratio(earnings, cashflow, assets)
fin_cash_conversion(cashflow, earnings)
fin_working_capital_change(working_capital, period_id, periods)
```

Domain names such as ROE, current ratio, operating margin, accruals, asset growth, inventory turnover, etc. are recipes over these field-agnostic primitives.

## Filing revision / staleness

```text
fin_revision_delta(x, period_id)
fin_revision_pct(x, period_id)
fin_revision_direction(x, period_id)
fin_revision_count(x, period_id, window_days)
fin_revision_magnitude(x, period_id, window_days)
fin_restated_flag(x, period_id, window_days)
fin_days_since_update(x, period_id, max_days)
fin_staleness(x, period_id, max_days)
```

A revision means the visible value changes while `period_id` remains the same. A newly available report period is not counted as a revision.

## Analyst expectations

```text
fin_surprise(actual, expected, scale_base)
fin_surprise_zscore(actual, expected, scale_base, window_days)
fin_expectation_revision(expected, target_period_id)
fin_expectation_revision_pct(expected, target_period_id)
fin_expectation_revision_speed(expected, target_period_id, window_days)
fin_expectation_dispersion(expected_std, expected_mean)
fin_actual_expectation_divergence(actual, expected, scale_base)
fin_beat_streak(actual, expected, period_id, max_periods)
fin_miss_streak(actual, expected, period_id, max_periods)
```

## Minute→daily operators

Every helper returns one daily value per instrument through a logical SourceRef. All support `bar_minutes`, `cutoff_time`, and `min_coverage`; subset-specific arguments such as `minutes`, `lag`, `q`, `threshold`, or `history_days` are explicit.

Families:

- realized risk: variance, vol, skew, kurtosis, quarticity, upside/downside semivariance, bipower variation, jump variation/ratio/count;
- path shape: first/last-N-minute returns, morning/afternoon reversal, trend slope/R², efficiency, autocorrelation, max drawdown/runup, time of high/low;
- opening/closing: opening range/drive, gap continuation/fill, closing return/ramp;
- VWAP: VWAP, close/high/low-to-VWAP, VWAP slope, deviation mean/std, cross count;
- volume profile: first/last share, peak time, HHI, entropy, profile skew/slope;
- intraday price-volume: return-volume correlation, signed imbalance, volume-weighted return, price impact, Amihud, turnover/volatility;
- time-of-day abnormality: profile z-score/deviation and abnormal volume/return/volatility profiles.

For non-A-share minute datasets, `minute_dataset`, `session_open`, `session_close`, and `session_minutes` must be explicitly configured. A US strategy must never silently fall back to the A-share minute mirror.

## Not implemented without a stronger data-source contract

The following are intentionally not faked from OHLCV/minute bars:

```text
quoted_spread
effective_spread
realized_spread
order_flow_imbalance
depth_imbalance
Kyle_lambda
VPIN_like
quote_update_intensity
```

They require real trade/quote/order-book schemas and timestamp semantics.

## Physical refactor policy

See `docs/operator_refactor_v2.json`.

The key distinction is:

- **public name retained + macro/recipe lowered**: cheap deterministic composites;
- **native kernel retained**: pivot geometry, stateful recursion, reporting-period walker, minute aggregation, adaptive candlestick engine;
- **legacy/unbounded implementation deprecated**: start-date-dependent cumulative indicators and retroactive pivot labeling;
- **reference implementation retained**: may exist for parity/evidence even when the normal DSL path lowers to primitives.
