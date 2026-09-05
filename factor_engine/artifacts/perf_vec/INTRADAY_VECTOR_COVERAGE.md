# Intraday Vector Coverage

Vectorized `daily_agg` kernel routing snapshot (GO_PROMPT §6.1 / §6.3).
Generated head SHA: `a18156566bb5c56504e4a6794dc1361346bf1877`
Bound WHITELIST ids: `29` (count_bound == len(bind_whitelist()) == 29)

Regenerate: `python3 scripts/generate_intraday_vector_coverage.py`

| canonical | vector_bound | scalar_impl | vector_impl | fallback_reason |
|---|---|---|---|---|
| intra_common_trading_intensity | True | smart_money:proven-scalar-kernel | _vec_common_trading_intensity | — |
| intra_continuous_variance | True | higher_moments:proven-scalar-kernel | _vec_param(_vec_bipower_and_jump) | — |
| intra_dynamic_stock_graph_features | True | smart_money:proven-scalar-kernel | _vec_stock_graph_features | — |
| intra_jump_variation | True | higher_moments:proven-scalar-kernel | _vec_param(_vec_bipower_and_jump) | — |
| intra_price_delay | True | true_gap_batch3:proven-scalar-kernel | _vec_price_delay | — |
| intra_realized_kurtosis | True | higher_moments:proven-scalar-kernel | _vec_param(_vec_realized_moments) | — |
| intra_realized_quarticity | True | higher_moments:proven-scalar-kernel | _vec_param(_vec_realized_moments) | — |
| intra_realized_skewness | True | higher_moments:proven-scalar-kernel | _vec_param(_vec_realized_moments) | — |
| intra_session_mean_reversion | True | true_gap_batch3:proven-scalar-kernel | _vec_session_mean_reversion | — |
| intra_time_above_vwap | True | vwap_path:proven-scalar-kernel | _vec_time_above_vwap | — |
| intra_tripower_quarticity | True | higher_moments:proven-scalar-kernel | _vec_tripower_quarticity | — |
| intra_ts_amount_weighted_mean | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_amount_weighted_mean | — |
| intra_ts_argmax | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_argmax | — |
| intra_ts_argmin | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_argmin | — |
| intra_ts_first | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_first | — |
| intra_ts_last | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_last | — |
| intra_ts_last_value | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_last | — |
| intra_ts_max | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_max | — |
| intra_ts_mean | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_mean | — |
| intra_ts_min | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_min | — |
| intra_ts_realized_covariance | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_realized_covariance | — |
| intra_ts_realized_variance | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_realized_variance | — |
| intra_ts_std | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_std | — |
| intra_ts_sum | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_sum | — |
| intra_ts_variance | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_variance | — |
| intra_ts_volume_weighted_return | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_volume_weighted_return | — |
| intra_ts_vwap | True | sufficient_stats_ops:proven-scalar-kernel | _vec_ts_vwap | — |
| intra_volume_imbalance | True | true_gap_batch3:proven-scalar-kernel | _vec_volume_imbalance | — |
| intra_vwap_reversion_speed | True | vwap_path:proven-scalar-kernel | _vec_vwap_reversion_speed | — |
| intra_covariance_manifold_shift | False | topology_manifold:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_critical_transition_score | False | topology_manifold:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_dmd_koopman_features | False | topology_manifold:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_drawdown_depth | False | vwap_path:make_drawdown_metric("depth") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_drawdown_duration | False | vwap_path:make_drawdown_metric("duration") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_drawdown_recovery_half_life | False | vwap_path:make_drawdown_metric("recovery") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_functional_motif_score | False | state_space:lambda p, v, _: _kernel(p, v, None) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_interval_amount_share | False | time_structure:lambda v, t: _interval_share(v, t, s, e) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_interval_realized_variance | False | time_structure:lambda v, t: _interval_rv(v, t, s, e) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_interval_return | False | time_structure:lambda v, t: _interval_return(v, t, s, e) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_interval_volume_share | False | time_structure:lambda v, t: _interval_share(v, t, s, e) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_jump_clustering | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_jump_concentration | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_jump_count | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_jump_first_time | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_jump_last_time | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_kalman_latent_price | False | state_space:_kernel | — | kernel not vectorized: raw kernel passed but not PERF-2-whitelisted (needs harness equivalence before binding) |
| intra_local_conditional_entropy | False | smart_money:lambda c, v: _local_conditional_entropy(c, v) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_longest_above_vwap_streak | False | vwap_path:make_longest_streak("above") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_longest_below_vwap_streak | False | vwap_path:make_longest_streak("below") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_matrix_profile_session_features | False | topology_manifold:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_max_drawdown | False | vwap_path:make_max_drawdown("down") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_max_drawup | False | vwap_path:make_max_drawdown("up") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_negative_jump_variation | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_positive_jump_variation | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_price_peak_ridge_valley_state | False | pattern_recognition:_kernel | — | kernel not vectorized: raw kernel passed but not PERF-2-whitelisted (needs harness equivalence before binding) |
| intra_price_vwap_max_negative_excursion | False | vwap_path:make_vwap_excursion("min") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_price_vwap_max_positive_excursion | False | vwap_path:make_vwap_excursion("max") | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_signed_jump_ratio | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_signed_tail_variation_ratio | False | higher_moments:_fn | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_smart_money_vwap_ratio | False | smart_money:lambda c, a, v: _smart_money_vwap_ratio(c, a, v) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intra_state_space_volume_components | False | state_space:_kernel | — | kernel not vectorized: raw kernel passed but not PERF-2-whitelisted (needs harness equivalence before binding) |
| intra_visibility_graph_features | False | state_space:_kernel | — | kernel not vectorized: raw kernel passed but not PERF-2-whitelisted (needs harness equivalence before binding) |
| intra_volume_peak_ridge_valley_state | False | pattern_recognition:_kernel | — | kernel not vectorized: raw kernel passed but not PERF-2-whitelisted (needs harness equivalence before binding) |
| intra_vwap_path_curvature | False | vwap_path:make_vwap_path(2, 2, False) | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_vwap_path_curvature_pct | False | vwap_path:make_vwap_path(2, 2, True) | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_vwap_path_slope | False | vwap_path:make_vwap_path(1, 1, False) | — | kernel not vectorized: per-(inst,day) Python loop |
| intra_vwap_path_slope_pct | False | vwap_path:make_vwap_path(1, 1, True) | — | kernel not vectorized: per-(inst,day) Python loop |
| intraday_rv_signature_slope | False | higher_moments:lambda v, t: _rv_signature_slope(v) | — | kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure carries no __vec__ |
| intraday_value_at_extreme_state | False | intra_state_space:_kernel | — | kernel not vectorized: raw kernel passed but not PERF-2-whitelisted (needs harness equivalence before binding) |

## Summary — total=69 · vectorized=29 · scalar_only=40

> Scalar-only operators are unvectorized because a fresh `lambda` / module-level `_fn` closure carries no `__vec__`; only the PERF-2 harness-proven raw kernel call convention routes to the vectorized fast path.