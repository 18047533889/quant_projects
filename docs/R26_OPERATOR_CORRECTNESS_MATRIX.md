# R26 Operator Correctness Matrix

| canonical | family | tie | missing | clock | state | numeric | default | unit | econ | fix |
|---|---|---|---|---|---|---|---|---|---|---|
| ts_chatterjee_xi | dependence | target_independent | ok | n/a | ok | ok | PASS | ok | ok | y-independent stable tie ordering (R26-005..007) |
| cs_rank_copula_mi | information | ok | ok | n/a | ok | ok | PASS | ok | ok | Jeffreys-only estimator, Miller-Madow removed (R26-009..011) |
| intra_realized_variance | intraday | ok | explicit_slots | official_grid | ok | ok | PASS | ok | ok | SessionPanel grid alignment + gated log returns (R26-013..020) |
| intra_path_efficiency | intraday | ok | break_on_gap | official_grid | ok | ok | PASS | ok | ok | contiguous complete run, no gap bridging (R26-028/029) |
| intra_high_time | intraday | latest_extreme | explicit_slots | slot_ordinal | ok | ok | PASS | ok | ok | official slot ordinal, not observed row index (R26-030) |
| intra_lunch_gap_return | intraday | ok | exact_endpoints | official_grid | ok | ok | PASS | ok | ok | EndpointPolicy.EXACT 11:30/13:01 (R26-027) |
| intra_limit_first_hit_time | intraday | ok | tri_state | slot_ordinal | ok | ok | PASS | ok | ok | per-side OHLC + slot ordinal + tri-state limit (R26-031..034) |
| session_event_recovery_score | intraday | ok | break_censor | official_grid | ok | ok | PASS | ok | ok | min_events=3 + EventMissingPolicy + session-close proof (R26-047..051) |
| ts_hill_tail_index | tail | ok | ok | n/a | ok | ok | PASS | ok | ok | positive-level lower tail unsupported; loss-magnitude Hill (R26-060..062) |
| ts_roll_effective_spread | liquidity | ok | ok | n/a | ok | ok | PASS | ok | ok | positive covariance -> NaN, positive-price gate (R26-069..071) |
| ts_glr_mean_shift_score | change_point | ok | ok | n/a | ok | recentered | PASS | ok | ok | recentered prefix moments (R26-065..068) |
| AROON_up | technical | latest_extreme | ok | n/a | ok | ok | PASS | ok | ok | latest-extreme tie policy (R26-076..078) |
| ts_last_pivot_high | technical | ok | ok | n/a | bounded_lookback | ok | PASS | ok | ok | pivot_lookback_bars bounds the state machine (R26-079..081) |
| cdl_spinning_top | candle | ok | tri_state | n/a | ok | ok | PASS | ok | ok | neutral-direction event -> NaN, never +1 (R26-085/086) |
| event_interval_memory | event_interval | ok | break | bar_window | ok | ok | PASS | ok | ok | BAR window semantics + doc==kernel min-support (R26-088..091) |
| ts_multiscale_permutation_entropy_slope | state_geometry | ok | ok | n/a | ok | ok | PASS | ok | ok | feasible default window=256 (R26-093..095) |
| composition_entropy | composition | ok | ok | n/a | ok | ok | PASS | ok | part_whole_only | invalid financial_statement schema removed (R26-096..102) |
| ts_effective_transfer_entropy | information | ok | fixed_mask | n/a | ok | ok | PASS | ok | ok | fixed NaN mask, finite-only surrogate rearrangement (R26-103/104) |
| ts_quantile_crossing_spectral_concentration | quantile | ok | contiguous_run_checked | n/a | ok | ok | PASS | ok | ok | post-truncation length re-check (R26-108/109) |
| ts_extremogram | quantile | ok | ok | n/a | ok | ok | PASS | signed_probability_difference | ok | signed probability difference unit (R26-110) |
| ts_dc_event_rate | directional_change | ok | clock_observable_denom | n/a | ok | ok | PASS | scale_mode_split | ok | scale_mode absolute/relative + clock-observable denominator (R26-113..117) |
| ts_dmd_energy_concentration | dmd | ok | ok | n/a | ok | log_domain | PASS | ok | ok | log-domain rho/energy, fail-closed on all-zero (R26-122..125) |
| intraday_impact_decay_rate | intraday | ok | ok | n/a | ok | censor_floor | PASS | ok | ok | detection-floor censor keeps recovery evidence (R26-057..059) |
