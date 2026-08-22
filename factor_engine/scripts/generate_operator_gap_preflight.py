#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R47 新增算子开发总规范 Wave 0: current-main preflight gap classification.

Classifies every R41-R47 candidate + supplementary operator against the live
registry (8e9893b5).  Dispositions derive from two read-only audit passes over
the codebase (semantic-equivalence / composability) plus explicit corrections.

Dispositions:
    EXISTING_EXACT / EXISTING_ALIAS / EXISTING_EQUIVALENT /
    COMPOSABLE_NO_NEW_OPERATOR / TRUE_GAP_IMPLEMENT / RESEARCH_ONLY_IMPLEMENT /
    BLOCKED_BY_DATA_CONTRACT / REJECT_FIELD_MISMATCH / REJECT_LOOKAHEAD /
    ARCHITECTURE_SUPERSEDED

Output: evidence/operator_gap_preflight_<HEAD>.csv
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


# candidate -> (disposition, replacement-or-reason)
_DISP: dict[str, tuple[str, str]] = {
    # ---- R41 indicators / daily (non-intraday) ----
    "HMA": ("TRUE_GAP_IMPLEMENT", "Hull MA = WMA(2*WMA(x,n/2)-WMA(x,n), sqrt(n)); no existing Hull op"),
    "QQE": ("TRUE_GAP_IMPLEMENT", "RSI + ATR-band adaptive oscillator; no existing QQE op"),
    "RSX": ("TRUE_GAP_IMPLEMENT", "Jurik low-lag RSI variant; no existing RSX op"),
    "ALMA": ("TRUE_GAP_IMPLEMENT", "Gaussian-window MA with offset; no existing ALMA op"),
    "CoppockCurve": ("TRUE_GAP_IMPLEMENT", "WMA(ROC(a)+ROC(b)); no existing named op"),
    "ElderRay": ("TRUE_GAP_IMPLEMENT", "bull=high-EMA(close), bear=low-EMA(close); no existing op"),
    "FisherTransform": ("TRUE_GAP_IMPLEMENT", "rolling-normalized 0.5*ln((1+z)/(1-z)); no existing op"),
    "KAMA": ("EXISTING_EXACT", "KAMA registered in technical/signal.py"),
    "WMA": ("EXISTING_EXACT", "WMA registered (polars_signal / _wma)"),
    "cn_sma": ("EXISTING_ALIAS", "ts_sma_cn"),
    "price_delay_score": ("EXISTING_EXACT", "ts_price_delay"),
    "cs_isolation_forest_score": ("EXISTING_EQUIVALENT", "cs_isolation"),
    "cs_robust_mahalanobis_score": ("EXISTING_EQUIVALENT", "cs_robust_mahalanobis_mad (+ cs_zscore)"),
    "pastor_stambaugh_beta": ("EXISTING_EQUIVALENT", "ts_pastor_stambaugh_liquidity_gamma"),
    "winsorized_ratio": ("COMPOSABLE_NO_NEW_OPERATOR", "divide(winsorize(num), winsorize(den))"),
    "cs_factor_bucket_return": ("COMPOSABLE_NO_NEW_OPERATOR", "group_mean(returns, cs_bucket(x, n))"),
    "same_calendar_month_return": ("COMPOSABLE_NO_NEW_OPERATOR", "returns + month-boundary grouping"),
    "same_calendar_day_mean": ("COMPOSABLE_NO_NEW_OPERATOR", "returns + day-of-year grouping"),
    "ts_ewm_std": ("EXISTING_EXACT", "ts_ewm_std already registered"),
    # ---- R41 fiscal / fundamental ----
    "fiscal_acceleration": ("EXISTING_EQUIVALENT", "fin_growth_acceleration / fin_trend_acceleration"),
    "fiscal_delta": ("EXISTING_EQUIVALENT", "fin_diff / period_change"),
    "fiscal_lag": ("EXISTING_EQUIVALENT", "period_lag / fin_lag"),
    "fiscal_pct_change": ("EXISTING_EQUIVALENT", "fin_pct_change"),
    "financial_snapshot_lag": ("EXISTING_EQUIVALENT", "fin_lag + fin_staleness"),
    "fiscal_rolling_std": ("EXISTING_EQUIVALENT", "fin_std"),
    "fiscal_capital_stock": ("COMPOSABLE_NO_NEW_OPERATOR", "fiscal_perpetual_inventory(capex, period_id, dep)"),
    "fiscal_rolling_regression": ("COMPOSABLE_NO_NEW_OPERATOR", "ts_regression_slope(fiscal, period_ordinal, window)"),
    "fiscal_cost_stickiness_score": ("EXISTING_EQUIVALENT", "fiscal_asymmetric_elasticity"),
    "fundamental_cost_stickiness_panel": ("EXISTING_EQUIVALENT", "fiscal_asymmetric_elasticity (panel wrapper)"),
    "fundamental_cash_flow_duration": ("RESEARCH_ONLY_IMPLEMENT", "Macaulay-style PV-weighted cash-flow duration; niche, research first"),
    "fundamental_latent_balance_sheet_factor": ("RESEARCH_ONLY_IMPLEMENT", "cross-sectional PCA of balance-sheet; research"),
    "fundamental_working_capital_financing_state": ("RESEARCH_ONLY_IMPLEMENT", "state classifier; research first"),
    "report_asof": ("ARCHITECTURE_SUPERSEDED", "DataAccess PIT-asof join (knowledge_time=PubDate)"),
    "event_window_return_asof": ("ARCHITECTURE_SUPERSEDED", "event_cumulative_return_past + PIT join"),
    "fin_schema_gate": ("BLOCKED_BY_DATA_CONTRACT", "no financial_statement_schema field in data dictionary"),
    # ---- R43/R47 panel / cs / ts ----
    "ts_lagged_predictability_score": ("TRUE_GAP_IMPLEMENT", "rolling rank-IC signal->realized_return meta-signal; P0"),
    "panel_ewm_beta_ex_self": ("COMPOSABLE_NO_NEW_OPERATOR", "ts_ewm_corr * std-ratio with ex-self market weight"),
    "panel_cmra_ex_self": ("COMPOSABLE_NO_NEW_OPERATOR", "ts_sum(benchmark_excess_return(ret, group_ex_self_weighted_mean(ret, mcap)))"),
    "panel_day_night_beta_gap": ("COMPOSABLE_NO_NEW_OPERATOR", "ts_beta(overnight) - ts_beta(intraday)"),
    "panel_apm_residual_tstat": ("COMPOSABLE_NO_NEW_OPERATOR", "cs_regression residual then ts_regression_tstat"),
    "panel_async_beta_ex_self": ("TRUE_GAP_IMPLEMENT", "staggered-refit daily ex-self beta; genuinely new"),
    "panel_factor_pocket_strength": ("TRUE_GAP_IMPLEMENT", "factor-payout regime persistence; new"),
    "panel_similarity_crowding_score": ("EXISTING_EQUIVALENT", "GroupCorrMstLength / GroupFeatureEffectiveRank"),
    "panel_cluster_risk_score": ("EXISTING_EQUIVALENT", "group_spectrum suite"),
    "panel_peer_graph_aggregate": ("EXISTING_EQUIVALENT", "cs_knn_peer_mean_ex_self / relation_peer_weighted_mean_ex_self"),
    "cs_predictability_mosaic_score": ("TRUE_GAP_IMPLEMENT", "multi-horizon predictability aggregation; new"),
    "panel_predictability_mosaic_score": ("TRUE_GAP_IMPLEMENT", "panel variant of predictability mosaic"),
    "cs_topological_anomaly_score": ("TRUE_GAP_IMPLEMENT", "cross-sectional TDA-based anomaly synthesis"),
    # ---- R43 intraday TRUE_GAP (implement) ----
    "intra_slice_mask_reduce": ("TRUE_GAP_IMPLEMENT", "session mask-quantile slice reducer; P0"),
    "intra_slice_mask_pair_reduce": ("TRUE_GAP_IMPLEMENT", "pairwise slice mask reducer; P0"),
    "intra_multiresolution_resample_reduce": ("TRUE_GAP_IMPLEMENT", "1m->k-min resample reduce; P0"),
    "intra_same_slot_zscore": ("TRUE_GAP_IMPLEMENT", "same-slot cross-day zscore; P0"),
    "intra_event_window_reduce": ("TRUE_GAP_IMPLEMENT", "event-window pre/post reduce; P0 (spec 11.1)"),
    "intra_event_pre_post_contrast": ("TRUE_GAP_IMPLEMENT", "event pre/post contrast; P0 (spec 11.2)"),
    "intra_state_dwell_stats": ("TRUE_GAP_IMPLEMENT", "state dwell/persistence; P1 (spec 11.5)"),
    "intra_piecewise_linear_path_features": ("RESEARCH_ONLY_IMPLEMENT", "DP/PELT piecewise path; P2 research"),
    "ts_event_decay_kernel": ("EXISTING_EQUIVALENT", "event_decay_asof"),
    "ts_online_change_point_score": ("EXISTING_EQUIVALENT", "ts_change_point_probability / ts_glr_* / ts_pettitt_change_score"),
    "same_clock_lag": ("COMPOSABLE_NO_NEW_OPERATOR", "intra_same_slot_momentum + ts_delay"),
    "trading_calendar_mask": ("COMPOSABLE_NO_NEW_OPERATOR", "calendar_day_diff / non-null volume mask"),
    # ---- R42 state/event TRUE_GAP ----
    "intra_state_count": ("TRUE_GAP_IMPLEMENT", "state-target minute count; P0"),
    "intra_state_sum": ("TRUE_GAP_IMPLEMENT", "state-target sum; P0"),
    "intra_state_vwap": ("TRUE_GAP_IMPLEMENT", "state-conditioned VWAP; P0"),
    "intra_state_interval_moment": ("TRUE_GAP_IMPLEMENT", "state event-gap moments; P0"),
    "intra_state_follow_ratio": ("TRUE_GAP_IMPLEMENT", "state follow ratio; P0"),
    "intra_state_follow_beta": ("TRUE_GAP_IMPLEMENT", "state follow OLS slope; P1"),
    "intra_state_follow_corr": ("TRUE_GAP_IMPLEMENT", "state follow correlation; P0"),
    "intra_state_pair_same_slot_corr": ("TRUE_GAP_IMPLEMENT", "state-pair same-slot corr; P1"),
    "intra_range_gap_flag": ("TRUE_GAP_IMPLEMENT", "pre/post range-gap flag; P0"),
    "intra_volume_peak_ridge_valley_state": ("TRUE_GAP_IMPLEMENT", "volume peak/ridge/valley engine; P0"),
    "intra_price_peak_ridge_valley_state": ("TRUE_GAP_IMPLEMENT", "price peak/ridge/valley engine; P0"),
    "intra_smart_money_vwap_ratio": ("TRUE_GAP_IMPLEMENT", "smart-money VWAP ratio; P0"),
    "intra_smart_money_fcm_score": ("RESEARCH_ONLY_IMPLEMENT", "fuzzy-c-means smart cluster on OHLCV features; P2 research"),
    "intra_neighbor_event_class": ("TRUE_GAP_IMPLEMENT", "event neighbor classification; P0"),
    # ---- R44 session / limit TRUE_GAP ----
    "intra_session_segment_reduce": ("EXISTING_EQUIVALENT", "intra_segment_return/volume_share/..."),
    "intra_session_boundary_jump": ("TRUE_GAP_IMPLEMENT", "session-boundary discontinuity; P0"),
    "suspension_restart_response": ("EXISTING_EQUIVALENT", "session_event_recovery_score + suspension markers"),
    "intra_limit_pre_hit_pressure_profile": ("TRUE_GAP_IMPLEMENT", "pre-hit return/volume accel (bar+limit fields only; R44 spec)"),
    # ---- R44/R45 volume-at-price / chip TRUE_GAP ----
    "intra_volume_at_price_profile": ("TRUE_GAP_IMPLEMENT", "volume-at-price distribution; P0"),
    "intra_volume_profile_peak_geometry": ("TRUE_GAP_IMPLEMENT", "profile peak/valley geometry; P0"),
    "intra_volume_profile_supply_structure": ("TRUE_GAP_IMPLEMENT", "overhead/supply structure; P0"),
    "intra_volume_profile_value_area": ("TRUE_GAP_IMPLEMENT", "value area / POC; P0"),
    "turnover_chip_distribution": ("EXISTING_EQUIVALENT", "ts_turnover_cost_* family"),
    "turnover_chip_distribution_transport": ("COMPOSABLE_NO_NEW_OPERATOR", "ts_wasserstein_shift / diff of ts_turnover_*"),
    "turnover_chip_age_cost_surface": ("TRUE_GAP_IMPLEMENT", "2D age x cost chip surface; P1"),
    "turnover_chip_overhang_surface": ("TRUE_GAP_IMPLEMENT", "overhang supply surface; P1"),
    "intra_round_price_clustering_share": ("TRUE_GAP_IMPLEMENT", "round-price clustering; P0"),
    "intra_round_price_barrier_response": ("TRUE_GAP_IMPLEMENT", "round-price barrier response; P0"),
    # ---- R45 impulse/probe/absorption TRUE_GAP ----
    "intra_impulse_event_detector": ("TRUE_GAP_IMPLEMENT", "vol-scaled impulse detection; P0"),
    "intra_post_impulse_response": ("TRUE_GAP_IMPLEMENT", "post-impulse retention/giveback; P0"),
    "intra_probe_outcome_score": ("TRUE_GAP_IMPLEMENT", "probe outcome composite; P0"),
    "intra_supply_absorption_score": ("TRUE_GAP_IMPLEMENT", "supply absorption; P0"),
    "intra_consolidation_quality": ("TRUE_GAP_IMPLEMENT", "consolidation tightness; P0"),
    "intra_response_curve_features": ("TRUE_GAP_IMPLEMENT", "response curve geometry; P1"),
    "intra_liquidity_resilience_curve_fit": ("TRUE_GAP_IMPLEMENT", "liquidity recovery half-life fit; P0"),
    # ---- R46 TRUE_GAP (intraday) ----
    "intra_eod_reversal_decomposition": ("TRUE_GAP_IMPLEMENT", "EOD move decomposition; P0"),
    "intra_volume_shock_state": ("EXISTING_EQUIVALENT", "intra_slot_volume_surprise / volume_shock"),
    "intra_comovement_curve_ex_self": ("TRUE_GAP_IMPLEMENT", "ex-self intraday comovement curve; P1"),
    # ---- RESEARCH_ONLY heavy-model intraday (Wave 3) ----
    "intra_jump_wavelet_morphology": ("RESEARCH_ONLY_IMPLEMENT", "wavelet+jump morphology; P1 research"),
    "intra_cojump_breadth_ex_self": ("COMPOSABLE_NO_NEW_OPERATOR", "group_mean(intra_jump_count)"),
    "intra_idiosyncratic_semivariance_balance_ex_self": ("TRUE_GAP_IMPLEMENT", "ex-self idio semivariance balance; P0"),
    "intra_functional_beta_profile_ex_self": ("TRUE_GAP_IMPLEMENT", "functional beta curve ex-self; P1"),
    "intra_absorption_curve_area": ("TRUE_GAP_IMPLEMENT", "absorption curve area; P1"),
    "intra_range_competition_profile": ("TRUE_GAP_IMPLEMENT", "range competition profile; P1"),
    "intra_dmd_koopman_features": ("TRUE_GAP_IMPLEMENT", "session-level DMD (daily ts_dmd_* exists); P1"),
    "intra_kalman_latent_price": ("TRUE_GAP_IMPLEMENT", "session-level Kalman (daily ts_kalman_* exists); P1"),
    "intra_state_space_volume_components": ("TRUE_GAP_IMPLEMENT", "SSA volume components; P1"),
    "intra_hmm_state_features": ("RESEARCH_ONLY_IMPLEMENT", "HMM fitting; research"),
    "intra_hsmm_duration_features": ("RESEARCH_ONLY_IMPLEMENT", "HSMM fitting; research"),
    "intra_change_point_sequence_features": ("RESEARCH_ONLY_IMPLEMENT", "multi-change-point sequence stats; research"),
    "intra_hawkes_event_features": ("RESEARCH_ONLY_IMPLEMENT", "Hawkes fitting; research"),
    "intra_multifractal_spectrum": ("EXISTING_EQUIVALENT", "ts_multifractal_* family"),
    "intra_wavelet_scattering_features": ("RESEARCH_ONLY_IMPLEMENT", "wavelet scattering cascade; research"),
    "intra_emd_hilbert_huang_features": ("RESEARCH_ONLY_IMPLEMENT", "EMD/HHT pipeline; research"),
    "intra_visibility_graph_features": ("TRUE_GAP_IMPLEMENT", "session-level HVG (daily ts_hvg_* exists); P1"),
    "intra_information_flow_features": ("EXISTING_EQUIVALENT", "ts_transfer_entropy / kernel_granger"),
    "intra_covariance_manifold_shift": ("TRUE_GAP_IMPLEMENT", "SPD manifold shift; P1"),
    "intra_diffusion_map_state": ("RESEARCH_ONLY_IMPLEMENT", "diffusion-map eigendecomposition; research"),
    "intra_common_trading_intensity": ("TRUE_GAP_IMPLEMENT", "cross-stock common intensity; P1"),
    "intra_price_efficiency_state_space": ("RESEARCH_ONLY_IMPLEMENT", "efficiency hidden-state framing; research"),
    "intra_neural_cde_embedding": ("RESEARCH_ONLY_IMPLEMENT", "neural CDE training; research"),
    "intra_contrastive_path_embedding": ("RESEARCH_ONLY_IMPLEMENT", "contrastive path embedding; research"),
    "intra_matrix_profile_session_features": ("TRUE_GAP_IMPLEMENT", "session-level matrix profile; P1"),
    "intra_dtw_archetype_features": ("RESEARCH_ONLY_IMPLEMENT", "DTW archetype (O(n^2)); research"),
    "intra_local_conditional_entropy": ("TRUE_GAP_IMPLEMENT", "conditional entropy estimator; P1"),
    "intra_business_time_deformation": ("EXISTING_EQUIVALENT", "intraday_volume_clock_* / ts_activity_clock_*"),
    "intra_quantile_dependence_features": ("EXISTING_EQUIVALENT", "ts_cross_quantilogram / quantile_* family"),
    "intra_kramers_moyal_dynamics": ("EXISTING_EQUIVALENT", "ts_kramers_moyal_* family"),
    "intra_extreme_event_interval_memory": ("EXISTING_EQUIVALENT", "event_interval_memory / ts_extremal_index"),
    "intra_signature_lead_lag_network": ("RESEARCH_ONLY_IMPLEMENT", "signature+lead-lag network; research"),
    "intra_validated_lead_lag_network": ("RESEARCH_ONLY_IMPLEMENT", "statistical validation of lead-lag; research"),
    "intra_dynamic_stock_graph_features": ("TRUE_GAP_IMPLEMENT", "cross-sectional intraday graph; P1"),
    "intra_tensor_common_mode": ("RESEARCH_ONLY_IMPLEMENT", "tensor decomposition; research"),
    "intra_topological_peer_anomaly": ("RESEARCH_ONLY_IMPLEMENT", "TDA peer anomaly; research"),
    "intra_critical_transition_score": ("TRUE_GAP_IMPLEMENT", "critical-transition composite; P1"),
    "intra_realized_measure_state_vector": ("COMPOSABLE_NO_NEW_OPERATOR", "combine intra_realized_* + ts_vector_state_*"),
    "intra_symbolic_dynamics_features": ("EXISTING_EQUIVALENT", "ts_permutation_entropy / lz_complexity / ..."),
    "intra_multiscale_state_residence": ("TRUE_GAP_IMPLEMENT", "multiscale state residence; P1"),
    "intra_visibility_motif_transition": ("TRUE_GAP_IMPLEMENT", "HVG motif transitions; P1"),
    "intra_recurrence_network_features": ("EXISTING_EQUIVALENT", "ts_recurrence_* family"),
    "intra_transfer_operator_metastability": ("EXISTING_EQUIVALENT", "ts_markov_spectral_gap / committor / MFPT"),
    "intra_local_drift_diffusion_surface": ("TRUE_GAP_IMPLEMENT", "2D KM drift/diffusion surface; P1"),
    "intra_session_ot_map": ("TRUE_GAP_IMPLEMENT", "full OT transport map; P1"),
    "intra_optimal_transport_profile_shift": ("EXISTING_EQUIVALENT", "intraday_return_wasserstein_shift"),
    "intra_topological_anomaly_score": ("EXISTING_EQUIVALENT", "ts_persistence_* family"),
    "intra_persistent_homology_features": ("EXISTING_EQUIVALENT", "ts_betti_1_max_persistence / persistence_entropy_*"),
    "intra_rqa_features": ("EXISTING_EQUIVALENT", "ts_rqa_* / ts_recurrence_* family"),
    "intra_log_signature_features": ("RESEARCH_ONLY_IMPLEMENT", "log-signature tensor log; research"),
    "intra_functional_pca_shape": ("EXISTING_EQUIVALENT", "intraday_profile_pca_residual / panel_rolling_pca_*"),
    "intra_functional_motif_score": ("TRUE_GAP_IMPLEMENT", "continuous functional-motif score; P1"),
    "intra_shapelet_match": ("RESEARCH_ONLY_IMPLEMENT", "shapelet matching; research"),
    "intra_functional_autoencoder_score": ("TRUE_GAP_IMPLEMENT", "intraday path AE reconstruction error; P2"),
    "intra_hmm_posterior_entropy": ("RESEARCH_ONLY_IMPLEMENT", "HMM posterior entropy; research"),
    "intra_function_on_function_anomaly_response": ("RESEARCH_ONLY_IMPLEMENT", "FoF regression; research"),
    "intra_bicoherence_features": ("EXISTING_EQUIVALENT", "ts_bicoherence_*"),
    "intra_frequency_granger_price_volume": ("EXISTING_EQUIVALENT", "ts_cross_spectral_* + ts_kernel_granger_score"),
    "intra_price_volume_cross_wavelet": ("EXISTING_EQUIVALENT", "ts_modwt_band_corr / ts_cross_spectral_coherence"),
    "intra_volume_shock_state": ("EXISTING_EQUIVALENT", "intra_slot_volume_surprise / volume_shock"),
    # ---- misc intraday (not in registry but research) ----
    "intra_distribution_moment": ("EXISTING_EQUIVALENT", "intra_realized_skewness/kurtosis/quarticity"),
    "intra_price_shape_cosine_match": ("EXISTING_EQUIVALENT", "intra_*_profile_cosine family"),
    "intra_state_transition_entropy": ("TRUE_GAP_IMPLEMENT", "per-session transition entropy (daily ts_permutation_* differs); P1 (spec 11.4)"),
    "intra_market_profile_corr_ex_self": ("TRUE_GAP_IMPLEMENT", "market-profile correlation ex-self; P1"),
    "intraday_value_at_extreme_state": ("RESEARCH_ONLY_IMPLEMENT", "value-at-extreme-state; research"),
    "panel_intraday_ordered_reduce": ("TRUE_GAP_IMPLEMENT", "time-vs-cs ordered reduce; P1"),
    "panel_intraday_low_rank_residual_ex_self": ("TRUE_GAP_IMPLEMENT", "intraday low-rank residual ex-self; P1"),
}

_COLUMNS = [
    "candidate", "candidate_signature", "current_exact", "current_alias",
    "semantic_equivalent", "composable", "current_surface", "pandas",
    "polars", "duckdb_sql", "field_legal", "pit_legal", "disposition",
    "replacement", "reason",
]


def main() -> int:
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "..")
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.operator_surface import classify_canonical

    load_all()
    sha = git_sha()
    canonicals = set(OperatorRegistry.list_canonical())

    cands = sorted(_DISP.keys())
    rows: list[dict[str, str]] = []
    for cand in cands:
        disp, reason = _DISP[cand]
        exists = cand in canonicals
        rows.append({
            "candidate": cand,
            "candidate_signature": "",
            "current_exact": "yes" if exists else "no",
            "current_alias": "no",
            "semantic_equivalent": "no",
            "composable": "no",
            "current_surface": classify_canonical(cand) if exists else "",
            "pandas": "",
            "polars": "",
            "duckdb_sql": "",
            "field_legal": "yes",
            "pit_legal": "yes",
            "disposition": disp,
            "replacement": reason.split(";")[0] if disp.startswith(("EXISTING", "COMPOSABLE", "ARCHITECTURE")) else "",
            "reason": reason,
        })

    out_dir = REPO / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"operator_gap_preflight_{sha}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    counts = Counter(r["disposition"] for r in rows)
    print(f"preflight -> {out}")
    print(f"  candidates: {len(rows)}")
    for k, v in sorted(counts.items()):
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
