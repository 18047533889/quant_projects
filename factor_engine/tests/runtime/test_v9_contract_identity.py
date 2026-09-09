"""Implemented semantic changes must reach the actual public identity path."""
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.backend.operator_semantic_version import semantic_version
from factor_engine.runtime.factor_identity import OperatorSemanticContractDigest
from factor_engine.runtime.execution_contract import execution_contract, own_history_requirement


@pytest.mark.parametrize("canonical", [
    "ts_bds_statistic", "event_historical_response_mean",
    "event_historical_response_sign_balance", "event_response_dispersion",
    "event_response_effective_events", "event_response_overlap_ratio",
    "ts_quantile_regression_beta", "ts_quantile_regression_slope",
    "ts_quantile_regression_coeff", "ts_quantile_regression_coeff_prior",
    "ts_quantile_regression_resid", "ts_quantile_beta_spread", "ts_quantile_beta_spread_prior",
    "ts_transfer_entropy", "ts_effective_transfer_entropy",
    "ts_transfer_entropy_peak_strength", "ts_transfer_entropy_peak_lag", "ts_transfer_entropy_peak_excess",
    "ts_distance_corr", "ts_distance_cov",
    "ts_huber_regression_in_sample_resid", "ts_huber_regression_predictive_resid",
    "ts_vector_state_mahalanobis", "ts_butterworth_lowpass_causal",
    "ts_local_lyapunov_exponent",
    "ts_cross_spectral_coherence", "ts_cross_spectral_phase",
    "ts_bicoherence_top_decile_mean", "ts_bicoherence_top_decile_excess",
    "ts_expectile", "ts_expectile_beta", "ts_kernel_granger_score",
    "ts_wavelet_low_frequency_ratio", "ts_wavelet_high_frequency_ratio",
    "ts_wavelet_entropy", "ts_wavelet_energy_slope", "ts_spectral_low_frequency_ratio",
    "ts_hill_tail_index", "ts_matrix_profile_motif_frequency",
    "ts_matrix_profile_neighbor_dispersion", "ts_residualized_hsic",
    "ts_causal_local_linear_smoother", "ts_fir_lowpass_causal",
    "ts_bessel_lowpass_causal", "ts_spectral_lowpass_trailing",
    "ts_causal_savgol_endpoint", "ts_total_variation_filter_trailing",
    "ts_l1_trend_filter_trailing", "cs_huber_resid", "cs_lad_resid",
    "group_peer_beta_deviation", "ts_first_passage_bias",
    "ts_first_passage_hit_probability", "ts_first_passage_conditional_time",
    "group_feature_mode_share", "group_feature_effective_rank",
    "group_feature_mode_localization", "group_feature_spectral_gap",
    "group_feature_second_mode_localization", "event_response_reversal_strength",
    "ts_turnover_reference_price", "ts_turnover_cost_dispersion",
    "ts_turnover_profit_share", "ts_turnover_holding_age",
    "ts_turnover_near_cost_mass", "ts_turnover_cost_quantile_distance",
    "ts_turnover_cost_entropy", "ts_turnover_cost_mode_distance",
    "ts_turnover_cost_skew", "ts_turnover_age_dispersion",
    "ts_turnover_old_mass", "ts_turnover_cost_entropy_vol_scaled",
])
def test_changed_semantics_reach_public_contract_identity(canonical):
    load_all()
    assert semantic_version(canonical) == 2
    digest = OperatorSemanticContractDigest.for_canonical(canonical)
    assert digest.semantic_version == "2.0"
    assert digest.backend_hashes, "changed kernel lacks implementation fingerprint"


def test_bessel_must_not_be_independently_restarted_from_finite_warmup():
    load_all()
    contract = execution_contract("ts_bessel_lowpass_causal")
    assert contract.state_model == "recursive"
    assert contract.chunking == "required_full_history"
    assert own_history_requirement("ts_bessel_lowpass_causal", {"order":8, "cutoff":.001}).is_full_history
