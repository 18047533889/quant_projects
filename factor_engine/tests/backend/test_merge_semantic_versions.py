from __future__ import annotations

import pytest

from factor_engine.backend.operator_semantic_version import semantic_version


@pytest.mark.parametrize(
    ("canonical", "version"),
    (
        ("ts_super_smoother", 2),
        ("ts_hodges_lehmann_location", 2),
        ("ts_binned_response_monotonicity", 2),
        ("ts_binned_response_curvature", 2),
        ("intraday_jump_test_stat", 2),
        ("ts_weighted_standardized_moment", 2),
        ("group_tail_centrality", 2),
        ("group_tail_lead_score", 2),
        ("ts_autocorrelation_time", 2),
        ("ts_autocorrelation_time_initial_positive_sequence", 2),
        ("ts_return_spectral_entropy", 2),
        ("ts_spectral_entropy", 2),
        ("ts_detrended_level_spectral_entropy", 2),
        ("ts_weighted_semivariance", 2),
        ("ts_weighted_downside_deviation", 2),
        ("ts_weighted_expected_shortfall", 2),
        ("ts_weighted_drawdown_area", 2),
        ("ts_markov_persistence", 2),
        ("ts_markov_state_entropy", 2),
        ("ts_markov_transition_surprisal", 2),
        ("ts_markov_entropy_production", 2),
        ("ts_kramers_moyal_local_stability", 2),
        ("ts_markov_committor", 2),
        ("ts_markov_mean_first_passage_time", 2),
        ("ts_markov_spectral_gap", 2),
        ("ts_markov_stationary_surprisal", 2),
        ("ts_km_equilibrium_distance", 2),
        ("ts_km_diffusion_gradient", 2),
        ("ts_km_quasipotential_depth", 2),
        ("ts_active_information_storage", 2),
        ("ts_beta_break_score", 2),
        ("ts_price_delay", 2),
        ("ts_best_lag_corr_excess", 2),
        ("ts_glr_mean_shift_score", 2),
        ("ts_glr_variance_shift_score", 2),
        ("ts_delay_intrinsic_dimension", 2),
        ("ts_vol_pvariation_roughness", 2),
        ("ts_vol_scaling_break", 2),
        ("intraday_volatility_time_centroid", 2),
        ("intraday_volatility_concentration", 2),
        ("intraday_volatility_entropy", 2),
        ("intraday_realized_semivariance_balance", 2),
        ("intraday_rv_signature_curvature", 2),
        ("ts_modwt_band_corr", 2),
        ("ts_fisher_information_shift", 2),
        ("cs_huber_resid", 3),
        ("ts_huber_regression_in_sample_resid", 3),
        ("ts_huber_regression_predictive_resid", 3),
    ),
)
def test_merge_numeric_changes_have_distinct_semantic_versions(
    canonical: str, version: int
) -> None:
    assert semantic_version(canonical) == version
