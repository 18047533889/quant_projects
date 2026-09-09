import numpy as np
import pytest

from quant_evaluator.metrics.statistical_evidence import (
    build_dsr_evidence,
    build_hac_evidence,
    build_horizon_curve_evidence,
    build_paired_block_bootstrap_difference,
    build_pbo_evidence,
    build_regime_evidence,
    build_retention_evidence,
)


def test_hac_counts_time_not_repeated_asset_cells_and_records_bandwidth():
    series = np.sin(np.arange(80) / 8) * 0.02
    evidence = build_hac_evidence(series, max_lag=5, min_periods=30)
    assert evidence.n_time == 80
    assert evidence.bandwidth == 5 and evidence.kernel == "bartlett"
    assert evidence.status == "VALID"
    with pytest.raises(ValueError):
        build_hac_evidence(np.tile(series[:, None], (1, 20)))


def test_hac_short_series_is_insufficient_not_fake_significance():
    evidence = build_hac_evidence([0.1, 0.2, 0.3], max_lag=2, min_periods=10)
    assert evidence.status == "INSUFFICIENT"
    assert evidence.t_statistic is None and evidence.standard_error is None


def test_hac_constant_series_does_not_certify_nan_statistics():
    evidence = build_hac_evidence(np.ones(60))
    assert evidence.status == "INSUFFICIENT"
    assert evidence.t_statistic is None and evidence.standard_error is None


def test_gapped_time_series_are_not_silently_compressed_into_valid_evidence():
    values = np.sin(np.arange(80) / 8)
    values[30] = np.nan
    hac = build_hac_evidence(values)
    paired = build_paired_block_bootstrap_difference(values, np.zeros(80))
    assert hac.status == paired.status == "INSUFFICIENT"
    assert hac.n_time == paired.n_time == 79
    assert hac.t_statistic is None
    assert paired.confidence_interval == (None, None)


def test_empty_warmup_and_tail_do_not_change_contiguous_time_evidence():
    values = np.sin(np.arange(80) / 8)
    padded = np.concatenate(([np.nan] * 3, values, [np.nan] * 4))
    assert build_hac_evidence(padded) == build_hac_evidence(values)
    assert build_paired_block_bootstrap_difference(padded, padded) == (
        build_paired_block_bootstrap_difference(values, values))


@pytest.mark.parametrize("kwargs", [
    {"max_lag": True}, {"max_lag": -1}, {"min_periods": 1.5},
    {"kernel": "unknown"}, {"bandwidth_rule": ""},
])
def test_hac_validates_policy_even_with_insufficient_observations(kwargs):
    with pytest.raises(ValueError):
        build_hac_evidence([], **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"confidence_level": np.nan}, {"confidence_level": True},
    {"confidence_level": 1}, {"repetitions": 2.5},
    {"block_length": True}, {"seed": -1},
])
def test_paired_bootstrap_rejects_invalid_sampling_policy(kwargs):
    with pytest.raises(ValueError):
        build_paired_block_bootstrap_difference([], [], **kwargs)


def test_dsr_cannot_flatten_assets_into_independent_time_observations():
    with pytest.raises(ValueError, match="one return time series"):
        build_dsr_evidence(
            np.ones((30, 20)), family_sharpes=[.1, .2], effective_trial_count=2,
            trial_ledger_ref="ledger", returns_frequency="daily", annualization_factor=252,
        )


def test_paired_identical_candidates_have_exact_zero_interval():
    rng = np.random.default_rng(7); values = rng.normal(size=100)
    evidence = build_paired_block_bootstrap_difference(
        values, values.copy(), block_length=8, repetitions=200, seed=19,
    )
    assert evidence.mean_difference == 0
    assert evidence.confidence_interval == (0.0, 0.0)


def test_paired_bootstrap_order_is_antisymmetric_with_shared_seed():
    rng = np.random.default_rng(4)
    base = rng.normal(size=120); candidate = base + 0.01
    ab = build_paired_block_bootstrap_difference(candidate, base, block_length=10, repetitions=250, seed=3)
    ba = build_paired_block_bootstrap_difference(base, candidate, block_length=10, repetitions=250, seed=3)
    assert ab.mean_difference == -ba.mean_difference
    np.testing.assert_allclose(ab.confidence_interval, tuple(-x for x in ba.confidence_interval[::-1]))


def _returns(n, raw_sharpe=0.1):
    centered = np.linspace(-1, 1, n)
    centered = centered / np.std(centered, ddof=1)
    return centered + raw_sharpe


def test_dsr_uses_actual_time_count_and_separates_sharpe_scales():
    raw_family = np.array([0.01, 0.03, 0.05, 0.09])
    short = build_dsr_evidence(
        _returns(80), family_sharpes=raw_family, effective_trial_count=4,
        trial_ledger_ref="ledger:v1", returns_frequency="daily",
        annualization_factor=252, family_sharpe_scale="raw_periodic",
    )
    long = build_dsr_evidence(
        _returns(240), family_sharpes=raw_family, effective_trial_count=4,
        trial_ledger_ref="ledger:v1", returns_frequency="daily",
        annualization_factor=252, family_sharpe_scale="raw_periodic",
    )
    annual = build_dsr_evidence(
        _returns(80), family_sharpes=raw_family * np.sqrt(252), effective_trial_count=4,
        trial_ledger_ref="ledger:v1", returns_frequency="daily",
        annualization_factor=252, family_sharpe_scale="annualized",
    )
    assert short.n_time == 80 and long.n_time == 240
    assert long.probability > short.probability
    np.testing.assert_allclose(short.benchmark_max_sharpe, annual.benchmark_max_sharpe)
    np.testing.assert_allclose(short.probability, annual.probability)


def test_dsr_rejects_incomplete_trial_family():
    with pytest.raises(ValueError, match="complete trial family"):
        build_dsr_evidence(
            _returns(80), family_sharpes=[0.1, 0.2], effective_trial_count=3,
            trial_ledger_ref="ledger:v1", returns_frequency="daily", annualization_factor=252,
        )


def test_pbo_rejects_winner_only_and_incomplete_matrices():
    with pytest.raises(ValueError, match="winner-only"):
        build_pbo_evidence(np.ones((30, 1)), candidate_universe_complete=True,
                           common_cost_spec_ref="cost:v1")
    incomplete = np.ones((30, 3)); incomplete[2, 1] = np.nan
    with pytest.raises(ValueError, match="complete on a common time axis"):
        build_pbo_evidence(incomplete, candidate_universe_complete=True,
                           common_cost_spec_ref="cost:v1")
    with pytest.raises(ValueError, match="complete candidate universe"):
        build_pbo_evidence(np.ones((30, 3)), candidate_universe_complete=False,
                           common_cost_spec_ref="cost:v1")


def test_pbo_known_overfit_has_positive_probability():
    rng = np.random.default_rng(5)
    matrix = rng.normal(0, 1, size=(60, 6))
    evidence = build_pbo_evidence(matrix, n_splits=6,
                                  candidate_universe_complete=True,
                                  common_cost_spec_ref="cost:v1")
    assert evidence.status == "VALID"
    assert len(evidence.split_logits) == 20
    assert 0 <= evidence.probability_backtest_overfit <= 1


def test_retention_near_zero_and_sign_reversal_disable_ratio():
    series = _returns(80)
    near_zero = build_retention_evidence(series, series, train_value=1e-5,
                                         validation_value=2e-5, block_length=5,
                                         repetitions=50, min_periods=30)
    reversal = build_retention_evidence(series, -series, train_value=0.1,
                                        validation_value=-0.05, block_length=5,
                                        repetitions=50, min_periods=30)
    assert not near_zero.ratio_applicable and near_zero.retention_ratio is None
    assert reversal.sign_reversal and not reversal.ratio_applicable
    assert reversal.signed_difference == pytest.approx(-0.15)


@pytest.mark.parametrize(
    "means, expected",
    [
        ([0.08, 0.04, -0.02, -0.01], "SIGN_CHANGE"),
        ([0.01, 0.05, 0.01, 0.06, 0.01], "MULTI_PEAK"),
        ([0.08, 0.04, 0.045, 0.02], "NOT_FIT"),
    ],
)
def test_horizon_curve_refuses_invalid_decay_shapes(means, expected):
    series = {i + 1: np.full(30, value) for i, value in enumerate(means)}
    refs = {h: f"label:{h}" for h in series}
    evidence = build_horizon_curve_evidence(series, label_refs=refs, min_periods=20)
    assert evidence.fit_status == expected and evidence.half_life is None


def test_horizon_curve_fits_clean_monotone_exponential():
    horizons = (1, 2, 3, 4)
    series = {h: np.full(30, 0.1 * np.exp(-h / 2)) for h in horizons}
    evidence = build_horizon_curve_evidence(
        series, label_refs={h: f"label:{h}" for h in horizons}, min_periods=20,
    )
    assert evidence.fit_status == "VALID"
    assert evidence.half_life == pytest.approx(2 * np.log(2))


def test_horizon_curve_rejects_multifactor_matrix_instead_of_flattening():
    series = {1: np.ones((30, 2)), 2: np.ones((30, 2)) * 0.5}
    with pytest.raises(ValueError, match="one-dimensional"):
        build_horizon_curve_evidence(
            series, label_refs={1: "label:1", 2: "label:2"}, min_periods=20,
        )


def test_regime_requires_counts_and_online_frozen_state():
    with pytest.raises(ValueError, match="frozen state_ref"):
        build_regime_evidence(np.ones(40), np.repeat([0, 1], 20), range(40),
                              regime_kind="online_causal")
    evidence = build_regime_evidence(
        np.arange(12.0), np.repeat([0, 1], [10, 2]), range(12),
        regime_kind="ex_post_diagnostic", min_periods=5,
    )
    assert evidence.status == "INSUFFICIENT"
    assert evidence.conditional_counts == {0: 10, 1: 2}
    assert evidence.conditional_values[1] is None
