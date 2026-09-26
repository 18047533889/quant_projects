"""Independent small-sample oracles for HAC, paired blocks, and BH FDR."""

import numpy as np
import pytest
from scipy.stats import norm

from quant_evaluator.metrics.multiple_testing import benjamini_hochberg_correction
from quant_evaluator.metrics.registry_adapters import compute_hac_pvalue_value
from quant_evaluator.metrics.statistical_evidence import (
    build_hac_evidence,
    build_paired_block_bootstrap_difference,
)


def _manual_hac_bartlett_lag1(values):
    n = len(values)
    mean = sum(values) / n
    centered = [float(x - mean) for x in values]
    gamma0 = sum(x * x for x in centered) / n
    gamma1 = sum(centered[i] * centered[i + 1] for i in range(n - 1)) / n
    se = ((gamma0 + gamma1) / n) ** 0.5  # Bartlett lag-1 weight is 1/2.
    return mean, se, mean / se


def test_hac_matches_manual_lag1_and_gaussian_two_sided_p():
    series = np.array([0., 0., 1., 1.] * 3)
    mean, se, t_stat = _manual_hac_bartlett_lag1(series)
    evidence = build_hac_evidence(series, max_lag=1, min_periods=2)
    assert evidence.status == "VALID"
    assert evidence.estimate == pytest.approx(mean)
    assert evidence.standard_error == pytest.approx(se)
    assert evidence.t_statistic == pytest.approx(t_stat)
    p = compute_hac_pvalue_value(series, max_lag=1, min_periods=2)
    assert p[0] == pytest.approx(2 * norm.sf(abs(t_stat)))


def test_hac_zero_short_gap_and_infinite_endpoints_fail_closed():
    assert build_hac_evidence(np.zeros(20), max_lag=1, min_periods=2).status == "INSUFFICIENT"
    assert build_hac_evidence(np.arange(9.), max_lag=1, min_periods=2).status == "INSUFFICIENT"
    base = .1 + np.sin(np.arange(40))
    for bad in (np.nan, np.inf, -np.inf):
        interior = base.copy()
        interior[10] = bad
        assert build_hac_evidence(interior, max_lag=1, min_periods=10).status == "INSUFFICIENT"
    for bad in (np.inf, -np.inf):
        assert build_hac_evidence(np.r_[base, bad], max_lag=1, min_periods=10).status == "INSUFFICIENT"
        assert build_hac_evidence(np.r_[bad, base], max_lag=1, min_periods=10).status == "INSUFFICIENT"
    assert build_hac_evidence(np.r_[np.nan, base, np.nan], max_lag=1, min_periods=10).status == "VALID"


def _manual_paired_blocks(diff, *, block_length, repetitions, confidence_level, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = (n + block_length - 1) // block_length
    means = []
    for _ in range(repetitions):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        draw = [diff[start + offset] for start in starts for offset in range(block_length)]
        means.append(sum(draw[:n]) / n)
    alpha = (1 - confidence_level) / 2
    return np.quantile(means, [alpha, 1 - alpha])


def test_paired_block_ci_matches_manual_shared_draws():
    baseline = np.array([.1, -.1, .2, 0., .3, -.2])
    candidate = baseline + np.array([0., .1, -.1, .2, 0., .3])
    diff = candidate - baseline
    expected = _manual_paired_blocks(diff, block_length=2, repetitions=19,
                                     confidence_level=.8, seed=17)
    evidence = build_paired_block_bootstrap_difference(
        candidate, baseline, block_length=2, repetitions=19,
        confidence_level=.8, seed=17, min_periods=2,
    )
    assert evidence.status == "VALID"
    assert evidence.mean_difference == pytest.approx(np.mean(diff))
    np.testing.assert_allclose(evidence.confidence_interval, expected, atol=1e-15, rtol=0)


def test_paired_blocks_zero_short_gap_and_infinite_endpoints_fail_closed():
    base = np.sin(np.arange(40))
    zeros = np.zeros(40)
    constant = build_paired_block_bootstrap_difference(base, base, block_length=4,
                                                        repetitions=20, min_periods=2)
    assert constant.confidence_interval == (0., 0.)
    assert build_paired_block_bootstrap_difference(base[:3], zeros[:3], block_length=4,
                                                    min_periods=2).status == "INSUFFICIENT"
    for bad in (np.nan, np.inf, -np.inf):
        interior = base.copy()
        interior[10] = bad
        assert build_paired_block_bootstrap_difference(interior, zeros, block_length=4,
                                                       min_periods=2).status == "INSUFFICIENT"
    for bad in (np.inf, -np.inf):
        for a, b in ((np.r_[base, bad], np.r_[zeros, 0.]),
                     (np.r_[base, 0.], np.r_[zeros, bad])):
            assert build_paired_block_bootstrap_difference(a, b, block_length=4,
                                                           min_periods=2).status == "INSUFFICIENT"


def test_bh_known_values_missing_and_extreme_probabilities():
    p = np.array([.2, np.nan, .03, .001, .01, np.inf, -np.inf])
    adjusted, reject, n = benjamini_hochberg_correction(p)
    np.testing.assert_allclose(adjusted[[0, 2, 3, 4]], [.2, .04, .004, .02])
    assert np.isnan(adjusted[[1, 5, 6]]).all()
    np.testing.assert_array_equal(reject, [False, False, True, True, True, False, False])
    assert n == 3
    tiny = np.nextafter(0., 1.)
    extreme, accepted, count = benjamini_hochberg_correction(np.array([0., tiny, 1.]))
    np.testing.assert_allclose(extreme, [0., tiny * 1.5, 1.], rtol=0, atol=tiny)
    np.testing.assert_array_equal(accepted, [True, True, False])
    assert count == 2
    all_missing = benjamini_hochberg_correction(np.array([np.nan, np.inf]))
    assert np.isnan(all_missing[0]).all() and not all_missing[1].any() and all_missing[2] == 0


@pytest.mark.parametrize("baseline_value", [0., -1e308])
def test_paired_difference_overflow_cannot_certify_interval(baseline_value):
    candidate = np.full(30, 1e308)
    baseline = np.full(30, baseline_value)
    evidence = build_paired_block_bootstrap_difference(
        candidate, baseline, block_length=3, repetitions=20,
    )
    assert evidence.status == "INSUFFICIENT"
    assert evidence.mean_difference is None and evidence.confidence_interval == (None, None)



def test_paired_resampled_mean_overflow_is_insufficient_without_warning():
    candidate = np.tile([1e308, -1e308], 15)
    with np.errstate(over="raise", invalid="raise"):
        evidence = build_paired_block_bootstrap_difference(
            candidate, np.zeros(30), block_length=1, repetitions=100, seed=2,
        )
    assert evidence.status == "INSUFFICIENT"
    assert evidence.mean_difference is None and evidence.confidence_interval == (None, None)
