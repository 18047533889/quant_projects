"""Independent finite-sample oracle for IC t-statistics and p-values."""

import numpy as np
import pytest
from scipy.stats import t as student_t

from quant_evaluator.metrics.ic_summary import compute_ic_summary_stats, compute_ic_tstat


def _manual_t(values):
    finite = np.asarray(values)[np.isfinite(values)]
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))
    statistic = mean * np.sqrt(len(finite)) / std
    p_value = 2.0 * student_t.sf(abs(statistic), df=len(finite) - 1)
    return statistic, p_value


def test_ic_tstat_uses_finite_samples_and_preserves_small_real_dispersion():
    ordinary = np.arange(1.0, 21.0) / 100.0
    tiny = ordinary * 1e-12
    series = np.column_stack((
        np.r_[ordinary, np.inf],
        np.r_[tiny, np.nan],
        np.r_[np.full(20, 0.1), -np.inf],
    ))
    actual_t, actual_p = compute_ic_tstat(series, min_periods=20)
    summary = compute_ic_summary_stats(series, min_periods=20)
    for index, source in enumerate((ordinary, tiny)):
        expected_t, expected_p = _manual_t(source)
        assert actual_t[index] == pytest.approx(expected_t, rel=1e-12)
        assert actual_p[index] == pytest.approx(expected_p, rel=1e-12)
        assert summary["t_stat"][index] == pytest.approx(expected_t, rel=1e-12)
        assert summary["p_value"][index] == pytest.approx(expected_p, rel=1e-12)
    assert np.isnan(actual_t[2]) and np.isnan(actual_p[2])


def test_ic_tstat_finite_minimum_and_parameter_validation():
    series = np.r_[np.arange(1.0, 20.0) / 100.0, np.inf][:, None]
    t_stat, p_value = compute_ic_tstat(series, min_periods=20)
    assert np.isnan(t_stat[0]) and np.isnan(p_value[0])
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="min_periods"):
            compute_ic_tstat(series, min_periods=invalid)
    with pytest.raises(ValueError, match="min_periods"):
        compute_ic_tstat(series, min_periods=1)
