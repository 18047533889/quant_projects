import numpy as np
import pytest

from quant_evaluator.metrics.ic_summary import compute_icir


def test_tiny_but_nonzero_sample_variance_remains_defined():
    values = np.array([0.3, 0.3 + 1e-12, 0.3 - 1e-12, 0.3 + 2e-12])[:, None]
    expected = np.mean(values[:, 0]) / np.std(values[:, 0], ddof=1)
    actual = compute_icir(values, min_periods=4)
    assert actual[0] == pytest.approx(expected, rel=1e-14)
    assert np.isfinite(actual[0])


def test_exact_constant_nonzero_ic_is_undefined():
    actual = compute_icir(np.full((5, 1), 0.3), min_periods=5)
    assert np.isnan(actual[0])


def test_infinities_are_not_counted_as_observations():
    values = np.array([0.1, 0.2, np.inf, -np.inf, np.nan])[:, None]
    assert np.isnan(compute_icir(values, min_periods=3)[0])
    expected = np.mean([0.1, 0.2]) / np.std([0.1, 0.2], ddof=1)
    assert compute_icir(values, min_periods=2)[0] == pytest.approx(expected)


@pytest.mark.parametrize("invalid", [0, 1, -2])
def test_min_periods_must_support_sample_standard_deviation(invalid):
    with pytest.raises(ValueError, match="at least 2"):
        compute_icir(np.array([[0.1], [0.2]]), min_periods=invalid)


@pytest.mark.parametrize("invalid", [True, 2.5, "2"])
def test_min_periods_must_be_an_integer(invalid):
    with pytest.raises(TypeError, match="integer"):
        compute_icir(np.array([[0.1], [0.2]]), min_periods=invalid)
