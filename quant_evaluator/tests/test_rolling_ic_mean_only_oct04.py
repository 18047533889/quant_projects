"""Protect the rolling_ic public mean-only computation contract."""
import numpy as np
import pytest

from quant_evaluator.metrics.ic_summary import compute_rolling_ic_stats
from quant_evaluator.registry.metrics import get_metric


def test_public_rolling_ic_does_not_compute_unrequested_standard_deviations(monkeypatch):
    # Defect: computing six rich statistics when the request needs only a mean.
    observed_std_calls = []
    original_std = np.std

    def record_std(*args, **kwargs):
        observed_std_calls.append(1)
        return original_std(*args, **kwargs)

    monkeypatch.setattr(np, "std", record_std)
    values = np.array([[.25, .25], [np.nan, .25], [.5, .25],
                       [np.inf, .25], [.75, .25], [-.25, .25]])
    before = values.copy()
    actual = get_metric("rolling_ic").compute_fn(values, window=3, min_periods=2)
    expected = np.array([[np.nan, np.nan], [np.nan, .25], [.375, .25],
                         [np.nan, .25], [.625, .25], [.25, .25]])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, before)
    assert observed_std_calls == [], "mean-only metric computed unrelated standard deviations"


@pytest.mark.parametrize("shape", [(0, 1), (1, 3), (7, 3), (31, 1), (96, 3)])
@pytest.mark.parametrize("window", [1, 7, 60])
@pytest.mark.parametrize("min_periods", [1, 2, 20])
def test_public_rolling_ic_preserves_rich_mean_for_strided_finite_and_missing_inputs(
    shape, window, min_periods
):
    rng = np.random.default_rng(20261004)
    base = rng.uniform(-1, 1, (shape[0], shape[1] * 2))
    values = base[:, ::2]
    if shape[0]:
        values[:, 0] = .25
        values[::11, 0] = np.nan
    if shape[1] > 1:
        values[::13, 1] = np.inf
        values[::17, 2] = -np.inf
    expected = compute_rolling_ic_stats(values, window, min_periods)["rolling_ic_mean"]
    actual = get_metric("rolling_ic").compute_fn(values, window, min_periods)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == shape and actual.dtype == np.float64


def test_public_rolling_ic_keeps_one_dimensional_input_as_one_factor():
    values = np.array([.25, np.nan, .5, .75])
    actual = get_metric("rolling_ic").compute_fn(values, window=2, min_periods=1)
    np.testing.assert_array_equal(actual, np.array([[.25], [.25], [.5], [.625]]))
    assert actual.shape == (4, 1)
