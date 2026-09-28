"""Calendar and positional grouping for IC dispersion."""

import numpy as np

from quant_evaluator.metrics.ic_summary import compute_yearly_quarterly_dispersion


def test_integer_time_index_uses_positional_quarters():
    ic = np.linspace(-0.5, 0.5, 252)[:, None]
    expected = compute_yearly_quarterly_dispersion(ic)
    actual = compute_yearly_quarterly_dispersion(ic, time_index=np.arange(252))

    for metric in expected:
        np.testing.assert_allclose(actual[metric], expected[metric], equal_nan=True)
    assert actual["n_quarters"][0] == 4


def test_datetime_time_index_uses_calendar_boundaries():
    ic = np.array([[0.1], [0.2], [0.3], [0.4]])
    dates = np.array(
        ["2024-03-30", "2024-03-31", "2024-04-01", "2025-01-01"],
        dtype="datetime64[D]",
    )
    actual = compute_yearly_quarterly_dispersion(ic, time_index=dates)

    assert actual["n_years"][0] == 2
    assert actual["n_quarters"][0] == 3
    np.testing.assert_allclose(
        actual["yearly_mean_ic_std"],
        np.std([0.2, 0.4], ddof=1),
    )
    np.testing.assert_allclose(
        actual["quarterly_mean_ic_std"],
        np.std([0.15, 0.3, 0.4], ddof=1),
    )
