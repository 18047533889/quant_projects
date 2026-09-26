"""Finite-universe oracle for factor membership change with nonfinite values."""

import numpy as np
import pytest

from quant_evaluator.metrics.temporal import compute_factor_turnover_rate


@pytest.mark.parametrize("nonfinite,quantile", [(np.inf, 0.9), (-np.inf, 0.1), (np.nan, 0.9)])
@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("universe_membership_change", 0.2),
        ("top_exit_fraction", 1.0),
        ("top_entry_fraction", 1.0),
        ("jaccard_distance", 1.0),
    ],
)
def test_nonfinite_signal_excluded_from_quantile_cutoff(nonfinite, quantile, measure, expected):
    # There are ten observable stocks. Their extreme member moves from asset
    # 9 to asset 0 for the upper tail, and from 0 to 9 for the lower tail.
    # Asset 10 is unobservable and must have no effect on either cutoff.
    panel = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, nonfinite],
        [9, 1, 2, 3, 4, 5, 6, 7, 8, 0, nonfinite],
    ], dtype=float)[:, :, None]
    result = compute_factor_turnover_rate(panel, quantile=quantile, measure=measure)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected)
