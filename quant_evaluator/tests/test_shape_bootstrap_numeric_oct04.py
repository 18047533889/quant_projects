"""Numeric guard regressions for shape bootstrap profiles near Float64 max."""

import numpy as np

from quant_evaluator.registry.metrics import get_metric


def _huge_monotone_windows(window_count=4):
    base = np.float64(1e308)
    # Integer ULP steps are exactly representable and preserve a strict rank
    # order independent of decimal rounding in the fixture.
    step = 8 * np.spacing(base)
    profile = base + np.arange(5, dtype=np.float64) * step
    return np.broadcast_to(
        profile[None, :, None], (window_count, profile.size, 1),
    ).copy()


def test_bootstrap_rank_metrics_preserve_huge_strict_profile_order():
    windows = _huge_monotone_windows()
    assert np.all(np.diff(windows[0, :, 0]) > 0)

    confidence = get_metric("shape_bootstrap_confidence").compute_fn(windows)
    rank_agreement = get_metric("shape_bootstrap_rank_agreement").compute_fn(windows)

    # Every sampled block has the same strictly increasing rank order.
    assert confidence[0] == 1.0
    assert rank_agreement[0] == 1.0


def test_bootstrap_rank_metrics_keep_three_finite_quantile_threshold():
    windows = _huge_monotone_windows(window_count=4)
    masked = np.repeat(windows, 2, axis=2)
    masked[:, 3:, 0] = np.nan  # exactly three common finite quantiles
    masked[:, 2:, 1] = np.nan  # two finite quantiles: insufficient evidence

    for metric_id in (
        "shape_bootstrap_confidence", "shape_bootstrap_rank_agreement",
    ):
        result = get_metric(metric_id).compute_fn(masked)
        assert result[0] == 1.0
        assert np.isnan(result[1])
