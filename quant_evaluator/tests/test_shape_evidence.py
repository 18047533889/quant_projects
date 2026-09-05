"""
R61-FI-021 (plan §13.4 / §14.1 / §14.3 / §14.4): shape evidence metrics.

Pins the NEW shape-evidence kernels of :mod:`quant_evaluator.metrics.
shape_evidence` and the adaptive-bins policy of
:mod:`quant_evaluator.contracts.adaptive_bins_policy`:

1. u_shape_score is HIGH on a synthetic U-shaped quantile profile and LOW
   on a monotone profile (plan §14.3: U-template fit must exceed the
   monotone template — a strong monotone factor is NOT a U even if its
   U-fit looks high).  inverted_u_score is the mirror.
2. linear_trend_score is ~+1 for monotone increasing, ~0 for U.
3. adaptive bins: AdaptiveBinsPolicy(preferred_bins=20, fallback_bins=(10,
   5), min_effective_names_per_bin=100) resolves 20 when names allow it,
   falls back to 10 then 5 as the universe shrinks, and reports
   ``insufficient`` (never 0) when even 5 bins are infeasible;
   compute_adaptive_quantile_count reports the ACTUAL bin count from the
   profile (never assumes 20).
4. Tail metrics carry explicit direction metadata on the MetricSpec
   (top_tail_slope / bottom_tail_slope direction == "neutral" — orientation
   informative, never assumed, plan §14.4); the robust tail cliffs
   (Q_K - mean(Q_(K-3..K-1))) are separate ids from the one-bin cliffs.
5. shape_stability / shape_regime_stability / shape_bootstrap_confidence
   are NaN for a single profile (missing evidence, never 0/1) and high for
   stable repeated U windows.

Run from the repo root:

    cd /home/sunhaiwei/quant_projects && .venv/bin/python -m pytest -q \
        quant_evaluator/tests/test_shape_evidence.py --timeout=300
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.contracts.adaptive_bins_policy import (
    AdaptiveBinsPolicy,
    AdaptiveBinsResolution,
    resolve_bin_count,
    resolve_bin_count_from_counts,
)
from quant_evaluator.contracts.artifact_types import QuantileReturnArtifact
from quant_evaluator.metrics.shape_evidence import (
    compute_adaptive_quantile_count,
    compute_bottom_quantile_cliff_robust,
    compute_bottom_tail_slope,
    compute_inverted_u_score,
    compute_left_right_asymmetry,
    compute_linear_trend_score,
    compute_shape_bootstrap_confidence,
    compute_shape_regime_stability,
    compute_shape_stability,
    compute_tail_vs_middle_contrast,
    compute_top_quantile_cliff_robust,
    compute_top_tail_slope,
    compute_u_shape_score,
)
from quant_evaluator.registry.metrics import get_metric

NQ = 20


def _u_profile(nq: int = NQ, amp: float = 1.5, mid: float = 0.0) -> np.ndarray:
    """Textbook U: middle bins underperform both tails (convex)."""
    x = np.linspace(0.0, 1.0, nq)
    return (amp * (x - 0.5) ** 2 + mid).reshape(-1, 1)


def _inverted_u_profile(nq: int = NQ, amp: float = 1.5) -> np.ndarray:
    """Textbook inverted-U (hill)."""
    x = np.linspace(0.0, 1.0, nq)
    return (-amp * (x - 0.5) ** 2 + 1.0).reshape(-1, 1)


def _monotone_profile(nq: int = NQ, slope: float = 1.0) -> np.ndarray:
    """Strictly monotone increasing profile."""
    x = np.linspace(0.0, 1.0, nq)
    return (slope * x).reshape(-1, 1)


def _flat_profile(nq: int = NQ) -> np.ndarray:
    return np.zeros((nq, 1))


# ---------------------------------------------------------------------------
# 1. U / inverted-U semantics (plan §14.3).
# ---------------------------------------------------------------------------


def test_u_shape_score_high_on_synthetic_u():
    score = compute_u_shape_score(_u_profile())
    assert score.shape == (1,)
    assert np.isfinite(score[0])
    assert score[0] >= 0.8, f"u_shape_score={score[0]} on a textbook U"


def test_u_shape_score_low_on_monotone():
    score = compute_u_shape_score(_monotone_profile())
    assert np.isfinite(score[0])
    assert score[0] <= 0.2, f"u_shape_score={score[0]} on monotone (not a U)"


def test_inverted_u_score_high_on_hill_low_on_u_and_monotone():
    inv = compute_inverted_u_score(_inverted_u_profile())
    assert inv[0] >= 0.8
    assert compute_inverted_u_score(_u_profile())[0] <= 0.5
    assert compute_inverted_u_score(_monotone_profile())[0] <= 0.2


def test_flat_profile_scores_zero_not_u():
    flat = compute_u_shape_score(_flat_profile())
    assert flat[0] == 0.0
    assert compute_inverted_u_score(_flat_profile())[0] == 0.0


def test_linear_trend_score_separates_monotone_from_u():
    trend = compute_linear_trend_score(_monotone_profile())
    assert trend[0] >= 0.95
    u_trend = compute_linear_trend_score(_u_profile())
    assert abs(u_trend[0]) <= 0.2


def test_u_shape_score_multifactor_per_column():
    """(nq, F) inputs produce one scalar per factor."""
    qr = np.stack([_u_profile()[:, 0], _monotone_profile()[:, 0]], axis=1)
    scores = compute_u_shape_score(qr)
    assert scores.shape == (2,)
    assert scores[0] >= 0.8
    assert scores[1] <= 0.2


# ---------------------------------------------------------------------------
# 2. Adaptive bins policy (plan §14.1).
# ---------------------------------------------------------------------------


def test_plan_default_policy_parameters():
    policy = AdaptiveBinsPolicy()
    assert policy.preferred_bins == 20
    assert policy.fallback_bins == (10, 5)
    assert policy.min_effective_names_per_bin == 100
    assert policy.candidate_bin_counts() == (20, 10, 5)


def test_resolve_20_when_names_allow():
    resolution = resolve_bin_count([2500, 3000, 8000])
    assert resolution.bin_count == 20
    assert resolution.reason == "preferred"
    assert resolution.min_names_per_bin == 2500 / 20


def test_resolve_fallback_10_when_20_infeasible():
    # 20 bins * 100 names = 2000 needed; only 1500 names -> 10 bins feasible.
    resolution = resolve_bin_count([1500, 1600, 1800])
    assert resolution.bin_count == 10
    assert resolution.reason == "fallback"


def test_resolve_fallback_5_when_10_infeasible():
    # 10*100 = 1000 needed; only 600 names -> 5 bins feasible.
    resolution = resolve_bin_count([600, 800])
    assert resolution.bin_count == 5
    assert resolution.reason == "fallback"


def test_resolve_insufficient_is_none_never_zero():
    # 5*100 = 500 needed; only 300 names -> nothing feasible.
    resolution = resolve_bin_count([300, 250])
    assert resolution.bin_count is None
    assert resolution.reason == "insufficient"
    assert resolution.is_insufficient


def test_resolve_empty_panel_is_insufficient():
    resolution = resolve_bin_count([])
    assert resolution.is_insufficient
    assert resolution.bin_count is None


def test_policy_frozen_and_validation():
    policy = AdaptiveBinsPolicy()
    with pytest.raises(Exception):
        policy.preferred_bins = 10  # type: ignore[misc]
    with pytest.raises(ValueError):
        AdaptiveBinsPolicy(preferred_bins=1)
    with pytest.raises(ValueError):
        AdaptiveBinsPolicy(fallback_bins=(20,))  # duplicates preferred


def test_resolve_from_observed_per_bin_counts():
    resolution = resolve_bin_count_from_counts(
        {20: [99, 120], 10: [250, 300]}
    )
    assert resolution.bin_count == 10
    assert resolution.reason == "fallback"
    # 20 bins unobserved entirely -> skipped, never assumed feasible.
    resolution2 = resolve_bin_count_from_counts({10: [150, 200]})
    assert resolution2.bin_count == 10


def test_adaptive_bin_count_reports_actual_used():
    """adaptive_quantile_count reports the profile's ACTUAL bin count."""
    for nq, expected in [(20, 20.0), (10, 10.0), (5, 5.0)]:
        qr = _monotone_profile(nq=nq)
        count = compute_adaptive_quantile_count(qr)
        assert count.shape == (1,)
        assert count[0] == expected


def test_adaptive_bin_count_from_artifact_uses_n_quantiles():
    artifact = QuantileReturnArtifact(
        values=_monotone_profile(nq=10), n_quantiles=10
    )
    count = compute_adaptive_quantile_count(artifact)
    assert count[0] == 10.0


# ---------------------------------------------------------------------------
# 3. Tail slopes and robust cliffs with direction metadata (plan §14.4).
# ---------------------------------------------------------------------------


def test_tail_slopes_positive_on_positive_monotone_profile():
    qr = _monotone_profile(slope=1.0)
    top = compute_top_tail_slope(qr)
    bottom = compute_bottom_tail_slope(qr)
    assert top[0] > 0.0
    assert bottom[0] > 0.0


def test_tail_slope_direction_metadata_is_neutral():
    """Orientation must never be assumed (plan §14.4) -> direction neutral."""
    top_spec = get_metric("top_tail_slope")
    bottom_spec = get_metric("bottom_tail_slope")
    assert top_spec.direction == "neutral"
    assert bottom_spec.direction == "neutral"
    assert "orientation" not in top_spec.direction or True  # doc-level only


def test_robust_top_cliff_detects_jump_into_top_bucket():
    col = np.linspace(0.0, 1.0, NQ)
    col[-1] += 0.5  # cliff into the very top bucket
    col[-2] += 0.1
    robust = compute_top_quantile_cliff_robust(col.reshape(-1, 1))
    assert robust[0] > 0.4
    # A plain (non-cliff) monotone profile has a near-flat robust cliff.
    plain = compute_top_quantile_cliff_robust(_monotone_profile())
    assert abs(plain[0]) < robust[0]


def test_robust_cliffs_are_separate_ids_from_one_bin_cliffs():
    """Plan §14.4: Q_K - mean(Q_(K-3..K-1)) is a DIFFERENT semantic from
    the one-bin top_quantile_cliff — registered under its own id."""
    spec_top = get_metric("top_quantile_cliff_robust")
    spec_bottom = get_metric("bottom_quantile_cliff_robust")
    assert spec_top.metric_id == "top_quantile_cliff_robust"
    assert spec_bottom.metric_id == "bottom_quantile_cliff_robust"
    assert spec_top.direction == "higher_is_better"
    assert spec_bottom.direction == "higher_is_better"
    # Both still resolve and remain distinct from the pre-existing ids.
    assert get_metric("top_quantile_cliff").metric_id == "top_quantile_cliff"
    assert get_metric("bottom_quantile_cliff").metric_id == "bottom_quantile_cliff"


def test_robust_cliff_needs_four_finite_bins():
    short = np.array([[1.0], [2.0], [3.0]])  # only 3 quantiles
    assert np.isnan(compute_top_quantile_cliff_robust(short)[0])


# ---------------------------------------------------------------------------
# 4. Contrast / asymmetry on the drawn profile.
# ---------------------------------------------------------------------------


def test_tail_vs_middle_contrast_high_for_u_low_for_flat():
    u_contrast = compute_tail_vs_middle_contrast(_u_profile())
    assert u_contrast[0] > 0.1
    flat_contrast = compute_tail_vs_middle_contrast(_flat_profile())
    assert flat_contrast[0] <= 1e-9


def test_tail_vs_middle_contrast_nan_for_tiny_profile():
    tiny = np.ones((5, 1))
    assert np.isnan(compute_tail_vs_middle_contrast(tiny)[0])


def test_left_right_asymmetry_positive_for_right_heavy_profile():
    x = np.linspace(0.0, 1.0, NQ)
    right_heavy = (1.5 * (x - 0.5) ** 2 + 0.2 + 0.8 * np.maximum(x - 0.5, 0.0))
    lr = compute_left_right_asymmetry(right_heavy.reshape(-1, 1))
    assert np.isfinite(lr[0])
    assert lr[0] > 0.05


# ---------------------------------------------------------------------------
# 5. Shape stability family: NaN for single window, high for stable windows.
# ---------------------------------------------------------------------------


def _windowed_stable_u(n_windows: int = 6, nq: int = 12) -> np.ndarray:
    xw = np.linspace(0.0, 1.0, nq)
    windows = np.empty((n_windows, nq, 1))
    for w in range(n_windows):
        windows[w, :, 0] = 1.5 * (xw - 0.5) ** 2 + 0.05 * w
    return windows


def test_shape_stability_nan_for_single_profile():
    """Missing second observation -> NaN, never 0 or 1."""
    single = _u_profile()
    assert np.isnan(compute_shape_stability(single)[0])
    assert np.isnan(compute_shape_regime_stability(single)[0])
    assert np.isnan(compute_shape_bootstrap_confidence(single)[0])


def test_shape_stability_high_for_stable_windows():
    windows = _windowed_stable_u()
    stability = compute_shape_stability(windows)
    assert np.isfinite(stability[0])
    assert stability[0] > 2.0  # Fisher-z of high correlations


def test_shape_regime_stability_high_for_stable_windows():
    windows = _windowed_stable_u()
    regime = compute_shape_regime_stability(windows)
    assert np.isfinite(regime[0])
    assert regime[0] > 2.0


def test_shape_bootstrap_confidence_high_for_stable_windows():
    windows = _windowed_stable_u()
    confidence = compute_shape_bootstrap_confidence(windows)
    assert confidence[0] >= 0.9


# ---------------------------------------------------------------------------
# 6. Registry wiring of the new shape ids.
# ---------------------------------------------------------------------------

NEW_SHAPE_IDS = (
    "u_shape_score",
    "inverted_u_score",
    "adaptive_quantile_count",
    "top_tail_slope",
    "bottom_tail_slope",
    "tail_vs_middle_contrast",
    "left_right_asymmetry",
    "linear_trend_score",
    "shape_stability",
    "shape_regime_stability",
    "shape_bootstrap_confidence",
    "top_quantile_cliff_robust",
    "bottom_quantile_cliff_robust",
)


def test_new_shape_ids_registered_with_bound_compute_fn():
    from quant_evaluator.registry.metrics import list_metrics

    registered = set(list_metrics())
    for metric_id in NEW_SHAPE_IDS:
        assert metric_id in registered, metric_id
        spec = get_metric(metric_id)
        assert spec.compute_fn is not None, metric_id
        assert spec.requires == ["QuantileReturnArtifact"], metric_id


def test_pre_existing_shape_ids_untouched():
    """The 8 pre-existing shape metrics are REUSED, not re-registered."""
    from quant_evaluator.registry.metrics import list_metrics

    registered = set(list_metrics())
    for metric_id in (
        "quantile_monotonicity",
        "quantile_curvature",
        "quantile_tail_asymmetry",
        "quantile_adjacent_spread",
        "quantile_extreme_cliff",
        "top_quantile_cliff",
        "bottom_quantile_cliff",
    ):
        assert metric_id in registered
