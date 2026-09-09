"""Independent numerical regressions for V9-M26."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import bds

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.research_spectral import (
    _bds_common_center_probability,
    _bds_statistic,
    _ts_bds_statistic,
)


def test_common_center_three_point_wedge_is_not_a_clique():
    values = np.array([-0.75, 0.0, 0.75])
    # Edges are left-center and center-right.  The two endpoints are not
    # adjacent, so clique probability is zero while the common-center wedge
    # probability is 2 / (3*2*1) = 1/3.
    assert _bds_common_center_probability(values, 1.0) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("embedding_dim", [2, 3, 4, 5])
def test_bds_matches_independent_statsmodels_conditioned_reference(embedding_dim):
    values = np.random.default_rng(2600 + embedding_dim).normal(size=120)
    multiplier = 1.5
    epsilon = multiplier * np.std(values, ddof=0)
    reference, _ = bds(values, max_dim=embedding_dim, epsilon=epsilon)
    reference = np.asarray(reference).reshape(-1)[-1]
    assert _bds_statistic(values, embedding_dim, multiplier) == pytest.approx(
        reference, rel=2e-12, abs=2e-12
    )


def test_bds_unit_invariance_constant_invalid_dimension_and_strict_boundary():
    values = np.random.default_rng(26).normal(size=80)
    original = _bds_statistic(values, 3, 1.0)
    for signed_scale in (1e-300, -1e-300, 1e300, -1e300):
        assert _bds_statistic(values * signed_scale, 3, 1.0) == pytest.approx(
            original, abs=2e-12
        )
    assert np.isnan(_bds_statistic(np.ones(80), 2, 1.5))
    assert np.isnan(_bds_statistic(values, values.size, 1.5))
    assert np.isnan(_bds_statistic(values, values.size + 1, 1.5))
    boundary = np.array([0.0, 1.0, 2.0])
    assert _bds_common_center_probability(boundary, 1.0) == 0.0


@pytest.mark.parametrize("seed", [261, 262, 263])
@pytest.mark.parametrize("origin", [1e300, -1e300])
def test_bds_large_offset_preserves_original_unit_deviation_graph(seed, origin):
    base = np.random.default_rng(seed).normal(size=80)
    shifted = origin + np.round(base * 16.0) * np.spacing(abs(origin))
    representable_deviations = shifted - shifted[0]
    expected = _bds_statistic(representable_deviations, 2, 1.5)
    assert _bds_statistic(shifted, 2, 1.5) == pytest.approx(expected, abs=2e-12)


def test_bds_public_calculate_matches_stable_kernel_at_large_offset():
    base = np.random.default_rng(261).normal(size=80)
    origin = 1e300
    shifted = origin + np.round(base * 16.0) * np.spacing(origin)

    ensure_cleaned_loaded()
    frame = pd.DataFrame({"A": shifted})
    public = OperatorRegistry.get("ts_bds_statistic", "pandas_numpy", mode="research")
    actual = public.calculate(
        frame, window=40, embedding_dim=2, distance_multiplier=1.5
    )
    direct = _ts_bds_statistic(
        frame, window=40, embedding_dim=2, distance_multiplier=1.5
    )
    pd.testing.assert_frame_equal(actual, direct)


def test_bds_opposite_sign_overflow_uses_bounded_fallback():
    values = np.random.default_rng(264).uniform(-1.0, 1.0, size=80) * 1e308
    bounded = values / np.max(np.abs(values))
    expected = _bds_statistic(bounded - bounded[0], 2, 1.5)
    assert _bds_statistic(values, 2, 1.5) == pytest.approx(expected, abs=2e-12)


def test_outer_window_gap_is_not_compressed_and_prefix_is_stable():
    ensure_cleaned_loaded()
    public = OperatorRegistry.get("ts_bds_statistic", "pandas_numpy", mode="research")
    values = np.random.default_rng(260).normal(size=80)
    frame = pd.DataFrame({"A": values})
    params = {"window": 40, "embedding_dim": 2, "distance_multiplier": 1.5}
    full = public.calculate(frame, **params)
    prefix = public.calculate(frame.iloc[:60], **params)
    pd.testing.assert_series_equal(full["A"].iloc[:60], prefix["A"])
    with_gap = frame.copy()
    with_gap.iloc[55, 0] = np.nan
    gap_result = public.calculate(with_gap, **params)
    assert gap_result["A"].iloc[55:80].isna().all()
