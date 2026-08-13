# -*- coding: utf-8 -*-
"""Hand-computed numeric fixtures for the alpha-language operators.

Each test asserts a known-output value from the operator specification, so a
regression is caught even if the panel-level contract tests stay green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _panel(vals):
    arr = np.asarray(vals, dtype=float).reshape(-1, 1)
    return pd.DataFrame(arr, index=range(arr.shape[0]), columns=["S0"])


def _col(vals):
    return np.round(_panel(vals)["S0"].to_numpy(), 6)


def _run(op, *frames, **kw):
    inst = OperatorRegistry.get(op, "pandas_numpy")
    f = [v if isinstance(v, pd.DataFrame) else _panel(v) for v in frames]
    return np.round(inst.calculate(*f, **kw)["S0"].to_numpy(), 6)


@pytest.mark.parametrize("op,kw,frames,expected,atol", [
    # ---- temporal state ----
    ("ts_hysteresis_state", dict(upper=1.0, lower=0.5), ([0.5, 1.2, 0.8, 0.3, -1.3, -0.5],), [0, 1, 1, 0, -1, -1], 0),
    ("ts_run_strength", dict(max_run=20), ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]), [1, 3, 6, 10, 15], 0),
    ("ts_run_efficiency", dict(max_run=20), ([1, -1, 2, -2, 3], [1, 1, 1, 1, 1]), [np.nan, 0.0, 0.5, 0.0, 1 / 3], 1e-6),
    ("ts_run_concentration", dict(max_run=20), ([1, 1, 1, 5, 1], [1, 1, 1, 1, 1]), [np.nan, 0.5, 1 / 3, 5 / 8, 5 / 9], 1e-6),
    ("ts_state_entry_strength", dict(upper=1.0, lower=0.5), ([0.5, 1.2, 0.8, 0.3, -1.3],), [0.0, 0.2, 0.2, 0.0, -0.3], 1e-6),
    ("ts_transition_intensity", dict(window=5), ([1, 2, 1, 2, 1], [1, 1, 0, 0, 1]), [np.nan, np.nan, 1.0, 1.0, 1.0], 0),
    # ---- shape geometry ----
    ("ts_monotonicity", dict(window=5), ([1, 2, 3, 4, 5],), [np.nan, np.nan, 1.0, 1.0, 1.0], 0),
    ("ts_turning_rate", dict(window=5), ([1, 2, 3, 4, 5],), [np.nan, np.nan, 0.0, 0.0, 0.0], 0),
    ("ts_path_efficiency", dict(window=5), ([1, 2, 3, 4, 5],), [np.nan, 1.0, 1.0, 1.0, 1.0], 0),
    ("ts_roughness", dict(window=5, min_periods=3), ([1, 2, 3, 4, 5],), [np.nan, np.nan, 0.0, 0.0, 0.0], 0),
    ("ts_trend_break_score", dict(window=6, split=0.5), ([1, 2, 3, 4, 5, 6],), [np.nan, np.nan, np.nan, 0.0, 0.0, 0.0], 0),
    ("ts_weighted_time_centroid", dict(window=4), ([0, 0, 0, 10],), [np.nan, np.nan, np.nan, 1.0], 1e-6),
    ("ts_endpoint_deviation", dict(window=5), ([1, 2, 3, 4, 5],), [np.nan, np.nan, 0.0, 0.0, 0.0], 0),
    ("ts_mass_concentration", dict(window=4), ([1, 1, 1, 1],), [np.nan, 0.0, 0.0, 0.0], 1e-6),
    # ---- distribution ----
    ("ts_location_shift", dict(recent_window=3, old_window=3, min_periods=2), ([1] * 8,), [np.nan] * 5 + [0.0, 0.0, 0.0], 1e-6),
    ("ts_scale_shift", dict(recent_window=3, old_window=3, min_periods=2), ([1] * 8,), [np.nan] * 5 + [0.0, 0.0, 0.0], 1e-6),
    ("ts_ks_shift", dict(recent_window=3, old_window=3, min_periods=2), ([1] * 8,), [np.nan] * 5 + [0.0, 0.0, 0.0], 1e-6),
    ("ts_tail_imbalance", dict(window=8, k=1.0, min_periods=4), ([1, 2, 3, 4, 5, 6, 7, 8],), [np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-6),
    # ---- volatility structure ----
    ("ts_semivariance_balance", dict(window=4, min_periods=2), ([0.01, 0.02, 0.03, 0.01],), [np.nan, 1.0, 1.0, 1.0], 1e-6),
    ("ts_realized_quarticity", dict(window=5, min_periods=3), ([0.01] * 5,), [np.nan, np.nan, 1 / 3, 1 / 3, 1 / 3], 1e-6),
    ("ts_vol_term_structure", dict(short_window=3, long_window=5), ([0.01] * 8,), [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-6),
    ("ts_sign_persistence", dict(window=6, min_periods=3), ([1, -1, 2, -2, 3, -3],), [np.nan, np.nan, np.nan, -1.0, -1.0, -1.0], 1e-6),
    ("ts_vol_clustering", dict(window=3, min_periods=3), ([0.01, 0.02, 0.03],), [np.nan, np.nan, np.nan], 1e-6),
    ("ts_jump_bipower_proxy", dict(window=5, min_periods=3), ([0.01] * 5,), [np.nan, np.nan, 0.0, 0.0, 0.0], 1e-6),
    # ---- events ----
    ("event_frequency", dict(window=5), ([1, 0, 1, 0, 1],), [1.0, 0.5, 2 / 3, 0.5, 0.6], 1e-6),
    ("event_cluster_count", dict(window=7, max_gap=1), ([1, 1, 0, 0, 1, 0, 1],), [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0], 0),
    ("event_cluster_mean_size", dict(window=7, max_gap=1), ([1, 1, 0, 0, 1, 0, 1],), [1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 4 / 3], 1e-6),
])
def test_alpha_language_semantics(op, kw, frames, expected, atol):
    out = _run(op, *frames, **kw)
    assert out.shape == (len(expected),), f"{op}: shape {out.shape} != {len(expected)}"
    np.testing.assert_allclose(out, expected, atol=atol, rtol=1e-6, equal_nan=True)


def test_ts_transition_intensity_normalize_degenerate_mad_is_nan():
    """Constant dx at transitions -> MAD=0 -> fail-closed NaN (was 1e12)."""
    out = _run("ts_transition_intensity", [1, 2, 1, 2, 1], [1, 1, 0, 0, 1], window=5, normalize=True)
    assert np.all(np.isnan(out)), f"expected all NaN, got {out}"


def test_cs_local_curvature_symmetric_is_zero():
    """Perfectly symmetric neighbors -> curvature 0."""
    x = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0], "C": [2.0, 2.0, 2.0]},
        index=range(3),
    )
    inst = OperatorRegistry.get("cs_local_curvature", "pandas_numpy")
    out = inst.calculate(x, k=1)
    # Row 0: A=1,B=3,C=2. For C (value 2): lower=1 (gap 1), upper=3 (gap 1) -> 0.
    assert abs(float(out.loc[0, "C"]) < 1e-9


def test_group_ex_self_quantile_matches_manual():
    x = pd.DataFrame({"A": [1.0, 5.0], "B": [2.0, 6.0], "C": [3.0, 7.0]}, index=range(2))
    g = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0], "C": [1.0, 1.0]}, index=range(2))
    inst = OperatorRegistry.get("group_ex_self_quantile", "pandas_numpy")
    out = inst.calculate(x, g, q=0.5)
    # Row 0 peers of A = {B:2, C:3} median = 2.5
    assert abs(float(out.loc[0, "A"]) - 2.5) < 1e-9
