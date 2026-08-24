# -*- coding: utf-8 -*-
"""R14 estimator-support / hidden-constant fixes.

Covers four audit items from the P1/P2 "support" section:

1. HVG graph-size coverage floor (``hvg_ext.py``): a window that leaves a tiny
   surviving graph (fewer than ``min_nodes`` finite nodes) or covers less than
   ``min_coverage_fraction`` of the nominal window must fail closed to NaN,
   not emit a statistic from a degenerate graph.
2. structural-level scale coverage (``structural_levels.py``): the level
   structure is not comparable with few finite observations — gate on
   ``min_periods`` / ``min_coverage_fraction`` and emit NaN.
3. structural-level hidden cutoff (``structural_levels.py``): the proximity
   bandwidth that decides "what counts as a nearby level" is now an explicit
   ``cutoff`` parameter in the operator contract (not a magic ``0.05``).
4. state-density bandwidth reject-not-clamp (``state_geometry.py``): a
   degenerate bandwidth (zero / negative / non-finite, e.g. an all-identical
   window) outputs NaN instead of being clamped to a tiny epsilon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    from factor_engine.cleaned_operators import load_all

    load_all()


def _frame(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"a": np.asarray(a, dtype=float)})


def _price_series(n: int = 160, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.01, n)
    return 100.0 * np.exp(np.cumsum(ret))


# ---------------------------------------------------------------------------
# 1. HVG graph-size coverage floor
# ---------------------------------------------------------------------------
def test_hvg_min_nodes_floor_fails_closed(_loaded):
    """8 finite nodes pass min_periods but fail the min_nodes=10 floor -> NaN."""
    c = np.full(60, np.nan)
    c[-8:] = np.arange(1.0, 9.0)
    op = OperatorRegistry.get("ts_hvg_degree_entropy", "pandas_numpy")
    out = op.calculate(
        _frame(c), window=50, min_periods=4, min_nodes=10, min_coverage_fraction=0.0
    ).to_numpy()[:, 0]
    assert np.isnan(out[-1])


def test_hvg_low_coverage_fails_closed(_loaded):
    """30 finite nodes cover 25% of a 120-bar window -> NaN (coverage gate)."""
    rng = np.random.default_rng(2)
    c = np.full(120, np.nan)
    c[-30:] = rng.standard_normal(30)
    op = OperatorRegistry.get("ts_hvg_degree_entropy", "pandas_numpy")
    out = op.calculate(
        _frame(c), window=120, min_periods=4, min_nodes=4
    ).to_numpy()[:, 0]
    assert np.isnan(out[-1])


def test_hvg_healthy_window_finite(_loaded):
    """~40 finite nodes with full coverage -> a real (finite) statistic."""
    rng = np.random.default_rng(0)
    op = OperatorRegistry.get("ts_hvg_degree_entropy", "pandas_numpy")
    out = op.calculate(_frame(rng.standard_normal(80)), window=40, min_periods=4).to_numpy()[:, 0]
    assert np.isfinite(out[-1])


def test_hvg_coverage_params_in_signature(_loaded):
    op = OperatorRegistry.get("ts_hvg_degree_entropy", "pandas_numpy")
    assert "min_nodes" in op.metadata.param_names
    assert "min_coverage_fraction" in op.metadata.param_names


# ---------------------------------------------------------------------------
# 2. structural-level scale coverage
# ---------------------------------------------------------------------------
def test_structural_levels_low_coverage_fails_closed(_loaded):
    """Window with ~9 finite observations -> NaN (few finite, low coverage)."""
    n = 160
    price = _price_series(n, seed=7)
    x = np.full(n, np.nan)
    x[:60] = price[:60]
    op = OperatorRegistry.get("ts_structural_level_density", "pandas_numpy")
    out = op.calculate(
        _frame(x), window=40, prominence=0.02, confirmation=3, bandwidth=0.03
    ).to_numpy()[:, 0]
    # trailing window [51:91] has only indices 51..59 finite (9 < min_periods=20)
    assert np.isnan(out[90])


def test_structural_levels_healthy_window_finite(_loaded):
    """Fully-finite trailing window -> the level structure is comparable (finite)."""
    op = OperatorRegistry.get("ts_structural_level_density", "pandas_numpy")
    out = op.calculate(
        _frame(_price_series(160, seed=7)), window=40, prominence=0.02,
        confirmation=3, bandwidth=0.03,
    ).to_numpy()[:, 0]
    assert np.isfinite(out[-1])


def test_structural_levels_coverage_params_in_signature(_loaded):
    for canon in (
        "ts_structural_level_density",
        "ts_nearest_structural_level_distance",
        "ts_structural_level_strength",
    ):
        op = OperatorRegistry.get(canon, "pandas_numpy")
        assert "min_periods" in op.metadata.param_names, canon
        assert "min_coverage_fraction" in op.metadata.param_names, canon


# ---------------------------------------------------------------------------
# 3. structural-level hidden cutoff is a real, visible parameter
# ---------------------------------------------------------------------------
def test_structural_level_strength_cutoff_is_parameter(_loaded):
    op = OperatorRegistry.get("ts_structural_level_strength", "pandas_numpy")
    # the proximity bandwidth is no longer a hidden magic 0.05 — it is in the contract
    assert "cutoff" in op.metadata.param_names
    # and it genuinely changes the output (a real parameter, not a constant)
    frame = _frame(_price_series(160, seed=7))
    base = dict(window=40, prominence=0.02, confirmation=3, decay=0.05)
    out_a = op.calculate(frame, cutoff=0.05, **base).to_numpy()
    out_b = op.calculate(frame, cutoff=0.30, **base).to_numpy()
    assert not np.allclose(
        np.nan_to_num(out_a, nan=0.0), np.nan_to_num(out_b, nan=0.0)
    )


# ---------------------------------------------------------------------------
# 4. state-density bandwidth reject-not-clamp
# ---------------------------------------------------------------------------
def test_state_density_all_identical_degenerate_bandwidth_nan(_loaded):
    """An all-identical history has zero spread -> degenerate bandwidth -> NaN."""
    op = OperatorRegistry.get("ts_state_density", "pandas_numpy")
    out = op.calculate(_frame(np.full(40, 3.0)), window=20, bandwidth=1.0, min_periods=5).to_numpy()[:, 0]
    assert np.isnan(out[-1])


def test_state_density_bandwidth_param_reject_not_clamp(_loaded):
    """A degenerate bandwidth parameter outputs NaN, not a clamped number."""
    rng = np.random.default_rng(1)
    op = OperatorRegistry.get("ts_state_density", "pandas_numpy")
    out = op.calculate(
        _frame(rng.normal(100.0, 2.0, 80)), window=60, bandwidth=0.0, min_periods=5
    ).to_numpy()[:, 0]
    assert np.isnan(out[-1])


def test_state_density_healthy_finite_positive(_loaded):
    """A normal varying window -> a finite, positive density."""
    rng = np.random.default_rng(0)
    op = OperatorRegistry.get("ts_state_density", "pandas_numpy")
    out = op.calculate(
        _frame(rng.normal(100.0, 2.0, 80)), window=60, bandwidth=1.0, min_periods=5
    ).to_numpy()[:, 0]
    tail = out[-3:]
    assert np.all(np.isfinite(tail))
    assert np.all(tail > 0.0)
