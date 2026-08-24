# -*- coding: utf-8 -*-
"""R15 master-prompt operator fixes — golden tests for the concrete P0 items.

Covers:
* R15-INC-171/172  KNN local operators: default ``k`` is the DOF floor (20 for
  the 3-feature default), ParamSpec min enforces it, and the effective
  neighbourhood is ``max(min_neigh, k)`` — never a silent smaller-k fit.
* R15-INC-174  rank-copula MI Miller–Madow correction sign: for independent
  data the corrected MI must be ≤ the upward-biased plug-in estimate and ≈ 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all


def _panel(data: np.ndarray) -> pd.DataFrame:
    cols = [f"s{i}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=cols)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


# ---------------------------------------------------------------------------
# R15-INC-174: rank-copula MI Miller–Madow correction sign.
# ---------------------------------------------------------------------------

def _plug_in_mi(a: np.ndarray, b: np.ndarray, grid: int, alpha: float = 0.5) -> float:
    """Raw plug-in MI (Jeffreys-smoothed joint, NO Miller–Madow correction)."""
    n = a.shape[0]
    from factor_engine.cleaned_operators.cross_section_local import _rank_transform

    u = np.clip((_rank_transform(np.stack([a, b], axis=1))[:, 0] * grid).astype(int), 0, grid - 1)
    v = np.clip((_rank_transform(np.stack([a, b], axis=1))[:, 1] * grid).astype(int), 0, grid - 1)
    raw = np.zeros((grid, grid), dtype=np.float64)
    for i in range(u.shape[0]):
        raw[u[i], v[i]] += 1.0
    joint = raw + alpha
    joint /= joint.sum()
    pu = joint.sum(axis=1)
    pv = joint.sum(axis=0)
    mi = 0.0
    for i in range(grid):
        for j in range(grid):
            p = joint[i, j]
            denom = pu[i] * pv[j]
            if p <= 1e-12 or denom <= 1e-12:
                continue
            mi += p * np.log(p / denom)
    return mi


def test_rank_copula_mi_miller_madow_sign_independent() -> None:
    """R15-INC-174: on INDEPENDENT uniforms the plug-in MI is upward biased;
    the Miller–Madow-corrected MI must be SMALLER (damped toward 0), not
    inflated by the pre-R15 opposite-sign correction."""
    from factor_engine.cleaned_operators.cross_section_local import _copula_cross_series

    rng = np.random.default_rng(2026)
    n, grid = 4000, 8
    a = _panel(rng.uniform(0.0, 1.0, n).reshape(1, -1))
    b = _panel(rng.uniform(0.0, 1.0, n).reshape(1, -1))
    corrected = float(
        _copula_cross_series(
            a.to_numpy(dtype=float), b.to_numpy(dtype=float), grid, entropy=False
        )[0, 0]
    )
    plug_in = _plug_in_mi(a.to_numpy(dtype=float).ravel(), b.to_numpy(dtype=float).ravel(), grid)
    # corrected must be <= plug-in (the correction subtracts, never adds)
    assert corrected <= plug_in + 1e-9
    # and both should be near zero for independent data
    assert abs(corrected) < 0.02
    assert abs(plug_in) < 0.05


def test_rank_copula_mi_positive_control_dependent() -> None:
    """R15-INC-174 positive control: strongly dependent data gives MI well above
    the independent null, so the correction does not over-damp a real signal."""
    from factor_engine.cleaned_operators.cross_section_local import _copula_cross_series

    rng = np.random.default_rng(7)
    n, grid = 4000, 8
    x = rng.uniform(0.0, 1.0, n)
    y = np.clip(x + 0.05 * rng.normal(size=n), 0.0, 1.0)
    a = _panel(x.reshape(1, -1))
    b = _panel(y.reshape(1, -1))
    mi = float(
        _copula_cross_series(
            a.to_numpy(dtype=float), b.to_numpy(dtype=float), grid, entropy=False
        )[0, 0]
    )
    assert mi > 0.05


# ---------------------------------------------------------------------------
# R15-INC-123/124: intraday session completeness is a multiset + grid check.
# ---------------------------------------------------------------------------

def _complete_session_runs(duplicate_slot: bool = False, off_grid_second: bool = False):
    """Build one full A-share 240-bar session and run ``_session_runs``.

    ``duplicate_slot`` inserts a second row at the same minute; ``off_grid_second``
    replaces the final row's timestamp with a stray seconds-level bar."""
    import numpy as np

    from factor_engine.cleaned_operators.intraday_session import _official_grid, _session_runs
    from factor_engine.runtime.session_calendar import SessionCalendar

    cal = SessionCalendar(market="CN", timestamp_convention="bar_start", bar_freq="1min")
    expected, close_mod, slot_set = _official_grid(cal)
    slots = sorted(slot_set)
    # base minute-of-day slots (official grid)
    base = slots[:]
    if duplicate_slot:
        base = slots[:-1] + [slots[-2]]  # replace last with a duplicate of the prev
    elif off_grid_second:
        pass
    n = len(base)
    x_col = np.arange(n, dtype=float)
    sid_col = np.ones(n, dtype=float)
    dates = np.ones(n, dtype=float) * 20000.0
    mods = np.asarray(base, dtype=float)
    on_grid = np.ones(n, dtype=bool)
    if off_grid_second:
        # a stray 09:46:30 bar floor-maps to 09:46 — off the minute grid
        on_grid[-1] = False
    return _session_runs(x_col, sid_col, dates, mods, expected, close_mod, slot_set, on_grid)


def test_session_duplicate_slot_fails_completeness() -> None:
    """R15-INC-123: a duplicated minute bar has the same SET as a complete
    session but a different count — it must never certify as complete."""
    runs = _complete_session_runs(duplicate_slot=True)
    assert len(runs) == 1
    assert runs[0]["completed"] is False


def test_session_off_grid_second_fails_completeness() -> None:
    """R15-INC-124: a stray seconds-level bar floor-mapped to a valid minute
    must not certify a session."""
    runs = _complete_session_runs(off_grid_second=True)
    assert len(runs) == 1
    assert runs[0]["completed"] is False


def test_session_full_grid_completes() -> None:
    """A genuine full 240-bar session is still completed (positive control)."""
    runs = _complete_session_runs()
    assert len(runs) == 1
    assert runs[0]["completed"] is True


# ---------------------------------------------------------------------------
# R15-INC-171/172: KNN default viability + ParamSpec floor.
# ---------------------------------------------------------------------------

def test_knn_default_k_is_dof_floor() -> None:
    """Default ``k`` must be the DOF floor (20 for the 3-feature operators) so a
    default call is not a guaranteed-NaN dead region."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for canonical in (
        "cs_knn_local_linear_residual",
        "cs_knn_local_gradient_norm",
        "cs_knn_tangent_residual",
    ):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        spec = op.metadata.param_specs.get("k")
        assert spec is not None, canonical
        assert spec.min == 20, (canonical, spec.min)
        assert spec.default == 20, (canonical, spec.default)
        assert spec.param_role.value == "estimator_resolution", canonical


def test_knn_effective_neighbourhood_is_max() -> None:
    """R15-INC-171: the kernel fits on ``max(min_neigh, k)`` peers — a k below
    the DOF floor must not silently degrade into a smaller fit.  With a
    cross-section of 40 valid stocks (≫ 20) the default k=20 produces finite
    values (the pre-R15 k=10 default was all-NaN for typical breadths)."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(3)
    rows, n = 1, 40  # a single cross-section day
    target = pd.DataFrame(rng.normal(size=(rows, n)), columns=[f"s{i}" for i in range(n)])
    feats = [
        pd.DataFrame(rng.normal(size=(rows, n)), columns=[f"s{i}" for i in range(n)])
        for _ in range(3)
    ]
    op = OperatorRegistry.get("cs_knn_local_linear_residual", "pandas_numpy")
    out = op._calculate_series(target, feats[0], feats[1], feats[2])
    finite = int(np.isfinite(out.to_numpy(dtype=float)).sum())
    assert finite > 0, "default k must yield finite values on a 40-name cross-section"
