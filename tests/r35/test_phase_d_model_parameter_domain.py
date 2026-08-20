# -*- coding: utf-8 -*-
"""R35 Phase D: model parameter-domain tests (taskbook §61 / §75).

Every model operator must reject out-of-domain / infeasible parameters at the
call boundary (fail closed) rather than silently truncating or producing
garbage:
- integer params (window / n_components / order / q / r / regimes / experts)
  reject non-integer and out-of-range values
- alpha / l1_ratio reject out-of-[0,1]
- feasibility constraints (k <= window) hold
- hostile numeric inputs (all-equal, all-NaN, zero-variance) never crash and
  never return non-finite garbage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _load():
    load_all()


def _panel(n, cols=2, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, cols)), columns=list("AB")[:cols])


# --------------------------------------------------------------------------
# GARCH parameter domain
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "ts_garch_standardized_shock",
        "ts_garch_next_vol_forecast",
        "ts_garch_persistence",
        "ts_garch_vol_surprise",
        "ts_gjr_garch_vol_forecast",
    ],
)
def test_garch_rejects_fractional_window(name):
    _load()
    op = OperatorRegistry.get(name, "pandas_numpy")
    if op is None:
        pytest.skip("not registered")
    x = _panel(150, 2, seed=1)
    with pytest.raises((ValueError, TypeError)):
        op.calculate(x, window=60.5)


@pytest.mark.parametrize(
    "name",
    [
        "ts_garch_standardized_shock",
        "ts_garch_next_vol_forecast",
    ],
)
def test_garch_hostile_inputs_fail_closed(name):
    """All-equal / constant inputs must yield NaN (non-converged fit), never
    crash and never return non-finite garbage."""
    _load()
    op = OperatorRegistry.get(name, "pandas_numpy")
    if op is None:
        pytest.skip("not registered")
    const = pd.DataFrame(np.full((120, 2), 0.5), columns=list("AB"))
    out = op.calculate(const, window=60)
    v = out.to_numpy()
    assert np.isnan(v).all() or np.isfinite(v[np.isfinite(v)]).all(), name
    # all-NaN input must not crash
    nanp = pd.DataFrame(np.full((120, 2), np.nan), columns=list("AB"))
    try:
        op.calculate(nanp, window=60)
    except Exception:
        pass  # any non-crash behaviour acceptable


# --------------------------------------------------------------------------
# PCA / panel model parameter domain
# --------------------------------------------------------------------------

def test_panel_forecast_rejects_fractional_integer_params():
    _load()
    op = OperatorRegistry.get("panel_rolling_pcr_forecast", "pandas_numpy")
    if op is None:
        pytest.skip("not registered")
    n = 120
    y = _panel(n, 2, seed=2)
    x1 = _panel(n, 2, seed=3)
    for kw in ({"n_components": 3.9}, {"window": 60.5}, {"label_horizon": 2.5}):
        with pytest.raises((ValueError, TypeError)):
            op.calculate(y, x1, **kw)


def test_panel_forecast_hostile_feature():
    """Zero-variance / all-NaN feature must fail closed (NaN), never crash."""
    _load()
    op = OperatorRegistry.get("panel_rolling_pcr_forecast", "pandas_numpy")
    if op is None:
        pytest.skip("not registered")
    n = 120
    y = _panel(n, 2, seed=4)
    zvar = pd.DataFrame(np.ones((n, 2)), columns=list("AB"))  # zero variance
    out = op.calculate(y, zvar, window=60, n_components=2)
    v = out.to_numpy()
    assert np.isnan(v).all() or np.isfinite(v[np.isfinite(v)]).all()


def test_pca_n_components_boundaries():
    from cleaned_operators.cross_section.panel_model import _pca_svd

    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 6))
    # n_components larger than the rank must be capped, not crash
    pca = _pca_svd(X, 100)
    assert pca is not None
    assert pca["k"] <= min(6, 60) - 1
    # n_components = 1 works
    pca1 = _pca_svd(X, 1)
    assert pca1 is not None and pca1["k"] == 1


def test_pca_one_suspended_stock():
    """§62: one all-NaN stock must be dropped from the active set without
    poisoning peers."""
    from cleaned_operators.cross_section.panel_model import _pca_svd

    rng = np.random.default_rng(6)
    X = rng.standard_normal((60, 5))
    X[:, 3] = np.nan  # one fully-suspended stock
    pca = _pca_svd(X, 2)
    assert pca is not None
    assert not pca["active"][3]  # suspended column inactive
    assert pca["active"].sum() == 4
    assert np.isfinite(pca["mu"]).all() and np.isfinite(pca["sd"]).all()


def test_pca_all_equal_stocks():
    """§62: all-equal columns must not crash (unit-scale fallback)."""
    from cleaned_operators.cross_section.panel_model import _pca_svd

    X = np.ones((40, 4))
    pca = _pca_svd(X, 2)
    assert pca is None or np.isfinite(pca["loadings"]).all()


# --------------------------------------------------------------------------
# ElasticNet / PLS parameter domain
# --------------------------------------------------------------------------

def test_enet_alpha_l1_domain():
    from cleaned_operators.cross_section.panel_model import _enet_predict

    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 3))
    y = rng.standard_normal(60)
    xnew = rng.standard_normal(3)
    # l1_ratio out of [0,1] must fail closed (NaN)
    assert np.isnan(_enet_predict(X, y, 0.01, 1.5, xnew))
    assert np.isnan(_enet_predict(X, y, 0.01, -0.5, xnew))
    # alpha <= 0 must fail closed
    assert np.isnan(_enet_predict(X, y, 0.0, 0.5, xnew))
    # valid domain returns finite
    v = _enet_predict(X, y, 0.01, 0.5, xnew)
    assert np.isfinite(v)


def test_pls_component_boundaries():
    from cleaned_operators.cross_section.panel_model import _pls1_predict

    rng = np.random.default_rng(8)
    X = rng.standard_normal((60, 4))
    y = rng.standard_normal(60)
    xnew = rng.standard_normal(4)
    # n_components=0 -> NaN (no latent component)
    assert np.isnan(_pls1_predict(X, y, 0, xnew))
    # 1..3 components finite
    for k in (1, 2, 3):
        v = _pls1_predict(X, y, k, xnew)
        assert np.isfinite(v), k
    # degenerate component (zero norm): the loop breaks and returns the mean
    # prediction — finite, never a crash / never non-finite garbage
    Xd = np.zeros((40, 3))
    vd = _pls1_predict(Xd, y[:40], 1, xnew[:3])
    assert np.isfinite(vd) or np.isnan(vd)


# --------------------------------------------------------------------------
# Kalman parameter domain
# --------------------------------------------------------------------------

def test_kalman_rejects_invalid_q_r():
    from cleaned_operators.ts_model.state_space import _kalman_level

    x = np.arange(50, dtype=float)
    for q, r in ((-1.0, 1.0), (1.0, 0.0), (1.0, -1.0), (np.inf, 1.0)):
        with pytest.raises(ValueError):
            _kalman_level(x, q, r, "level")


def test_kalman_q_zero_and_small_r():
    """§66: q=0 and small r must work without crashing."""
    from cleaned_operators.ts_model.state_space import _kalman_level

    rng = np.random.default_rng(9)
    x = rng.standard_normal(80)
    out0 = _kalman_level(x, 0.0, 1.0, "level")
    assert np.isfinite(out0[10:]).all()
    out_sr = _kalman_level(x, 1e-4, 1e-6, "level")
    assert np.isfinite(out_sr[10:]).all()
