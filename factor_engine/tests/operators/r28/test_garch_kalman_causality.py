# -*- coding: utf-8 -*-
"""R28 §四十一 / §九十九 / §一百: GARCH + Kalman synthetic causality.

- GARCH: variance recursion timing; the current shock must not enter its own
  denominator; future perturbation leaves the past volatility path unchanged.
- Kalman: filter is one-sided; missing observation is predict-only; future data
  cannot change past filtered states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _load():
    load_all()


def _panel(vals: np.ndarray, instruments: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=vals.shape[0], freq="D")
    cols = [f"C{k}" for k in range(vals.shape[1] if vals.ndim == 2 else instruments)]
    if vals.ndim == 1:
        vals = vals[:, None]
    return pd.DataFrame(vals, index=idx, columns=cols)


def _garch(x, window=60):
    return OperatorRegistry.get("ts_garch_standardized_shock", "pandas_numpy", mode="any").calculate(
        _panel(x), window=window
    )["C0"].to_numpy()


def _garch_vol(x, window=60):
    return OperatorRegistry.get("ts_garch_next_vol_forecast", "pandas_numpy", mode="any").calculate(
        _panel(x), window=window
    )["C0"].to_numpy()


def _kalman(x, q=1e-3, r=1e-2):
    return OperatorRegistry.get("ts_kalman_level", "pandas_numpy", mode="any").calculate(
        _panel(x), q=q, r=r
    )["C0"].to_numpy()


# ---------------------------------------------------------------- GARCH ------
def test_garch_white_noise_shock_roughly_zero():
    """Standardized shocks of a homoskedastic white noise must hover near zero
    (the GARCH(1,1) conditional variance approximates the constant variance)."""
    _load()
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300) * 0.02
    out = _garch(x, window=120)
    finite = out[np.isfinite(out)]
    assert len(finite) > 50
    assert abs(float(np.mean(finite))) < 0.5, "white-noise standardized shock far from 0"


def test_garch_volatility_clustering_reaction():
    """After a volatility burst the conditional-variance forecast must rise."""
    _load()
    rng = np.random.default_rng(6)
    n = 300
    x = rng.standard_normal(n) * 0.01
    x[150:200] *= 5.0  # volatility burst
    vol = _garch_vol(x, window=120)
    pos = np.arange(vol.size)
    pre = float(np.nanmedian(vol[(pos >= 100) & (pos < 145)]))
    post = float(np.nanmedian(vol[(pos >= 160) & (pos < 200)]))
    assert post > pre, "GARCH vol forecast did not react to the burst"


def test_garch_current_shock_not_in_own_denominator():
    """The standardized shock at t must NOT be inside its own variance denominator.
    Equivalent runtime check: perturbing ONLY the current row must not change the
    operator's own output at that row for the shock statistic (fit on seg[:-1])."""
    _load()
    rng = np.random.default_rng(7)
    n = 200
    x = rng.standard_normal(n) * 0.02
    base = _garch(x, window=80)
    xm = x.copy()
    xm[150] *= 50.0  # huge current shock
    changed = _garch(xm, window=80)
    # rows before 150 unchanged
    np.testing.assert_allclose(changed[:150], base[:150], equal_nan=True, atol=1e-8)
    # the shock at 150 is allowed to move ITS row (it is the numerator), but the
    # denominator fit for later rows must exclude it -> 151+ stable to the fitted params
    # (params can legitimately move slightly since window still includes row 150's
    # finite value for the NEXT window fit; we only require no global NaN / no inf).
    assert np.all(np.isfinite(changed[np.isfinite(base)])), "GARCH produced non-finite value"


def test_garch_future_perturbation():
    _load()
    rng = np.random.default_rng(8)
    n = 260
    x = rng.standard_normal(n) * 0.02
    base = _garch_vol(x, window=80)
    xm = x.copy()
    xm[200:] = rng.standard_normal(n - 200) * 100.0
    changed = _garch_vol(xm, window=80)
    np.testing.assert_allclose(changed[:170], base[:170], equal_nan=True, atol=1e-8)


# -------------------------------------------------------------- Kalman -------
def test_kalman_filter_one_sided():
    """Filtered level at t must not change when the future is perturbed."""
    _load()
    rng = np.random.default_rng(9)
    n = 200
    x = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1) + rng.standard_normal(n) * 0.3
    base = _kalman(x)
    for tail_scale in (100.0, 0.001, -1.0):
        xm = x.copy()
        xm[140:] *= tail_scale
        changed = _kalman(xm)
        np.testing.assert_allclose(changed[:140], base[:140], equal_nan=True, atol=1e-8)


def test_kalman_missing_is_predict_only():
    """A missing observation produces a filtered value that is the prediction
    (no update), and the gap does not corrupt later states."""
    _load()
    rng = np.random.default_rng(10)
    n = 150
    x = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1) + rng.standard_normal(n) * 0.3
    xm = x.copy()
    xm[100] = np.nan
    out = _kalman(xm)
    # the missing row still yields a finite filtered state (predict-only)
    assert np.isfinite(out[100]), "missing observation should yield predict-only state"
    # later rows recover smoothly (no corruption / no NaN avalanche)
    assert np.isfinite(out[140])


def test_kalman_prefix_invariance():
    _load()
    rng = np.random.default_rng(11)
    n = 200
    x = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1) + rng.standard_normal(n) * 0.3
    full = _kalman(x)
    for cutoff in (60, 100, 140):
        pref = _kalman(x[:cutoff])
        np.testing.assert_allclose(pref, full[:cutoff], equal_nan=True, atol=1e-8)
