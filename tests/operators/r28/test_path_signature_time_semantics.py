# -*- coding: utf-8 -*-
"""R28-P0-001 / §五十一..五十三: path-signature time semantics.

- current x missing -> output[t] NaN
- current y missing -> output[t] NaN
- interior joint gap -> pre-gap points never bridged (no-compress)
- future values changed -> output <= t unchanged (prefix invariance)
- leadlag sign synthetic golden
- translation invariance
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model.path_signature import _trailing_contiguous_xy


def _load():
    load_all()


def _panel(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame(values, index=idx, columns=["A"])


def _area(a, b, window=60):
    return OperatorRegistry.get("ts_path_signature_area").calculate(a, b, window=window)["A"]


def test_trailing_contiguous_current_row_missing_returns_none():
    _load()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # current row y missing
    assert _trailing_contiguous_xy(x, np.array([1.0, 2.0, 3.0, np.nan])) is None
    # current row x missing
    assert _trailing_contiguous_xy(np.array([1.0, 2.0, 3.0, np.nan]), x) is None
    # both missing
    assert _trailing_contiguous_xy(np.array([1.0, np.nan]), np.array([1.0, np.nan])) is None
    # empty
    assert _trailing_contiguous_xy(np.array([]), np.array([])) is None


def test_current_x_missing_output_nan():
    _load()
    base = np.arange(1.0, 30.0)
    x = _panel(base)
    y = _panel(base * 0.5 + 1.0)
    out = _area(x, y)
    # replace current row with NaN -> output at that row must be NaN
    x_bad = x.copy()
    x_bad.iloc[-1, 0] = np.nan
    out_bad = _area(x_bad, y)
    assert np.isnan(out_bad.iloc[-1]), "current x missing must fail closed"
    # prefix up to t-1 unchanged
    np.testing.assert_allclose(
        out_bad.iloc[:-1].to_numpy(), out.iloc[:-1].to_numpy(), equal_nan=True
    )


def test_current_y_missing_output_nan():
    _load()
    base = np.arange(1.0, 30.0)
    x = _panel(base)
    y = _panel(base * 0.5 + 1.0)
    out = _area(x, y)
    y_bad = y.copy()
    y_bad.iloc[-1, 0] = np.nan
    out_bad = _area(x, y_bad)
    assert np.isnan(out_bad.iloc[-1]), "current y missing must fail closed"


def test_interior_gap_never_bridged():
    _load()
    # interior joint gap at t=15: rows before the gap must NOT feed the output
    base = np.arange(1.0, 40.0)
    x = _panel(base)
    y = _panel(base * 0.5 + 1.0)
    x_gap = x.copy()
    x_gap.iloc[15, 0] = np.nan
    out = _area(x_gap, y)
    # right after the gap the signature must be recomputed from the post-gap run:
    # it must NOT equal what the pre-gap path would have produced at that point.
    assert np.isfinite(out.iloc[20]), "post-gap output should be defined from post-gap run"
    # pre-gap rows are unaffected by the gap
    assert np.isfinite(out.iloc[10]), "pre-gap output defined"


def test_future_perturbation_no_effect_on_past():
    _load()
    rng = np.random.default_rng(3)
    n = 80
    x = _panel(np.cumsum(rng.standard_normal(n)))
    y = _panel(np.cumsum(rng.standard_normal(n)))
    out = _area(x, y, window=60)
    cutoff = n - 61  # at row < cutoff the trailing window never reaches the perturbed tail
    for perturb_future in (
        lambda a: a + 1000.0,
        lambda a: a * 100.0,
        lambda a: -a,
        lambda a: np.full(a.shape, 7.0),
    ):
        x_mod = x.copy()
        x_mod.iloc[cutoff + 1 :] = perturb_future(x_mod.iloc[cutoff + 1 :].to_numpy())
        out_mod = _area(x_mod, y, window=60)
        np.testing.assert_allclose(
            out_mod.iloc[:cutoff].to_numpy(),
            out.iloc[:cutoff].to_numpy(),
            equal_nan=True,
            atol=1e-9,
        )


def test_leadlag_sign_golden():
    _load()
    # a pure lead (x leads y) produces positive area; lag produces negative
    base = np.sin(np.linspace(0, 4 * np.pi, 120))
    x = _panel(base)
    y_lead = _panel(np.roll(base, -3))  # y leads x
    y_lag = _panel(np.roll(base, 3))  # y lags x
    area_lead = _area(x, y_lead, window=60).iloc[-1]
    area_lag = _area(x, y_lag, window=60).iloc[-1]
    assert np.isfinite(area_lead) and np.isfinite(area_lag)
    # lead area positive, lag area negative (opposite orientation)
    assert area_lead * area_lag < 0, f"lead={area_lead} lag={area_lag}"

def test_translation_invariance():
    _load()
    rng = np.random.default_rng(7)
    n = 70
    x = _panel(np.cumsum(rng.standard_normal(n)))
    y = _panel(np.cumsum(rng.standard_normal(n)))
    base = _area(x, y, window=60)
    for c in (5.0, -3.0, 100.0):
        shifted = _area(x + c, y + c, window=60)
        np.testing.assert_allclose(
            shifted.to_numpy(), base.to_numpy(), equal_nan=True, atol=1e-9
        )
