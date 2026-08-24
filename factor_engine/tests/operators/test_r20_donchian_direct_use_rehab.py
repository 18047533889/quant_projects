# -*- coding: utf-8 -*-
"""R20 Donchian-family DirectUse rehabilitation oracles.

The raw Donchian upper/lower/mid price levels stay intermediate (price-scale).
These canonicals are the dimensionless / causal counterparts promoted for
direct alpha use:

* ``donchian_width_pct`` — (upper - lower) / close, strict-positive close masked
* ``donchian_channel_position`` — (close - lower) / (upper - lower), NaN on zero width
* ``donchian_breakout_up`` — close / prior-window rolling-max(high) - 1, only on fresh N-bar highs
* ``donchian_breakout_down`` — mirror on fresh N-bar lows

All windows are trailing and END at t (PIT-safe, inclusive of the current bar);
rolling-only (bounded state) => NOT in ``_RECURSIVE_EWM``, and the outputs are
prefix-invariant under truncation.  The module bootstraps only the technical
chain (same import order as ``load_all``) so the KAMA override pin resolves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("donchian_width_pct", "pandas_numpy") is not None:
        return
    from factor_engine.cleaned_operators.technical import signal  # noqa: F401
    from factor_engine.cleaned_operators.technical import polars_signal  # noqa: F401
    from factor_engine.cleaned_operators import composite_fastpath  # noqa: F401
    from factor_engine.cleaned_operators.technical import indicators_v2  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()


def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


def _ohlc(values, spread=1.0):
    close = pd.DataFrame({"A": [float(v) for v in values]})
    return close + spread, close - spread, close


def _trending_panel(n=40, seed=7):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.05, 0.6, n))})
    high = close + 0.5
    low = close - 0.5
    return high, low, close


def test_donchian_width_pct_matches_manual_oracle():
    high, low, close = _trending_panel()
    w = 5
    upper = high.rolling(w, min_periods=w).max()
    lower = low.rolling(w, min_periods=w).min()
    expected = (upper - lower) / close.where(close > 0.0)
    out = _op("donchian_width_pct").calculate(high, low, close, window=w)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: first w-1 bars are NaN.
    assert out.iloc[: w - 1].isna().all().all()
    assert out.iloc[w:].notna().all().all()


def test_donchian_width_pct_masks_non_positive_close():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 30), spread=0.4)
    close.iloc[10, 0] = -1.0
    out = _op("donchian_width_pct").calculate(high, low, close, window=5)
    assert np.isnan(out.iloc[10, 0])
    assert np.isfinite(out.iloc[-1, 0])


def test_donchian_width_pct_is_strictly_positive():
    high, low, close = _trending_panel(n=60, seed=3)
    out = _op("donchian_width_pct").calculate(high, low, close, window=8)
    valid = out.dropna()
    assert (valid > 0.0).all().all()


def test_donchian_channel_position_matches_manual_oracle():
    high, low, close = _trending_panel()
    w = 5
    upper = high.rolling(w, min_periods=w).max()
    lower = low.rolling(w, min_periods=w).min()
    expected = (close - lower) / (upper - lower).replace(0.0, np.nan)
    out = _op("donchian_channel_position").calculate(high, low, close, window=w)
    pd.testing.assert_frame_equal(out, expected)


def test_donchian_channel_position_is_nan_on_zero_width():
    # Constant high == low == close keeps width == 0 => NaN, never inf/0.
    frame = pd.DataFrame({"A": [10.0] * 10})
    out = _op("donchian_channel_position").calculate(frame, frame, frame, window=4)
    assert out.iloc[3:].isna().all().all()


def test_donchian_breakout_up_fresh_high_semantics():
    # Deterministic rally: every bar sets a new 3-bar high after warmup.
    close = pd.DataFrame({"A": np.linspace(10.0, 20.0, 12)})
    high = close + 0.1
    low = close - 0.5
    out = _op("donchian_breakout_up").calculate(high, low, close, window=3)
    upper = high.rolling(3, min_periods=3).max().shift(1)
    expected = (close / upper - 1.0).where(close >= upper)
    # Every post-warmup bar breaks the PRIOR channel: strictly positive
    # magnitude, never clipped by the inclusive-t window.
    pd.testing.assert_frame_equal(out.iloc[3:], expected.iloc[3:])
    assert (out.iloc[3:] > 0.0).all().all()
    # Warmup: the prior channel is rolling(w).max().shift(1), so it is first
    # valid at index w — indices 0..w-1 are NaN (here w=3).
    assert out.iloc[:3].isna().all().all()


def test_donchian_breakout_up_is_zero_without_fresh_high():
    # Monotonically falling closes never make a new 4-bar high.
    close = pd.DataFrame({"A": np.linspace(20.0, 10.0, 15)})
    high = close + 0.5
    low = close - 0.5
    out = _op("donchian_breakout_up").calculate(high, low, close, window=4)
    assert np.allclose(out.iloc[4:].to_numpy(), 0.0)
    assert out.iloc[:4].isna().all().all()


def test_donchian_breakout_down_is_mirror_and_nonpositive():
    close = pd.DataFrame({"A": np.linspace(20.0, 10.0, 15)})
    high = close + 0.5
    low = close - 0.5
    out = _op("donchian_breakout_down").calculate(high, low, close, window=4)
    lower = low.rolling(4, min_periods=4).min().shift(1)
    expected = (close / lower - 1.0).where(close <= lower)
    pd.testing.assert_frame_equal(out.iloc[4:], expected.iloc[4:])
    assert (out.iloc[4:] <= 0.0).all().all()
    # Fresh lows => strictly negative; never 0 unless close == lower.
    assert (out.iloc[4:] < 0.0).all().all()
    # No fresh highs on this panel: breakout_up stays exactly 0 post-warmup.
    up = _op("donchian_breakout_up").calculate(high, low, close, window=4)
    assert np.allclose(up.iloc[4:].to_numpy(), 0.0)
    assert out.iloc[:4].isna().all().all()


def test_donchian_breakout_up_masks_non_positive_close():
    close = pd.DataFrame({"A": np.linspace(20.0, 10.0, 12)})
    high = close + 0.5
    low = close - 0.5
    close.iloc[8, 0] = -1.0
    out = _op("donchian_breakout_up").calculate(high, low, close, window=4)
    assert np.isnan(out.iloc[8, 0])
    down = _op("donchian_breakout_down").calculate(high, low, close, window=4)
    assert np.isnan(down.iloc[8, 0])


def test_donchian_family_is_prefix_invariant_under_truncation():
    # Rolling windows end at t and carry bounded state: a truncated panel must
    # reproduce every overlapping prefix value bit-exactly (PIT-safe, no leak).
    high, low, close = _trending_panel(n=50, seed=11)
    for name in ("donchian_width_pct", "donchian_channel_position",
                 "donchian_breakout_up", "donchian_breakout_down"):
        full = _op(name).calculate(high, low, close, window=6)
        head = _op(name).calculate(high.iloc[:35], low.iloc[:35], close.iloc[:35], window=6)
        pd.testing.assert_frame_equal(full.iloc[:35], head)


def test_donchian_family_param_specs_and_relative_alpha_classification():
    expected_specs = {
        "donchian_width_pct": {"window"},
        "donchian_channel_position": {"window"},
        "donchian_breakout_up": {"window"},
        "donchian_breakout_down": {"window"},
    }
    for name, params in expected_specs.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is not None, f"{name}.{p} has no dtype"

    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {
        "donchian_width_pct", "donchian_channel_position",
        "donchian_breakout_up", "donchian_breakout_down",
    }
    assert promoted <= _RELATIVE_ALPHA_OPS
    # Raw price-scale Donchian levels must NOT be promoted by this slice.
    for raw in ("donchian_upper", "donchian_lower", "donchian_mid"):
        assert raw not in _RELATIVE_ALPHA_OPS


def test_donchian_family_not_recursive_ewm_and_daily_surface():
    # Rolling-only: bounded state, no full_replay governance tag.
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    for name in ("donchian_width_pct", "donchian_channel_position",
                 "donchian_breakout_up", "donchian_breakout_down"):
        assert name not in _RECURSIVE_EWM
        assert "stateful" not in _op(name).metadata.tags

    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    for name in ("donchian_width_pct", "donchian_channel_position",
                 "donchian_breakout_up", "donchian_breakout_down"):
        assert classify_canonical(name) == "daily"
