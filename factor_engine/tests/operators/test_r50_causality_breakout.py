# -*- coding: utf-8 -*-
"""R50 causality (future-leak) oracles for the breakout / support-resistance /
candlestick / limit operator families.

Rehabilitation Rule 3: a "today's breakout" reference must use ONLY ``[t-L, t-1]``
(strict prior), never include t itself or future confirmation.  Support/resistance
must use only structure known as of t.  Candlestick binary patterns are
EVENT/CONDITION, not terminal alphas.

The "future poison is causal" test pattern: run the operator on a prefix, then
re-run it on the prefix PLUS a large future value (poison).  If the operator is
causal, the prefix output must be bit-identical (the future value cannot reach
backwards).  Any prefix row that changes proves a future leak.

All operators audited here were verified causal by this oracle; the test locks
that property in so a future regression cannot silently reintroduce look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("KAMA", "pandas_numpy") is not None:
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


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame({"A": [float(v) for v in values]})


def _ohlc_det(values):
    """Deterministic OHLC panel: high=close+0.1, low=close-0.1, open=close.

    Deterministic (no RNG) so the prefix is identical whether or not a poison
    row is appended — the only difference between the two runs is the future
    poison value itself.
    """
    close = _frame(values)
    high = close + 0.1
    low = close - 0.1
    open_ = close.copy()
    return open_, high, low, close


def _assert_causal(name, prefix_args, full_args, n):
    """Run on prefix and on prefix+poison; prefix output must be unchanged."""
    op = _op(name)
    out_prefix = pd.DataFrame(op.calculate(*prefix_args))
    out_full = pd.DataFrame(op.calculate(*full_args))
    a = out_prefix.iloc[:n, 0].to_numpy(dtype=float)
    b = out_full.iloc[:n, 0].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True,
                               err_msg=f"{name} leaks future data into its prefix")


# ---------------------------------------------------------------------------
# Breakout / new-high-low / prev-extreme family (technical_extensions +
# polars_misc_v2).  Reference window is [t-w, t-1] via shift(1) rolling.
# ---------------------------------------------------------------------------
def _breakout_cases(n=40, w=5):
    base = [10.0 + 2.0 * np.sin(i * 0.5) + 0.05 * i for i in range(n)]
    poisoned = base + [1e6]
    return base, poisoned, w


@pytest.mark.parametrize("name", [
    "ts_prev_high", "ts_prev_low",
    "ts_breakout_high", "ts_breakdown_low",
    "ts_new_high", "ts_new_low",
    "ts_distance_to_high", "ts_distance_to_low",
    "ts_channel_position", "ts_days_since_high", "ts_days_since_low",
])
def test_breakout_family_future_poison_is_causal(name):
    base, poisoned, w = _breakout_cases()
    _assert_causal(name, (_frame(base), w), (_frame(poisoned), w), len(base))


# ---------------------------------------------------------------------------
# Confirmed-pivot / support / resistance family (polars_structure +
# technical_structure_repairs + structure_patterns_v2).  Pivots emit only on
# their confirmation timestamp; levels use only confirmed pivots known as of t.
# ---------------------------------------------------------------------------
def _pivot_cases(n=40, lw=2, rw=2, hw=12, pts=3):
    base = [10.0 + 2.0 * np.sin(i * 0.5) + 0.05 * i for i in range(n)]
    poisoned = base + [1e6]
    return base, poisoned, lw, rw, hw, pts


@pytest.mark.parametrize("name", [
    "ts_nth_pivot_high", "ts_nth_pivot_low",
    "ts_resistance_level", "ts_support_level",
    "ts_resistance_slope", "ts_support_slope",
    "ts_resistance_fit_r2", "ts_support_fit_r2",
])
def test_pivot_family_future_poison_is_causal(name):
    base, poisoned, lw, rw, hw, pts = _pivot_cases()
    _assert_causal(name, (_frame(base), lw, rw, hw, pts),
                   (_frame(poisoned), lw, rw, hw, pts), len(base))


@pytest.mark.parametrize("name", [
    "ts_last_pivot_high", "ts_last_pivot_low",
    "ts_pivot_high_count", "ts_pivot_low_count",
    "ts_pivot_high_spacing", "ts_pivot_low_spacing",
])
def test_pivot_4param_family_future_poison_is_causal(name):
    # These operators take (x, left_window, right_window, history_window) — no
    # ``points``/``n`` parameter.
    base, poisoned, lw, rw, hw, _pts = _pivot_cases()
    _assert_causal(name, (_frame(base), lw, rw, hw),
                   (_frame(poisoned), lw, rw, hw), len(base))


@pytest.mark.parametrize("name", [
    "ts_resistance_break", "ts_support_break",
    "ts_distance_to_resistance", "ts_distance_to_support",
])
def test_resistance_break_family_future_poison_is_causal(name):
    base, poisoned, lw, rw, hw, pts = _pivot_cases()
    _assert_causal(name, (_frame(base), _frame(base), lw, rw, hw, pts),
                   (_frame(poisoned), _frame(poisoned), lw, rw, hw, pts), len(base))


# ---------------------------------------------------------------------------
# SR / donchian / keltner / retest family (indicators_v2 + polars_structure).
# ---------------------------------------------------------------------------
def _sr_cases(n=40, w=5):
    base = [10.0 + 2.0 * np.sin(i * 0.5) + 0.05 * i for i in range(n)]
    poisoned = base + [1e6]
    return base, poisoned, w


def test_sr_distance_pct_future_poison_is_causal():
    base, poisoned, w = _sr_cases()
    o, h, l, c = _ohlc_det(base)
    o2, h2, l2, c2 = _ohlc_det(poisoned)
    _assert_causal("sr_distance_pct", (h, l, c, w), (h2, l2, c2, w), len(base))


def test_sr_touch_count_future_poison_is_causal():
    base, poisoned, w = _sr_cases()
    o, h, l, c = _ohlc_det(base)
    o2, h2, l2, c2 = _ohlc_det(poisoned)
    _assert_causal("sr_touch_count", (h, l, c, w, 0.05), (h2, l2, c2, w, 0.05), len(base))


@pytest.mark.parametrize("name", [
    "donchian_breakout_up", "donchian_breakout_down",
    "donchian_width_pct", "donchian_channel_position",
])
def test_donchian_family_future_poison_is_causal(name):
    base, poisoned, w = _sr_cases()
    o, h, l, c = _ohlc_det(base)
    o2, h2, l2, c2 = _ohlc_det(poisoned)
    _assert_causal(name, (h, l, c, w), (h2, l2, c2, w), len(base))


def test_keltner_breakout_strength_future_poison_is_causal():
    base, poisoned, w = _sr_cases()
    o, h, l, c = _ohlc_det(base)
    o2, h2, l2, c2 = _ohlc_det(poisoned)
    _assert_causal("keltner_breakout_strength", (h, l, c, 5, 5, 2.0),
                   (h2, l2, c2, 5, 5, 2.0), len(base))


@pytest.mark.parametrize("name", ["pattern_breakout_retest", "pattern_breakdown_retest"])
def test_retest_family_future_poison_is_causal(name):
    base, poisoned, w = _sr_cases()
    o, h, l, c = _ohlc_det(base)
    o2, h2, l2, c2 = _ohlc_det(poisoned)
    _assert_causal(name, (c, w, 3, 0.05), (c2, w, 3, 0.05), len(base))


# ---------------------------------------------------------------------------
# Candlestick binary patterns (polars_candle).  EVENT/CONDITION: a pattern at t
# uses only bars up to t (known at close of t).  A future poison must not change
# any prefix row.
# ---------------------------------------------------------------------------
_CDL_NAMES = [
    "cdl_doji", "cdl_hammer", "cdl_inverted_hammer", "cdl_shooting_star",
    "cdl_marubozu", "cdl_spinning_top", "cdl_engulfing", "cdl_inside_bar",
    "cdl_outside_bar", "cdl_dragonfly_doji", "cdl_gravestone_doji",
    "cdl_hanging_man", "cdl_harami", "cdl_harami_cross", "cdl_piercing",
    "cdl_dark_cloud_cover", "cdl_morning_star", "cdl_evening_star",
    "cdl_three_white_soldiers", "cdl_three_black_crows",
    "cdl_tweezer_top", "cdl_tweezer_bottom",
]


def _cdl_cases(n=40):
    rng = np.random.default_rng(3)
    close = [10.0 + 2.0 * np.sin(i * 0.5) + 0.05 * i for i in range(n)]
    o = [c - 0.3 * rng.random() for c in close]
    h = [max(o[i], close[i]) + 0.4 * rng.random() for i in range(n)]
    l = [min(o[i], close[i]) - 0.4 * rng.random() for i in range(n)]
    # poison: append a huge future bar
    o2 = o + [o[-1]]
    h2 = h + [h[-1] + 1e6]
    l2 = l + [l[-1]]
    c2 = close + [1e6]
    return (o, h, l, close), (o2, h2, l2, c2)


@pytest.mark.parametrize("name", _CDL_NAMES)
def test_candlestick_family_future_poison_is_causal(name):
    (o, h, l, c), (o2, h2, l2, c2) = _cdl_cases()
    _assert_causal(name, (_frame(o), _frame(h), _frame(l), _frame(c)),
                   (_frame(o2), _frame(h2), _frame(l2), _frame(c2)), len(c))
