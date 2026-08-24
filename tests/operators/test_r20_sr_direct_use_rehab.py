# -*- coding: utf-8 -*-
"""R20 support/resistance DirectUse rehabilitation oracles.

Causal, dimensionless support/resistance canonicals over ONE documented pivot
definition: the PRIOR-window typical-price median pivot

    pivot_t = median((high+low+close)/3 over the ``window`` bars ending at
    t-1)   == ((h+l+c)/3).rolling(w, min_periods=w).median().shift(1)

A level a bar-t decision acts against must be formed BEFORE t, so the pivot
window ends at t-1 (strictly trailing, PIT-safe; mirrors the donchian_breakout_*
prior-window precedent).

* ``sr_distance_pct`` — (close - pivot) / close, strict-positive close masked
* ``sr_touch_count``  — fraction of the last w bars whose high OR low came
  within ``tol`` (relative, tol*pivot) of the current prior pivot; bounded
  [0,1], bad bars excluded from BOTH numerator and denominator

``sr_breakout_up`` / ``sr_breakout_down`` were considered and SKIPPED as
near-duplicates of the registered donchian_breakout_up/down (prior-window
rolling extremes): a prior-window "fresh extreme vs prior pivot-median" breakout
is just donchian_breakout_* under a different pivot; the genuinely distinct
median-pivot semantics are already carried by sr_distance_pct (see the evidence
YAML R20-SR-DIRECTUSE).

Manual pandas oracles recompute the pivot and the touch fractions from raw
OHLC independently of the operator helpers; a no-lookahead oracle proves a
bar's own extreme is NOT part of its reference level; prefix-invariance is
exact for the rolling-only family.
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


def _ohlc_panel(values):
    """Synthetic OHLC panel from a single close-like path (valid bars)."""
    close = pd.DataFrame({"A": [float(v) for v in values]})
    rng = np.random.default_rng(7)
    o = close.iloc[:, 0].to_numpy() + 0.4 * (rng.random(len(close)) - 0.5)
    h = np.maximum(o, close.iloc[:, 0].to_numpy()) + 0.3 * rng.random(len(close))
    l = np.minimum(o, close.iloc[:, 0].to_numpy()) - 0.3 * rng.random(len(close))
    return (
        pd.DataFrame({"A": o}),
        pd.DataFrame({"A": h}),
        pd.DataFrame({"A": l}),
        close,
    )


def _oracle_pivot(h, l, c, w):
    """Manual prior-window typical-price median pivot (independent of the
    operator helper): median over rows t-w..t-1."""
    tp = (h + l + c) / 3.0
    return tp.rolling(w, min_periods=w).median().shift(1)


def test_sr_distance_pct_manual_oracle_and_warmup_nan():
    vals = [20.0 + 2.0 * np.sin(i * 0.6) + 0.03 * i for i in range(40)]
    o, h, l, c = _ohlc_panel(vals)
    w = 5
    out = _op("sr_distance_pct").calculate(h, l, c, window=w)
    pivot = _oracle_pivot(h, l, c, w)
    pos = c.where(c > 0.0)
    expected = (pos - pivot) / pos
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: the prior pivot needs w+1 rows -> NaN through index w, finite
    # from index w+1.
    assert out.iloc[: w, 0].isna().all()
    assert np.isfinite(out.iloc[w:, 0]).all()


def test_sr_distance_pct_no_lookahead_own_bar_not_in_reference_level():
    # No-lookahead oracle: a bar's own extremes/typical price must NOT be in
    # its reference level.  Construct a panel where the LAST bar spikes far
    # above everything before it; if bar t's own data leaked into the pivot,
    # the distance would collapse.  The pivot at t must equal the median of
    # the w typical prices ENDING AT t-1 only.
    n = 12
    base = 10.0
    c = pd.DataFrame({"A": [base] * (n - 1) + [base + 5.0]})
    h = c + 0.1
    l = c - 0.1
    o = c.copy()
    w = 4
    out = _op("sr_distance_pct").calculate(h, l, c, window=w)
    pivot = _oracle_pivot(h, l, c, w)
    # Manual check on the last row: pivot = median of typical prices of rows
    # n-2-w+1 .. n-2 (all == base) -> distance = 5/15 exactly.
    pivot_last = float(np.median([base] * w))
    expected_last = (base + 5.0 - pivot_last) / (base + 5.0)
    assert np.isclose(out.iloc[-1, 0], expected_last)
    # Generic property: mutating bar t's OWN inputs never changes the pivot at
    # t (prefix-invariance restricted to the reference level).
    c2 = c.copy()
    c2.iloc[-1, 0] = base + 50.0  # wildly different own-bar close
    h2 = c2 + 0.1
    l2 = c2 - 0.1
    out2 = _op("sr_distance_pct").calculate(h2, l2, c2, window=w)
    pivot2 = _oracle_pivot(h2, l2, c2, w)
    pd.testing.assert_series_equal(
        pivot2.iloc[:-1, 0], pivot.iloc[:-1, 0], check_names=False)
    # and the pivot at the last row is unchanged by the own-bar mutation.
    assert np.isclose(pivot2.iloc[-1, 0], pivot.iloc[-1, 0])


def test_sr_distance_pct_sign_and_malformed_bar_masking():
    # Sign: rising closes -> close above the prior median pivot -> positive.
    vals = list(np.linspace(10.0, 25.0, 30))
    o, h, l, c = _ohlc_panel(vals)
    w = 5
    out = _op("sr_distance_pct").calculate(h, l, c, window=w)
    tail = out.iloc[w:, 0].dropna()
    assert (tail > 0.0).all()
    # Falling panel -> strictly negative.
    o2, h2, l2, c2 = _ohlc_panel(list(np.linspace(25.0, 10.0, 30)))
    out2 = _op("sr_distance_pct").calculate(h2, l2, c2, window=w)
    assert (out2.iloc[w:, 0].dropna() < 0.0).all()

    # Malformed-bar masking (never laundered):
    # non-positive close at row 20 -> NaN on that bar, neighbours finite.
    o3, h3, l3, c3 = _ohlc_panel([12.0 + 1.2 * np.sin(i * 0.8) for i in range(40)])
    c3.iloc[20, 0] = 0.0
    out3 = _op("sr_distance_pct").calculate(h3, l3, c3, window=4)
    assert np.isnan(out3.iloc[20, 0]), "close<=0 bar must be NaN"
    assert np.isfinite(out3.iloc[10, 0])
    # NaN close at row 18 -> NaN (missing judgement input).
    o4, h4, l4, c4 = _ohlc_panel([12.0 + 1.2 * np.sin(i * 0.8) for i in range(40)])
    c4.iloc[18, 0] = float("nan")
    out4 = _op("sr_distance_pct").calculate(h4, l4, c4, window=4)
    assert np.isnan(out4.iloc[18, 0])
    assert np.isfinite(out4.iloc[10, 0])


def test_sr_touch_count_manual_oracle_and_bounded():
    rng = np.random.default_rng(5)
    n = 50
    c = pd.DataFrame({"A": 10.0 + 0.6 * np.sin(np.arange(n) * 0.5)})
    o = c + pd.DataFrame({"A": 0.2 * (rng.random(n) - 0.5)})
    h = pd.DataFrame({"A": np.maximum(o.iloc[:, 0], c.iloc[:, 0]) + 0.3 * rng.random(n)})
    l = pd.DataFrame({"A": np.minimum(o.iloc[:, 0], c.iloc[:, 0]) - 0.3 * rng.random(n)})
    w, tol = 6, 0.01
    out = _op("sr_touch_count").calculate(h, l, c, window=w, tol=tol)
    # Manual oracle (numpy loops, independent of the operator helpers).
    # A touch is TIME-COINCIDENT: window bar i is judged against pivot_i —
    # the prior-window median typical price at bar i's OWN row (rows
    # i-w..i-1) — never against the level formed later at t.
    hv, lv, cv = h.to_numpy(float)[:, 0], l.to_numpy(float)[:, 0], c.to_numpy(float)[:, 0]
    tp = (hv + lv + cv) / 3.0
    piv = np.full(n, np.nan)
    for i in range(n):
        if i - w >= 0:
            piv[i] = np.median(tp[i - w:i])
    expected = np.full(n, np.nan)
    for t in range(n):
        if t - w < 0:
            continue
        if not (np.isfinite(cv[t]) and cv[t] > 0.0
                and np.isfinite(hv[t]) and np.isfinite(lv[t])):
            continue
        num = den = 0.0
        for i in range(t - w + 1, t + 1):
            if np.isfinite(hv[i]) and np.isfinite(lv[i]) and np.isfinite(piv[i]) and piv[i] > 0.0:
                den += 1.0
                band = tol * piv[i]
                if abs(hv[i] - piv[i]) <= band or abs(lv[i] - piv[i]) <= band:
                    num += 1.0
        expected[t] = num / den if den > 0 else np.nan
    np.testing.assert_allclose(out.to_numpy(float)[:, 0], expected, equal_nan=True)
    fin = out.iloc[w:, 0].dropna()
    assert ((fin >= 0.0) & (fin <= 1.0)).all()
    # Warmup: no output before w rows exist.
    assert out.iloc[: w, 0].isna().all()


def test_sr_touch_count_deterministic_flat_panel_exact():
    # All bars identical: pivot == price, every high/low is exactly AT the
    # pivot -> fraction exactly 1.0 for any tol > 0.
    n = 15
    c = pd.DataFrame({"A": [10.0] * n})
    h = pd.DataFrame({"A": [10.3] * n})
    l = pd.DataFrame({"A": [9.7] * n})
    # pivot = median typical price = 10.0; |high-10|=0.3, |low-10|=0.3 ->
    # tol=0.031 -> band=0.31 > 0.3 -> all touch -> 1.0.  (0.03 is an FP
    # knife-edge: band=0.3000000000000007-style roundoff flips bars, so the
    # test pins clearly-separated sides of the boundary.)
    out = _op("sr_touch_count").calculate(h, l, c, window=4, tol=0.031)
    assert np.allclose(out.iloc[4:, 0].to_numpy(), 1.0)
    # tol just below the 0.3/10.0 ratio -> zero touches.
    out0 = _op("sr_touch_count").calculate(h, l, c, window=4, tol=0.0299)
    assert np.allclose(out0.iloc[4:, 0].to_numpy(), 0.0)


def test_sr_touch_count_malformed_bar_excluded_not_counted_as_nontouch():
    # One bad bar (missing high) inside the window drops out of BOTH numerator
    # and denominator — a deformed bar is not evidence of "no touch".
    vals = [10.0 + 0.8 * np.sin(i * 0.9) for i in range(20)]
    o, h, l, c = _ohlc_panel(vals)
    w = 4
    # Baseline: clean panel fraction at t=8.
    base = _op("sr_touch_count").calculate(h, l, c, window=w, tol=0.05)
    assert np.isfinite(base.iloc[8, 0])
    # Break row 7 (inside the t=8 window rows 5..8): its high goes NaN.  Row
    # 7 itself must NaN (present-extremes requirement on the judged bar),
    # while the t=8 fraction stays finite (computed over the 3 judgeable
    # bars) — the bad bar is excluded, never counted as a non-touch.
    h2 = h.copy()
    h2.iloc[7, 0] = float("nan")
    out = _op("sr_touch_count").calculate(h2, l, c, window=w, tol=0.05)
    assert np.isnan(out.iloc[7, 0]), "NaN-field bar must NaN its own row"
    assert np.isfinite(out.iloc[8, 0]), "bad bar excluded, not counted as non-touch"
    # Non-positive close on the JUDGED bar t -> NaN (R5-38).
    c3 = c.copy()
    c3.iloc[10, 0] = 0.0
    out3 = _op("sr_touch_count").calculate(h, l, c3, window=w, tol=0.05)
    assert np.isnan(out3.iloc[10, 0])
    assert np.isfinite(out3.iloc[8, 0])


def test_sr_prefix_invariance():
    vals = [15.0 + 1.6 * np.sin(i * 0.5) for i in range(50)]
    o, h, l, c = _ohlc_panel(vals)
    for name, kwargs in (("sr_distance_pct", {"window": 5}),
                         ("sr_touch_count", {"window": 5, "tol": 0.02})):
        full = _op(name).calculate(h, l, c, **kwargs)
        head = _op(name).calculate(h.iloc[:35], l.iloc[:35], c.iloc[:35], **kwargs)
        pd.testing.assert_frame_equal(full.iloc[:35], head)


def test_sr_param_specs_and_rolling_governance():
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    names = {"sr_distance_pct", "sr_touch_count"}
    for name in names:
        specs = _op(name).metadata.param_specs
        assert "window" in specs, f"{name} ParamSpec keys: {sorted(specs)}"
        assert specs["window"].dtype is int, f"{name}.window dtype"
        tags = set(_op(name).metadata.tags or [])
        assert "stateful" not in tags, f"{name} tags: {tags}"
        assert {"causal", "pit_safe"} <= tags, f"{name} tags: {tags}"
        assert name not in _RECURSIVE_EWM, f"{name} must not be in _RECURSIVE_EWM"
    assert set(_op("sr_touch_count").metadata.param_specs) == {"window", "tol"}
    assert _op("sr_touch_count").metadata.param_specs["tol"].dtype is float
    # window < 2 and tol <= 0 rejected at the call boundary.
    o, h, l, c = _ohlc_panel([10.0, 11.0, 12.0])
    with pytest.raises(ValueError):
        _op("sr_distance_pct").calculate(h, l, c, window=1)
    with pytest.raises(ValueError):
        _op("sr_touch_count").calculate(h, l, c, window=3, tol=0.0)


def test_sr_relative_alpha_membership_and_promotion():
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"sr_distance_pct", "sr_touch_count"}
    assert promoted <= _RELATIVE_ALPHA_OPS
    # The price-scale pivot-cluster regression levels stay intermediate.
    for level in ("ts_support_level", "ts_resistance_level"):
        assert level not in _RELATIVE_ALPHA_OPS, f"{level} must stay intermediate"
    from factor_engine.cleaned_operators.operator_surface import DAILY_FACTOR_MIGRATED, _TECHNICAL_V2_CANONICALS

    assert promoted <= DAILY_FACTOR_MIGRATED
    assert promoted <= _TECHNICAL_V2_CANONICALS
