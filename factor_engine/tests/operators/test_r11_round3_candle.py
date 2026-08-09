# -*- coding: utf-8 -*-
"""Round-3 candle audit tests (items 15 & 16).

Item 15 — unified OHLC invariant: every candle operator must fail closed
(NaN/None) when High < max(Open, Close), Low > min(Open, Close), High < Low, or
any of O/H/L/C <= 0.  Otherwise shadow lengths go negative, rejection fractions
leave [0,1] and close strength leaves [-1,1], and multi-candle patterns fire on
garbage bars.

Item 16 — ``candle_gap_atr`` must use ATR as of t-1 in the denominator (the
morning gap should not be weakened by today's late-session range).

These tests import the owned candle modules directly: ``load_all()`` may be
blocked by a concurrent session's layer-governance surface mismatch (a shared,
out-of-scope file), so the tests never depend on the full registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators.price_volume.candle_geometry_v2 import (  # noqa: E402
    _validate_ohlc,
    candle_body_position,
    candle_close_strength,
    candle_gap_atr,
    candle_inside_ratio,
    candle_lower_shadow_zscore,
    candle_overlap_ratio,
    candle_rejection_lower,
    candle_rejection_upper,
    candle_upper_shadow_zscore,
)
from cleaned_operators.price_volume.candle_pattern_engine_v2 import (  # noqa: E402
    _engine as cdl_engine,
)
from cleaned_operators.price_volume.candle_patterns_extended import (  # noqa: E402
    dragonfly_doji,
    gravestone_doji,
    hammer,
    harami,
    morning_star,
    three_white_soldiers,
    tweezer_top,
)
from cleaned_operators.price_volume import polars_candle as pc  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frame(values, start="2024-01-01"):
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    return pd.DataFrame(v, index=pd.date_range(start, periods=v.shape[0], freq="B"), columns=["A"])


def _valid_ohlc(n=8, base=10.0, span=2.0):
    """A clean bullish/bearish mixed panel with valid OHLC relationships."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = ["A"]
    o = pd.DataFrame(base + (np.arange(n) % 3) * 0.5, index=idx, columns=cols)
    c = o + pd.DataFrame(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=idx, columns=cols)
    h = pd.DataFrame(np.maximum(o.to_numpy(), c.to_numpy()), index=idx, columns=cols) + span
    l = pd.DataFrame(np.minimum(o.to_numpy(), c.to_numpy()), index=idx, columns=cols) - span
    return o, h, l, c


def _to_pl(df):
    return pl.DataFrame({c: df[c].to_numpy() for c in df.columns})


def _assert_pd_pl_parity(name, pd_out, pl_out, rtol=1e-6, atol=1e-8):
    assert list(pd_out.columns) == list(pl_out.columns)
    for col in pd_out.columns:
        a = np.asarray(pd_out[col].to_numpy(), dtype=float)
        b = np.asarray(pl_out[col].to_numpy(), dtype=float)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


# ---------------------------------------------------------------------------
# item 15 — _validate_ohlc mask
# ---------------------------------------------------------------------------

def test_validate_ohlc_rejects_every_broken_relationship():
    o, h, l, c = _valid_ohlc(n=8)
    valid = _validate_ohlc(o, h, l, c)
    assert bool(valid.all().all())

    # High < max(Open, Close)
    o2, h2, l2, c2 = _valid_ohlc(n=8)
    h2.iloc[3, 0] = np.maximum(o2.iloc[3, 0], c2.iloc[3, 0]) - 0.1
    assert not bool(_validate_ohlc(o2, h2, l2, c2).iloc[3, 0])
    assert bool(_validate_ohlc(o2, h2, l2, c2).iloc[2, 0])

    # Low > min(Open, Close)
    o3, h3, l3, c3 = _valid_ohlc(n=8)
    l3.iloc[4, 0] = np.minimum(o3.iloc[4, 0], c3.iloc[4, 0]) + 0.1
    assert not bool(_validate_ohlc(o3, h3, l3, c3).iloc[4, 0])

    # High < Low
    o4, h4, l4, c4 = _valid_ohlc(n=8)
    h4.iloc[5, 0] = l4.iloc[5, 0] - 0.1
    assert not bool(_validate_ohlc(o4, h4, l4, c4).iloc[5, 0])

    # non-positive raw price
    o5, h5, l5, c5 = _valid_ohlc(n=8)
    c5.iloc[6, 0] = -1.0
    assert not bool(_validate_ohlc(o5, h5, l5, c5).iloc[6, 0])

    # NaN fails closed
    o6, h6, l6, c6 = _valid_ohlc(n=8)
    o6.iloc[7, 0] = np.nan
    assert not bool(_validate_ohlc(o6, h6, l6, c6).iloc[7, 0])

    # subset (no low/high) only checks positivity of the provided fields
    sub = _validate_ohlc(o, None, None, c)
    assert bool(sub.iloc[3, 0])


# ---------------------------------------------------------------------------
# item 15 — pandas operators fail closed on invalid bars
# ---------------------------------------------------------------------------

def test_shadow_and_strength_operators_fail_closed():
    o, h, l, c = _valid_ohlc(n=8)
    # break bar 3: High below the body
    h.iloc[3, 0] = np.maximum(o.iloc[3, 0], c.iloc[3, 0]) - 0.1

    for fn, args in (
        (candle_rejection_upper, (o, h, l, c)),
        (candle_rejection_lower, (o, h, l, c)),
        (candle_body_position, (o, h, l, c)),
        (candle_upper_shadow_zscore, (o, h, c)),
    ):
        out = fn(*args, **({"window": 3} if fn is candle_upper_shadow_zscore else {}))
        assert np.isnan(out.iloc[3, 0]), f"{fn.__name__} invalid bar must be NaN"
        assert np.isfinite(out.iloc[5, 0]), f"{fn.__name__} valid bar stays finite"

    # close_strength only sees (high, low, close) — it fails closed when the
    # close leaves the daily range (otherwise strength would exceed [-1,1]).
    o9, h9, l9, c9 = _valid_ohlc(n=8)
    c9.iloc[3, 0] = h9.iloc[3, 0] + 0.5  # close above high
    out = candle_close_strength(h9, l9, c9)
    assert np.isnan(out.iloc[3, 0]), "close_strength must be NaN when close > high"
    assert np.isfinite(out.iloc[5, 0]), "close_strength valid bar stays finite"

    # lower shadow z-score (open, low, close) — break Low above body
    o2, h2, l2, c2 = _valid_ohlc(n=8)
    l2.iloc[3, 0] = np.minimum(o2.iloc[3, 0], c2.iloc[3, 0]) + 0.1
    out = candle_lower_shadow_zscore(o2, l2, c2, 3)
    assert np.isnan(out.iloc[3, 0])

    # two-bar geometry (overlap / inside) only sees (high, low) — it fails
    # closed when the prior bar's high < low (a detectably broken bar).
    o3, h3, l3, c3 = _valid_ohlc(n=8)
    h3.iloc[3, 0] = l3.iloc[3, 0] - 0.1  # high below low
    out = candle_overlap_ratio(h3, l3)
    assert np.isnan(out.iloc[4, 0]), "overlap ratio must be NaN when prior bar invalid"
    out = candle_inside_ratio(h3, l3)
    assert np.isnan(out.iloc[4, 0]), "inside ratio must be NaN when prior bar invalid"


def test_extended_patterns_fail_closed_on_invalid_bars():
    o, h, l, c = _valid_ohlc(n=10)
    h.iloc[4, 0] = np.maximum(o.iloc[4, 0], c.iloc[4, 0]) - 0.1

    assert np.isnan(dragonfly_doji(o, h, l, c).iloc[4, 0])
    assert np.isnan(gravestone_doji(o, h, l, c).iloc[4, 0])
    assert np.isnan(hammer(o, h, l, c).iloc[4, 0])
    # two-bar patterns: an invalid prior bar blocks the current row too
    assert np.isnan(harami(o, h, l, c).iloc[5, 0])
    assert np.isnan(tweezer_top(o, h, l, c).iloc[5, 0])
    # three-bar patterns
    assert np.isnan(morning_star(o, h, l, c).iloc[6, 0])
    assert np.isnan(three_white_soldiers(o, h, l, c).iloc[6, 0])


def test_candlestick_engine_fail_closed_on_invalid_bar():
    o, h, l, c = _valid_ohlc(n=12)
    # make bar 4 invalid (High below body) so the engine cannot classify it
    h.iloc[4, 0] = np.maximum(o.iloc[4, 0], c.iloc[4, 0]) - 0.1
    frames = tuple(pd.DataFrame({"A": v["A"]}) for v in (o, h, l, c))
    for pattern in ("long_line", "long_legged_doji", "3_inside", "homing_pigeon"):
        out = pd.DataFrame(cdl_engine(*frames, pattern, 3, 3, 0.3))
        assert np.isnan(out.iloc[4, 0]), f"{pattern} invalid bar must be NaN"


def test_polars_twin_fail_closed_matches_pandas():
    o, h, l, c = _valid_ohlc(n=12)
    h.iloc[3, 0] = np.maximum(o.iloc[3, 0], c.iloc[3, 0]) - 0.1
    l.iloc[6, 0] = np.minimum(o.iloc[6, 0], c.iloc[6, 0]) + 0.1

    pd_ops = (
        (candle_rejection_upper, (o, h, l, c), {}),
        (candle_rejection_lower, (o, h, l, c), {}),
        (candle_close_strength, (h, l, c), {}),
        (candle_body_position, (o, h, l, c), {}),
        (candle_upper_shadow_zscore, (o, h, c), {"window": 3}),
        (candle_overlap_ratio, (h, l), {}),
        (dragonfly_doji, (o, h, l, c), {}),
        (harami, (o, h, l, c), {}),
    )
    pl_ops = (
        (pc.candle_rejection_upper, (o, h, l, c), {}),
        (pc.candle_rejection_lower, (o, h, l, c), {}),
        (pc.candle_close_strength, (h, l, c), {}),
        (pc.candle_body_position, (o, h, l, c), {}),
        (pc.candle_upper_shadow_zscore, (o, h, c), {"window": 3}),
        (pc.candle_overlap_ratio, (h, l), {}),
        (pc.cdl_dragonfly_doji, (o, h, l, c), {}),
        (pc.cdl_harami, (o, h, l, c), {}),
    )
    for name, (pfn, pargs, pkw), (plfn, plargs, plkw) in zip(
        ("rejection_upper", "rejection_lower", "close_strength", "body_position",
         "upper_shadow_zscore", "overlap_ratio", "dragonfly_doji", "harami"),
        pd_ops, pl_ops,
    ):
        pd_out = pfn(*pargs, **pkw)
        pl_out = plfn(*[_to_pl(a) for a in plargs], **plkw)
        _assert_pd_pl_parity(name, pd_out, pl_out)


# ---------------------------------------------------------------------------
# item 16 — candle_gap_atr denominator is ATR as of t-1
# ---------------------------------------------------------------------------

def test_gap_atr_uses_prior_atr_denominator():
    n = 12
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = ["A"]
    o = pd.DataFrame(10.0, index=idx, columns=cols)
    h = pd.DataFrame(11.0, index=idx, columns=cols)
    l = pd.DataFrame(9.0, index=idx, columns=cols)
    c = pd.DataFrame(10.0, index=idx, columns=cols)
    # bar 5 opens a huge one-off range -> inflates ATR_5, but ATR_4 is untouched
    h.iloc[5, 0] = 50.0
    l.iloc[5, 0] = 5.0
    c.iloc[5, 0] = 30.0
    # bar 6 opens with a gap against close_5 and stays structurally valid
    o.iloc[6, 0] = 12.0
    h.iloc[6, 0] = 13.0
    l.iloc[6, 0] = 9.0

    w = 3
    pc_prev = c.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce(
            [(h - l).to_numpy(), (h - pc_prev).abs().to_numpy(), (l - pc_prev).abs().to_numpy()]
        ),
        index=idx, columns=cols,
    )
    atr = tr.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    atr_prev = atr.shift(1)
    gap = o - c.shift(1)
    expected = (gap / atr_prev.replace(0, np.nan)).iloc[6, 0]

    out = candle_gap_atr(o, h, l, c, w).iloc[6, 0]
    assert np.isfinite(out), "gap_atr must be finite at bar 6"
    assert np.isclose(out, expected, rtol=1e-9, atol=1e-12), (
        f"denominator is not ATR as of t-1: got {out}, expected gap/ATR_prev = {expected}"
    )
    # the fix must actually be observable: the output differs from gap/ATR_t
    # (the old, wrong denominator that includes today's bar).
    old_wrong = (gap / atr.replace(0, np.nan)).iloc[6, 0]
    assert not np.isclose(out, old_wrong, rtol=1e-6, atol=1e-9), (
        "candle_gap_atr still divides by today's ATR"
    )


def test_gap_atr_pandas_polars_parity():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2022-01-01", periods=60, freq="D")
    cols = ["A", "B"]
    o = pd.DataFrame(rng.uniform(5, 50, (60, 2)), index=idx, columns=cols)
    c = o + pd.DataFrame(rng.normal(0, 2.5, (60, 2)), index=idx, columns=cols)
    h = pd.DataFrame(np.maximum(o.to_numpy(), c.to_numpy()), index=idx, columns=cols) + pd.DataFrame(rng.uniform(0.1, 2.0, (60, 2)), index=idx, columns=cols)
    l = pd.DataFrame(np.minimum(o.to_numpy(), c.to_numpy()), index=idx, columns=cols) - pd.DataFrame(rng.uniform(0.1, 2.0, (60, 2)), index=idx, columns=cols)

    pd_out = candle_gap_atr(o, h, l, c, 7)
    pl_out = pc.candle_gap_atr(_to_pl(o), _to_pl(h), _to_pl(l), _to_pl(c), 7)
    _assert_pd_pl_parity("candle_gap_atr", pd_out, pl_out)


def test_gap_atr_pandas_polars_parity_nan():
    # same shape as the repo NaN-warmup parity test (seed 31), ensures the
    # shifted-ATR denominator stays in pandas/polars lock-step over null bars.
    rng = np.random.default_rng(31)
    idx = pd.date_range("2023-01-01", periods=80, freq="D")
    mask = rng.random((80, 2)) < 0.10
    o = pd.DataFrame(rng.uniform(5, 50, (80, 2)), index=idx, columns=["A", "B"])
    o[mask] = np.nan
    c = o + pd.DataFrame(rng.normal(0, 2.5, (80, 2)), index=idx, columns=["A", "B"])
    h = pd.DataFrame(np.maximum(np.nan_to_num(o.to_numpy(), nan=-1e9), np.nan_to_num(c.to_numpy(), nan=-1e9)), index=idx, columns=["A", "B"])
    l = pd.DataFrame(np.minimum(np.nan_to_num(o.to_numpy(), nan=1e9), np.nan_to_num(c.to_numpy(), nan=1e9)), index=idx, columns=["A", "B"])
    h[mask] = np.nan
    l[mask] = np.nan

    pd_out = candle_gap_atr(o, h, l, c, 5)
    pl_out = pc.candle_gap_atr(_to_pl(o), _to_pl(h), _to_pl(l), _to_pl(c), 5)
    _assert_pd_pl_parity("candle_gap_atr(nan)", pd_out, pl_out)
