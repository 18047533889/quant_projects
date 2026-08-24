# -*- coding: utf-8 -*-
"""R20 candlestick-pattern DirectUse rehabilitation oracles.

AGGREGATE per-bar candlestick-pattern STRENGTH over a trailing window (w bars
ending at t, min_periods=window, rolling-only bounded state — no full-replay
governance): these are NOT one-off binary cdl_* detectors.

* ``candle_body_strength``  — mean of sign(close-open)*body/range over w
* ``candle_wick_balance``   — mean of (lower_wick-upper_wick)/range over w
* ``candle_range_pct``      — mean of (high-low)/close over w
* ``candle_pattern_count``  — fraction of VALID w-bars classified doji /
  one-sided-rejection (|body/range|<0.1 or a >=70% dominant wick whose
  opposite wick <= 15% of range)

Data-integrity contract (never laundered): a bar is structurally valid only
when high > low (STRICT — a zero range cannot normalize the geometry),
high >= max(open,close), low <= min(open,close) and close > 0; bad bars are
NaN and excluded from the trailing mean/fraction — never counted as 0.
Manual pandas oracles recompute the geometry from raw OHLC independently of
the operator helpers; prefix-invariance (truncated panel == full-panel
prefix) is exact for the rolling-only family.
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
    rng = np.random.default_rng(11)
    o = close.iloc[:, 0].to_numpy() + 0.4 * (rng.random(len(close)) - 0.5)
    h = np.maximum(o, close.iloc[:, 0].to_numpy()) + 0.3 * rng.random(len(close))
    l = np.minimum(o, close.iloc[:, 0].to_numpy()) - 0.3 * rng.random(len(close))
    return (
        pd.DataFrame({"A": o}),
        pd.DataFrame({"A": h}),
        pd.DataFrame({"A": l}),
        close,
    )


def _oracle_valid(o, h, l, c):
    hi = np.maximum(o.to_numpy(float), c.to_numpy(float))
    lo = np.minimum(o.to_numpy(float), c.to_numpy(float))
    valid = (
        o.notna() & h.notna() & l.notna() & c.notna()
        & (c > 0.0).to_numpy(bool) & (h > l).to_numpy(bool)
        & (h.to_numpy(float) >= hi) & (l.to_numpy(float) <= lo)
    )
    return pd.DataFrame(valid, index=o.index, columns=o.columns), hi, lo


def _oracle_body(o, h, l, c):
    valid, hi, lo = _oracle_valid(o, h, l, c)
    rng = (h - l).where(valid)
    sign = np.sign((c - o).to_numpy(float))
    body = pd.DataFrame(sign * np.abs(c.to_numpy(float) - o.to_numpy(float)),
                        index=o.index, columns=o.columns) / rng
    return body.where(valid)


def _oracle_wick(o, h, l, c):
    valid, hi, lo = _oracle_valid(o, h, l, c)
    rng = (h - l).where(valid).replace(0, np.nan)
    lower = pd.DataFrame(lo - l.to_numpy(float), index=o.index, columns=o.columns)
    upper = pd.DataFrame(h.to_numpy(float) - hi, index=o.index, columns=o.columns)
    return ((lower - upper) / rng).where(valid)


def _oracle_range_pct(o, h, l, c):
    valid, _hi, _lo = _oracle_valid(o, h, l, c)
    rng = (h - l).where(valid)
    return (rng / c.where(c > 0.0)).where(valid)


def test_candle_body_strength_manual_oracle_and_warmup_nan():
    vals = [10.0 + 1.5 * np.sin(i * 0.7) + 0.05 * i for i in range(40)]
    o, h, l, c = _ohlc_panel(vals)
    w = 5
    out = _op("candle_body_strength").calculate(o, h, l, c, window=w)
    expected = _oracle_body(o, h, l, c).rolling(w, min_periods=w).mean()
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: window=5 -> first four rows NaN, finite from index 4.
    assert out.iloc[:4, 0].isna().all()
    assert np.isfinite(out.iloc[4:, 0]).all()
    # Bounded by construction: |mean signed body/range| <= 1.
    assert out.abs().max().max() <= 1.0


def test_candle_body_strength_sign_on_directional_panel():
    # Monotone rising closes with opens below every close -> every bar is an
    # up-candle -> the aggregate body strength is strictly positive.
    vals = list(np.linspace(10.0, 20.0, 30))
    c = pd.DataFrame({"A": vals})
    o = c - 0.5
    h = c + 0.1
    l = o - 0.1
    w = 4
    out = _op("candle_body_strength").calculate(o, h, l, c, window=w)
    tail = out.iloc[w:, 0].dropna()
    assert len(tail) > 0
    assert (tail > 0.0).all()
    # Mirror: falling panel -> strictly negative.
    c2 = pd.DataFrame({"A": list(np.linspace(20.0, 10.0, 30))})
    o2 = c2 + 0.5
    h2 = o2 + 0.1
    l2 = c2 - 0.1
    out2 = _op("candle_body_strength").calculate(o2, h2, l2, c2, window=w)
    assert (out2.iloc[w:, 0].dropna() < 0.0).all()


def test_candle_wick_balance_manual_oracle_and_sign():
    vals = [10.0 + 1.2 * np.sin(i * 0.55) for i in range(35)]
    o, h, l, c = _ohlc_panel(vals)
    w = 6
    out = _op("candle_wick_balance").calculate(o, h, l, c, window=w)
    expected = _oracle_wick(o, h, l, c).rolling(w, min_periods=w).mean()
    pd.testing.assert_frame_equal(out, expected)
    assert np.isfinite(out.iloc[w:, 0]).all()
    assert out.abs().max().max() <= 1.0
    # Adversarial sign panel: long lower wicks (hammer-like) -> positive
    # wick balance (buyers rejecting the low); inverted -> negative.
    n = 24
    cc = pd.DataFrame({"A": [10.0] * n})
    oo = pd.DataFrame({"A": [10.0] * n})
    hh = pd.DataFrame({"A": [10.2] * n})
    ll = pd.DataFrame({"A": [9.0] * n})  # 0.8 lower wick vs 0.2 upper
    w = 5
    up = _op("candle_wick_balance").calculate(oo, hh, ll, cc, window=w)
    assert (up.iloc[w:, 0] > 0.0).all()
    up2 = _op("candle_wick_balance").calculate(oo, hh, ll, cc, window=w)
    exact = (1.0 - 0.2) / 1.2  # (lower=1.0 - upper=0.2) / range=1.2
    assert np.allclose(up2.iloc[w:, 0].to_numpy(), exact)


def test_candle_range_pct_manual_oracle_and_distinct_from_true_range():
    vals = [30.0 + 2.0 * np.sin(i * 0.6) for i in range(40)]
    o, h, l, c = _ohlc_panel(vals)
    w = 5
    out = _op("candle_range_pct").calculate(o, h, l, c, window=w)
    expected = _oracle_range_pct(o, h, l, c).rolling(w, min_periods=w).mean()
    pd.testing.assert_frame_equal(out, expected)
    assert np.isfinite(out.iloc[w:, 0]).all()
    assert (out.iloc[w:, 0] > 0.0).all()
    # Distinctness vs the registered gap-inclusive true_range_pct kernel
    # (numerator max(h-l, |h-prev_close|, |l-prev_close|) / prev_close): on a
    # gapping panel the plain-range mean is materially different — they are
    # different canonicals, not a rename.
    tr_pct = _op("true_range_pct").calculate(h, l, c)
    tr_mean = tr_pct.rolling(w, min_periods=w).mean()
    both = out.iloc[w:, 0].dropna().index.intersection(tr_mean.iloc[w:, 0].dropna().index)
    diff = (out.loc[both, "A"] - tr_mean.loc[both, "A"]).abs()
    assert (diff > 1e-6).mean() > 0.5, "plain-range kernel must differ from true-range"


def test_candle_pattern_count_manual_oracle_and_bounded():
    rng = np.random.default_rng(3)
    n = 50
    o = pd.DataFrame({"A": 10.0 + 0.5 * (rng.random(n) - 0.5)})
    c = pd.DataFrame({"A": o.iloc[:, 0].to_numpy() + 0.5 * (rng.random(n) - 0.5)})
    h = pd.DataFrame({"A": np.maximum(o.iloc[:, 0], c.iloc[:, 0]) + 0.4 * rng.random(n)})
    l = pd.DataFrame({"A": np.minimum(o.iloc[:, 0], c.iloc[:, 0]) - 0.4 * rng.random(n)})
    w = 5
    out = _op("candle_pattern_count").calculate(o, h, l, c, window=w)
    # Manual oracle.
    valid, hi, lo = _oracle_valid(o, h, l, c)
    r = (h - l).to_numpy(float)
    body = np.abs(c.to_numpy(float) - o.to_numpy(float))
    upper = h.to_numpy(float) - hi
    lower = lo - l.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = body / r
        ur = upper / r
        lr = lower / r
    pat = (br < 0.1) | ((ur >= 0.7) & (lr <= 0.15)) | ((lr >= 0.7) & (ur <= 0.15))
    pat = pat & valid.to_numpy(bool)
    num = pd.DataFrame(np.where(valid, pat.astype(float), np.nan),
                       index=o.index, columns=o.columns).rolling(w, min_periods=1).sum()
    den = valid.astype(float).rolling(w, min_periods=1).sum()
    rows = (o.notna() & h.notna() & l.notna() & c.notna()).astype(float).rolling(w, min_periods=w).sum()
    expected = (num / den.where(den > 0.0)).where(rows >= float(w))
    pd.testing.assert_frame_equal(out, expected)
    fin = out.iloc[w:, 0].dropna()
    assert ((fin >= 0.0) & (fin <= 1.0)).all()
    # Deterministic doji panel: open == close == mid, both wicks 0.5/0.5 of a
    # unit range -> every bar a doji -> fraction exactly 1.
    n2 = 12
    oo = pd.DataFrame({"A": [10.0] * n2})
    cc = pd.DataFrame({"A": [10.0] * n2})
    hh = pd.DataFrame({"A": [10.5] * n2})
    ll = pd.DataFrame({"A": [9.5] * n2})
    out2 = _op("candle_pattern_count").calculate(oo, hh, ll, cc, window=4)
    assert np.allclose(out2.iloc[3:, 0].to_numpy(), 1.0)
    # Deterministic marubozu panel: body == range -> NO pattern bars -> 0.
    oo3 = pd.DataFrame({"A": [10.0 + 0.2 * i for i in range(n2)]})
    cc3 = pd.DataFrame({"A": [10.2 + 0.2 * i for i in range(n2)]})
    hh3 = pd.DataFrame({"A": [10.2 + 0.2 * i for i in range(n2)]})
    ll3 = pd.DataFrame({"A": [10.0 + 0.2 * i for i in range(n2)]})
    out3 = _op("candle_pattern_count").calculate(oo3, hh3, ll3, cc3, window=4)
    assert np.allclose(out3.iloc[3:, 0].to_numpy(), 0.0)


def test_candle_ops_mask_malformed_ohlc():
    vals = [12.0 + 1.2 * np.sin(i * 0.8) for i in range(40)]

    def fresh():
        return _ohlc_panel(vals)

    ops = ["candle_body_strength", "candle_wick_balance",
           "candle_range_pct", "candle_pattern_count"]

    def run(name, o, h, l, c, w=4):
        return _op(name).calculate(o, h, l, c, window=w)

    # high < low at row 15: bad bar NaN in every op (excluded, not laundered).
    o, h, l, c = fresh()
    h.iloc[15, 0], l.iloc[15, 0] = 9.0, 15.0
    for name in ops:
        out = run(name, o, h, l, c)
        if name == "candle_pattern_count":
            # the bad bar drops out of BOTH numerator and denominator: with the
            # other 3 window bars valid the fraction stays finite at row 15.
            assert np.isfinite(out.iloc[15, 0])
        else:
            assert np.isnan(out.iloc[15, 0]), f"{name}: high<low bar must be NaN"
            # Scoped-review P2: min_periods=window means the bad bar NaNs every
            # window covering it — with window=4 the NaN bleeds to rows 15..18
            # (fail-closed window kill, not plain skipna exclusion).
            assert out.iloc[15:19, 0].isna().all(), (
                f"{name}: bad bar must NaN its whole covering window"
            )
            assert np.isfinite(out.iloc[19, 0]), f"{name}: window must recover after the bad bar"
        assert np.isfinite(out.iloc[10, 0]), f"{name}: neighbours stay finite"

    # non-positive close at row 20.
    o, h, l, c = fresh()
    c.iloc[20, 0] = 0.0
    for name in ops:
        out = run(name, o, h, l, c)
        if name == "candle_pattern_count":
            assert np.isfinite(out.iloc[20, 0])
        else:
            assert np.isnan(out.iloc[20, 0]), f"{name}: close<=0 bar must be NaN"
        assert np.isfinite(out.iloc[10, 0])

    # zero range (high == low == open == close) at row 25: excluded, not a
    # fabricated 0-geometry contribution.
    o, h, l, c = fresh()
    o.iloc[25, 0] = h.iloc[25, 0] = l.iloc[25, 0] = c.iloc[25, 0] = 11.0
    for name in ops:
        out = run(name, o, h, l, c)
        if name == "candle_pattern_count":
            assert np.isfinite(out.iloc[25, 0])
        else:
            assert np.isnan(out.iloc[25, 0]), f"{name}: zero-range bar must be NaN"

    # NaN field at row 18: rows-presence fails -> pattern_count NaN too.
    o, h, l, c = fresh()
    h.iloc[18, 0] = float("nan")
    for name in ops:
        out = run(name, o, h, l, c)
        assert np.isnan(out.iloc[18, 0]), f"{name}: NaN-field bar must be NaN"


def test_candle_pattern_count_bad_bar_excluded_not_counted_as_nonpattern():
    # 4 valid bars where exactly 2 are dojis + 1 structurally-bad bar inside a
    # window=4: the fraction must be computed over the 3 VALID bars (2/3), not
    # 2/4 — a deformed bar is not evidence of "no pattern".
    o = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]})
    c = pd.DataFrame({"A": [10.0, 10.0, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2]})
    # bars: 0 doji, 1 doji, 2.. marubozu-ish (body/range=1 -> not pattern)
    h = pd.DataFrame({"A": [10.5, 10.5, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2]})
    l = pd.DataFrame({"A": [9.5, 9.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]})
    # make row 0 structurally bad (high < low) -> window covering rows 0..3 at
    # t=3 has 3 valid bars: doji(row1), marubozu(row2), marubozu(row3) -> 1/3.
    h.iloc[0, 0] = 9.0
    l.iloc[0, 0] = 11.0
    out = _op("candle_pattern_count").calculate(o, h, l, c, window=4)
    assert np.isclose(out.iloc[3, 0], 1.0 / 3.0)
    # t=4: window rows 1..4 all valid — the surviving doji (row 1) is still
    # inside -> 1/4; t>=5: all-valid all-marubozu windows -> exactly 0.
    assert np.isclose(out.iloc[4, 0], 0.25)
    assert np.allclose(out.iloc[5:, 0].to_numpy(), 0.0)


def test_candle_prefix_invariance():
    vals = [10.0 + 1.4 * np.sin(i * 0.6) for i in range(50)]
    o, h, l, c = _ohlc_panel(vals)
    names = ["candle_body_strength", "candle_wick_balance",
             "candle_range_pct", "candle_pattern_count"]
    for name in names:
        full = _op(name).calculate(o, h, l, c, window=5)
        head = _op(name).calculate(o.iloc[:35], h.iloc[:35], l.iloc[:35], c.iloc[:35], window=5)
        pd.testing.assert_frame_equal(full.iloc[:35], head)


def test_candle_param_specs_and_rolling_governance():
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    names = {"candle_body_strength", "candle_wick_balance",
             "candle_range_pct", "candle_pattern_count"}
    for name in names:
        specs = _op(name).metadata.param_specs
        assert set(specs) == {"window"}, f"{name} ParamSpec keys: {sorted(specs)}"
        assert specs["window"].dtype is int, f"{name}.window dtype"
        # Rolling-only bounded window state — deliberately NOT stateful.
        assert name not in _RECURSIVE_EWM, f"{name} must not be in _RECURSIVE_EWM"
        tags = set(_op(name).metadata.tags or [])
        assert "stateful" not in tags, f"{name} tags: {tags}"
        assert {"causal", "pit_safe"} <= tags, f"{name} tags: {tags}"
    # window < 2 rejected at the call boundary.
    o, h, l, c = _ohlc_panel([10.0, 11.0, 12.0])
    with pytest.raises(ValueError):
        _op("candle_body_strength").calculate(o, h, l, c, window=1)


def test_relative_alpha_membership_and_binary_detector_exclusion():
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"candle_body_strength", "candle_wick_balance",
                "candle_range_pct", "candle_pattern_count"}
    assert promoted <= _RELATIVE_ALPHA_OPS
    # One-off binary detectors are NOT aggregate strength — must stay out.
    for binary in ("cdl_doji", "cdl_hammer", "cdl_engulfing", "cdl_shooting_star"):
        assert binary not in _RELATIVE_ALPHA_OPS, f"{binary} must not be promoted"
    from factor_engine.cleaned_operators.operator_surface import DAILY_FACTOR_MIGRATED, _TECHNICAL_V2_CANONICALS

    assert promoted <= DAILY_FACTOR_MIGRATED
    assert promoted <= _TECHNICAL_V2_CANONICALS
