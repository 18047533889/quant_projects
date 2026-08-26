# -*- coding: utf-8 -*-
"""R20 Keltner-family DirectUse rehabilitation oracles (P0 slice 2).

Raw KeltnerMid / KeltnerUpper / KeltnerLower levels stay INTERMEDIATE
(price-scale).  These canonicals are the causal, dimensionless counterparts
promoted for direct alpha use:

* ``keltner_width_pct`` — (Upper-Lower)/Mid, strict-positive mid masking
* ``keltner_compression`` — trailing percentile rank (0..1) of the width ratio
* ``keltner_breakout_strength`` — signed breakout distance / band width

All rolling windows end at ``t`` (PIT-safe); no future data, no shift.  The
module bootstraps only the technical chain (same import order as ``load_all``)
so the KAMA override pin resolves regardless of registry state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("KeltnerPosition", "pandas_numpy") is not None:
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


def test_keltner_width_pct_matches_manual_oracle():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 40), spread=0.4)
    from factor_engine.cleaned_operators.technical.indicators_v2 import (
        KeltnerMid, KeltnerUpper, KeltnerLower,
    )

    u = KeltnerUpper(high, low, close, 20, 14, 2.0)
    l = KeltnerLower(high, low, close, 20, 14, 2.0)
    m = KeltnerMid(close, 20)
    expected = (u - l) / m.where(m > 0.0)
    out = _op("keltner_width_pct").calculate(
        high, low, close, ema_window=20, atr_window=14, multiplier=2.0
    )
    pd.testing.assert_frame_equal(out, expected)


def test_keltner_width_pct_masks_non_positive_mid():
    # A negative close drives the EMA mid negative -> width must be NaN, not a
    # meaningless negative ratio laundered by abs().  The EMA recovers to a
    # positive mid a few bars later, so the tail must be finite again.
    n = 40
    closes = [10.0] * n
    closes[10] = -50.0
    close = pd.DataFrame({"A": [float(v) for v in closes]})
    high = close + 0.4
    low = close - 0.4
    out = _op("keltner_width_pct").calculate(
        high, low, close, ema_window=3, atr_window=3, multiplier=2.0
    )
    # Every bar whose EMA mid is <= 0 must be NaN (strict-positive masking).
    from factor_engine.cleaned_operators.technical.indicators_v2 import KeltnerMid

    mid = KeltnerMid(close, 3).iloc[:, 0]
    for i in range(n):
        if np.isfinite(mid.iloc[i]) and mid.iloc[i] <= 0.0:
            assert np.isnan(out.iloc[i, 0]), f"row {i}: mid={mid.iloc[i]}"
    # The EMA has recovered by the tail -> strictly positive finite width.
    assert np.isfinite(out.iloc[-1, 0]) and out.iloc[-1, 0] > 0.0


def test_keltner_compression_is_bounded_and_trailing():
    high, low, close = _ohlc(
        np.linspace(10.0, 15.0, 40) + np.sin(np.arange(40) * 0.9) * 0.3
    )
    out = _op("keltner_compression").calculate(
        high, low, close, ema_window=10, atr_window=10, multiplier=2.0,
        score_window=10,
    )
    valid = out.dropna()
    assert ((valid >= 0.0) & (valid <= 1.0)).all().all()
    # Truncating the panel must not change any overlapping prefix value
    # (strictly trailing window ending at t — no future leakage).
    head = _op("keltner_compression").calculate(
        high.iloc[:30], low.iloc[:30], close.iloc[:30],
        ema_window=10, atr_window=10, multiplier=2.0, score_window=10,
    )
    pd.testing.assert_frame_equal(out.iloc[:30], head)


def test_keltner_breakout_strength_above_band():
    # An oscillating series keeps the EMA mid lagging the swings, then a final
    # spike pushes close far above the upper band -> positive
    # (close-Upper)/(Upper-Lower); the oscillating prefix stays inside (0).
    vals = [10 + 1.5 * np.sin(i * 0.9) for i in range(40)] + [40.0]
    close = pd.DataFrame({"A": [float(v) for v in vals]})
    high = close + 0.05
    low = close - 0.05
    from factor_engine.cleaned_operators.technical.indicators_v2 import (
        KeltnerUpper, KeltnerLower,
    )

    u = KeltnerUpper(high, low, close, 5, 5, 2.0)
    l = KeltnerLower(high, low, close, 5, 5, 2.0)
    span = (u - l).replace(0.0, np.nan)
    expected = (close - u).where(close > u, 0.0) / span
    out = _op("keltner_breakout_strength").calculate(
        high, low, close, ema_window=5, atr_window=5, multiplier=2.0
    )
    pd.testing.assert_frame_equal(out, expected)
    # The last bar is a genuine breakout -> strictly positive.
    assert out.iloc[-1, 0] > 0.0
    # Mid-panel oscillating bars (inside the band) are exactly zero.
    inside = out.iloc[10:35, 0].dropna()
    assert (inside == 0.0).all()


def test_keltner_breakout_strength_below_band():
    # Mirror of the above: oscillation then a final gap down drives close far
    # below the lower band -> negative (close-Lower)/(Upper-Lower).
    vals = [10 + 1.5 * np.sin(i * 0.9) for i in range(40)] + [-20.0]
    close = pd.DataFrame({"A": [float(v) for v in vals]})
    high = close + 0.05
    low = close - 0.05
    from factor_engine.cleaned_operators.technical.indicators_v2 import (
        KeltnerUpper, KeltnerLower,
    )

    u = KeltnerUpper(high, low, close, 5, 5, 2.0)
    l = KeltnerLower(high, low, close, 5, 5, 2.0)
    span = (u - l).replace(0.0, np.nan)
    expected = (close - l).where(close < l, 0.0) / span
    out = _op("keltner_breakout_strength").calculate(
        high, low, close, ema_window=5, atr_window=5, multiplier=2.0
    )
    pd.testing.assert_frame_equal(out, expected)
    # The last bar is below the lower band -> strictly negative.
    assert out.iloc[-1, 0] < 0.0
    # Mid-panel oscillating bars (inside the band) are exactly zero.
    inside = out.iloc[10:35, 0].dropna()
    assert (inside == 0.0).all()


def test_keltner_breakout_strength_inside_band_is_zero():
    # A flat, tight-bar panel keeps close at the midline -> inside the band -> 0.
    base = 10.0
    high = pd.DataFrame({"A": [base + 0.05] * 30})
    low = pd.DataFrame({"A": [base - 0.05] * 30})
    close = pd.DataFrame({"A": [base] * 30})
    out = _op("keltner_breakout_strength").calculate(
        high, low, close, ema_window=5, atr_window=5, multiplier=2.0
    )
    # After warmup, close sits at the midline inside the band -> exactly 0.
    assert np.allclose(out.iloc[10:].to_numpy(), 0.0, equal_nan=True)


def test_keltner_compression_direction_low_means_compressed():
    # Scoped-review P2: pin the ranking DIRECTION.  A smoothly decaying
    # oscillation amplitude shrinks the Keltner width monotonically, so the
    # current width is the minimum of its trailing window -> the trailing
    # pct-rank must sit near the bottom (a sign-inverted 1-rank impl would
    # sit near the top and fail this).
    n = 60
    amp = 2.0 * np.exp(-np.arange(n) / 20.0)
    vals = 10.0 + amp * np.sin(np.arange(n) * 0.9)
    close = pd.DataFrame({"A": vals})
    high = close + 0.05
    low = close - 0.05
    out = _op("keltner_compression").calculate(
        high, low, close, ema_window=5, atr_window=5, multiplier=2.0,
        score_window=10,
    )
    tail = out.iloc[-3:, 0].dropna()
    assert len(tail) == 3
    assert (tail <= 0.2).all(), f"decaying width must rank low, got {tail.tolist()}"


def test_keltner_breakout_strength_nan_close_stays_nan():
    # Scoped-review P2: pin "cannot judge != no breakout".  A NaN close bar
    # must produce NaN output, never a false 0 (inside-band), regardless of
    # which internal guard catches it (band NaN, span NaN, or the explicit
    # close/band notna re-mask).
    vals = [10 + 1.5 * np.sin(i * 0.9) for i in range(40)]
    close = pd.DataFrame({"A": [float(v) for v in vals]})
    close.iloc[20, 0] = np.nan
    high = close + 0.05
    low = close - 0.05
    out = _op("keltner_breakout_strength").calculate(
        high, low, close, ema_window=5, atr_window=5, multiplier=2.0
    )
    assert np.isnan(out.iloc[20, 0])
    # Bars well away from the NaN island are finite again.
    assert np.isfinite(out.iloc[10, 0])


def test_keltner_family_param_specs_and_recursive_stateful_tag():
    expected_specs = {
        "keltner_width_pct": {"ema_window", "atr_window", "multiplier"},
        "keltner_compression": {"ema_window", "atr_window", "multiplier", "score_window"},
        "keltner_breakout_strength": {"ema_window", "atr_window", "multiplier"},
    }
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    for name, params in expected_specs.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is not None, f"{name}.{p} has no dtype"
        # EMA/Wilder-derived state must be governed stateful/full_replay.
        assert name in _RECURSIVE_EWM, f"{name} missing from _RECURSIVE_EWM"
        tags = set(_op(name).metadata.tags or [])
        assert {"stateful", "full_replay"} <= tags, f"{name} tags: {tags}"


def test_keltner_compression_multiplier_is_rank_scale_invariant():
    """R50 injectivity audit: ``multiplier`` genuinely feeds the Keltner band
    width, but ``keltner_compression`` then takes a *trailing percentile rank*
    of that width, so any positive multiplier rescales the width monotonically
    and is normalised away by the rank.  The parameter is therefore NOT an
    injective alpha dimension on this transform — it is a band-width knob whose
    scale is washed out by the rank step (not a dead code path, and not a bug).
    This pins the behaviour so a future change to the transform (e.g. removing
    the rank) that would make the multiplier injective is caught here."""
    high, low, close = _ohlc(
        np.linspace(10.0, 15.0, 60) + np.sin(np.arange(60) * 0.9) * 0.4,
        spread=0.4,
    )
    base = _op("keltner_compression").calculate(
        high, low, close, ema_window=10, atr_window=10, multiplier=2.0, score_window=20
    )
    base_np = base.to_numpy(dtype=float)
    assert np.isfinite(base_np).any()
    for m in (0.5, 5.0, 20.0):
        out = _op("keltner_compression").calculate(
            high, low, close, ema_window=10, atr_window=10, multiplier=m, score_window=20
        )
        np.testing.assert_array_equal(out.to_numpy(dtype=float), base_np,
                                      err_msg=f"multiplier={m} changed output")


def test_keltner_relative_alpha_membership():
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"keltner_width_pct", "keltner_compression", "keltner_breakout_strength"}
    assert promoted <= _RELATIVE_ALPHA_OPS
    # Raw price-scale Keltner levels must NOT be promoted by this slice.
    assert "KeltnerMid" not in _RELATIVE_ALPHA_OPS
    assert "KeltnerUpper" not in _RELATIVE_ALPHA_OPS
    assert "KeltnerLower" not in _RELATIVE_ALPHA_OPS
