# -*- coding: utf-8 -*-
"""R20 rolling-VWAP DirectUse rehabilitation oracles.

Causal, dimensionless rolling-VWAP canonicals (trailing window ENDS at t,
min_periods=window, rolling-only bounded state — no full-replay governance):

* ``vwap_distance_pct`` — (close - rolling VWAP(close*volume, volume, w))/close
* ``vwap_slope_pct`` — rolling VWAP.diff()/close (dimensionless slope)
* ``vwap_premium_pct`` — (close - HLC3 rolling VWAP)/close (typical-price
  variant; distinct from the close-only kernel)

Data-integrity contract: a non-positive close bar is NaN (strict-positive
masking, R5-38) and a bad volume bar (volume <= 0 or NaN) is excluded from
BOTH rolling sums; a window with no valid volume is NaN — bad data is never
laundered into a fake price.  Manual pandas oracles recompute every kernel
independently of the operator helper; prefix-invariance (truncated panel ==
full-panel prefix) is exact for the rolling-only family.
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


def _panel(values, volumes=None):
    close = pd.DataFrame({"A": [float(v) for v in values]})
    if volumes is None:
        rng = np.random.default_rng(7)
        volumes = 1.0e6 + rng.random(len(values)) * 5.0e5
    vol = pd.DataFrame({"A": [float(v) for v in volumes]})
    return close, vol


def _oracle_vwap(price, volume, window):
    """Independent manual rolling-VWAP oracle (strict valid-volume cohort)."""
    valid = price.notna() & volume.notna() & (volume > 0.0)
    num = (price * volume).where(valid).rolling(window, min_periods=window).sum()
    den = volume.where(valid).rolling(window, min_periods=window).sum()
    return num / den.where(den > 0.0)


def test_vwap_distance_pct_manual_oracle_and_warmup_nan():
    vals = [10.0 + 1.5 * np.sin(i * 0.7) + 0.05 * i for i in range(40)]
    close, vol = _panel(vals)
    w = 5
    out = _op("vwap_distance_pct").calculate(close, vol, window=w)
    pos = close.where(close > 0.0)
    expected = (pos - _oracle_vwap(close, vol, w)) / pos
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: window=5 needs 5 valid bars — first four are NaN.
    assert out.iloc[:4, 0].isna().all()
    assert np.isfinite(out.iloc[4:, 0]).all()


def test_vwap_slope_pct_manual_oracle_and_sign():
    # Steadily rising close -> the VWAP cost basis rises bar over bar -> the
    # per-bar slope is positive once past warmup.
    vals = list(np.linspace(10.0, 20.0, 35))
    close, vol = _panel(vals)
    w = 4
    out = _op("vwap_slope_pct").calculate(close, vol, window=w)
    vwap = _oracle_vwap(close, vol, w)
    pd.testing.assert_frame_equal(out, vwap.diff() / close.where(close > 0.0))
    # index w-1 is the first legal VWAP value (0-based, min_periods=4); its
    # diff is NaN there, and index w carries the first finite slope.
    assert np.isnan(out.iloc[w - 1, 0])
    assert np.isfinite(out.iloc[w, 0])
    tail = out.iloc[w + 1 :, 0].dropna()
    assert (tail > 0.0).all()
    # Dimensionless sanity: per-bar slope magnitude far below 1.
    assert tail.max() < 0.5


def test_vwap_premium_pct_manual_oracle_and_distinctness():
    vals = [10.0 + 1.2 * np.sin(i * 0.5) + 0.02 * i for i in range(45)]
    close, vol = _panel(vals)
    high = close + 0.3
    low = close - 0.25
    w = 6
    out = _op("vwap_premium_pct").calculate(high, low, close, vol, window=w)
    tp = (high + low + close) / 3.0
    pos = close.where(close > 0.0)
    expected = (pos - _oracle_vwap(tp, vol, w)) / pos
    pd.testing.assert_frame_equal(out, expected)
    # Distinctness: with a wide high/low spread the HLC3 premium differs from
    # the close-only distance on materially many bars (not a near-duplicate).
    dist = _op("vwap_distance_pct").calculate(close, vol, window=w)
    both = out.iloc[w:, 0].dropna().index.intersection(dist.iloc[w:, 0].dropna().index)
    diff = (out.loc[both, "A"] - dist.loc[both, "A"]).abs()
    assert (diff > 1e-4).mean() > 0.5, "HLC3 variant must be materially distinct"


def test_vwap_ops_mask_non_positive_and_missing_volume():
    # A volume <= 0 (or NaN) bar is excluded from BOTH sums; with
    # min_periods=window the whole overlapping window goes NaN.  A non-positive
    # close is NaN in every op while volume-valid neighbours stay finite.
    vals = [10.0 + 1.5 * np.sin(i * 0.8) for i in range(30)]
    vols = [1.0e6 + 1.0e5 * ((i % 5) + 1) for i in range(30)]
    close, vol = _panel(vals, vols)
    # Bad close at 15 (volume fine): distance/premium/slope all NaN there.
    close.iloc[15, 0] = 0.0
    out_d = _op("vwap_distance_pct").calculate(close, vol, window=4)
    out_s = _op("vwap_slope_pct").calculate(close, vol, window=4)
    assert np.isnan(out_d.iloc[15, 0])
    assert np.isnan(out_s.iloc[15, 0])
    # Neighbouring valid bars stay finite (window=4 fully valid around 15).
    assert np.isfinite(out_d.iloc[10, 0])
    assert np.isfinite(out_s.iloc[10, 0])

    # Zero / negative / NaN volume on a bar kills every window covering it
    # (min_periods=4, one invalid bar -> NaN for 4 bars).
    close2, vol2 = _panel(
        [10.0 + 1.5 * np.sin(i * 0.8) for i in range(30)],
        [1.0e6 + 1.0e5 * ((i % 5) + 1) for i in range(30)],
    )
    # Clean baseline computed BEFORE poisoning (scoped-review P1: computing it
    # from vol2.copy() after the mutation made the prefix comparison vacuous).
    clean = _op("vwap_distance_pct").calculate(close2, vol2.copy(), window=4)
    for bad in (0.0, -5.0e3, float("nan")):
        vol2.iloc[12, 0] = bad
        out = _op("vwap_distance_pct").calculate(close2, vol2, window=4)
        assert np.isnan(out.iloc[12, 0]), f"volume={bad}: bad bar must be NaN"
        assert np.isnan(out.iloc[13, 0]), f"volume={bad}: covering window must be NaN"
        # Before the poisoned window the values are finite and equal to the
        # clean-panel prefix (bad data must not leak backwards).
        pd.testing.assert_frame_equal(out.iloc[:9], clean.iloc[:9])


def test_vwap_window_with_all_zero_volume_is_nan():
    # Every volume in the window <= 0 -> volume sum <= 0 -> NaN (never 0/0
    # laundered into a fake price via a spurious 0.0).
    vals = list(np.linspace(10.0, 12.0, 12))
    close, _ = _panel(vals)
    vol = pd.DataFrame({"A": [0.0] * 12})
    out = _op("vwap_distance_pct").calculate(close, vol, window=3)
    assert out.isna().all().all()


def test_vwap_prefix_invariance():
    # Rolling-only kernels are prefix-computable: truncating the panel must
    # not change any overlapping value.
    vals = [10.0 + 1.4 * np.sin(i * 0.6) for i in range(50)]
    close, vol = _panel(vals)
    high = close + 0.2
    low = close - 0.2
    cases = {
        "vwap_distance_pct": lambda c, v: _op("vwap_distance_pct").calculate(c, v, window=5),
        "vwap_slope_pct": lambda c, v: _op("vwap_slope_pct").calculate(c, v, window=5),
    }
    for name, run in cases.items():
        full = run(close, vol)
        head = run(close.iloc[:35], vol.iloc[:35])
        pd.testing.assert_frame_equal(full.iloc[:35], head)
    full_p = _op("vwap_premium_pct").calculate(high, low, close, vol, window=5)
    head_p = _op("vwap_premium_pct").calculate(
        high.iloc[:35], low.iloc[:35], close.iloc[:35], vol.iloc[:35], window=5
    )
    pd.testing.assert_frame_equal(full_p.iloc[:35], head_p)


def test_vwap_param_specs_and_rolling_governance():
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    expected = {
        "vwap_distance_pct": {"window"},
        "vwap_slope_pct": {"window"},
        "vwap_premium_pct": {"window"},
    }
    for name, params in expected.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is not None, f"{name}.{p} has no dtype"
        # Rolling-only bounded window state — deliberately NOT stateful.
        assert name not in _RECURSIVE_EWM, f"{name} must not be in _RECURSIVE_EWM"
        tags = set(_op(name).metadata.tags or [])
        assert "stateful" not in tags, f"{name} tags: {tags}"
        assert {"causal", "pit_safe"} <= tags, f"{name} tags: {tags}"
    # window < 2 rejected at the call boundary (ParamSpec-style contract).
    close, vol = _panel([10.0, 11.0, 12.0])
    with pytest.raises(ValueError):
        _op("vwap_distance_pct").calculate(close, vol, window=1)


def test_relative_alpha_membership_and_raw_vwap_exclusion():
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"vwap_distance_pct", "vwap_slope_pct", "vwap_premium_pct"}
    assert promoted <= _RELATIVE_ALPHA_OPS
    # The raw price-scale VWAP level must stay intermediate, not promoted here.
    assert "rolling_vwap" not in _RELATIVE_ALPHA_OPS
    from factor_engine.cleaned_operators.operator_surface import DAILY_FACTOR_MIGRATED

    assert promoted <= DAILY_FACTOR_MIGRATED
