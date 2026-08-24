# -*- coding: utf-8 -*-
"""R20 Ichimoku-family DirectUse rehabilitation oracles.

CAUSAL, dimensionless Ichimoku canonicals — NO future shift.  The classic
Senkou A/B 26-bar forward displacement is DISPLAY-ONLY (honouring it would
read future bars), and the classic Chikou span (close compared against bars
26 ahead) is likewise future-looking.  Every canonical here uses trailing
windows ENDING at t:

* ``tenkan_kijun_cross`` — sign((Tenkan - Kijun)/close) in {-1,0,+1}
* ``chikou_distance_pct`` — (close - Kijun)/close, the causal equilibrium
  distance (the classic shifted Chikou comparison is NOT used)
* ``senkou_span_causal_pct`` — ((Tenkan_t + Kijun_t)/2 - close)/close, the
  un-shifted cloud-mid vs price gap

All windows are rolling-only with min_periods=window (bounded state) => NOT
in ``_RECURSIVE_EWM``, and the outputs are prefix-invariant under truncation
(the no-future-shift oracle).  The module bootstraps only the technical chain
(same import order as ``load_all``) so the KAMA override pin resolves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("tenkan_kijun_cross", "pandas_numpy") is not None:
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


def _trending_panel(n=60, seed=7):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.05, 0.6, n))})
    high = close + 0.5
    low = close - 0.5
    return high, low, close


def _midpoint(high, low, window):
    """Manual oracle for a trailing Ichimoku midpoint line."""
    w = int(window)
    return (high.rolling(w, min_periods=w).max() + low.rolling(w, min_periods=w).min()) / 2.0


def test_tenkan_kijun_cross_matches_manual_oracle():
    high, low, close = _trending_panel()
    t, k = 5, 12
    tenkan = _midpoint(high, low, t)
    kijun = _midpoint(high, low, k)
    diff = (tenkan - kijun) / close.where(close > 0.0)
    expected = pd.DataFrame(np.sign(diff.to_numpy(float)), index=diff.index, columns=diff.columns)
    out = _op("tenkan_kijun_cross").calculate(high, low, close, tenkan_window=t, kijun_window=k)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: Kijun (the longer window) is first valid at index k-1.
    assert out.iloc[: k - 1].isna().all().all()
    assert out.iloc[k:].notna().all().all()
    # Bounded regime values only.
    vals = out.dropna().to_numpy()
    assert set(np.unique(vals)) <= {-1.0, 0.0, 1.0}


def test_tenkan_kijun_cross_regime_flips():
    # Deterministic V: falling then rising.  Tenkan(3) reacts first, so the
    # cross must go negative then positive — proving both regimes are live.
    n = 30
    leg = np.linspace(20.0, 10.0, n // 2)
    close = pd.DataFrame({"A": np.concatenate([leg, leg[::-1]])})
    high = close + 0.2
    low = close - 0.2
    out = _op("tenkan_kijun_cross").calculate(high, low, close, tenkan_window=3, kijun_window=10)
    vals = out.iloc[10:].dropna().to_numpy()
    assert (vals < 0).any() and (vals > 0).any()


def test_tenkan_kijun_cross_tie_emits_zero():
    # Flat OHLC keeps Tenkan == Kijun exactly -> sign 0, never NaN/inf.
    frame = pd.DataFrame({"A": [10.0] * 20})
    out = _op("tenkan_kijun_cross").calculate(frame, frame, frame, tenkan_window=3, kijun_window=6)
    assert out.iloc[5:].notna().all().all()
    assert np.allclose(out.iloc[5:].to_numpy(), 0.0)


def test_tenkan_kijun_cross_rejects_window_swap():
    high, low, close = _trending_panel(n=20)
    with pytest.raises(ValueError):
        _op("tenkan_kijun_cross").calculate(high, low, close, tenkan_window=9, kijun_window=4)


def test_chikou_distance_pct_matches_manual_oracle():
    high, low, close = _trending_panel()
    k = 9
    kijun = _midpoint(high, low, k)
    pos = close.where(close > 0.0)
    expected = (pos - kijun) / pos
    out = _op("chikou_distance_pct").calculate(high, low, close, kijun_window=k)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: first k-1 bars are NaN; post-warmup is fully populated.
    assert out.iloc[: k - 1].isna().all().all()
    assert out.iloc[k - 1:].notna().all().all()


def test_chikou_distance_pct_is_causal_no_chikou_shift():
    # The classic Chikou compares close against the series 26 bars AHEAD
    # (future).  This canonical must read only trailing data: change the LAST
    # 26 bars' future (none exists) — instead assert the value at t depends
    # only on bars <= t by perturbing bars AFTER t and requiring t unchanged.
    n = 60
    rng = np.random.default_rng(5)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.0, 0.8, n))})
    high = close + 0.4
    low = close - 0.4
    base = _op("chikou_distance_pct").calculate(high, low, close, kijun_window=9)
    cut = 40
    trunc = _op("chikou_distance_pct").calculate(
        high.iloc[:cut], low.iloc[:cut], close.iloc[:cut], kijun_window=9
    )
    pd.testing.assert_frame_equal(base.iloc[:cut], trunc)


def test_chikou_distance_pct_masks_non_positive_close():
    high, low, close = _trending_panel(n=30, seed=3)
    close.iloc[10, 0] = -1.0
    out = _op("chikou_distance_pct").calculate(high, low, close, kijun_window=6)
    assert np.isnan(out.iloc[10, 0])
    assert np.isfinite(out.iloc[-1, 0])


def test_senkou_span_causal_pct_matches_manual_oracle():
    high, low, close = _trending_panel()
    t, k = 4, 10
    mid = (_midpoint(high, low, t) + _midpoint(high, low, k)) / 2.0
    pos = close.where(close > 0.0)
    expected = (mid - pos) / pos
    out = _op("senkou_span_causal_pct").calculate(high, low, close, tenkan_window=t, kijun_window=k)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: governed by the LONGER window (Kijun): first k-1 bars NaN.
    assert out.iloc[: k - 1].isna().all().all()
    assert out.iloc[k - 1:].notna().all().all()


def test_senkou_span_causal_pct_has_no_future_shift():
    # NO shift(26): on a constant panel the cloud mid equals close, so the
    # gap is exactly 0 at every post-warmup bar (a shifted implementation
    # would also be 0 here, so ALSO check a step panel where shift != no-shift:
    # price steps 10 -> 12 at index 25; a no-shift op jumps immediately at 25,
    # a shift(26) op would stay flat there).
    n = 60
    close = pd.DataFrame({"A": np.concatenate([np.full(25, 10.0), np.full(n - 25, 12.0)])})
    high = close + 0.1
    low = close - 0.1
    out = _op("senkou_span_causal_pct").calculate(high, low, close, tenkan_window=3, kijun_window=9)
    # Before the step: mid == 10, close == 10 -> exactly 0.
    assert np.allclose(out.iloc[8:25].to_numpy(), 0.0)
    # AT the step bar (t=25) the trailing mid still averages the old regime,
    # so the gap reacts IMMEDIATELY (negative: price above the trailing mid)
    # — a display-shifted implementation would delay this by 26 bars.  The
    # gap then decays back to exactly 0 once both windows are fully post-step
    # (t >= 25 + 9), which only a trailing-window op can do.
    assert out.iloc[25, 0] < 0.0
    assert (out.iloc[25:33] < 0.0).all().all()
    assert np.allclose(out.iloc[33:].to_numpy(), 0.0)


def test_senkou_span_causal_pct_masks_non_positive_close():
    high, low, close = _trending_panel(n=30, seed=11)
    close.iloc[15, 0] = 0.0  # exactly zero is also bad data
    out = _op("senkou_span_causal_pct").calculate(high, low, close, tenkan_window=3, kijun_window=9)
    assert np.isnan(out.iloc[15, 0])
    assert np.isfinite(out.iloc[-1, 0])


def test_senkou_span_causal_pct_rejects_window_swap():
    high, low, close = _trending_panel(n=20)
    with pytest.raises(ValueError):
        _op("senkou_span_causal_pct").calculate(high, low, close, tenkan_window=9, kijun_window=3)


def test_ichimoku_family_is_prefix_invariant_under_truncation():
    # Rolling trailing windows, bounded state: a truncated panel must
    # reproduce every overlapping prefix value bit-exactly — the direct
    # no-future-shift oracle (any leak from t+1.. would break equality).
    high, low, close = _trending_panel(n=50, seed=13)
    cut = 32
    for name, kwargs in (
        ("tenkan_kijun_cross", {"tenkan_window": 5, "kijun_window": 14}),
        ("chikou_distance_pct", {"kijun_window": 11}),
        ("senkou_span_causal_pct", {"tenkan_window": 5, "kijun_window": 14}),
    ):
        full = _op(name).calculate(high, low, close, **kwargs)
        head = _op(name).calculate(
            high.iloc[:cut], low.iloc[:cut], close.iloc[:cut], **kwargs
        )
        pd.testing.assert_frame_equal(full.iloc[:cut], head)


def test_ichimoku_family_param_specs_relational_and_promotion():
    expected_params = {
        "tenkan_kijun_cross": {"tenkan_window", "kijun_window"},
        "chikou_distance_pct": {"kijun_window"},
        "senkou_span_causal_pct": {"tenkan_window", "kijun_window"},
    }
    for name, params in expected_params.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is not None, f"{name}.{p} has no dtype"

    # Relational specs: tenkan < kijun pruning (mirror PPO fast/slow).
    for name in ("tenkan_kijun_cross", "senkou_span_causal_pct"):
        rels = _op(name).metadata.relational_specs
        assert rels, f"{name} has no relational spec"
        assert any("tenkan_window < kijun_window" in str(r.expression) for r in rels)
    assert _op("chikou_distance_pct").metadata.relational_specs == []

    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"tenkan_kijun_cross", "chikou_distance_pct", "senkou_span_causal_pct"}
    assert promoted <= _RELATIVE_ALPHA_OPS
    # Raw price-scale Ichimoku span levels must stay intermediate-only.
    for raw in ("ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b"):
        assert raw not in _RELATIVE_ALPHA_OPS


def test_ichimoku_family_not_recursive_ewm_and_daily_surface():
    # Rolling-only: bounded state, no full_replay governance tag.
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    for name in ("tenkan_kijun_cross", "chikou_distance_pct", "senkou_span_causal_pct"):
        assert name not in _RECURSIVE_EWM
        assert "stateful" not in _op(name).metadata.tags

    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    for name in ("tenkan_kijun_cross", "chikou_distance_pct", "senkou_span_causal_pct"):
        assert classify_canonical(name) == "daily"
