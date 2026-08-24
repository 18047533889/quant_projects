# -*- coding: utf-8 -*-
"""R20 MA-distance/slope/crossover DirectUse rehabilitation oracles.

Causal, dimensionless MA-family canonicals (NOT new MAs): every op reuses
existing MA machinery (EMA / DEMA / TEMA helpers and the registered KAMA) and
normalizes by strict-positive close so a bad (non-positive) close bar becomes
NaN — never a laundered value.

* ``ema_distance_pct`` / ``sma_distance_pct`` / ``dema_distance_pct`` /
  ``tema_distance_pct`` / ``kama_distance_pct`` — (close - MA) / close
* ``ma_slope_pct`` — EMA.diff() / close (dimensionless slope)
* ``ema_crossover`` — sign((fast EMA - slow EMA) / close) in {-1, 0, +1}

All windows end at t (PIT); EMA-family results are prefix-computable, so
truncated-panel equality is an exact oracle.  The module bootstraps the
technical chain (same import order as ``load_all``) so the KAMA override pin
resolves regardless of registry state.
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


def _close(values):
    return pd.DataFrame({"A": [float(v) for v in values]})


def test_ema_distance_pct_manual_oracle_and_prefix_invariance():
    vals = list(np.linspace(10.0, 15.0, 40)) + [9.0]
    close = _close(vals)
    # pandas EMA oracle computed independently of the operator's helper call.
    expected = close.ewm(span=5, adjust=False, min_periods=5).mean()
    expected = (close - expected) / close.where(close > 0.0)
    out = _op("ema_distance_pct").calculate(close, window=5)
    pd.testing.assert_frame_equal(out, expected)
    # Prefix-computable: the EMA uses only bars up to t, so truncating the
    # panel must not change any overlapping value.
    head = _op("ema_distance_pct").calculate(close.iloc[:30], window=5)
    pd.testing.assert_frame_equal(out.iloc[:30], head)


def test_sma_distance_pct_manual_oracle_and_warmup_nan():
    vals = list(np.linspace(10.0, 16.0, 30))
    close = _close(vals)
    out = _op("sma_distance_pct").calculate(close, window=4)
    expected = (close - close.rolling(4, min_periods=4).mean()) / close.where(close > 0.0)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: an SMA with window=4 needs 4 bars — first three are NaN.
    assert out.iloc[:3, 0].isna().all()
    assert np.isfinite(out.iloc[3:, 0]).all()


def test_dema_and_tema_distance_manual_oracles():
    # DEMA / TEMA recomputed from an INDEPENDENT EMA chain (oracle does not
    # call the module's DEMA/TEMA helpers).
    vals = [10.0 + 1.5 * np.sin(i * 0.7) for i in range(40)]
    close = _close(vals)
    e1 = close.ewm(span=6, adjust=False, min_periods=6).mean()
    e2 = e1.ewm(span=6, adjust=False, min_periods=6).mean()
    e3 = e2.ewm(span=6, adjust=False, min_periods=6).mean()
    dema = 2.0 * e1 - e2
    tema = 3.0 * e1 - 3.0 * e2 + e3
    pos = close.where(close > 0.0)
    out_d = _op("dema_distance_pct").calculate(close, window=6)
    pd.testing.assert_frame_equal(out_d, (pos - dema) / pos)
    out_t = _op("tema_distance_pct").calculate(close, window=6)
    pd.testing.assert_frame_equal(out_t, (pos - tema) / pos)
    # Cross-check against the registered DEMA/TEMA canonicals themselves.
    from factor_engine.cleaned_operators.technical.indicators_v2 import DEMA as _DEMA, TEMA as _TEMA

    pd.testing.assert_frame_equal(
        out_d, (pos - _DEMA(close, 6)) / pos
    )
    pd.testing.assert_frame_equal(
        out_t, (pos - _TEMA(close, 6)) / pos
    )


def test_kama_distance_reuses_registered_kama():
    vals = [10.0 + 1.2 * np.sin(i * 0.5) for i in range(60)]
    close = _close(vals)
    out = _op("kama_distance_pct").calculate(close, er_window=8, fast_window=2, slow_window=30)
    from factor_engine.cleaned_operators.technical.indicators_v2 import KAMA

    pos = close.where(close > 0.0)
    expected = (pos - KAMA(close, 8, 2, 30)) / pos
    pd.testing.assert_frame_equal(out, expected)
    # KAMA needs er_window+1 contiguous finite prices: the warmup prefix is NaN.
    assert out.iloc[:8, 0].isna().all()
    assert np.isfinite(out.iloc[20:, 0]).all()
    # Scoped-review F1 (P1): the fast<slow relational contract must prune the
    # search space up front, mirroring the KAMA canonical's own spec.
    rel = _op("kama_distance_pct").metadata.relational_specs or []
    assert rel, "kama_distance_pct has no relational_specs"


def test_ema_family_prefix_invariance():
    # Scoped-review F2: prefix-invariance for the whole EMA-recursive family,
    # not just ema_distance_pct — EMA state is prefix-computable by
    # construction, so truncation must not change any overlapping value.
    vals = [10.0 + 1.5 * np.sin(i * 0.6) for i in range(50)]
    close = _close(vals)
    cases = {
        "dema_distance_pct": dict(window=6),
        "tema_distance_pct": dict(window=6),
        "kama_distance_pct": dict(er_window=8, fast_window=2, slow_window=30),
        "ma_slope_pct": dict(window=5),
        "ema_crossover": dict(fast_window=3, slow_window=10),
    }
    for name, kw in cases.items():
        full = _op(name).calculate(close, **kw)
        head = _op(name).calculate(close.iloc[:35], **kw)
        pd.testing.assert_frame_equal(full.iloc[:35], head)


def test_ma_slope_pct_manual_oracle_and_sign():
    # Steadily rising close -> EMA rises bar over bar -> positive slope;
    # the manual oracle recomputes EMA.diff()/close independently.
    vals = list(np.linspace(10.0, 20.0, 30))
    close = _close(vals)
    out = _op("ma_slope_pct").calculate(close, window=5)
    ema = close.ewm(span=5, adjust=False, min_periods=5).mean()
    pd.testing.assert_frame_equal(out, ema.diff() / close.where(close > 0.0))
    # First bar has no EMA difference (diff -> NaN), second is post-seed.
    assert np.isnan(out.iloc[0, 0])
    tail = out.iloc[2:, 0].dropna()
    assert (tail > 0.0).all()
    # Dimensionless sanity: slope magnitude is far below 1 per bar.
    assert tail.max() < 0.5


def test_ema_crossover_semantics_and_ordering_guard():
    # Fast EMA(3) tracks the ramps earlier than slow EMA(10): during a rise
    # fast > slow -> +1, during a fall fast < slow -> -1.  Cross-check against
    # an independently computed sign oracle.
    rising = list(np.linspace(10.0, 20.0, 15))
    falling = list(np.linspace(20.0, 8.0, 25))
    close = _close(rising + falling)
    out = _op("ema_crossover").calculate(close, fast_window=3, slow_window=10)
    ef = close.ewm(span=3, adjust=False, min_periods=3).mean()
    es = close.ewm(span=10, adjust=False, min_periods=10).mean()
    expected = np.sign(((ef - es) / close.where(close > 0.0)).to_numpy(float))
    np.testing.assert_array_equal(out.to_numpy(float), expected)
    # Warmup (before the slow EMA has its 10 observations) is NaN, never a
    # fake sign: indices 0..8 are NaN, index 9 is the first legal value.
    assert np.isnan(out.iloc[8, 0])
    assert np.isfinite(out.iloc[9, 0])
    # Late in the sustained fall the fast EMA is strictly below -> -1.
    assert out.iloc[-1, 0] == -1.0
    # All finite values are in {-1, 0, +1} (bounded regime sign).
    finite = out.dropna().to_numpy(float)
    assert set(np.unique(finite)) <= {-1.0, 0.0, 1.0}
    # fast_window >= slow_window is rejected (search-space contract).
    with pytest.raises(ValueError):
        _op("ema_crossover").calculate(close, fast_window=10, slow_window=3)
    with pytest.raises(ValueError):
        _op("ema_crossover").calculate(close, fast_window=5, slow_window=5)


def test_distance_ops_mask_non_positive_close():
    # A single non-positive close bar must be NaN in EVERY distance/slope op
    # (strict-positive masking — never abs()), while neighbours stay finite.
    vals = [10.0 + 1.5 * np.sin(i * 0.8) for i in range(30)]
    close = _close(vals)
    close.iloc[15, 0] = 0.0
    for name, kwargs in [
        ("ema_distance_pct", {"window": 5}),
        ("sma_distance_pct", {"window": 5}),
        ("dema_distance_pct", {"window": 5}),
        ("tema_distance_pct", {"window": 5}),
        ("kama_distance_pct", {"er_window": 5, "fast_window": 2, "slow_window": 30}),
        ("ma_slope_pct", {"window": 5}),
        ("ema_crossover", {"fast_window": 3, "slow_window": 8}),
    ]:
        out = _op(name).calculate(close, **kwargs)
        assert np.isnan(out.iloc[15, 0]), f"{name}: non-positive close must be NaN"
        # TEMA's third EMA layer has the longest warmup (3*(window-1)+1 bars),
        # so a TEMA-family op's own warmup extends past index 10; every other
        # op must be finite at the pre-gap neighbour bar.
        if name != "tema_distance_pct":
            assert np.isfinite(out.iloc[10, 0]), f"{name}: neighbour bar must stay finite"


def test_ma_family_param_specs_and_recursive_governance():
    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    expected_specs = {
        "ema_distance_pct": {"window"},
        "sma_distance_pct": {"window"},
        "dema_distance_pct": {"window"},
        "tema_distance_pct": {"window"},
        "kama_distance_pct": {"er_window", "fast_window", "slow_window"},
        "ma_slope_pct": {"window"},
        "ema_crossover": {"fast_window", "slow_window"},
    }
    for name, params in expected_specs.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is not None, f"{name}.{p} has no dtype"
        # EMA/DEMA/TEMA/KAMA-recursive ops are stateful/full_replay; the pure
        # rolling SMA distance is NOT (bounded window state).  kama_distance_pct
        # is governed by the hard-coded stateful set next to KAMA in the tag
        # loop (NOT _RECURSIVE_EWM) — either container yields the same tags,
        # so the tag assertion below is the actual contract.
        if name == "kama_distance_pct":
            assert name not in _RECURSIVE_EWM
        elif name != "sma_distance_pct":
            assert name in _RECURSIVE_EWM, f"{name} missing from _RECURSIVE_EWM"
        if name != "sma_distance_pct":
            tags = set(_op(name).metadata.tags or [])
            assert {"stateful", "full_replay"} <= tags, f"{name} tags: {tags}"
        else:
            assert name not in _RECURSIVE_EWM
            tags = set(_op(name).metadata.tags or [])
            assert "stateful" not in tags
    # ema_crossover carries the fast<slow relational constraint.
    rels = _op("ema_crossover").metadata.relational_specs or []
    assert any("fast_window < slow_window" in r.expression for r in rels)


def test_relative_alpha_membership_and_raw_ma_exclusion():
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {
        "ema_distance_pct", "sma_distance_pct", "dema_distance_pct",
        "tema_distance_pct", "kama_distance_pct", "ma_slope_pct", "ema_crossover",
    }
    assert promoted <= _RELATIVE_ALPHA_OPS
    # Raw MA levels (price-scale) must NOT be promoted by this slice.
    for raw in ("KAMA", "DEMA", "TEMA", "ts_ema", "ts_sma"):
        assert raw not in _RELATIVE_ALPHA_OPS, f"{raw} must stay intermediate"
