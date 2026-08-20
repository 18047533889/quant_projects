# -*- coding: utf-8 -*-
"""R20 PSAR-family DirectUse rehabilitation oracles (P0).

The raw PSAR *level* stays intermediate (price-scale, recursive state).  These
canonicals are the causal, dimensionless, terminal-usable counterparts:

* ``psar_direction`` — PSAR regime sign (+1 bull / -1 bear)
* ``psar_distance_pct`` — (close - PSAR) / close, strict-positive close masked
* ``psar_flip`` — +1 on flip up, -1 on flip down, 0 hold
* ``psar_days_since_flip`` — bars since the last regime flip

All values are read from the same recursive PSAR state machine with no future
displacement; PSAR is ``stateful`` / ``full_replay`` so prefix-invariance does
NOT hold by design — the test instead proves full-history determinism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("psar_direction", mode="any") is not None:
        return
    from cleaned_operators.technical import signal  # noqa: F401
    from cleaned_operators.technical import polars_signal  # noqa: F401
    from cleaned_operators import composite_fastpath  # noqa: F401
    from cleaned_operators.technical import indicators_v2  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()


def _op(name: str):
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, mode="any")
    assert op is not None, f"{name} not registered"
    return op


def _panel(seed=7, n=200):
    rng = np.random.default_rng(seed)
    px = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    close = pd.DataFrame({"A": px})
    return close + 0.5, close - 0.5, close


def test_psar_direction_matches_close_minus_sar_sign():
    high, low, close = _panel()
    from cleaned_operators.technical.indicators_v2 import PSAR

    sar = PSAR(high, low, 0.02, 0.2)
    out = _op("psar_direction").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    valid = sar.notna() & close.notna()
    assert valid.sum().iloc[0] > 20  # non-vacuous
    expected = pd.DataFrame(
        np.where(close > sar, 1.0, -1.0), index=sar.index, columns=sar.columns
    ).where(valid)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup (SAR NaN) must stay NaN, never a manufactured sign.
    assert np.isnan(out.iloc[0, 0])


def test_psar_distance_pct_formula_and_positive_close_masking():
    high, low, close = _panel()
    from cleaned_operators.technical.indicators_v2 import PSAR

    sar = PSAR(high, low, 0.02, 0.2)
    out = _op("psar_distance_pct").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    pos = close.where(close > 0)
    expected = (pos - sar).div(pos)
    pd.testing.assert_frame_equal(out, expected)

    # Bad close (<= 0) must become NaN, never a silently flipped ratio.
    close.iloc[100, 0] = -5.0
    out_bad = _op("psar_distance_pct").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    assert np.isnan(out_bad.iloc[100, 0])


def test_psar_flip_on_constructed_reversal():
    # Monotone uptrend then hard monotone downtrend -> exactly one up->down flip.
    up = np.linspace(10.0, 30.0, 40)
    down = np.linspace(30.0, 10.0, 40)
    close = pd.DataFrame({"A": np.concatenate([up, down])})
    high, low = close + 0.2, close - 0.2
    flip = _op("psar_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    vals = flip.iloc[:, 0].to_numpy()
    valid = vals[~np.isnan(vals)]
    assert valid.size > 10  # non-vacuous
    assert set(np.unique(valid)) <= {-1.0, 0.0, 1.0}
    downs = np.where(vals == -1.0)[0]
    assert len(downs) >= 1
    # Every flip must coincide with a sign change of psar_direction.
    direction = _op("psar_direction").calculate(high, low, close, acceleration=0.02, maximum=0.2).iloc[:, 0].to_numpy()
    for t in downs:
        assert direction[t - 1] == 1.0 and direction[t] == -1.0
    # No flip before direction is twice-finite (warmup NaN region).
    assert np.isnan(vals[0])


def test_psar_flip_is_zero_when_direction_holds():
    high, low, close = _panel(seed=11)
    flip = _op("psar_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    direction = _op("psar_direction").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    d = direction.iloc[:, 0].to_numpy()
    f = flip.iloc[:, 0].to_numpy()
    holds = 0
    for t in range(1, len(d)):
        if np.isfinite(d[t]) and np.isfinite(d[t - 1]) and d[t] == d[t - 1]:
            assert f[t] == 0.0
            holds += 1
    assert holds > 10  # non-vacuous


def test_psar_days_since_flip_arithmetic():
    high, low, close = _panel()
    flip = _op("psar_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    days = _op("psar_days_since_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    f = flip.iloc[:, 0].to_numpy()
    d = days.iloc[:, 0].to_numpy()
    seen = 0
    for t in range(len(f)):
        if np.isnan(f[t]):
            assert np.isnan(d[t])
            continue
        if f[t] != 0.0:
            assert d[t] == 0.0
            seen += 1
        else:
            # hold bar after a flip: exactly one more than the previous count.
            # A NaN previous slot means pre-first-flip warmup: stays NaN.
            if not np.isnan(d[t - 1]):
                assert d[t] == d[t - 1] + 1.0
            else:
                assert np.isnan(d[t])
    assert seen >= 2  # non-vacuous: at least two flips in this panel


def test_psar_family_param_specs_and_stateful_governance():
    expected = {
        "psar_direction": {"acceleration", "maximum"},
        "psar_distance_pct": {"acceleration", "maximum"},
        "psar_flip": {"acceleration", "maximum"},
        "psar_days_since_flip": {"acceleration", "maximum"},
    }
    for name, params in expected.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is float, f"{name}.{p} dtype"
            assert specs[p].min is not None and specs[p].min > 0
        tags = set(_op(name).metadata.tags)
        assert {"stateful", "full_replay", "causal", "pit_safe"} <= tags, name


def test_psar_rejects_bad_acceleration_maximum():
    high, low, close = _panel(n=40)
    with pytest.raises(ValueError):
        _op("psar_direction").calculate(high, low, close, acceleration=0.0, maximum=0.2)
    with pytest.raises(ValueError):
        _op("psar_distance_pct").calculate(high, low, close, acceleration=0.5, maximum=0.2)


def test_psar_full_history_determinism():
    # PSAR is recursive full-replay: prefix truncation is NOT required to be
    # invariant; instead the same full history must reproduce bit-identically.
    high, low, close = _panel(seed=21)
    a = _op("psar_days_since_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    b = _op("psar_days_since_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    pd.testing.assert_frame_equal(a, b)
    f1 = _op("psar_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    f2 = _op("psar_flip").calculate(high, low, close, acceleration=0.02, maximum=0.2)
    pd.testing.assert_frame_equal(f1, f2)
    # Determinism must extend to a duplicated column (state independent per column).
    close2 = close.copy(); close2["B"] = close["A"].to_numpy()
    high2 = high.copy(); high2["B"] = high["A"].to_numpy()
    low2 = low.copy(); low2["B"] = low["A"].to_numpy()
    multi = _op("psar_direction").calculate(high2, low2, close2, acceleration=0.02, maximum=0.2)
    pd.testing.assert_series_equal(multi["A"], multi["B"], check_names=False)


def test_psar_relative_alpha_classification():
    from mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"psar_direction", "psar_distance_pct", "psar_flip", "psar_days_since_flip"}
    assert promoted <= _RELATIVE_ALPHA_OPS
    # Raw price-scale PSAR level must NOT be promoted by this slice.
    assert "PSAR" not in _RELATIVE_ALPHA_OPS
