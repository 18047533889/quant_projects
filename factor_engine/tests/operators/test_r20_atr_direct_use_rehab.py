# -*- coding: utf-8 -*-
"""R20 ATR-family DirectUse rehabilitation oracles (P0 slice 1).

The raw Wilder ATR / true-range levels stay intermediate (price-scale).  These
canonicals are the dimensionless / causal counterparts promoted for direct
alpha use:

* ``atr_pct`` — ATR / close (NATR / 100)
* ``atr_zscore`` / ``atr_percentile`` — trailing standardization of ``atr_pct``
* ``true_range_pct`` — true range / previous close
* ``true_range_surprise`` / ``true_range_zscore`` — current-vs-trailing regime
* ``atr_short_long_ratio`` — short vs long ATR regime
* ``atr_acceleration`` — first difference of ``atr_pct``

All rolling windows end at ``t`` (PIT-safe); no future data, no shift.  The
module bootstraps only the technical chain (same import order as
``load_all``) so the KAMA override pin resolves regardless of registry state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("atr_pct", "pandas_numpy") is not None:
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

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


def _ohlc(values, spread=1.0):
    close = pd.DataFrame({"A": [float(v) for v in values]})
    return close + spread, close - spread, close


def test_atr_pct_is_natr_over_100():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 40), spread=0.4)
    ratio = _op("atr_pct").calculate(high, low, close, window=5)
    natr = _op("NATR").calculate(high, low, close, window=5)
    pd.testing.assert_frame_equal(ratio, natr / 100.0)


def test_atr_pct_masks_non_positive_close():
    high, low, close = _ohlc([10.0, 11.0, 12.0, 13.0, 14.0], spread=0.4)
    close.iloc[3, 0] = -1.0
    out = _op("atr_pct").calculate(high, low, close, window=3)
    assert np.isnan(out.iloc[3, 0])
    assert np.isfinite(out.iloc[-1, 0])


def test_atr_zscore_matches_manual_trailing_oracle():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 40) + np.sin(np.arange(40) * 0.7) * 0.2)
    from cleaned_operators.technical.indicators_v2 import atr_pct

    ratio = atr_pct(high, low, close, 5)
    mean = ratio.rolling(10, min_periods=10).mean()
    std = ratio.rolling(10, min_periods=10).std(ddof=1)
    expected = (ratio - mean) / std.replace(0.0, np.nan)
    out = _op("atr_zscore").calculate(high, low, close, window=5, score_window=10)
    pd.testing.assert_frame_equal(out, expected)


def test_atr_percentile_is_bounded_and_trailing():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 40) + np.sin(np.arange(40) * 0.9) * 0.3)
    out = _op("atr_percentile").calculate(high, low, close, window=5, score_window=10)
    valid = out.dropna()
    assert ((valid >= 0.0) & (valid <= 1.0)).all().all()
    # Truncating the panel must not change any overlapping prefix value.
    head = _op("atr_percentile").calculate(high.iloc[:30], low.iloc[:30], close.iloc[:30], window=5, score_window=10)
    pd.testing.assert_frame_equal(out.iloc[:30], head)


def test_true_range_pct_uses_previous_close():
    high, low, close = _ohlc([10.0, 12.0, 14.0, 16.0], spread=0.5)
    out = _op("true_range_pct").calculate(high, low, close)
    parts = [
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ]
    tr = pd.concat(parts, axis=1).max(axis=1).to_frame(name="A")
    expected = tr.div(close.shift(1).iloc[:, 0], axis=0)
    pd.testing.assert_frame_equal(out, expected)


def test_true_range_surprise_is_zero_for_constant_range():
    # Constant bar geometry + constant previous close keeps ratio / mean - 1 == 0.
    base = 10.0
    high = pd.DataFrame({"A": [base + 1.0] * 30})
    low = pd.DataFrame({"A": [base - 1.0] * 30})
    close = pd.DataFrame({"A": [base] * 30})
    out = _op("true_range_surprise").calculate(high, low, close, window=5)
    assert np.allclose(out.iloc[6:].to_numpy(), 0.0, equal_nan=True)


def test_atr_short_long_ratio_rejects_reversed_windows():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 40))
    with pytest.raises(ValueError, match="short_window"):
        _op("atr_short_long_ratio").calculate(high, low, close, short_window=10, long_window=3)


def test_atr_acceleration_is_first_difference():
    high, low, close = _ohlc(np.linspace(10.0, 15.0, 40) + np.sin(np.arange(40) * 0.5) * 0.2)
    from cleaned_operators.technical.indicators_v2 import atr_pct

    expected = atr_pct(high, low, close, 5).diff()
    out = _op("atr_acceleration").calculate(high, low, close, window=5)
    pd.testing.assert_frame_equal(out, expected)


def test_atr_family_param_specs_and_relative_alpha_classification():
    expected_specs = {
        "atr_pct": {"window"},
        "atr_zscore": {"window", "score_window"},
        "atr_percentile": {"window", "score_window"},
        "true_range_surprise": {"window"},
        "true_range_zscore": {"window"},
        "atr_short_long_ratio": {"short_window", "long_window"},
        "atr_acceleration": {"window"},
    }
    for name, params in expected_specs.items():
        specs = _op(name).metadata.param_specs
        assert params == set(specs), f"{name} ParamSpec keys: {sorted(specs)}"
        for p in params:
            assert specs[p].dtype is not None, f"{name}.{p} has no dtype"

    from mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {
        "atr_pct", "atr_zscore", "atr_percentile",
        "true_range_pct", "true_range_surprise", "true_range_zscore",
        "atr_short_long_ratio", "atr_acceleration",
    }
    assert promoted <= _RELATIVE_ALPHA_OPS
    # Raw price-scale ATR levels must NOT be promoted by this slice.
    assert "ATR_WILDER" not in _RELATIVE_ALPHA_OPS
    assert "true_range" not in _RELATIVE_ALPHA_OPS
