from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.api.dsl_parser import parse_factor
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.ir.analyzer import Analyzer


def _frame(values, *, columns=("A",)) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(
        arr,
        index=pd.date_range("2024-01-02", periods=arr.shape[0], freq="B"),
        columns=list(columns),
    )


def _op(name: str):
    load_all()
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None
    return op


def test_previous_high_and_breakout_exclude_current_bar() -> None:
    close = _frame([10, 11, 12, 13, 20, 19])
    prev = _op("ts_prev_high").calculate(close, 3)
    breakout = _op("ts_breakout_high").calculate(close, 3)

    assert prev.iloc[3, 0] == 12.0
    assert prev.iloc[4, 0] == 13.0
    assert breakout.iloc[4, 0] == 20.0 / 13.0 - 1.0
    assert breakout.iloc[5, 0] == 0.0


def test_confirmed_pivot_emits_only_on_confirmation_timestamp() -> None:
    # Index 3 is the high. With left=2/right=2 it is only knowable at index 5.
    high = _frame([1, 2, 3, 10, 4, 3, 2, 1])
    out = _op("ts_confirmed_pivot_high").calculate(high, 2, 2)

    assert np.isnan(out.iloc[3, 0])
    assert np.isnan(out.iloc[4, 0])
    assert out.iloc[5, 0] == 10.0


def test_bounded_resistance_does_not_depend_on_remote_start_history() -> None:
    # Repeated peaks provide enough confirmed points in the local history.
    high = _frame([1,3,1,2,4,2,3,5,3,4,6,4,5,7,5,6,8,6,7,9,7,8,10,8,9,11,9,10,12,10])
    op = _op("ts_resistance_level")
    full = op.calculate(high, 1, 1, 12, 3)

    # Prefix invariance: appending future observations may not change the past.
    prefix = op.calculate(high.iloc[:24], 1, 1, 12, 3)
    pd.testing.assert_series_equal(full.iloc[:24, 0], prefix.iloc[:, 0])


def test_relative_volume_uses_only_prior_baseline() -> None:
    volume = _frame([1, 1, 1, 10, 10])
    out = _op("relative_volume").calculate(volume, 3)
    assert out.iloc[3, 0] == 10.0


def test_ohlc_volatility_estimators_are_nonnegative_and_causal() -> None:
    close = _frame(np.linspace(100, 130, 64) + np.sin(np.arange(64)))
    open_ = close * 0.998
    high = pd.DataFrame(np.maximum(open_, close) * 1.01, index=close.index, columns=close.columns)
    low = pd.DataFrame(np.minimum(open_, close) * 0.99, index=close.index, columns=close.columns)

    for name, args in {
        "parkinson_vol": (high, low, 20),
        "garman_klass_vol": (open_, high, low, close, 20),
        "rogers_satchell_vol": (open_, high, low, close, 20),
        "yang_zhang_vol": (open_, high, low, close, 20),
    }.items():
        op = _op(name)
        full = op.calculate(*args)
        finite = full.to_numpy(dtype=float)
        assert np.nanmin(finite) >= 0.0
        sliced = tuple(x.iloc[:48] if isinstance(x, pd.DataFrame) else x for x in args)
        prefix = op.calculate(*sliced)
        np.testing.assert_allclose(
            full.iloc[:48].to_numpy(), prefix.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
        )


def test_candle_geometry_and_engulfing_are_numeric_factors() -> None:
    open_ = _frame([10, 9])
    close = _frame([9, 11])
    high = _frame([10.5, 11.5])
    low = _frame([8.5, 8.5])

    body_ratio = _op("candle_body_ratio").calculate(open_, high, low, close)
    engulf = _op("cdl_engulfing").calculate(open_, high, low, close)

    assert 0.0 <= body_ratio.iloc[1, 0] <= 1.0
    assert engulf.iloc[1, 0] == 1.0


def test_analyzer_accounts_for_hidden_technical_warmup() -> None:
    cases = {
        "ts_prev_high(close, 20)": 20,
        "ts_confirmed_pivot_high(high, 3, 3)": 6,
        "ts_resistance_level(high, 3, 3, 60, 3)": 65,
        "ulcer_index(close, 20)": 38,
    }
    analyzer = Analyzer()
    for source, minimum in cases.items():
        factor = parse_factor(source, surface="extended")
        result = analyzer.lower(factor.expr)
        assert result.lookback >= minimum, (source, result.lookback, minimum)


def test_extension_surface_has_no_unregistered_names() -> None:
    load_all()
    from factor_engine.cleaned_operators.operator_surface import _TECHNICAL_EXTENSION_CANONICALS

    missing = sorted(name for name in _TECHNICAL_EXTENSION_CANONICALS if OperatorRegistry.get(name) is None)
    assert missing == []
