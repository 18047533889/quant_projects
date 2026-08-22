# -*- coding: utf-8
"""日内 / 时序新增算子 golden tests。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.microstructure.session import (
    pct_change_by_session,
    session_cum_vwap,
    session_key_from_index,
    session_vwap_deviation,
)
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_ts_sharpe_insufficient_window(_loaded):
    op = OperatorRegistry.get("ts_sharpe")
    ret = pd.DataFrame({"A": [0.01, -0.02, 0.015]})
    out = op.calculate(ret, window=60, min_periods=10)
    assert out.isna().all().all()


def test_ts_sharpe_zero_std_is_null_for_constant_returns(_loaded):
    op = OperatorRegistry.get("ts_sharpe")
    ret = pd.DataFrame({"A": [0.01] * 30})
    out = op.calculate(ret, window=20, min_periods=5)
    assert pd.isna(out.iloc[-1, 0])


def test_ts_autocorr_perfect_for_monotonic_series(_loaded):
    op = OperatorRegistry.get("ts_autocorr")
    x = pd.DataFrame({"A": np.arange(50, dtype=float)})
    out = op.calculate(x, window=20, lag=1, min_periods=5)
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_ts_autocorr_negative_lag_is_rejected(_loaded):
    from backend.operator_errors import OperatorParameterError

    op = OperatorRegistry.get("ts_autocorr")
    x = pd.DataFrame({"A": np.arange(10, dtype=float)})
    with pytest.raises(OperatorParameterError, match="lag"):
        op.calculate(x, window=5, lag=-1)


def test_session_key_splits_at_calendar_midnight():
    idx = pd.DatetimeIndex(["2024-01-02 23:59", "2024-01-03 00:01"])
    keys = session_key_from_index(idx)
    assert keys.iloc[0].normalize() != keys.iloc[1].normalize()


def test_pct_change_by_session_no_cross_day_leak():
    idx = pd.DatetimeIndex(["2024-01-02 15:00", "2024-01-03 09:31", "2024-01-03 09:32"])
    close = pd.Series([100.0, 105.0, 107.0], index=idx)
    out = pct_change_by_session(close)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(107.0 / 105.0 - 1.0)


def test_session_cum_vwap_zero_volume_bar():
    idx = pd.DatetimeIndex(["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-02 09:33"])
    price = pd.Series([100.0, 102.0, 104.0], index=idx)
    volume = pd.Series([10.0, 0.0, 10.0], index=idx)
    vwap = session_cum_vwap(price, volume)
    assert vwap.iloc[0] == pytest.approx(100.0)
    assert vwap.iloc[1] == pytest.approx(100.0)
    assert vwap.iloc[2] == pytest.approx((100 * 10 + 104 * 10) / 20.0)


def test_session_cum_vwap_resets_each_day():
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-03 09:31",
            "2024-01-03 09:32",
        ]
    )
    price = pd.Series([100.0, 102.0, 200.0, 204.0], index=idx)
    volume = pd.Series([10.0, 10.0, 5.0, 5.0], index=idx)
    vwap = session_cum_vwap(price, volume)
    assert vwap.iloc[0] == pytest.approx(100.0)
    assert vwap.iloc[2] == pytest.approx(200.0)


def test_intraday_vwap_deviation(_loaded):
    op = OperatorRegistry.get("intraday_vwap_deviation")
    idx = pd.DatetimeIndex(["2024-01-02 09:31", "2024-01-02 09:32"])
    close = pd.DataFrame({"A": [100.0, 104.0]}, index=idx)
    price = close.copy()
    volume = pd.DataFrame({"A": [10.0, 10.0]}, index=idx)
    out = op.calculate(close, price, volume)
    assert out.iloc[0, 0] == pytest.approx(0.0)
    expected = session_vwap_deviation(close["A"], price["A"], volume["A"]).iloc[1]
    assert out.iloc[1, 0] == pytest.approx(expected)


def test_atr_wilder_requires_full_window(_loaded):
    op = OperatorRegistry.get("ATR_WILDER")
    high = pd.DataFrame({"A": [101.0, 102.0]})
    low = pd.DataFrame({"A": [99.0, 100.0]})
    close = pd.DataFrame({"A": [100.0, 101.0]})
    out = op.calculate(high, low, close, window=14)
    assert pd.isna(out.iloc[0, 0])
    assert pd.isna(out.iloc[1, 0])
