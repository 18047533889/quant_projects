# -*- coding: utf-8 -*-
"""Tests for the 2026-08 intraday feature extensions and ``intra_*`` aliases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from storage.sources.intraday_feature_extension import _calc


_TIMES = ["09:31", "09:36", "09:41", "09:46", "09:51", "09:56", "13:01", "13:06", "13:11", "13:16"]


def _bar(rows: int | None = None, seed: int = 0, cross_lunch: bool = False) -> pd.DataFrame:
    if rows is None:
        rows = len(_TIMES)
    times = _TIMES[:rows]
    idx = pd.to_datetime(["2024-01-02 " + h for h in times])
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0, 0.01, rows))
    open_px = np.concatenate([[10.0], close[:-1]])
    high = np.maximum(open_px, close) + 0.01
    low = np.minimum(open_px, close) - 0.01
    volume = rng.integers(100, 500, rows).astype(float)
    amount = volume * close
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )


def test_intraday_features_and_intra_native_ops_are_exposed() -> None:
    from api.intraday_daily import INTRADAY_DAILY_DSL_FUNCTIONS
    from cleaned_operators.registry import OperatorRegistry

    for name in (
        "intraday_lunch_gap_return",
        "intraday_return_activity_corr",
        "intraday_vwap_above_ratio",
        "intraday_kyle_lambda_proxy",
        "intraday_extreme_bar_return",
        "intraday_segment_return",
        "intraday_segment_volume_share",
        "intraday_segment_amount_share",
        "intraday_segment_vwap_deviation",
        "intraday_segment_realized_vol",
        "intraday_limit_first_hit_time",
        "intraday_limit_duration",
        "intraday_limit_reopen_count",
    ):
        assert name in INTRADAY_DAILY_DSL_FUNCTIONS, name
    for name in (
        "intra_realized_variance",
        "intra_bipower_variation",
        "intra_jump_ratio",
        "intra_path_efficiency",
        "intra_high_time",
        "intra_low_time",
        "intra_vwap_cross_count",
        "intra_amihud",
        "intra_lunch_gap_return",
        "intra_segment_return",
        "intra_limit_reopen_count",
    ):
        assert OperatorRegistry.get(name) is not None, name


def test_lunch_gap_return_is_afternoon_open_over_morning_close_minus_one() -> None:
    bar = _bar(rows=8)  # 6 morning bars then 2 afternoon bars
    out = _calc("lunch_gap_return", bar, {})
    morning_last_close = bar["close"].iloc[5]
    afternoon_first_open = bar["open"].iloc[6]
    assert out == pytest.approx(afternoon_first_open / morning_last_close - 1.0)


def test_vwap_above_ratio_is_mean_close_over_bar_vwap() -> None:
    bar = _bar(rows=10)
    out = _calc("vwap_above_ratio", bar, {})
    bar_vwap = (bar["amount"] / bar["volume"]).to_numpy()
    expected = float(np.mean(bar["close"].to_numpy() >= bar_vwap))
    assert out == pytest.approx(expected)


def test_extreme_bar_return_side() -> None:
    bar = _bar(rows=10)
    r = np.log(bar["close"].to_numpy() / np.roll(bar["close"].to_numpy(), 1))[1:]
    assert _calc("extreme_bar_return", bar, {"side": "max"}) == pytest.approx(float(np.nanmax(r)))
    assert _calc("extreme_bar_return", bar, {"side": "min"}) == pytest.approx(float(np.nanmin(r)))


def test_segment_return_first_bars() -> None:
    bar = _bar(rows=10)
    out = _calc("segment_return", bar, {"segment": "first", "minutes": 10, "bar_minutes": 5})
    first_two = bar.iloc[:2]
    expected = first_two["close"].iloc[-1] / first_two["open"].iloc[0] - 1.0
    assert out == pytest.approx(expected)


def test_segment_volume_share() -> None:
    bar = _bar(rows=10)
    out = _calc("segment_volume_share", bar, {"segment": "first", "minutes": 10, "bar_minutes": 5})
    expected = bar["volume"].iloc[:2].sum() / bar["volume"].sum()
    assert out == pytest.approx(expected)


def test_kyle_lambda_proxy_is_positive() -> None:
    bar = _bar(rows=10)
    out = _calc("kyle_lambda_proxy", bar, {})
    assert np.isfinite(out) and out >= 0.0


def test_return_activity_corr_matches_signed_corr() -> None:
    bar = _bar(rows=10)
    out = _calc("return_activity_corr", bar, {"activity": "volume"})
    close = bar["close"].to_numpy()
    r = np.full(len(close), 0.0)
    r[1:] = np.log(close[1:] / close[:-1])
    corr = np.corrcoef(r, bar["volume"].to_numpy())[0, 1]
    assert out == pytest.approx(float(corr))


def test_limit_features_nan_without_limit_price() -> None:
    bar = _bar(rows=10)
    # prev_close=0 -> ashare_limit_prices returns NaN -> features return NaN
    assert np.isnan(_calc("limit_first_hit_time", bar, {}))
    assert np.isnan(_calc("limit_duration", bar, {}))
    assert np.isnan(_calc("limit_reopen_count", bar, {}))
