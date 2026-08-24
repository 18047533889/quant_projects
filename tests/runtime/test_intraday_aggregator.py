# -*- coding: utf-8
"""IntradayAggregator 日频聚合测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.runtime.intraday_aggregator import IntradayAggregator


def _intraday_close_volume() -> tuple[pd.Series, pd.Series]:
    rows = []
    for day in ("2024-01-02", "2024-01-03"):
        for hour, px, vol in ((10, 100.0, 10.0), (11, 102.0, 20.0), (15, 99.0, 30.0)):
            ts = pd.Timestamp(f"{day} {hour:02d}:30:00")
            rows.append((ts, "AAA", px, vol))
    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in rows],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([r[2] for r in rows], index=idx)
    volume = pd.Series([r[3] for r in rows], index=idx)
    return close, volume


def test_intraday_last_close_is_end_of_day():
    close, volume = _intraday_close_volume()
    agg = IntradayAggregator.from_panel({"close": close, "volume": volume})
    daily = agg.last("close")
    assert len(daily) == 2
    assert daily.iloc[-1] == pytest.approx(99.0)


def test_intraday_vwap_uses_amount_over_volume():
    close, volume = _intraday_close_volume()
    # Deliberately differs from close*volume to prove that bar VWAP is not
    # reconstructed from the closing print.
    amount = pd.Series(
        [1005.0, 2050.0, 3000.0] * 2,
        index=close.index,
    )
    agg = IntradayAggregator.from_panel(
        {"close": close, "volume": volume, "amount": amount}
    )
    vwap = agg.vwap()
    assert vwap.iloc[0] == pytest.approx((1005 + 2050 + 3000) / 60)


def test_intraday_vwap_keeps_legacy_price_volume_fallback():
    close, volume = _intraday_close_volume()
    agg = IntradayAggregator.from_panel({"close": close, "volume": volume})
    expected = (100 * 10 + 102 * 20 + 99 * 30) / 60
    assert agg.vwap().iloc[0] == pytest.approx(expected)


def test_aggregate_features_keys():
    close, volume = _intraday_close_volume()
    agg = IntradayAggregator.from_panel({"close": close, "volume": volume})
    feats = agg.aggregate_features()
    assert "close" in feats
    assert "volume" in feats
    assert "vwap" in feats
