"""Composite PIT：asof_backward 不得读取未来基本面。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.storage.composite_source import CompositeDataSource
from tests.storage.test_composite_source import CountingSeriesSource, _build_series


def test_asof_backward_does_not_use_future_fundamental():
    """2024-01-02 只能看到 2024-01-01 及之前的基本面，不能用 2024-01-05 的值。"""
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 100.0),
                    ("2024-01-03", "AAA", 101.0),
                    ("2024-01-04", "AAA", 102.0),
                ]
            )
        }
    )
    # 未来 PE 在 2024-01-05 才披露；更早日期不应被用到
    fundamental = CountingSeriesSource(
        {
            "pe": _build_series(
                [
                    ("2024-01-01", "AAA", 10.0),
                    ("2024-01-05", "AAA", 999.0),
                ]
            )
        }
    )

    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        aliases={"pe": "fundamental.pe"},
        joins={"fundamental": {"method": "asof_backward"}},
    )

    pe = source.load_column("pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)
    assert pe.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(10.0)
    assert pe.loc[(pd.Timestamp("2024-01-04"), "AAA")] == pytest.approx(10.0)


def test_exact_join_leaves_gap_without_same_day_fundamental():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}
    )
    fundamental = CountingSeriesSource(
        {"pe": _build_series([("2024-01-05", "AAA", 999.0)])}
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "exact"},
    )
    pe = source.load_column("fundamental.pe")
    assert pd.isna(pe.loc[(pd.Timestamp("2024-01-02"), "AAA")])
