"""
adapters 单元测试：Arrow → MultiIndex Series 契约
（和老 ParquetSource.load_column 返回格式必须一致）。
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa

from data_access.adapters import arrow_table_to_multiindex_columns, arrow_to_multiindex_series


def _make_table(data: dict) -> pa.Table:
    return pa.table(data)


def test_basic_conversion():
    tbl = _make_table({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "instrument": ["AAPL", "AAPL", "MSFT"],
        "close": [150.0, 151.0, 300.0],
    })
    series = arrow_to_multiindex_series(
        tbl,
        timestamp_column="timestamp",
        instrument_column="instrument",
        value_column="close",
    )
    assert series.name == "close"
    assert series.index.names == ["timestamp", "instrument"]
    assert len(series) == 3
    # 排序了：AAPL 在 MSFT 前面
    first_key = series.index[0]
    assert first_key[1] == "AAPL"


def test_dedup_keeps_last():
    """同 (ts, instrument) 有两行时，默认保留最后一行（和老实现一致）。"""
    tbl = _make_table({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "instrument": ["AAPL", "AAPL"],
        "close": [100.0, 200.0],
    })
    series = arrow_to_multiindex_series(
        tbl,
        timestamp_column="timestamp",
        instrument_column="instrument",
        value_column="close",
    )
    assert len(series) == 1
    assert series.iloc[0] == 200.0  # keep=last


def test_drops_null_keys():
    """timestamp 或 instrument 为空的行被丢弃。"""
    tbl = _make_table({
        "timestamp": pd.to_datetime(["2024-01-01", None, "2024-01-02"]),
        "instrument": ["AAPL", "AAPL", None],
        "close": [100.0, 200.0, 300.0],
    })
    series = arrow_to_multiindex_series(
        tbl,
        timestamp_column="timestamp",
        instrument_column="instrument",
        value_column="close",
    )
    assert len(series) == 1
    assert series.iloc[0] == 100.0


def test_missing_column_raises():
    tbl = _make_table({"a": [1], "b": [2]})
    import pytest
    with pytest.raises(KeyError, match="缺少必要列"):
        arrow_to_multiindex_series(
            tbl,
            timestamp_column="timestamp",
            instrument_column="instrument",
            value_column="close",
        )


def test_output_name_override():
    tbl = _make_table({
        "t": pd.to_datetime(["2024-01-01"]),
        "i": ["X"],
        "v": [1.0],
    })
    series = arrow_to_multiindex_series(
        tbl,
        timestamp_column="t",
        instrument_column="i",
        value_column="v",
        output_name="my_signal",
    )
    assert series.name == "my_signal"


def test_timestamp_unit_conversion():
    """timestamp 是 int epoch 时，按 unit 转成 datetime。"""
    # 纳秒 epoch
    tbl = _make_table({
        "ts_ns": [1704067200_000_000_000, 1704153600_000_000_000],
        "inst": ["X", "X"],
        "val": [1.0, 2.0],
    })
    series = arrow_to_multiindex_series(
        tbl,
        timestamp_column="ts_ns",
        instrument_column="inst",
        value_column="val",
        timestamp_unit="ns",
    )
    assert len(series) == 2
    first_ts = series.index[0][0]
    assert str(first_ts).startswith("2024-01-01")


def test_timezone_stripped():
    """和老 ParquetSource 契约一致：tz-aware 的时间列要去掉 tz，下游 groupby 不会炸。"""
    tbl = _make_table({
        "t": pd.to_datetime(["2024-01-01"], utc=True),
        "i": ["X"],
        "v": [1.0],
    })
    series = arrow_to_multiindex_series(
        tbl,
        timestamp_column="t",
        instrument_column="i",
        value_column="v",
    )
    ts = series.index[0][0]
    assert ts.tz is None


def test_batch_columns_single_conversion():
    """批量路径：多列只做一次 Arrow→pandas。"""
    tbl = _make_table({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "instrument": ["AAPL", "MSFT"],
        "close": [150.0, 300.0],
        "volume": [1_000_000.0, 2_000_000.0],
    })
    cols = arrow_table_to_multiindex_columns(
        tbl,
        timestamp_column="timestamp",
        instrument_column="instrument",
        value_columns=["close", "volume"],
        output_names={"close": "c", "volume": "v"},
    )
    assert set(cols) == {"c", "v"}
    assert cols["c"].name == "c"
    assert cols["v"].name == "v"
    assert len(cols["c"]) == 2
    assert cols["c"].index.names == ["timestamp", "instrument"]
