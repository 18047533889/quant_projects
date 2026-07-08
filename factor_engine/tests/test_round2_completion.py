# -*- coding: utf-8 -*-
"""日内窗口精度、mining 标签、data_access delete_rows 测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pd = pytest.importorskip("pandas")


def test_windowed_datasource_preserves_intraday_timestamp():
    from storage.time_window import WindowedDataSource
    from storage.datasource import DataSource

    class _Inner(DataSource):
        def load_column(self, name: str):
            idx = pd.MultiIndex.from_product(
                [
                    pd.to_datetime(
                        ["2024-06-01 09:35:00", "2024-06-01 10:00:00", "2024-06-01 15:55:00"]
                    ),
                    ["A"],
                ],
                names=["timestamp", "instrument"],
            )
            return pd.Series([1.0, 2.0, 3.0], index=idx, name=name)

    inner = _Inner()
    wrapped = WindowedDataSource(
        inner,
        start_date="2024-06-01 09:40:00",
        end_date="2024-06-01 16:00:00",
        intraday=True,
    )
    out = wrapped.load_column("x")
    assert len(out) == 2
    assert out.iloc[0] == pytest.approx(2.0)
    assert out.iloc[1] == pytest.approx(3.0)


def test_narrow_kline_keeps_intraday_start_string(tmp_path):
    from storage.kline_parquet_source import KlineParquetSource
    from storage.time_window import narrow_data_source_for_window

    root = tmp_path / "k"
    root.mkdir()
    src = KlineParquetSource(root=str(root), bar_freq="5m")
    narrowed = narrow_data_source_for_window(
        src,
        start_date=pd.Timestamp("2024-06-01 14:00:00"),
        bar_freq="5m",
    )
    assert isinstance(narrowed, KlineParquetSource)
    assert "14:00:00" in str(narrowed.start_date)


def test_mining_integration_label_config():
    from api.mining_integration import default_mining_label_config, validate_mining_label_formula

    cfg = default_mining_label_config()
    assert cfg["gap_bars"] >= 1
    ok, msg = validate_mining_label_formula("ts_pct(close, 5)")
    assert ok, msg


def test_store_delete_rows(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    from data_access.registry import ParametricDataset
    from data_access.upsert import delete_rows_from_dataset
    from data_access.paths import PathAuthorizer

    root = tmp_path / "staging" / "f1"
    part = root / "year=2024"
    part.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "asset": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )
    pa.parquet.write_table(pa.Table.from_pandas(df, preserve_index=False), part / "data.parquet")

    ds = ParametricDataset(
        name="factor_lake_staging",
        access_mode="staging",
        layout="hive",
        root_template=str(tmp_path / "staging" / "{factor_id}"),
        glob_template="year=*/*.parquet",
        params_schema={"factor_id": "str"},
        time_column="datetime",
        instrument_column="asset",
        hive_partitioning=True,
        union_by_name=True,
    )
    authorizer = PathAuthorizer(allowed_roots=[tmp_path])
    result = delete_rows_from_dataset(
        ds=ds,
        authorizer=authorizer,
        target_dir=root,
        time_column="datetime",
        after="2024-01-02",
        params={"factor_id": "f1"},
    )
    assert result["rows_deleted"] == 1
    kept = pd.read_parquet(part / "data.parquet")
    assert len(kept) == 1
