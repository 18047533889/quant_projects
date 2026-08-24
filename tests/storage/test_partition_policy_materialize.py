"""分区策略与宽表物化 round-trip 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.materializer import ParquetMaterializer
from factor_engine.storage.partition_policy import PartitionPolicy, attach_partition_columns, partition_key
from factor_engine.storage.result_store import PandasResultStore


def _series():
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2023-12-28", "2024-01-02", "2024-06-01"]),
            ["A", "B"],
        ],
        names=["timestamp", "instrument"],
    )
    return pd.Series(np.arange(6, dtype=float), index=idx)


def test_attach_partition_columns_year_month():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-06-01"]),
            "asset": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )
    policy = PartitionPolicy.from_config(partition_columns=["year", "month"])
    out = attach_partition_columns(df, policy)
    assert list(out["year"]) == [2024, 2024]
    assert list(out["month"]) == [1, 6]
    assert partition_key({"year": 2024, "month": 6}) == "month=6|year=2024"


def test_materialize_wide_round_trip(tmp_path):
    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake, catalog=FactorCatalog(lake / "_catalog.sqlite"))
    result = mat.materialize(
        "wide_demo",
        _series(),
        author="test",
        storage_format="wide",
    )
    assert result["storage_format"] == "wide"
    assert result["rows_written"] == 6
    assert (lake / "factors" / "wide_demo" / "year=2023" / "panel.parquet").exists()
    assert (lake / "factors" / "wide_demo" / "year=2024" / "panel.parquet").exists()

    store = PandasResultStore(lake)
    wide = store.load_factor_wide("wide_demo")
    assert wide.shape[1] == 2
    assert len(wide) == 3


def test_materialize_year_month_partitions(tmp_path):
    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake, catalog=FactorCatalog(lake / "_catalog.sqlite"))
    result = mat.materialize(
        "month_demo",
        _series(),
        author="test",
        partition_columns=["year", "month"],
    )
    assert set(result["partition_keys"]) == {
        "month=1|year=2024",
        "month=6|year=2024",
        "month=12|year=2023",
    }
    assert (lake / "factors" / "month_demo" / "year=2024" / "month=1" / "data.parquet").exists()


def test_partition_policy_rejects_unknown_column():
    with pytest.raises(ValueError, match="暂不支持"):
        PartitionPolicy.from_config(partition_columns=["bucket"])
