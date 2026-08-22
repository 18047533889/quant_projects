"""输入 DQ 与 materializer 元数据列测试。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from runtime.input_dq import InputDQError, assert_input_dq, evaluate_input_columns
from storage.materializer import ParquetMaterializer
from tests.helpers import InMemorySeriesSource
from storage.datasource import DataSource


def _panel(values: list[float]) -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(values, index=idx)


def test_input_dq_passes_with_valid_column():
    src = InMemorySeriesSource(data={"close": _panel([1, 2, 3, 4])})
    report = assert_input_dq(src, ["close"], raise_on_fail=True)
    assert report.passed


def test_input_dq_fails_on_empty_column():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime([]), []],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(data={"close": pd.Series([], index=idx, dtype=float)})
    with pytest.raises(InputDQError):
        assert_input_dq(src, ["close"], raise_on_fail=True)


@dataclass
class _BatchFailSource(DataSource):
    """load_columns 失败时应逐列 fallback 到 load_column。"""

    data: dict[str, pd.Series]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        raise RuntimeError("batch unavailable")

    def load_column(self, name: str) -> pd.Series:
        return self.data[name]


@dataclass
class _PartialBatchSource(DataSource):
    """批量只返回部分列时，缺失列应单独加载。"""

    data: dict[str, pd.Series]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        return {n: self.data[n] for n in names if n == "close"}

    def load_column(self, name: str) -> pd.Series:
        return self.data[name]


def test_input_dq_batch_failure_falls_back_per_column():
    panel = _panel([1, 2, 3, 4])
    src = _BatchFailSource(data={"close": panel, "open": panel + 1})
    report = evaluate_input_columns(src, ["close", "open"])
    assert report.passed
    assert {c.column for c in report.columns} == {"close", "open"}


def test_input_dq_partial_batch_loads_missing_columns():
    panel = _panel([1, 2, 3, 4])
    src = _PartialBatchSource(data={"close": panel, "open": panel + 1})
    report = evaluate_input_columns(src, ["close", "open"])
    assert report.passed
    assert len(report.columns) == 2


def test_input_dq_allows_nan_but_rejects_inf():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    with_nan = pd.Series([1.0, np.nan], index=idx)
    src = InMemorySeriesSource(data={"close": with_nan})
    assert assert_input_dq(src, ["close"], raise_on_fail=True).passed

    with_inf = pd.Series([1.0, np.inf], index=idx)
    src_inf = InMemorySeriesSource(data={"close": with_inf})
    with pytest.raises(InputDQError):
        assert_input_dq(src_inf, ["close"], raise_on_fail=True)


def test_materializer_writes_metadata_columns(tmp_path):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-15"]), ["000001.SZ"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.5], index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(
        factor_id="meta_test",
        result=series,
        ast_hash="abc123def456",
        data_snapshot_id="snap001",
        write_metadata=True,
    )
    pq = tmp_path / "factors" / "meta_test" / "year=2024" / "data.parquet"
    df = pd.read_parquet(pq)
    assert "calc_time" in df.columns
    assert "factor_version" in df.columns
    # R11 #6: factor_version = 语义身份 digest 前缀（不再直接等于 ast_hash）。
    # 只传 ast_hash（无 ir_node）时仍计算身份（含 frequency），版本是其 digest。
    from runtime.factor_identity import compute_identity_from_materialize_ctx

    expected = compute_identity_from_materialize_ctx(
        ir_node=None,
        ast_hash="abc123def456",
        data_source_config=None,
        run_lineage=None,
        frequency="1d",
    ).identity_digest()[:16]
    assert df["factor_version"].iloc[0] == expected
    assert df["data_snapshot_id"].iloc[0] == "snap001"
    assert df["is_valid"].iloc[0] == 1
