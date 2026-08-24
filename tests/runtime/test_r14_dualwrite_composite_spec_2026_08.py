# -*- coding: utf-8 -*-
"""R14 #4 / #6 测试。

#4  ClickHouse 双写 version parity：orchestrator 只算一次 canonical
    factor_version（semantic_identity digest 前缀），Parquet/catalog/ClickHouse
    用同一份；不再在 ClickHouse 侧退化成 ``ast_hash[:16]``。
#6  Composite ``execution_spec()`` 不再伪造 ``{"type": "data_access"}``：
    任一 child 无法序列化完整 source contract → production 抛
    ``UnreconstructableDataSource``；research 返回 ``None``（不发明 config）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest import mock

from factor_engine.runtime.incremental_scheduler import DataEvent  # noqa: F401 (import sanity)
from factor_engine.storage.exceptions import UnreconstructableDataSource
from factor_engine.storage.sources.composite_source import CompositeDataSource


def _ser():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01"]), ["A"]], names=["timestamp", "instrument"]
    )
    return pd.Series([1.0], index=idx)


# ---------------------------------------------------------------------------
# #4 ClickHouse 双写 version parity
# ---------------------------------------------------------------------------


def test_r14_dual_write_uses_canonical_factor_version():
    from factor_engine.runtime.reconcile.dual_write_service import append_clickhouse_to_summary
    from factor_engine.storage.write_targets import ClickHouseWriteTarget

    summary = {"factor_id": "f", "rows_written": 1}
    with mock.patch.object(
        ClickHouseWriteTarget,
        "write_factor_series",
        return_value={
            "factor_id": "f",
            "table": "t",
            "rows_written": 1,
            "database": "db",
        },
    ) as m:
        append_clickhouse_to_summary(
            summary,
            factor_id="f",
            result=_ser(),
            ast_hash="a" * 64,
            factor_version="canonical-v1-16",
            write_target="clickhouse",
        )
    kwargs = m.call_args[1]
    assert kwargs["factor_version"] == "canonical-v1-16"


def test_r14_dual_write_falls_back_to_ast_hash():
    """orchestrator 未传 factor_version（缺省）→ 才回退 ast_hash 前缀。"""
    from factor_engine.runtime.reconcile.dual_write_service import append_clickhouse_to_summary
    from factor_engine.storage.write_targets import ClickHouseWriteTarget

    with mock.patch.object(
        ClickHouseWriteTarget,
        "write_factor_series",
        return_value={
            "factor_id": "f",
            "table": "t",
            "rows_written": 1,
            "database": "db",
        },
    ) as m:
        append_clickhouse_to_summary(
            {"factor_id": "f", "rows_written": 1},
            factor_id="f",
            result=_ser(),
            ast_hash="b" * 64,
            write_target="clickhouse",
        )
    kwargs = m.call_args[1]
    assert kwargs["factor_version"] == ("b" * 64)[:16]


# ---------------------------------------------------------------------------
# #6 Composite execution_spec 不伪造残缺 spec
# ---------------------------------------------------------------------------


class _UnserializableChild:
    """非 DataSource / 无 execution_spec 的 child。"""


def test_r14_composite_production_raises_unreconstructable(tmp_path):
    comp = CompositeDataSource(
        anchor_source="anchor",
        anchor_column="date",
        sources={"anchor": _UnserializableChild()},
        production=True,
    )
    with pytest.raises(UnreconstructableDataSource, match="无法"):
        comp.execution_spec()


def test_r14_composite_research_returns_none(tmp_path):
    comp = CompositeDataSource(
        anchor_source="anchor",
        anchor_column="date",
        sources={"anchor": _UnserializableChild()},
        production=False,
    )
    assert comp.execution_spec() is None


def test_r14_composite_production_child_spec_none_raises():
    """child 是 DataSource 但其 execution_spec() 返回 None → 同样不得伪造。"""

    class _NoSpecSource:
        def execution_spec(self) -> None:
            return None

    comp = CompositeDataSource(
        anchor_source="child",
        anchor_column="date",
        sources={"child": _NoSpecSource()},
        production=True,
    )
    with pytest.raises(UnreconstructableDataSource):
        comp.execution_spec()
    comp_r = CompositeDataSource(
        anchor_source="child",
        anchor_column="date",
        sources={"child": _NoSpecSource()},
        production=False,
    )
    assert comp_r.execution_spec() is None


def test_r14_composite_all_serializable_passes(tmp_path):
    """全部 child 可序列化 → production 仍返回完整 canonical spec（含 production 标记）。"""

    from factor_engine.storage.sources.datasource import DataSource

    class _SpecSource(DataSource):
        def execution_spec(self) -> dict:
            return {"type": "parquet", "dataset": "ds_x", "columns": ["close"]}

        def load_column(self, name: str) -> Any:
            raise NotImplementedError

    comp = CompositeDataSource(
        anchor_source="child",
        anchor_column="date",
        sources={"child": _SpecSource()},
        production=True,
    )
    spec = comp.execution_spec()
    assert spec["type"] == "composite"
    assert spec["sources"]["child"] == {
        "type": "parquet",
        "dataset": "ds_x",
        "columns": ["close"],
    }
    assert spec["production"] is True
