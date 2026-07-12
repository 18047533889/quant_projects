# -*- coding: utf-8 -*-
"""SQL 下推 executor：QueryBudget / view_columns 传参测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from backend.sql_pushdown.emitter import CompiledSql, SqlDialect
from backend.sql_pushdown.executor import (
    PushdownContext,
    _build_duckdb_store_kwargs,
    _execute_duckdb_table,
)
from storage.datasource import DataSource


@dataclass
class _FakeDataAccessSource(DataSource):
    dataset: str = "test_ds"
    fields: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def dataset_axis_columns(self) -> tuple[str, str]:
        return ("align_time", "ticker")

    def load_column(self, name: str):
        raise NotImplementedError


def test_build_duckdb_store_kwargs_maps_fields_and_params():
    src = _FakeDataAccessSource(
        fields={"close": "close_px"},
        params={"factor_id": "mom_3d"},
    )
    compiled = CompiledSql(
        query="SELECT 1",
        read_datasets=("factor_lake",),
        referenced_columns=frozenset({"close", "volume"}),
        dialect=SqlDialect.DUCKDB,
    )
    pctx = PushdownContext(
        dialect=SqlDialect.DUCKDB,
        dataset="factor_lake",
        time_column="align_time",
        instrument_column="ticker",
    )
    kwargs = _build_duckdb_store_kwargs(compiled, pctx, src)
    assert kwargs["view_columns"]["factor_lake"] == sorted(
        {"align_time", "ticker", "close_px", "volume"}
    )
    assert kwargs["read_params"]["factor_lake"] == {"factor_id": "mom_3d"}


def test_execute_duckdb_table_passes_view_columns():
    compiled = CompiledSql(
        query="WITH x AS (SELECT 1) SELECT 1",
        read_datasets=("prices",),
        referenced_columns=frozenset({"close"}),
        dialect=SqlDialect.DUCKDB,
    )
    pctx = PushdownContext(
        dialect=SqlDialect.DUCKDB,
        dataset="prices",
        time_column="datetime",
        instrument_column="asset",
    )
    src = _FakeDataAccessSource(dataset="prices")
    mock_store = MagicMock()
    mock_store.sql.return_value = MagicMock(num_rows=1)

    from data_access.read.query_budget import QueryBudget

    with patch("data_access.get_store", return_value=mock_store):
        _execute_duckdb_table(
            compiled,
            pctx,
            src,
            query_budget=QueryBudget(max_rows=123),
        )

    call_kw = mock_store.sql.call_args.kwargs
    assert call_kw["view_columns"]["prices"] == sorted({"datetime", "asset", "close"})
    assert call_kw["query_budget"].max_rows == 123
