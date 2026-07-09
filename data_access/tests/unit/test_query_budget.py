"""query_budget 单元测试。"""

from __future__ import annotations

import pyarrow as pa
import pytest

from data_access.exceptions import ValidationError
from data_access.query_budget import (
    QueryBudget,
    apply_sql_row_limit,
    enforce_arrow_budget,
    enforce_result_budget,
    enforce_stream_budget,
    resolve_query_budget,
    validate_query_request,
    validate_sql_view_columns,
)


def test_research_default_budget_has_no_row_cap(monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("DATA_ACCESS_DEFAULT_MAX_ROWS", raising=False)
    budget = resolve_query_budget()
    assert budget.max_rows is None
    assert budget.require_columns is False


def test_research_budget_respects_default_max_rows_env(monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("DATA_ACCESS_DEFAULT_MAX_ROWS", "1000000")
    budget = resolve_query_budget()
    assert budget.max_rows == 1_000_000


def test_production_mode_requires_columns(monkeypatch):
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    budget = resolve_query_budget()
    assert budget.require_columns is True
    with pytest.raises(ValidationError, match="columns"):
        validate_query_request(budget, columns=None, time_range=None)


def test_strict_read_mode_requires_columns(monkeypatch):
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    budget = resolve_query_budget()
    assert budget.require_columns is True
    with pytest.raises(ValidationError, match="columns"):
        validate_query_request(budget, columns=None, time_range=None)


def test_sql_view_columns_required_in_production(monkeypatch):
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    budget = resolve_query_budget()
    with pytest.raises(ValidationError, match="view_columns"):
        validate_sql_view_columns(budget, ["factors"], None)
    with pytest.raises(ValidationError, match="缺少数据集"):
        validate_sql_view_columns(
            budget,
            ["factors", "prices"],
            {"factors": ["asset", "value"]},
        )
    validate_sql_view_columns(
        budget,
        ["factors"],
        {"factors": ["asset", "value"]},
    )


def test_enforce_max_rows():
    budget = QueryBudget(max_rows=10)
    enforce_result_budget(budget, rows=5, elapsed_ms=1.0)
    with pytest.raises(ValidationError, match="超过预算"):
        enforce_result_budget(budget, rows=11, elapsed_ms=1.0)


def test_enforce_arrow_budget_bytes():
    table = pa.table({"a": list(range(100))})
    budget = QueryBudget(max_rows=1000, max_result_bytes=table.nbytes - 1)
    with pytest.raises(ValidationError, match="字节"):
        enforce_arrow_budget(budget, table, elapsed_ms=1.0)
    enforce_arrow_budget(
        QueryBudget(max_result_bytes=table.nbytes + 1),
        table,
        elapsed_ms=1.0,
    )


def test_enforce_stream_budget_bytes():
    batch = pa.record_batch([pa.array([1, 2, 3])], names=["a"])
    budget = QueryBudget(max_rows=10, max_result_bytes=batch.nbytes - 1)
    with pytest.raises(ValidationError, match="字节"):
        enforce_stream_budget(
            budget,
            total_rows=batch.num_rows,
            total_bytes=batch.nbytes,
            elapsed_ms=1.0,
        )


def test_apply_sql_row_limit_wraps_without_existing_limit():
    sql = "SELECT a FROM t WHERE x = 1"
    wrapped = apply_sql_row_limit(sql, 1000)
    assert "LIMIT 1000" in wrapped
    assert "__da_bounded" in wrapped


def test_apply_sql_row_limit_respects_existing_limit():
    sql = "SELECT a FROM t LIMIT 5"
    assert apply_sql_row_limit(sql, 1000) == sql


def test_apply_sql_row_limit_tightens_existing_limit():
    sql = "SELECT a FROM t LIMIT 500"
    tightened = apply_sql_row_limit(sql, 100)
    assert "LIMIT 100" in tightened
    assert "LIMIT 500" not in tightened
