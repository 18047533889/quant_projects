"""read/predicate_ast.py：Filter AST 单测（DuckDB / Polars / Arrow 三编译器）。"""
from __future__ import annotations

import pytest

import pyarrow as pa

from data_access.read.predicate_ast import (
    And,
    Between,
    Eq,
    Ge,
    Gt,
    In,
    IsNotNull,
    IsNull,
    Le,
    Lt,
    Ne,
    Not,
    NotIn,
    Or,
    compile_filter_arrow,
    compile_filter_duckdb,
    compile_filter_polars,
    parse_filters,
)


def test_parse_scalar_in_comparison():
    f = parse_filters({"col": 5})
    assert isinstance(f, Eq)
    f = parse_filters({"col": [1, 2]})
    assert isinstance(f, In)
    f = parse_filters({"col": {"gt": 1, "lte": 3}})
    assert isinstance(f, And)
    f = parse_filters({"col": {"between": [1, 3]}})
    assert isinstance(f, Between)


def test_parse_nested():
    f = parse_filters({"a": {"in": [1, 2]}, "b": {"isnull": None}})
    assert isinstance(f, And)
    cols = f.children[1]
    assert isinstance(cols, IsNull)


def test_compile_duckdb_bind_params():
    f = And([Gt("close", 100), In("symbol", ["AAPL", "MSFT"]), IsNull("note")])
    sql, params = compile_filter_duckdb(f)
    assert "close" in sql and "symbol" in sql and "IS NULL" in sql
    assert params == [100, ["AAPL", "MSFT"]]


def test_compile_duckdb_allowed_columns():
    f = Gt("bogus", 1)
    with pytest.raises(ValueError):
        compile_filter_duckdb(f, allowed_columns={"close", "symbol"})


def test_compile_polars():
    pytest.importorskip("polars")
    import polars as pl

    expr = compile_filter_polars(And([Gt("a", 1), Le("a", 5)]), pl=pl)
    df = pl.DataFrame({"a": [1, 2, 6]})
    assert df.filter(expr)["a"].to_list() == [2]


def test_compile_arrow():
    pc = pytest.importorskip("pyarrow.compute")
    table = pa.table({"symbol": ["AAPL", "MSFT", "AAPL"], "close": [90.0, 200.0, 150.0]})
    expr = And([In("symbol", ["AAPL"]), Gt("close", 100)])
    arrow_expr = compile_filter_arrow(expr, pc=pc)
    filtered = table.filter(arrow_expr)
    assert filtered.num_rows == 1


def test_not():
    f = Not(Eq("a", 1))
    sql, _ = compile_filter_duckdb(f)
    assert sql.startswith("NOT (")
