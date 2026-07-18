"""
predicate 单元测试：SQL 片段 + 参数绑定正确性。
"""
from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.predicate import Predicate, compile_predicate


def test_empty_predicate():
    compiled = compile_predicate(
        Predicate(),
        time_column="align_time",
        instrument_column="ticker",
    )
    assert compiled.where_sql == ""
    assert compiled.params == []


def test_time_range_both_bounds():
    compiled = compile_predicate(
        Predicate(time_range=("2024-01-01", "2024-12-31")),
        time_column="align_time",
        instrument_column="ticker",
    )
    assert "WHERE" in compiled.where_sql
    assert '"align_time" >= ?' in compiled.where_sql
    assert '"align_time" <= ?' in compiled.where_sql
    assert compiled.params == ["2024-01-01", "2024-12-31"]


def test_time_range_start_only():
    compiled = compile_predicate(
        Predicate(time_range=("2024-01-01", None)),
        time_column="t",
        instrument_column="i",
    )
    assert ">= ?" in compiled.where_sql
    assert "<= ?" not in compiled.where_sql
    assert compiled.params == ["2024-01-01"]


def test_instrument_filter():
    compiled = compile_predicate(
        Predicate(instrument_filter=["AAPL", "MSFT"]),
        time_column="t",
        instrument_column="ticker",
    )
    # list 作为单个参数绑定
    assert '"ticker" IN ?' in compiled.where_sql
    assert compiled.params == [["AAPL", "MSFT"]]


def test_combined_predicate():
    """时间 + 标的同时用：两个条件 AND 连接，参数顺序必须和 ? 一一对应。"""
    compiled = compile_predicate(
        Predicate(
            time_range=("2024-01-01", "2024-12-31"),
            instrument_filter=["AAPL"],
        ),
        time_column="align_time",
        instrument_column="ticker",
    )
    assert " AND " in compiled.where_sql
    # 参数顺序：start, end, instrument_list
    assert compiled.params == ["2024-01-01", "2024-12-31", ["AAPL"]]


def test_extra_predicate_not_supported_in_pr1():
    """PR1 不开放 extra，有人传了要抛错提醒。"""
    with pytest.raises(ValidationError, match="extra"):
        compile_predicate(
            Predicate(extra=[{"col": "x", "op": "=", "value": 1}]),
            time_column="t",
            instrument_column="i",
        )


def test_column_name_with_special_char():
    """列名虽然一般不会有引号，但用了 quote_ident 就要经得起考验。"""
    compiled = compile_predicate(
        Predicate(time_range=(1, 2)),
        time_column='weird"name',
        instrument_column="i",
    )
    # 引号被转义成两个引号
    assert '"weird""name"' in compiled.where_sql
