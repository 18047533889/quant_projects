"""R32 P0-083..086: change_impact.py 保留完整 changed_columns 列表。

旧实现：to_fe_input 只取 changed_columns[0]，多列修订信息丢失。
新实现：保留完整 changed_columns 列表，FE 侧可按需处理所有变化列。
"""
from __future__ import annotations

from data_access.r30.change_impact import to_fe_input
from data_access.r30.data_change import ChangeKind, DataChangeSet


def test_to_fe_input_preserves_all_changed_columns():
    """to_fe_input 保留完整 changed_columns 列表（R32 P0-083..086）。"""
    change = DataChangeSet(
        dataset="financials",
        changed_columns=("revenue", "net_income", "eps"),  # 三列同时修订
        changed_time_range=("2024-01-01", "2024-12-31"),
        changed_instruments=frozenset({"000001.SZ", "600000.SH"}),
        change_kind=ChangeKind.revision,
        revision_availability={"latest_revision_date": "2024-12-31"},
    )

    result = to_fe_input(change)

    # R32: changed_columns 必须完整保留（非只取 [0]）
    assert "changed_columns" in result
    assert result["changed_columns"] == ["revenue", "net_income", "eps"]

    # field 仍是第一列（向后兼容）
    assert result["field"] == "revenue"

    # column_identity 也用第一列（向后兼容）
    assert result["column_identity"]["field"] == "revenue"


def test_to_fe_input_single_column_backward_compatible():
    """单列变化时，行为向后兼容。"""
    change = DataChangeSet(
        dataset="price",
        changed_columns=("close",),
        changed_time_range=("2024-01-01", "2024-01-31"),
        changed_instruments=frozenset(),
        change_kind=ChangeKind.append,
    )

    result = to_fe_input(change)

    assert result["field"] == "close"
    assert result["changed_columns"] == ["close"]
    assert result["column_identity"]["field"] == "close"


def test_to_fe_input_empty_changed_columns():
    """changed_columns 为空时，field 为空字符串（防御性）。"""
    change = DataChangeSet(
        dataset="metadata",
        changed_columns=(),  # 空列表
        changed_time_range=("2024-01-01", "2024-01-01"),
        changed_instruments=frozenset(),
        change_kind=ChangeKind.schema_change,
    )

    result = to_fe_input(change)

    assert result["field"] == ""
    assert result["changed_columns"] == []
    # When field is empty, it's not added to column_identity (防御性)
    assert "field" not in result["column_identity"] or result["column_identity"]["field"] == ""


def test_to_fe_input_with_calendar():
    """calendar 传入时出现在输出中。"""
    change = DataChangeSet(
        dataset="price",
        changed_columns=("open", "high"),
        changed_time_range=("2024-01-01", "2024-01-31"),
        changed_instruments=frozenset(),
        change_kind=ChangeKind.revision,
    )

    from unittest.mock import Mock
    mock_calendar = Mock()

    result = to_fe_input(change, calendar=mock_calendar)

    assert "calendar" in result
    assert result["calendar"] is mock_calendar
    assert result["changed_columns"] == ["open", "high"]


def test_to_fe_input_preserves_revision_identity():
    """revision_identity 正确传递到 column_identity。"""
    change = DataChangeSet(
        dataset="financials",
        changed_columns=("revenue", "net_income"),
        changed_time_range=("2024-01-01", "2024-12-31"),
        changed_instruments=frozenset({"000001.SZ"}),
        change_kind=ChangeKind.revision,
        revision_availability={"latest_revision_date": "2024-12-31T23:59:59"},
    )

    result = to_fe_input(change)

    assert result["column_identity"]["revision_identity"] == "2024-12-31T23:59:59"
    assert result["changed_columns"] == ["revenue", "net_income"]


def test_to_fe_input_all_output_keys():
    """to_fe_input 输出包含所有必需键。"""
    change = DataChangeSet(
        dataset="test_ds",
        changed_columns=("col1", "col2", "col3"),
        changed_time_range=("2024-01-01", "2024-01-31"),
        changed_instruments=frozenset(),
        change_kind=ChangeKind.append,
    )

    result = to_fe_input(change)

    # R32: 必须包含 changed_columns
    assert "field" in result
    assert "changed_start" in result
    assert "changed_end" in result
    assert "column_identity" in result
    assert "changed_columns" in result

    assert result["field"] == "col1"
    assert result["changed_start"] == "2024-01-01"
    assert result["changed_end"] == "2024-01-31"
    assert isinstance(result["column_identity"], dict)
    assert result["changed_columns"] == ["col1", "col2", "col3"]
