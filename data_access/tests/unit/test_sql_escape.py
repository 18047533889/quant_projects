"""sql_escape 表名重写单元测试。"""

from __future__ import annotations

import pytest

from data_access.exceptions import ValidationError
from data_access.sql_escape import _rewrite_query_tables


def test_rewrite_query_tables_placeholder_syntax():
    view_map = {"factor_lake": "__da_cd34_factor_lake"}
    sql = "SELECT * FROM {{factor_lake}} WHERE value > 0"
    out = _rewrite_query_tables(sql, view_map)
    assert "__da_cd34_factor_lake" in out
    assert "{{" not in out


def test_rewrite_query_tables_join_with_aliases():
    view_map = {"factors": "__da_ab12_factors", "prices": "__da_ab12_prices"}
    sql = (
        "SELECT f.asset FROM {{factors}} f "
        "JOIN {{prices}} p USING (asset)"
    )
    out = _rewrite_query_tables(sql, view_map)
    assert "__da_ab12_factors" in out
    assert "__da_ab12_prices" in out


def test_rewrite_query_tables_rejects_missing_placeholder():
    view_map = {"factors": "__da_ab12_factors"}
    sql = "SELECT f.asset FROM factors f"
    with pytest.raises(ValidationError, match="缺少占位符"):
        _rewrite_query_tables(sql, view_map)


def test_rewrite_query_tables_does_not_touch_string_literals():
    view_map = {"factor_lake": "__da_x_factor_lake"}
    sql = (
        "SELECT * FROM {{factor_lake}} "
        "WHERE note = 'factor_lake' AND source = 'us_stocks'"
    )
    out = _rewrite_query_tables(sql, view_map)
    assert "note = 'factor_lake'" in out
    assert "source = 'us_stocks'" in out
    assert "__da_x_factor_lake" in out
