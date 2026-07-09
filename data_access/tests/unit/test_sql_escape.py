"""sql_escape 表名重写单元测试。"""

from __future__ import annotations

from data_access.sql_escape import _rewrite_query_tables


def test_rewrite_query_tables_bare_name():
    view_map = {"factors": "__da_ab12_factors", "prices": "__da_ab12_prices"}
    sql = "SELECT f.asset FROM factors f JOIN prices p USING (asset)"
    out = _rewrite_query_tables(sql, view_map)
    assert "__da_ab12_factors" in out
    assert "__da_ab12_prices" in out
    assert " factors " not in out
    assert " prices " not in out


def test_rewrite_query_tables_placeholder_syntax():
    view_map = {"factor_lake": "__da_cd34_factor_lake"}
    sql = "SELECT * FROM {{factor_lake}} WHERE value > 0"
    out = _rewrite_query_tables(sql, view_map)
    assert "__da_cd34_factor_lake" in out
    assert "{{" not in out
