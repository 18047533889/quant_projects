"""sql_escape 表名重写单元测试。"""

from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.sql_escape import _rewrite_query_tables, assert_sql_from_scope


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


def test_rewrite_query_tables_does_not_touch_comments_or_string_placeholder():
    view_map = {"factor_lake": "__da_x_factor_lake"}
    sql = "SELECT '{{factor_lake}}' AS literal -- {{factor_lake}}\nFROM {{factor_lake}}"
    out = _rewrite_query_tables(sql, view_map)
    assert "'{{factor_lake}}'" in out
    assert "-- {{factor_lake}}" in out
    assert out.endswith("FROM __da_x_factor_lake")


@pytest.mark.parametrize(
    "query",
    [
        "WITH c AS (SELECT * FROM _sub) SELECT * FROM c",
        "WITH C AS (SELECT * FROM _sub) SELECT * FROM c",
        "WITH C AS (SELECT * FROM _sub) SELECT * FROM c AS d",
        (
            "WITH a AS (SELECT * FROM _sub), b AS (SELECT * FROM a) "
            "SELECT * FROM b"
        ),
        (
            "WITH shadowed_external AS (SELECT * FROM _sub) "
            "SELECT $$FROM real_external$$ AS note FROM shadowed_external "
            "-- JOIN another_external"
        ),
    ],
)
def test_relation_scope_allows_local_cte_aliases(query):
    assert_sql_from_scope(query, allowed=("_sub",))


@pytest.mark.parametrize(
    "query",
    [
        "WITH c AS (SELECT * FROM real_external) SELECT * FROM c",
        "WITH c AS (SELECT * FROM _sub) SELECT * FROM c JOIN real_external ON TRUE",
        (
            "WITH real_external AS (SELECT * FROM real_external) "
            "SELECT * FROM real_external"
        ),
        "WITH a AS (SELECT * FROM later), later AS (SELECT * FROM _sub) SELECT * FROM a",
        (
            "SELECT * FROM (WITH c AS (SELECT * FROM real_external) SELECT * FROM c) q"
        ),
        "WITH C AS (SELECT * FROM _sub) SELECT * FROM real_external AS c",
        "WITH C AS (SELECT * FROM _sub) SELECT * FROM evil.c AS c",
    ],
)
def test_relation_scope_rejects_external_tables_inside_or_beside_cte(query):
    with pytest.raises(ValidationError, match="scope 外"):
        assert_sql_from_scope(query, allowed=("_sub",))
