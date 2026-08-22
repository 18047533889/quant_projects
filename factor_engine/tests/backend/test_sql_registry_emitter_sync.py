# -*- coding: utf-8
"""SQL registry 白名单与 emitter 编译能力一致性 CI。"""

from __future__ import annotations

import pytest

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.plan_fixtures import minimal_plan
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS, register_sql_backends
from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS, SQL_PRODUCTION_SAFE_CANONICALS


@pytest.fixture(scope="module", autouse=True)
def _load_sql_markers():
    register_sql_backends()


_SQL_OPS = sorted(SQL_IMPLEMENTED_CANONICALS - {"column", "literal"})
assert SQL_CAPABLE_CANONICALS == SQL_IMPLEMENTED_CANONICALS


@pytest.mark.parametrize("op", _SQL_OPS)
def test_sql_capable_ops_emit_non_empty_sql(op: str):
    plan = minimal_plan(op)
    assert plan_is_sql_capable(plan), f"{op} 在 registry 中但 plan 不可 SQL 下推"
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None, f"{op} plan_is_sql_capable 但 emitter 返回 None"
    assert compiled.query.strip(), f"{op} 编译 SQL 为空"
