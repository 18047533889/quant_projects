# -*- coding: utf-8
"""SQL registry / emitter 一致性 + SQL_PRODUCTION_SAFE 约束。"""
from __future__ import annotations

import pytest

from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from factor_engine.backend.sql_pushdown.plan_fixtures import minimal_plan
from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
from factor_engine.backend.sql_tiers import (
    SQL_IMPLEMENTED_CANONICALS,
    SQL_PRODUCTION_SAFE_CANONICALS,
)


@pytest.fixture(scope="module", autouse=True)
def _load_sql_markers():
    register_sql_backends()


_SQL_OPS = sorted(SQL_IMPLEMENTED_CANONICALS - {"column", "literal"})
_SQL_PROD_OPS = sorted(SQL_PRODUCTION_SAFE_CANONICALS - {"column", "literal"})


@pytest.mark.parametrize("op", _SQL_OPS)
def test_sql_implemented_ops_emit_non_empty_sql(op: str):
    plan = minimal_plan(op)
    assert plan_is_sql_capable(plan), f"{op} 在 SQL_IMPLEMENTED 中但 plan 不可 SQL 下推"
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None, f"{op} plan_is_sql_capable 但 emitter 返回 None"
    assert compiled.query.strip(), f"{op} 编译 SQL 为空"


@pytest.mark.parametrize("op", _SQL_PROD_OPS)
def test_sql_production_safe_ops_emitter_ok(op: str):
    """SQL_PRODUCTION_SAFE 必须能 minimal_plan 编译出非空 SQL。"""
    plan = minimal_plan(op)
    assert plan_is_sql_capable(plan), f"{op} production_safe 但 plan 不可 SQL"
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None and compiled.query.strip(), f"{op} production_safe emitter 失败"
