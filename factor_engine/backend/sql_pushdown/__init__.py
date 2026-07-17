# -*- coding: utf-8 -*-
"""SQL pushdown: PlanNode compilation, execution and backend registration."""

from backend.sql_pushdown.emitter import (
    CompiledSql,
    SqlDialect,
    SqlPushdownFilter,
    compile_plan_to_sql,
    plan_is_sql_capable,
)
from backend.sql_pushdown.cos_semantic_fixes import install_sql_semantic_fixes
from backend.sql_pushdown.executor import (
    PushdownContext,
    execute_compiled_sql,
    extract_pushdown_context,
    try_execute_sql_pushdown,
)
from backend.sql_pushdown.sql_registry import (
    SQL_CAPABLE_CANONICALS,
    is_sql_capable,
    register_sql_backends,
    sql_backends_for,
)
from backend.sql_pushdown.active_contracts import apply_active_sql_contracts

apply_active_sql_contracts()
install_sql_semantic_fixes()
register_sql_backends()

__all__ = [
    "SQL_CAPABLE_CANONICALS",
    "CompiledSql",
    "SqlDialect",
    "SqlPushdownFilter",
    "PushdownContext",
    "compile_plan_to_sql",
    "plan_is_sql_capable",
    "is_sql_capable",
    "register_sql_backends",
    "sql_backends_for",
    "execute_compiled_sql",
    "extract_pushdown_context",
    "try_execute_sql_pushdown",
]
