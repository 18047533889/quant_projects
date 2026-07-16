# -*- coding: utf-8 -*-
"""SQL 下推子包：PlanNode 编译、DuckDB/ClickHouse 执行与 registry 对齐。

对外导出 ``compile_plan_to_sql``、``try_execute_sql_pushdown`` 等入口；
导入时自动调用 ``register_sql_backends()`` 登记 SQL backend 元数据。
"""

from backend.sql_pushdown.emitter import (
    CompiledSql,
    SqlDialect,
    SqlPushdownFilter,
    compile_plan_to_sql,
    plan_is_sql_capable,
)
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
