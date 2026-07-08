# -*- coding: utf-8
"""SQL 下推：DuckDB 内执行可编译子集，其余回退 Python。"""

from backend.sql_pushdown.emitter import (
    SQL_CAPABLE_OPS,
    compile_plan_to_sql,
    plan_is_sql_capable,
)
from backend.sql_pushdown.executor import try_execute_sql_pushdown

__all__ = [
    "SQL_CAPABLE_OPS",
    "compile_plan_to_sql",
    "plan_is_sql_capable",
    "try_execute_sql_pushdown",
]
