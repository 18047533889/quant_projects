# -*- coding: utf-8
"""SQL long-lazy 下推的 strict 模式与失败 telemetry。

当 ``FACTOR_ENGINE_STRICT_SQL_LONG`` 或 production fastpath 开启时，
下推失败会抛出 ``SqlLongPushdownError`` 而非静默回退 Python/Polars。
"""
from __future__ import annotations

import os
from typing import Any


class SqlLongPushdownError(RuntimeError):
    """production / strict 下 SQL long-lazy 下推失败。"""


def strict_sql_long(*, ctx: Any | None = None) -> bool:
    """是否启用 SQL long-lazy 下推 strict 模式。

    当环境变量 ``FACTOR_ENGINE_STRICT_SQL_LONG`` 或
    ``FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH`` 为真时返回 ``True``。
    """
    if os.environ.get("FACTOR_ENGINE_STRICT_SQL_LONG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    if os.environ.get("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return False


def record_sql_long_pushdown_failure(
    ctx: Any,
    *,
    sid: str | None,
    exc: BaseException | None = None,
    phase: str = "execute",
) -> None:
    """将 SQL long-lazy 下推失败信息写入 ``ctx.runtime_stats``。"""
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    if sid is not None:
        runtime["sql_long_pushdown_failed_sid"] = str(sid)
    if exc is not None:
        runtime["sql_long_pushdown_error_type"] = type(exc).__name__
        runtime["sql_long_pushdown_error_message"] = str(exc)[:500]
    runtime["sql_long_pushdown_failed_phase"] = phase
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


def handle_sql_long_pushdown_failure(
    ctx: Any,
    *,
    sid: str | None,
    exc: BaseException,
    phase: str = "execute",
) -> None:
    """记录失败 telemetry；strict 模式下重新抛出 ``SqlLongPushdownError``。"""
    record_sql_long_pushdown_failure(ctx, sid=sid, exc=exc, phase=phase)
    if strict_sql_long(ctx=ctx):
        sid_part = f" sid={sid!r}" if sid else ""
        raise SqlLongPushdownError(
            f"SQL long-lazy pushdown failed{sid_part}: {type(exc).__name__}: {exc}"
        ) from exc
