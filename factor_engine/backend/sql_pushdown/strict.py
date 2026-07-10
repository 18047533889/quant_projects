# -*- coding: utf-8
"""SQL long-lazy pushdown strict 模式与失败 telemetry。"""
from __future__ import annotations

import os
from typing import Any


class SqlLongPushdownError(RuntimeError):
    """production / strict 下 SQL long-lazy 下推失败。"""


def strict_sql_long(*, ctx: Any | None = None) -> bool:
    """``FACTOR_ENGINE_STRICT_SQL_LONG=1`` 或与 production fastpath 联用。"""
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
    record_sql_long_pushdown_failure(ctx, sid=sid, exc=exc, phase=phase)
    if strict_sql_long(ctx=ctx):
        sid_part = f" sid={sid!r}" if sid else ""
        raise SqlLongPushdownError(
            f"SQL long-lazy pushdown failed{sid_part}: {type(exc).__name__}: {exc}"
        ) from exc
