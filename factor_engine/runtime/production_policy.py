"""Production 硬策略：research 宽松 / production 强制门禁。"""

from __future__ import annotations

import os
from typing import Any, Iterable

PRODUCTION_MODE = "production"


class ProductionPolicyViolation(ValueError):
    """production 模式下违反硬策略。"""


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_run_mode(mode: str | None = None) -> str:
    if mode is not None and str(mode).strip():
        return str(mode).lower()
    fe = os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower()
    if fe == "research":
        return "research"
    if fe == PRODUCTION_MODE:
        return PRODUCTION_MODE
    if _truthy_env("QUANT_PRODUCTION_MODE"):
        return PRODUCTION_MODE
    return "research"


def is_production_mode(mode: str | None = None) -> bool:
    return resolve_run_mode(mode) == PRODUCTION_MODE


def assert_columns_explicit(
    columns: Iterable[str] | None,
    *,
    mode: str | None = None,
    context: str = "read",
) -> None:
    """production 禁止 SELECT * / 空列列表。"""
    if not is_production_mode(mode):
        return
    if columns is None:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 必须显式指定 columns，禁止 SELECT *"
        )
    cols = list(columns)
    if not cols or "*" in cols:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 必须显式列名，禁止 SELECT * 或空 columns"
        )


def assert_production_run_flags(
    *,
    mode: str | None = None,
    input_dq_check: bool = False,
    auto_warmup: bool = False,
    pit_enforce: bool = False,
    context: str = "run",
) -> None:
    """production 强制 input_dq / auto_warmup / PIT。"""
    if not is_production_mode(mode):
        return
    missing: list[str] = []
    if not input_dq_check:
        missing.append("input_dq_check")
    if not auto_warmup:
        missing.append("auto_warmup")
    if not pit_enforce:
        missing.append("pit_enforce")
    if missing:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 必须开启: {', '.join(missing)}"
        )


def assert_no_stub_operators(plan: Any, *, mode: str | None = None) -> None:
    """production 禁止 ``*_stub`` 占位算子。"""
    if not is_production_mode(mode):
        return
    stub_ops: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op.endswith("_stub"):
            stub_ops.append(op)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    if stub_ops:
        unique = sorted(set(stub_ops))
        raise ProductionPolicyViolation(
            f"production 模式禁止 stub 算子: {', '.join(unique)}"
        )


def record_production_pandas_fallback(
    ctx: Any,
    *,
    op: str,
    requested_backend: str,
    actual_backend: str,
    mode: str | None = None,
) -> None:
    """production 下 Polars 热路径回退 pandas 时记录/告警。"""
    if not is_production_mode(mode):
        return
    if actual_backend != "pandas_numpy" or requested_backend == actual_backend:
        return
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    fallbacks = list(runtime.get("production_pandas_fallbacks", []))
    entry = {"op": op, "requested": requested_backend, "actual": actual_backend}
    if entry not in fallbacks:
        fallbacks.append(entry)
    runtime["production_pandas_fallbacks"] = fallbacks
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
    import logging

    logging.getLogger("runtime.production_policy").warning(
        "production pandas fallback: op=%s requested=%s actual=%s",
        op,
        requested_backend,
        actual_backend,
    )


def assert_production_plan_ops(
    plan: Any,
    *,
    mode: str | None = None,
    context: str = "compile",
) -> None:
    """production 模式：逻辑计划中的算子须 ``allow_in_production``。"""
    if not is_production_mode(mode):
        return
    from cleaned_operators.operator_spec import check_production_plan_ops

    violations = check_production_plan_ops(plan)
    if violations:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 含非 production 算子: {'; '.join(violations)}"
        )


def _fastpath_gate_enabled() -> bool:
    return _truthy_env("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH")


def assert_production_fastpath_plan(
    plan: Any,
    *,
    mode: str | None = None,
    context: str = "compile",
) -> None:
    """production + ``FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH=1``：须可走 fast path。"""
    if not is_production_mode(mode) or not _fastpath_gate_enabled():
        return
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops, fastpath_gate_strict

    result = check_production_fastpath_plan_ops(plan, strict=fastpath_gate_strict())
    if not result.ok:
        raise ProductionPolicyViolation(
            f"production fast path {context} 未通过: {'; '.join(result.violations)}"
        )


def assert_production_fastpath_runtime(
    ctx: Any,
    *,
    mode: str | None = None,
    context: str = "execute",
) -> None:
    """production fastpath：执行后不得发生 fallback / 非 native tier。"""
    if not is_production_mode(mode) or not _fastpath_gate_enabled():
        return
    from backend.production_fastpath_gate import audit_runtime_fastpath_violations

    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    violations = audit_runtime_fastpath_violations(runtime)
    if violations:
        raise ProductionPolicyViolation(
            f"production fast path {context} runtime 违规: {'; '.join(violations)}"
        )


def assert_no_unapproved_map_groups_in_production(
    plan: Any,
    *,
    mode: str | None = None,
    context: str = "compile",
) -> None:
    """production 默认禁止 map_groups（除非在批准白名单）。"""
    if not is_production_mode(mode):
        return
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.polars_long_production import APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION
    from cleaned_operators.registry import OperatorRegistry

    bad: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if not op:
            return
        canon = OperatorRegistry._aliases.get(op, op)
        if infer_polars_long_tier(canon) == "map_groups" and canon not in APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION:
            bad.append(canon)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    if bad:
        unique = sorted(set(bad))
        raise ProductionPolicyViolation(
            f"production 模式 {context} 含未批准 map_groups 算子: {', '.join(unique)}"
        )


def record_production_fastpath_check(ctx: Any, plan: Any, *, mode: str | None = None) -> None:
    """记录 fast path 检查结果到 runtime_stats（不强制）。"""
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops, fastpath_gate_strict

    result = check_production_fastpath_plan_ops(plan, strict=fastpath_gate_strict())
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["production_fastpath_ok"] = result.ok
    if result.violations:
        runtime["production_fastpath_violations"] = list(result.violations)
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


def assert_production_factors(
    factors: Iterable[Any],
    *,
    mode: str | None = None,
    context: str = "run",
) -> None:
    """production 模式：因子 DSL / source_expr 须通过 production 校验。"""
    if not is_production_mode(mode):
        return
    from api.mining_integration import validate_production_dsl

    violations: list[str] = []
    for factor in factors:
        src = getattr(factor, "source_expr", None)
        if src:
            ok, msg = validate_production_dsl(str(src))
            if not ok:
                violations.append(f"{getattr(factor, 'name', '?')}: {msg}")
    if violations:
        raise ProductionPolicyViolation(
            f"production 模式 {context} DSL 校验失败: {'; '.join(violations)}"
        )


def assert_no_production_pandas_fallbacks(
    ctx: Any,
    *,
    mode: str | None = None,
    context: str = "execute",
) -> None:
    """production 严格模式：禁止 Polars 热路径回退 pandas。"""
    if not is_production_mode(mode):
        return
    strict = os.environ.get("FACTOR_ENGINE_PRODUCTION_STRICT_POLARS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not strict:
        return
    runtime = getattr(ctx, "runtime_stats", None) or {}
    fallbacks = runtime.get("production_pandas_fallbacks") or []
    if fallbacks:
        ops = sorted({str(x.get("op", "")) for x in fallbacks if x.get("op")})
        raise ProductionPolicyViolation(
            f"production 严格模式 {context} 禁止 pandas fallback，算子: {', '.join(ops)}"
        )


def summarize_pandas_fallbacks(ctx: Any) -> list[dict[str, str]]:
    """汇总 production 运行中的 Polars→pandas 回退记录（供报表 / 监控）。"""
    runtime = getattr(ctx, "runtime_stats", None) or {}
    raw = runtime.get("production_pandas_fallbacks") or []
    return [dict(x) for x in raw if isinstance(x, dict)]


def format_pandas_fallback_report(fallbacks: Iterable[dict[str, str]]) -> str:
    """将 fallback 列表格式化为单行摘要（日志 / 批跑报表）。"""
    items = [dict(x) for x in fallbacks if isinstance(x, dict)]
    if not items:
        return ""
    lines = [f"production pandas fallbacks ({len(items)}):"]
    for entry in items:
        op = entry.get("op", "?")
        requested = entry.get("requested", "?")
        actual = entry.get("actual", "?")
        lines.append(f"  - {op}: requested={requested} actual={actual}")
    return "\n".join(lines)
