"""Production hard policy: semantic admission is backend independent.

A factor operator may be production-safe on one certified backend without being
portable to every backend. Backend-specific fast-path constraints are enforced
only when that fast path is requested/selected; they must never reject an
otherwise production-certified Pandas/Numpy execution path.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

PRODUCTION_MODE = "production"


class ProductionPolicyViolation(ValueError):
    """Production policy violation."""


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_run_mode(mode: str | None = None) -> str:
    if mode is not None and str(mode).strip():
        return str(mode).lower()
    fe = os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower()
    if fe in {"research", PRODUCTION_MODE}:
        return fe
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
        raise ProductionPolicyViolation(
            f"production 模式禁止 stub 算子: {', '.join(sorted(set(stub_ops)))}"
        )


def record_production_pandas_fallback(
    ctx: Any,
    *,
    op: str,
    requested_backend: str,
    actual_backend: str,
    mode: str | None = None,
) -> None:
    """Record a real fallback only; intentional cost routing is not fallback."""
    if mode is None:
        mode = getattr(ctx, "run_mode", None)
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


def assert_production_plan_ops(
    plan: Any,
    *,
    mode: str | None = None,
    context: str = "compile",
) -> None:
    """Semantic production gate + bounded production call signatures."""
    if not is_production_mode(mode):
        return
    from cleaned_operators.operator_spec import check_production_plan_ops
    from backend.pandas_first_signature import check_pandas_first_plan_signatures

    violations = check_production_plan_ops(plan)
    violations.extend(check_pandas_first_plan_signatures(plan))
    if violations:
        raise ProductionPolicyViolation(
            f"production 模式 {context} 含非 production/未认证调用: {'; '.join(violations)}"
        )


def _fastpath_gate_enabled() -> bool:
    return _truthy_env("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH")


def assert_production_fastpath_plan(
    plan: Any,
    *,
    mode: str | None = None,
    context: str = "compile",
) -> None:
    if not is_production_mode(mode) or not _fastpath_gate_enabled():
        return
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops, fastpath_gate_strict

    result = check_production_fastpath_plan_ops(
        plan, strict=fastpath_gate_strict(), mode=mode
    )
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
    """Backend-specific map_groups gate.

    This check used to run for every production plan and therefore rejected a
    Pandas-certified operator merely because an optional Polars map_groups
    implementation was not certified.  It now applies only when the caller
    explicitly requires production fast paths. Normal production routing is
    constrained later by per-backend capability evidence.
    """
    if not is_production_mode(mode) or not _fastpath_gate_enabled():
        return
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.polars_long_production import APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION
    from cleaned_operators.registry import OperatorRegistry

    bad: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op:
            canon = OperatorRegistry._aliases.get(op, op)
            if (
                infer_polars_long_tier(canon) == "map_groups"
                and canon not in APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION
            ):
                bad.append(canon)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    if bad:
        raise ProductionPolicyViolation(
            f"production fast path {context} 含未批准 Polars map_groups 算子: "
            + ", ".join(sorted(set(bad)))
        )


def record_production_fastpath_check(ctx: Any, plan: Any, *, mode: str | None = None) -> None:
    # A diagnostic fastpath check must not silently reintroduce the strict dual
    # backend policy when the production job did not request that policy.
    if not _fastpath_gate_enabled():
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["production_fastpath_required"] = False
        ctx.runtime_stats = runtime  # type: ignore[attr-defined]
        return
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops, fastpath_gate_strict

    result = check_production_fastpath_plan_ops(
        plan, strict=fastpath_gate_strict(), mode=mode
    )
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["production_fastpath_required"] = True
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
    """Validate restricted syntax; canonical plan owns production admission."""
    if not is_production_mode(mode):
        return
    from api.mining_integration import validate_factor_engine_dsl

    violations: list[str] = []
    for factor in factors:
        src = getattr(factor, "source_expr", None)
        if src:
            ok, msg = validate_factor_engine_dsl(str(src), surface="lqtp")
            if not ok:
                violations.append(f"{getattr(factor, 'name', '?')}: {msg}")
    if violations:
        raise ProductionPolicyViolation(
            f"production 模式 {context} DSL 语法/兼容校验失败: {'; '.join(violations)}"
        )


def assert_no_production_pandas_fallbacks(
    ctx: Any,
    *,
    mode: str | None = None,
    context: str = "execute",
) -> None:
    if mode is None:
        mode = getattr(ctx, "run_mode", None)
    if not is_production_mode(mode):
        return
    runtime = getattr(ctx, "runtime_stats", None) or {}
    fallbacks = runtime.get("production_pandas_fallbacks") or []
    policy = str(getattr(ctx, "production_fallback_policy", "error") or "error")
    if fallbacks and policy != "warn":
        ops = sorted({str(x.get("op", "")) for x in fallbacks if x.get("op")})
        raise ProductionPolicyViolation(
            f"production 模式 {context} 禁止未计划 pandas fallback，算子: {', '.join(ops)}"
        )


def summarize_pandas_fallbacks(ctx: Any) -> list[dict[str, str]]:
    runtime = getattr(ctx, "runtime_stats", None) or {}
    raw = runtime.get("production_pandas_fallbacks") or []
    return [dict(x) for x in raw if isinstance(x, dict)]


def format_pandas_fallback_report(fallbacks: Iterable[dict[str, str]]) -> str:
    items = [dict(x) for x in fallbacks if isinstance(x, dict)]
    if not items:
        return ""
    lines = [f"production pandas fallbacks ({len(items)}):"]
    for entry in items:
        lines.append(
            f"  - {entry.get('op','?')}: requested={entry.get('requested','?')} "
            f"actual={entry.get('actual','?')}"
        )
    return "\n".join(lines)
