"""Production hard policy: semantic admission is backend independent."""
from __future__ import annotations

import os
from typing import Any, Iterable

PRODUCTION_MODE = "production"


class ProductionPolicyViolation(ValueError):
    """Production policy violation."""


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_run_mode(mode: str | None = None) -> str:
    supported={"research",PRODUCTION_MODE,"paper"}
    if mode is not None:
        resolved=str(mode).strip().lower()
        if resolved not in supported:
            raise ProductionPolicyViolation(f"unknown run mode {mode!r}")
        return resolved
    factor_engine_mode = os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower()
    if factor_engine_mode:
        if factor_engine_mode not in supported:
            raise ProductionPolicyViolation(f"unknown FACTOR_ENGINE_RUN_MODE={factor_engine_mode!r}")
        return factor_engine_mode
    if _truthy_env("QUANT_PRODUCTION_MODE"):
        return PRODUCTION_MODE
    return "research"


def is_production_mode(mode: str | None = None) -> bool:
    return resolve_run_mode(mode) == PRODUCTION_MODE


def production_data_event_auto_publish_enabled() -> bool:
    """production DataEvent 自动 stage+publish 是否启用（R14 复查 P0-1 方案 A）。

    R34 P0-039：严格 production 下 ``DATA_EVENT_PRODUCTION_AUTO_PUBLISH`` 这个
    env bypass **必须不存在**。production DataEvent 统一走原子两阶段
    GenerationTransaction——全部 factor stage 到 staging、全部成功后才逐
    factor publish（任一 stage 失败即 reject，published 因子湖零 mixed
    generation）。该路径由 ``incremental_scheduler`` 强制执行（write_target
    强制 staging），不再由 env 门控：

    - production 模式：env 被忽略，恒返回 ``True``（production 事件只有原子
      两阶段一条路，无 bypass 可关）。
    - research 模式：保留显式 env 启用的实验性两阶段发布。
    """
    if is_production_mode():
        return True  # production: 唯一路径即原子两阶段，无 env escape hatch
    return _truthy_env("DATA_EVENT_PRODUCTION_AUTO_PUBLISH")


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
    names = list(columns)
    if not names or "*" in names:
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

    # A normal incremental call has already loaded analysis lookback + buffer.
    # The planner issues a task-local, one-shot certificate immediately before
    # ``run``. Direct runs cannot forge or reuse it, and full-history factors do
    # not receive it because they require explicit auto-warmup from origin.
    incremental_history_satisfied = False
    if not auto_warmup:
        try:
            from factor_engine.runtime.incremental import consume_incremental_history_certificate

            incremental_history_satisfied = consume_incremental_history_certificate()
        except ImportError:
            incremental_history_satisfied = False

    missing: list[str] = []
    if not input_dq_check:
        missing.append("input_dq_check")
    if not auto_warmup and not incremental_history_satisfied:
        missing.append("auto_warmup_or_incremental_history_contract")
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
    ctx.runtime_stats = runtime


def assert_production_plan_ops(
    plan: Any,
    *,
    mode: str | None = None,
    context: str = "compile",
) -> None:
    if not is_production_mode(mode):
        return
    from factor_engine.backend.pandas_first_signature import check_pandas_first_plan_signatures
    from factor_engine.cleaned_operators.operator_spec import check_production_plan_ops

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
    from factor_engine.backend.production_fastpath_gate import (
        check_production_fastpath_plan_ops,
        fastpath_gate_strict,
    )

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
    from factor_engine.backend.production_fastpath_gate import audit_runtime_fastpath_violations

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
    if not is_production_mode(mode) or not _fastpath_gate_enabled():
        return
    from factor_engine.backend.polars_long_policy import infer_polars_long_tier
    from factor_engine.backend.polars_long_production import (
        APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    bad: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op:
            canonical = OperatorRegistry._aliases.get(op, op)
            if (
                infer_polars_long_tier(canonical) == "map_groups"
                and canonical not in APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION
            ):
                bad.append(canonical)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    if bad:
        raise ProductionPolicyViolation(
            f"production fast path {context} 含未批准 Polars map_groups 算子: "
            + ", ".join(sorted(set(bad)))
        )


def record_production_fastpath_check(
    ctx: Any, plan: Any, *, mode: str | None = None
) -> None:
    if not _fastpath_gate_enabled():
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["production_fastpath_required"] = False
        ctx.runtime_stats = runtime
        return
    from factor_engine.backend.production_fastpath_gate import (
        check_production_fastpath_plan_ops,
        fastpath_gate_strict,
    )

    result = check_production_fastpath_plan_ops(
        plan, strict=fastpath_gate_strict(), mode=mode
    )
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["production_fastpath_required"] = True
    runtime["production_fastpath_ok"] = result.ok
    if result.violations:
        runtime["production_fastpath_violations"] = list(result.violations)
    ctx.runtime_stats = runtime


def assert_production_factors(
    factors: Iterable[Any],
    *,
    mode: str | None = None,
    context: str = "run",
) -> None:
    if not is_production_mode(mode):
        return
    from factor_engine.api.mining_integration import validate_production_dsl
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.storage.catalog import compute_ir_hash

    def _factor_market(factor: Any) -> str | None:
        """Resolve the factor's explicit market (semantic_identity → factor).

        R40 #174: production validation must never silently default a market;
        a factor without one fails closed below."""
        semantic = getattr(factor, "semantic_identity", None)
        for cand in (
            getattr(semantic, "market", None),
            getattr(factor, "market", None),
        ):
            if cand is not None and str(cand) != "":
                return str(cand)
        return None

    violations: list[str] = []
    for factor in factors:
        source = getattr(factor, "source_expr", None)
        if not source:
            violations.append(
                f"{getattr(factor, 'name', '?')}: missing source_expr required for production validation"
            )
            continue
        market = _factor_market(factor)
        if not market:
            violations.append(
                f"{getattr(factor, 'name', '?')}: production DSL validation requires "
                "an explicit market (semantic_identity.market or factor.market); "
                "market=None 只在 research/compat 模式合法（R40 #174）"
            )
            continue
        ok, message = validate_production_dsl(str(source), market=market)
        if not ok:
            violations.append(f"{getattr(factor, 'name', '?')}: {message}")
            continue
        try:
            source_hash = compute_ir_hash(
                Analyzer(production=True, market=market).lower(
                    parse_expr(str(source), surface="daily")
                ).ir,
                structural_only=True
            )
            actual_hash = compute_ir_hash(
                Analyzer(production=True, market=market).lower(factor.expr).ir,
                structural_only=True
            )
        except Exception as exc:
            violations.append(
                f"{getattr(factor, 'name', '?')}: source_expr consistency check failed: {exc}"
            )
            continue
        if source_hash != actual_hash:
            violations.append(
                f"{getattr(factor, 'name', '?')}: source_expr does not match factor.expr "
                f"({source_hash[:12]} != {actual_hash[:12]})"
            )
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
        operators = sorted(
            {str(item.get("op", "")) for item in fallbacks if item.get("op")}
        )
        raise ProductionPolicyViolation(
            f"production 模式 {context} 禁止未计划 pandas fallback，算子: "
            + ", ".join(operators)
        )


def summarize_pandas_fallbacks(ctx: Any) -> list[dict[str, str]]:
    runtime = getattr(ctx, "runtime_stats", None) or {}
    raw = runtime.get("production_pandas_fallbacks") or []
    return [dict(item) for item in raw if isinstance(item, dict)]


def format_pandas_fallback_report(
    fallbacks: Iterable[dict[str, str]],
) -> str:
    items = [dict(item) for item in fallbacks if isinstance(item, dict)]
    if not items:
        return ""
    lines = [f"production pandas fallbacks ({len(items)}):"]
    for entry in items:
        lines.append(
            f"  - {entry.get('op', '?')}: requested={entry.get('requested', '?')} "
            f"actual={entry.get('actual', '?')}"
        )
    return "\n".join(lines)
