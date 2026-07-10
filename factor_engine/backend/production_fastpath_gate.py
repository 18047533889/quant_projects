# -*- coding: utf-8
"""Production vs Production Fast Path 双层门禁。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Sequence

FastpathBackend = Literal["duckdb_sql", "polars_long_native"]
RequireMode = Literal["any", "all"]


@dataclass(frozen=True)
class FastpathGateResult:
    """Production fast path 门禁检查结果。"""

    ok: bool
    violations: tuple[str, ...]
    ops_checked: tuple[str, ...]
    original_ops: tuple[str, ...] = ()
    lowered_ops: tuple[str, ...] = ()
    lowering_trace: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。

        返回:
            含 ok、violations、ops_checked 的字典。
        """
        return {
            "ok": self.ok,
            "violations": list(self.violations),
            "ops_checked": list(self.ops_checked),
            "original_ops": list(self.original_ops),
            "lowered_ops": list(self.lowered_ops),
            "lowering_trace": [{"source": s, "primitives": list(p)} for s, p in self.lowering_trace],
        }


def check_original_operator_policy(plan: Any) -> list[str]:
    """阶段 A：原始 plan（lowering 前）production policy 门禁。"""
    from backend.composite_evidence import composite_production_safe
    from cleaned_operators.operator_spec import (
        build_operator_spec,
        infer_production_policy,
        is_production_denied,
        is_production_permanently_forbidden,
    )
    from planner.composite_lowering import has_composite_lowering

    violations: list[str] = []
    for canon in _iter_plan_ops(plan):
        if is_production_permanently_forbidden(canon):
            violations.append(f"{canon}: permanently forbidden")
            continue
        policy = infer_production_policy(canon)
        if policy == "permanently_forbidden":
            violations.append(f"{canon}: production_policy=permanently_forbidden")
            continue
        if policy == "pending":
            violations.append(f"{canon}: production_policy=pending（research composite，不可 production）")
            continue
        if policy == "denied" or is_production_denied(canon):
            violations.append(f"{canon}: production_policy=denied")
            continue
        spec = build_operator_spec(canon)
        if spec is None:
            violations.append(f"{canon}: 无 runtime 实现")
            continue
        if not spec.allow_in_production:
            violations.append(f"{canon}: 不允许 production（status={spec.status}）")
            continue
        if has_composite_lowering(canon):
            if policy == "allowed" and not composite_production_safe(canon):
                violations.append(
                    f"{canon}: composite policy allowed but production evidence incomplete"
                )
                continue
        if not spec.pit_safe:
            violations.append(f"{canon}: pit_safe=False")
        if not spec.deterministic:
            violations.append(f"{canon}: non-deterministic")
        if not spec.shape_preserving:
            violations.append(f"{canon}: shape_preserving=False")
    return violations


_SKIP_OPS = frozenset({"column", "literal", "materialized_series", "plan_ref"})


def fastpath_gate_strict(*, strict: bool | None = None) -> bool:
    """``FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH=1`` 时启用 strict 规则。"""
    if strict is not None:
        return bool(strict)
    return os.environ.get("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def dual_backend_gate_required(*, require_dual: bool | None = None) -> bool:
    """``FACTOR_ENGINE_PRODUCTION_REQUIRE_DUAL_BACKEND=1`` 时要求双后端同时 production-safe。"""
    if require_dual is not None:
        return bool(require_dual)
    return os.environ.get("FACTOR_ENGINE_PRODUCTION_REQUIRE_DUAL_BACKEND", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_require_mode(
    require_mode: RequireMode,
    *,
    require_dual: bool | None = None,
) -> RequireMode:
    if dual_backend_gate_required(require_dual=require_dual):
        return "all"
    return require_mode


def _iter_plan_ops(plan: Any) -> list[str]:
    """深度优先遍历 plan，收集去重后的 canonical 算子列表。"""
    from cleaned_operators.registry import OperatorRegistry

    seen: set[str] = set()
    ops: list[str] = []

    def walk(node: Any) -> None:
        """深度优先遍历 plan 节点收集 canonical 算子。"""
        op = str(getattr(node, "op", "") or "")
        if not op:
            return
        canon = OperatorRegistry._aliases.get(op, op)
        if canon not in seen and canon not in _SKIP_OPS:
            seen.add(canon)
            ops.append(canon)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return ops


def _duckdb_fastpath_ok(canon: str) -> bool:
    """判断 canonical 是否满足 DuckDB production-safe 且 emitter 可编译。"""
    from backend.operator_capability import _sql_emitter_ok
    from backend.sql_tiers import effective_sql_production_safe

    return effective_sql_production_safe(canon) and _sql_emitter_ok(canon)


def _polars_native_fastpath_ok(canon: str) -> bool:
    """判断 canonical 是否满足 PolarsLong native production-safe。"""
    from backend.polars_long_production import is_polars_long_native_production_safe

    return is_polars_long_native_production_safe(canon)


def _check_full_plan_compilation(plan: Any, *, require: frozenset[FastpathBackend]) -> list[str]:
    """对完整 plan 做端到端 compile + 执行探测（非 minimal_plan 单算子）。"""
    violations: list[str] = []
    if "duckdb_sql" in require:
        try:
            from backend.fastpath_plan_probe import probe_duckdb_full_plan

            probe_duckdb_full_plan(plan)
        except Exception as exc:
            violations.append(f"full_plan: duckdb execute failed: {exc}")
    if "polars_long_native" in require:
        try:
            from backend.fastpath_plan_probe import probe_polars_full_plan

            probe_polars_full_plan(plan)
        except Exception as exc:
            violations.append(f"full_plan: polars_long execute failed: {exc}")
    return violations


def check_lowered_backend_fastpath(
    plan: Any,
    *,
    require: Sequence[FastpathBackend] = ("duckdb_sql", "polars_long_native"),
    require_mode: RequireMode = "any",
    strict: bool | None = None,
    require_dual: bool | None = None,
    check_full_plan: bool | None = None,
) -> FastpathGateResult:
    """阶段 B：lowering 后 plan 的 backend fastpath 门禁。"""
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.polars_long_production import POLARS_LONG_FASTPATH_DEFERRED
    from backend.sql_tiers import SQL_PRODUCTION_DEFERRED_CANONICALS
    from cleaned_operators.operator_spec import build_operator_spec
    from planner.composite_lowering import has_composite_lowering

    is_strict = fastpath_gate_strict(strict=strict)
    mode = _resolve_require_mode(require_mode, require_dual=require_dual)
    require_set = frozenset(require)
    violations: list[str] = []
    ops = _iter_plan_ops(plan)

    for canon in ops:
        if is_strict and has_composite_lowering(canon):
            violations.append(f"{canon}: 未下降的 composite 算子（optimizer 须先 lower_composite_operators）")
            continue

        spec = build_operator_spec(canon)
        if spec is None:
            violations.append(f"{canon}: 无 runtime 实现")
            continue
        if not spec.allow_in_production:
            violations.append(f"{canon}: 不允许 production（status={spec.status}）")
            continue

        if canon in POLARS_LONG_FASTPATH_DEFERRED or canon in SQL_PRODUCTION_DEFERRED_CANONICALS:
            violations.append(f"{canon}: deferred，不可 production fast path")
            continue

        tier = infer_polars_long_tier(canon)
        if is_strict and tier in {"map_groups", "registry", "passthrough", "python_rolling", "blocked_causal"}:
            violations.append(f"{canon}: polars_long_{tier} 不可 production fast path（strict）")
            continue

        duck_ok = _duckdb_fastpath_ok(canon)
        polars_ok = _polars_native_fastpath_ok(canon)

        if not is_strict:
            from backend.polars_long_production import APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION

            if tier == "map_groups" and not duck_ok and canon not in APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION:
                violations.append(f"{canon}: map_groups 未批准 production fast path")
                continue
            if tier == "registry" and not duck_ok:
                violations.append(f"{canon}: registry bridge 不可 production fast path")
                continue
            if tier == "passthrough" and not duck_ok:
                violations.append(f"{canon}: passthrough 不可 production fast path")
                continue

        need_duck = "duckdb_sql" in require_set
        need_polars = "polars_long_native" in require_set
        if mode == "all":
            allowed = (not need_duck or duck_ok) and (not need_polars or polars_ok)
        else:
            allowed = False
            if need_duck and duck_ok:
                allowed = True
            if need_polars and polars_ok:
                allowed = True
        if not allowed:
            parts: list[str] = []
            if need_duck and not duck_ok:
                parts.append("duckdb_sql 非 production_safe 或 emitter 失败")
            if need_polars and not polars_ok:
                parts.append("polars_long_native 非 production_safe")
            if mode == "all":
                parts.append("require_mode=all 需双后端同时满足")
            violations.append(f"{canon}: {'; '.join(parts)}")

    do_full_plan = check_full_plan if check_full_plan is not None else is_strict
    if do_full_plan and not violations:
        violations.extend(_check_full_plan_compilation(plan, require=require_set))

    return FastpathGateResult(
        ok=not violations,
        violations=tuple(violations),
        ops_checked=tuple(ops),
    )


def check_production_fastpath_plan_ops(
    plan: Any,
    *,
    original_plan: Any | None = None,
    require: Sequence[FastpathBackend] = ("duckdb_sql", "polars_long_native"),
    require_mode: RequireMode = "any",
    strict: bool | None = None,
    require_dual: bool | None = None,
    check_full_plan: bool | None = None,
    lowering_trace: Sequence[tuple[str, tuple[str, ...]]] | None = None,
) -> FastpathGateResult:
    """两阶段 production gate：原始 policy + lowered backend fastpath。"""
    violations: list[str] = []
    original_ops: tuple[str, ...] = ()
    if original_plan is not None:
        original_ops = tuple(_iter_plan_ops(original_plan))
        violations.extend(check_original_operator_policy(original_plan))

    backend = check_lowered_backend_fastpath(
        plan,
        require=require,
        require_mode=require_mode,
        strict=strict,
        require_dual=require_dual,
        check_full_plan=check_full_plan,
    )
    violations.extend(backend.violations)
    trace = tuple(lowering_trace or ())
    return FastpathGateResult(
        ok=not violations,
        violations=tuple(dict.fromkeys(violations)),
        ops_checked=backend.ops_checked,
        original_ops=original_ops,
        lowered_ops=backend.ops_checked,
        lowering_trace=trace,
    )


def check_production_fastpath_formula_ops(
    formula: str,
    *,
    use_real_plan: bool = True,
    **kwargs: Any,
) -> FastpathGateResult:
    """解析公式并检查 fast path（两阶段：原始 policy + lowered backend）。"""
    if use_real_plan:
        try:
            from api.dsl_parser import parse_expr
            from ir.analyzer import Analyzer
            from planner.lowerer import Lowerer
            from planner.optimizer import Optimizer

            expr = parse_expr(str(formula or ""))
            analysis = Analyzer().lower(expr)
            original = Lowerer().to_logical_plan(analysis.ir)
            optimized, pre_lowering, trace = Optimizer().optimize_with_trace(original)
            return check_production_fastpath_plan_ops(
                optimized,
                original_plan=pre_lowering,
                lowering_trace=trace,
                **kwargs,
            )
        except Exception as exc:
            return FastpathGateResult(
                ok=False,
                violations=(f"真实 plan 编译失败: {exc}",),
                ops_checked=(),
            )

    import ast

    from cleaned_operators.registry import OperatorRegistry
    from planner.logical_plan import PlanNode

    try:
        tree = ast.parse(str(formula or ""), mode="eval")
    except SyntaxError as exc:
        return FastpathGateResult(ok=False, violations=(f"公式语法错误: {exc}",), ops_checked=())

    ops: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "col":
                continue
            canon = OperatorRegistry._aliases.get(name, name)
            if canon not in _SKIP_OPS:
                ops.append(canon)

    if not ops:
        return FastpathGateResult(ok=True, violations=(), ops_checked=())

    from backend.sql_pushdown.plan_fixtures import column, minimal_plan

    violations: list[str] = []
    for op in sorted(set(ops)):
        try:
            sub = minimal_plan(op)
        except Exception:
            sub = PlanNode(op=op, inputs=[column("close")], attrs={"d": 3})
        result = check_production_fastpath_plan_ops(sub, **kwargs)
        violations.extend(result.violations)

    return FastpathGateResult(
        ok=not violations,
        violations=tuple(dict.fromkeys(violations)),
        ops_checked=tuple(sorted(set(ops))),
    )


def audit_runtime_fastpath_violations(runtime: dict[str, Any] | None) -> list[str]:
    """执行后审计：pandas fallback / polars_long fallback / SQL fallback / 非 native tier。"""
    r = dict(runtime or {})
    violations: list[str] = []
    if r.get("polars_long_fallback_reason"):
        violations.append(f"polars_long_fallback: {r['polars_long_fallback_reason']}")
    if r.get("polars_expr_fallback"):
        violations.append("polars_expr_fallback")
    if r.get("used_polars_long_map_groups"):
        violations.append("used_polars_long_map_groups")
    if r.get("used_polars_long_python_rolling"):
        violations.append("used_polars_long_python_rolling")
    if r.get("used_polars_long_registry"):
        violations.append("used_polars_long_registry")
    if r.get("used_polars_long_passthrough"):
        violations.append("used_polars_long_passthrough")
    blocked_causal = list(r.get("polars_long_blocked_causal_ops") or [])
    if blocked_causal:
        violations.append(f"polars_long_blocked_causal:{','.join(blocked_causal)}")
    sql_fallback = int(r.get("sql_fallback_subtree_count") or 0)
    if sql_fallback > 0:
        violations.append(f"sql_fallback_subtrees:{sql_fallback}")
    if r.get("sql_full_execution_failed"):
        violations.append("fully_sql_plan_execution_failed")
    if r.get("used_sql_pushdown") and int(r.get("sql_query_count") or 0) <= 0:
        violations.append("false_sql_pushdown_telemetry")
    if r.get("sql_long_pushdown_error_type"):
        violations.append(f"sql_long_pushdown_failed:{r['sql_long_pushdown_error_type']}")
    for fb in r.get("production_pandas_fallbacks") or []:
        if isinstance(fb, dict) and fb.get("op"):
            violations.append(f"pandas_fallback:{fb['op']}")
    if r.get("production_fastpath_ok") is False:
        for v in r.get("production_fastpath_violations") or []:
            violations.append(f"plan_gate:{v}")
    route = str(r.get("primary_route") or "")
    if route in {
        "pandas_fallback",
        "polars_panel_fallback",
        "polars_expr_fallback",
        "sql_full_execution_failed",
        "sql_to_python_fallback",
    }:
        violations.append(f"primary_route:{route}")
    return violations
