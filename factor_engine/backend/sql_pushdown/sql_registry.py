# -*- coding: utf-8 -*-
"""SQL 下推能力与 OperatorRegistry 对齐。

维护 SQL 可编译 canonical 集合，并为每个算子登记 ``backend='sql'`` 占位元数据，
供 planner / emitter 判断计划树是否可下推。
"""
from __future__ import annotations

from dataclasses import dataclass

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode

from backend.contracts import ExecutionKind, PhysicalImplementationSpec
from backend.sql_tiers import (
    SQL_IMPLEMENTED_CANONICALS,
    SQL_PARITY_VERIFIED_CANONICALS,
    SQL_PRODUCTION_SAFE_CANONICALS,
)

# 向后兼容：emitter 有实现 ≠ production-safe
SQL_CAPABLE_CANONICALS: frozenset[str] = SQL_IMPLEMENTED_CANONICALS

__all__ = [
    "SQL_CAPABLE_CANONICALS",
    "SQL_IMPLEMENTED_CANONICALS",
    "SQL_PARITY_VERIFIED_CANONICALS",
    "SQL_PRODUCTION_SAFE_CANONICALS",
    "SqlCapableOperator",
    "is_sql_capable",
    "is_sql_parity_verified",
    "is_sql_production_safe",
    "register_sql_backends",
    "resolve_canonical",
    "sql_backends_for",
]

_SQL_MARKERS_REGISTERED = False
_SQL_MARKERS_INCOMPLETE = False


@dataclass(frozen=True)
class SqlCapableOperator:
    """Registry 中 ``backend='sql'`` 的占位算子（编译走 emitter）。

    .. important::（审计 #363）
        SQL 实现**不拥有** parameter signature：capability 一律读 OperatorRegistry
        的 logical contract（``_catalog[canon]`` 的 ``param_names`` / ``param_specs`` /
        生产签名）。这里 metadata 的 ``param_names=[]`` 只是占位，**不做**签名审计
        ——签名审计走 registry contract（见 ``backend/operator_capability.py`` 的
        ``check_call_capability`` / ``_sql_bound_param_names``）。
    """

    canonical: str

    @property
    def _physical_specs(self) -> dict[str, PhysicalImplementationSpec]:
        """Declare the concrete SQL implementation for each supported dialect."""
        return {
            "duckdb_sql": PhysicalImplementationSpec(
                canonical=self.canonical,
                backend="duckdb_sql",
                execution_kind=ExecutionKind.DUCKDB_NATIVE_SQL,
                supports_lazy=True,
                supports_streaming=True,
                supports_nulls=True,
                supports_nan=True,
                supports_inf=True,
                implementation_source_hash="backend.sql_pushdown.sql_registry:SqlCapableOperator:v2",
                emitter_identity="backend.sql_pushdown.emitter:duckdb:v1",
                parameter_domain_hash="sql:parameter-domain:v1",
                semantic_contract_hash="sql:semantic-contract:v1",
            ),
            "clickhouse_sql": PhysicalImplementationSpec(
                canonical=self.canonical,
                backend="clickhouse_sql",
                execution_kind=ExecutionKind.NATIVE_EXPR,
                supports_lazy=True,
                supports_streaming=True,
                supports_nulls=True,
                supports_nan=True,
                supports_inf=True,
                implementation_source_hash="backend.sql_pushdown.sql_registry:SqlCapableOperator:v2",
                emitter_identity="backend.sql_pushdown.emitter:clickhouse:v1",
                parameter_domain_hash="sql:parameter-domain:v1",
                semantic_contract_hash="sql:semantic-contract:v1",
            ),
        }

    @property
    def metadata(self):
        from cleaned_operators.base import OperatorMetadata

        return OperatorMetadata(
            name=self.canonical,
            category="sql",
            description=f"SQL 下推算子 {self.canonical}",
            param_names=[],
            return_type="series",
            tags=["sql", "pushdown"],
        )


def register_sql_backends() -> None:
    """为 SQL 可编译 canonical 登记 ``backend='sql'`` 元数据。

    可在 ``load_all()`` 之前被 ``sql_pushdown`` 包 import 触发；若当时尚无
    runtime 实现，会标记 incomplete，并在后续再次调用时补登记。
    """
    global _SQL_MARKERS_REGISTERED, _SQL_MARKERS_INCOMPLETE
    if _SQL_MARKERS_REGISTERED and not _SQL_MARKERS_INCOMPLETE:
        return
    incomplete = False
    for canon in SQL_CAPABLE_CANONICALS:
        if canon in {"column", "literal"}:
            continue
        # Attach sql marker to the resolved runtime canonical when possible, so we
        # never create an empty new primary name that later blocks rename merges.
        resolved = OperatorRegistry.resolve_canonical_optional(canon)
        if resolved in OperatorRegistry._operators:
            target = resolved
        elif canon in OperatorRegistry._operators:
            target = canon
        else:
            incomplete = True
            continue
        if "sql" not in OperatorRegistry.backends_for(target):
            OperatorRegistry.register(
                SqlCapableOperator(target),
                canonical=target,
                backend="sql",
                source="backend/sql_pushdown",
            )
        # SQL capability is an explicit implementation contract.  Production
        # reports must not infer these flags from an operator name or defaults.
        from cleaned_operators.operator_policy import infer_operator_policy

        entry = OperatorRegistry._catalog[target]
        policy = infer_operator_policy(target, canonical=target)
        scope = getattr(policy, "scope", "unknown")
        params = set(entry.get("param_names") or ())
        backend_meta = dict(entry.get("backend_meta") or {})
        sql_meta = dict(backend_meta.get("sql") or {})
        sql_meta.update({
            "execution_kind": "duckdb_native_sql",
            "supports_lazy": True,
            "supports_streaming": True,
            "materializes_full_panel": False,
            "supports_nulls": True,
            "supports_nan": True,
            "supports_inf": True,
            "supports_scalar_broadcast": True,
            "supports_group": scope == "group",
            "supports_window": scope == "ts",
            "supports_min_periods": "min_periods" in params,
        })
        backend_meta["sql"] = sql_meta
        entry["backend_meta"] = backend_meta
    _SQL_MARKERS_REGISTERED = True
    _SQL_MARKERS_INCOMPLETE = incomplete


def resolve_canonical(op: str) -> str:
    """将算子别名解析为 canonical 名称。"""
    resolved = OperatorRegistry._aliases.get(op, op)
    # During early imports legacy aliases may temporarily point at a pre-dedupe
    # name (for example ts_regression_slope -> ts_regression).  Keep an explicit
    # SQL canonical when the temporary target is not itself SQL implemented.
    if op in SQL_CAPABLE_CANONICALS and resolved not in SQL_CAPABLE_CANONICALS:
        return op
    return resolved


def is_sql_capable(plan: PlanNode) -> bool:
    """递归判断计划树是否全部由 SQL backend 支持的算子构成。"""
    register_sql_backends()
    canon = resolve_canonical(plan.op)
    if canon not in SQL_CAPABLE_CANONICALS:
        return False
    return all(is_sql_capable(c) for c in plan.inputs)


def _plan_has_tier(plan: PlanNode, allowed: frozenset[str]) -> bool:
    canon = resolve_canonical(plan.op)
    if canon not in allowed:
        return False
    return all(_plan_has_tier(child, allowed) for child in plan.inputs)


def is_sql_parity_verified(plan: PlanNode) -> bool:
    """Return whether every plan node has real DuckDB parity evidence."""
    return _plan_has_tier(plan, SQL_PARITY_VERIFIED_CANONICALS)


def is_sql_production_safe(plan: PlanNode) -> bool:
    """Return whether every plan node is certified for production SQL."""
    return _plan_has_tier(plan, SQL_PRODUCTION_SAFE_CANONICALS)


def sql_backends_for(name: str) -> list[str]:
    """返回算子名在 OperatorRegistry 中登记的后端列表（含 ``sql``）。"""
    register_sql_backends()
    return OperatorRegistry.backends_for(name)
