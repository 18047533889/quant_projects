# -*- coding: utf-8 -*-
"""Mixed SQL + Polars backend with production-safe physical routing."""
from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode
from .base import Backend
from .context import ExecutionContext
from .sql_backend import SqlBackend


def _supports_polars_long_scan(ctx: ExecutionContext) -> bool:
    ds = getattr(ctx, "data_source", None)
    scan = getattr(ds, "scan_polars_long", None)
    return callable(scan)


def _production_long_plan_safe(plan: PlanNode, ctx: ExecutionContext) -> tuple[bool, tuple[str, ...]]:
    """Require every physical Polars-long node to have production evidence.

    SourceRef columns intentionally stay on the logical/Pandas-Arrow boundary;
    attempting to route them through ``scan_polars_long`` would either fail or
    bypass source-specific PIT contracts.
    """
    if str(getattr(ctx, "run_mode", "research") or "research").lower() != "production":
        return True, ()

    from api.source_ref import decode_source_ref
    from backend.polars_long_production import is_polars_long_native_production_safe
    from cleaned_operators.registry import OperatorRegistry

    bad: set[str] = set()

    def walk(node: PlanNode) -> None:
        op = str(getattr(node, "op", "") or "")
        if op == "column":
            name = str((getattr(node, "attrs", None) or {}).get("name") or "")
            if name and decode_source_ref(name) is not None:
                bad.add("SourceRef")
        elif op not in {"", "literal", "plan_ref", "materialized_series"}:
            canon = OperatorRegistry._aliases.get(op, op)
            if not is_polars_long_native_production_safe(canon):
                bad.add(canon)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return not bad, tuple(sorted(bad))


class HybridBackend(Backend):
    """SQL pushdown plus certified Polars/Pandas operator execution."""

    runtime_backend_label = "hybrid"

    def __init__(self) -> None:
        self._sql = SqlBackend(operator_backend="auto")
        self._long: Backend | None = None

    def _long_backend(self) -> Backend:
        if self._long is None:
            from .hybrid_long_backend import HybridLongBackend
            self._long = HybridLongBackend()
        return self._long

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        if _supports_polars_long_scan(ctx):
            safe, blocked = _production_long_plan_safe(plan, ctx)
            if safe:
                return self._long_backend().execute(plan, ctx)
            runtime = dict(getattr(ctx, "runtime_stats", None) or {})
            runtime["hybrid_long_skipped_unverified"] = True
            runtime["hybrid_long_blocked_ops"] = list(blocked)
            ctx.runtime_stats = runtime  # type: ignore[attr-defined]
        # SqlBackend lowers only production-safe SQL subtrees in production and
        # delegates residual nodes through BackendRouter, which can select only
        # independently certified Pandas/Polars implementations.
        return self._sql.execute(plan, ctx)
