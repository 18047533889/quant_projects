# -*- coding: utf-8 -*-
"""Cost-routed production backend over Pandas, Polars and SQL plans."""
from __future__ import annotations

from typing import Any

from planner.logical_plan import PlanNode
from .base import Backend
from .context import ExecutionContext
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .sql_backend import SqlBackend


def _supports_polars_long_scan(ctx: ExecutionContext) -> bool:
    ds = getattr(ctx, "data_source", None)
    return callable(getattr(ds, "scan_polars_long", None))


class HybridBackend(Backend):
    """Choose the cheapest certified physical plan for ``backend.type=auto``."""

    runtime_backend_label = "hybrid"

    def __init__(self) -> None:
        self._sql = SqlBackend(operator_backend="auto")
        self._pandas = PandasBackend()
        self._polars = PolarsBackend()
        self._long: Backend | None = None

    def _long_backend(self) -> Backend:
        if self._long is None:
            from .hybrid_long_backend import HybridLongBackend
            self._long = HybridLongBackend()
        return self._long

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        from backend.plan_cost_router import choose_plan_route, record_plan_route

        route = choose_plan_route(plan, ctx)
        if route.backend == "polars_long" and not _supports_polars_long_scan(ctx):
            from dataclasses import replace

            remaining = tuple((k, v) for k, v in route.candidate_costs if k != "polars_long")
            if remaining:
                chosen = min(remaining, key=lambda item: (item[1], item[0]))
                route = replace(
                    route,
                    backend=chosen[0],
                    estimated_cost=chosen[1],
                    candidate_costs=remaining,
                    routing_basis="estimated",
                    reason="polars_long candidate removed: data source lacks scan_polars_long",
                )
            else:
                route = replace(
                    route,
                    backend="hybrid",
                    routing_basis="estimated",
                    reason="polars_long unavailable on data plane; using certified hybrid",
                )

        record_plan_route(ctx, route)
        if route.backend == "pandas_numpy":
            return self._pandas.execute(plan, ctx)
        if route.backend == "polars_panel":
            return self._polars.execute(plan, ctx)
        if route.backend == "polars_long":
            return self._long_backend().execute(plan, ctx)
        # Full DuckDB and mixed plans share SqlBackend. The physical lowerer
        # distinguishes a fully-pushed root from certified partial subtrees.
        return self._sql.execute(plan, ctx)
