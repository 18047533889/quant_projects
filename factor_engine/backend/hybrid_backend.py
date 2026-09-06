# -*- coding: utf-8 -*-
"""Cost-routed production backend over Pandas, Polars and SQL plans."""
from __future__ import annotations

from typing import Any

from factor_engine.planner.logical_plan import PlanNode
from .base import Backend
from .context import ExecutionContext
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend
from .sql_backend import SqlBackend


def _inner_data_source(ctx: ExecutionContext) -> Any:
    """Return the real underlying data source, unwrapping the LQTP logical layer."""
    ds = getattr(ctx, "data_source", None)
    return getattr(ds, "inner", ds)


def _supports_polars_long_scan(ctx: ExecutionContext) -> bool:
    return callable(getattr(_inner_data_source(ctx), "scan_polars_long", None))


def _supports_load_column(ctx: ExecutionContext) -> bool:
    return callable(getattr(_inner_data_source(ctx), "load_column", None))


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
        from factor_engine.backend.plan_cost_router import choose_plan_route, record_plan_route, plan_requires_data_source

        route = choose_plan_route(plan, ctx)
        # Data-plane eligibility is part of selection, before certificates are
        # built. Never relabel an already certified route in the executor.
        if (plan_requires_data_source(plan) and route.backend in {"pandas_numpy", "polars_panel"} and not _supports_load_column(ctx)) or (
            route.backend == "polars_long" and not _supports_polars_long_scan(ctx)
        ):
            raise RuntimeError("selected route no longer matches source capabilities; replan required")
        certificate = route.certificate
        if certificate is not None and certificate.requested_backend != route.backend:
            raise RuntimeError("selected route differs from execution certificate")
        if certificate is not None:
            from factor_engine.planner.plan_hash import structural_key

            if certificate.certificate_hash != certificate.recompute_hash() or certificate.structural_hash != structural_key(plan):
                raise RuntimeError("execution certificate integrity or plan binding mismatch")
        if str(getattr(ctx, "run_mode", "research")) == "production" and certificate is None:
            raise RuntimeError("production route requires an execution certificate")

        record_plan_route(ctx, route)
        # R21-P022: stamp selected_backend on the context so per-operator
        # reroute in cleaned_bridge is suppressed for this execution.
        from dataclasses import replace as _ctx_replace
        ctx = _ctx_replace(ctx, selected_backend=route.backend)
        if route.backend == "pandas_numpy":
            return self._pandas.execute(plan, ctx)
        if route.backend == "polars_panel":
            return self._polars.execute(plan, ctx)
        if route.backend == "polars_long":
            return self._long_backend().execute(plan, ctx)
        # Full DuckDB and mixed plans share SqlBackend. The physical lowerer
        # distinguishes a fully-pushed root from certified partial subtrees.
        return self._sql.execute(plan, ctx)
