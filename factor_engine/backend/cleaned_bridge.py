"""FactorEngine ↔ cleaned_operators execution bridge.

The bridge converts engine Series/panels to the selected physical operator
backend and normalises results back to the FactorEngine panel contract.
Production backend selection is evidence constrained and workload-cost aware.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .pandas_compat import pd
from planner.logical_plan import PlanNode
from .context import ExecutionContext
from .panel_native import panel_native_enabled, to_panel
from cache.panel_cache import series_panel_cache_key

_CLEANED_LOADED = False


def ensure_cleaned_loaded() -> None:
    global _CLEANED_LOADED
    if _CLEANED_LOADED:
        return
    from cleaned_operators import load_all
    load_all()
    _CLEANED_LOADED = True


def series_to_panel(s: pd.Series, ctx: ExecutionContext) -> pd.DataFrame:
    if not isinstance(s.index, pd.MultiIndex):
        raise TypeError("cleaned_bridge expects MultiIndex (timestamp, instrument) Series")
    cache = ctx.panel_cache
    cache_key = series_panel_cache_key(s)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
    tcol, icol = ctx.timestamp_col, ctx.instrument_col
    if s.index.names != [tcol, icol]:
        s = s.copy()
        s.index = s.index.set_names([tcol, icol])
    panel = s.unstack(level=icol)
    if not panel.index.is_monotonic_increasing:
        panel = panel.sort_index()
    if cache is not None:
        cache[cache_key] = panel
    return panel


def panel_to_series(
    panel: pd.DataFrame,
    ctx: ExecutionContext,
    *,
    template: "pd.Series | pd.Index | None" = None,
) -> pd.Series:
    stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]
    if template is None:
        return stacked
    if isinstance(template, pd.Index):
        target = template
    else:
        target = template.index
    if len(stacked) == len(target) and stacked.index.equals(target):
        return stacked
    return stacked.reindex(target)


def _resolve_canonical(op: str) -> str:
    from cleaned_operators.registry import OperatorRegistry
    return OperatorRegistry.resolve_canonical_strict(op)


def _call_cleaned_operator(canonical: str, operator: Any, call_args: list[Any], kw: dict[str, Any]) -> Any:
    from backend.parameter_aliases import reject_runtime_parameter_aliases
    reject_runtime_parameter_aliases(canonical, kw)
    # R5 P0-01 (defense-in-depth): the dispatch layer runs the central logical
    # call validator for every cleaned operator, so an operator that overrides
    # ``calculate`` directly (bypassing ``SeriesOperator._prepare_call``) still
    # cannot escape the integer ParamSpec / panel-axis / validate_params gate.
    # SeriesOperator-derived operators re-validate inside ``calculate``; that
    # second pass is pure and cheap.  Polars panels are left untouched by the
    # pandas-axis validator (only ``pd.DataFrame`` frames are checked).
    from cleaned_operators.base import validate_operator_call
    metadata = getattr(operator, "metadata", None)
    if metadata is not None:
        validate_operator_call(operator, tuple(call_args), dict(kw))
    return operator.calculate(*call_args, **kw)


def _operator_backend_preference(ctx: ExecutionContext) -> str:
    perf = getattr(ctx, "perf", None)
    if perf is not None and getattr(perf, "operator_backend", None):
        return str(perf.operator_backend)
    return "auto"


def _data_source_kind(ctx: ExecutionContext) -> str:
    ds = getattr(ctx, "data_source", None)
    seen: set[int] = set()
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        name = type(ds).__name__.lower()
        if "clickhouse" in name:
            return "clickhouse"
        if "duckdb" in name or "dataaccess" in name or "parquet" in name:
            return "duckdb"
        ds = getattr(ds, "inner", None) or getattr(ds, "_inner", None)
    return "memory"


def _estimate_row_count(evaluated: list[Any]) -> int | None:
    """Estimate scalar operator workload after child evaluation.

    A wide panel is costed by cells rather than dates because Pandas↔Polars
    conversion and elementwise/rolling work scale with date×instrument cells.
    """
    from .panel_polars import is_polars_frame

    estimates: list[int] = []
    for val in evaluated:
        if isinstance(val, pd.Series):
            estimates.append(int(len(val)))
        elif isinstance(val, pd.DataFrame):
            estimates.append(int(val.shape[0] * max(1, val.shape[1])))
        elif is_polars_frame(val):
            height = int(getattr(val, "height", 0) or 0)
            width = int(getattr(val, "width", 1) or 1)
            estimates.append(height * max(1, width))
    return max(estimates) if estimates else None


def _record_backend_route(
    ctx: ExecutionContext,
    *,
    canonical: str,
    backend: str,
    row_count_estimate: int | None,
) -> None:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    routes = dict(runtime.get("operator_backend_route_counts") or {})
    key = f"{canonical}:{backend}"
    routes[key] = int(routes.get(key, 0)) + 1
    runtime["operator_backend_route_counts"] = routes
    runtime["last_operator_backend_route"] = {
        "canonical": canonical,
        "backend": backend,
        "row_count_estimate": row_count_estimate,
    }
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


def _resolve_operator(
    canonical: str,
    ctx: ExecutionContext,
    *,
    row_count_estimate: int | None = None,
):
    from backend.backend_router import BackendRouter

    prefer = _operator_backend_preference(ctx)
    run_mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
    fallback = "allow" if getattr(ctx, "production_fallback_policy", "error") == "warn" else "error"
    selection = BackendRouter.select(
        canonical,
        requested_backend=prefer,
        run_mode=run_mode,
        fallback_policy=fallback,
        data_source_kind=_data_source_kind(ctx),
        row_count_estimate=row_count_estimate,
        allow_unverified_backend=run_mode != "production",
    )
    _record_backend_route(
        ctx,
        canonical=selection.canonical,
        backend=selection.backend,
        row_count_estimate=row_count_estimate,
    )
    return selection.operator, selection.backend


def _record_polars_op(ctx: ExecutionContext, op: str) -> None:
    runtime = getattr(ctx, "runtime_stats", None) or {}
    cache_stats = runtime.get("cache")
    if cache_stats is not None and hasattr(cache_stats, "record_polars_op"):
        cache_stats.record_polars_op(op)


def _prepare_call_args(
    evaluated: list[Any],
    ctx: ExecutionContext,
    *,
    backend: str,
    broadcast_scalars: bool = False,
) -> tuple[list[Any], pd.Series | None, pd.DataFrame | None]:
    from .panel_polars import is_polars_frame

    call_args: list[Any] = []
    template: pd.Series | None = None
    template_panel: pd.DataFrame | None = None
    for val in evaluated:
        if isinstance(val, (pd.Series, pd.DataFrame)) or is_polars_frame(val):
            panel = to_panel(val, ctx) if isinstance(val, (pd.Series, pd.DataFrame)) else val
            if is_polars_frame(panel):
                if template_panel is None and isinstance(val, pd.Series):
                    template_panel = val.unstack(level=ctx.instrument_col)
                call_args.append(panel)
            else:
                if template_panel is None:
                    template_panel = panel
                if backend == "polars":
                    from .panel_polars import panel_to_polars
                    call_args.append(panel_to_polars(panel))
                else:
                    call_args.append(panel)
            if template is None and isinstance(val, pd.Series):
                template = val
            elif template is None and _ctx_template(ctx) is not None:
                template = _ctx_template(ctx)
        else:
            if broadcast_scalars and template_panel is not None:
                scalar_panel = pd.DataFrame(
                    val, index=template_panel.index, columns=template_panel.columns
                )
                if backend == "polars":
                    # WS-B #245: keep every panel in a polars call a polars frame
                    # carrying the same ``__fe_time__`` metadata column so the
                    # central validator / PanelIdentity gate sees one consistent
                    # axis (a bare pandas frame would be column-misaligned with
                    # the ``__fe_time__``-carrying polars inputs).
                    from .panel_polars import panel_to_polars
                    call_args.append(panel_to_polars(scalar_panel))
                else:
                    call_args.append(scalar_panel)
            else:
                # Scalar literals used by panel operators (e.g. ts_delay(..., n))
                # remain scalar; cleaned operators validate them directly.
                call_args.append(val)
    return call_args, template, template_panel


def _target_index(template: Any) -> Any:
    """从 Series 或 MultiIndex 模板提取对齐用的 target index（R16）。"""
    if isinstance(template, pd.Index):
        return template
    return template.index


def _ctx_template(ctx: ExecutionContext) -> Any:
    """Phase 5 R16：优先取 axis-only ``template_index``，兼容旧 ``template_series``。"""
    idx = getattr(ctx, "template_index", None)
    if idx is not None:
        return idx
    return getattr(ctx, "template_series", None)


def _normalize_operator_result(
    result: Any,
    *,
    backend: str,
    template: "pd.Series | pd.Index",
    template_panel: pd.DataFrame | None,
    ctx: ExecutionContext,
) -> Any:
    if backend == "polars":
        from .panel_polars import is_polars_frame, polars_to_panel
        if is_polars_frame(result):
            if template_panel is None:
                raise ValueError("polars operator requires a DataFrame template panel")
            result = polars_to_panel(result, template=template_panel)

    from backend.operator_errors import OperatorShapeError
    if isinstance(result, pd.DataFrame):
        if template_panel is None:
            raise OperatorShapeError("DataFrame result requires a panel template")
        if not result.index.equals(template_panel.index):
            raise OperatorShapeError("operator DataFrame index does not match input panel")
        if not result.columns.equals(template_panel.columns):
            raise OperatorShapeError("operator DataFrame columns do not match input panel")
        if panel_native_enabled(ctx):
            return result
        return panel_to_series(result, ctx, template=template)
    target = _target_index(template)
    if isinstance(result, pd.Series):
        if isinstance(result.index, pd.MultiIndex):
            if not result.index.equals(target):
                raise OperatorShapeError("operator result index does not match the input template")
            return result
        if len(result) != len(target):
            raise OperatorShapeError(f"operator result length {len(result)} does not match template {len(target)}")
        if not result.index.equals(target):
            raise OperatorShapeError("operator returned an indexless/misaligned Series")
        return result
    if pd.api.types.is_scalar(result):
        return pd.Series(result, index=target)
    array = np.asarray(result)
    if array.ndim == 1:
        if len(array) != len(template):
            raise OperatorShapeError(f"operator 1D result length {len(array)} does not match template {len(template)}")
        return pd.Series(array, index=template.index)
    if array.ndim == 2:
        if template_panel is None:
            raise OperatorShapeError("operator 2D result requires a panel template")
        if tuple(array.shape) != tuple(template_panel.shape):
            raise OperatorShapeError(f"operator 2D result shape {array.shape} does not match panel {template_panel.shape}")
        frame = pd.DataFrame(array, index=template_panel.index, columns=template_panel.columns)
        if panel_native_enabled(ctx):
            return frame
        return panel_to_series(frame, ctx, template=template)
    raise OperatorShapeError(f"unsupported operator result type/shape: {type(result).__name__} {array.shape}")


def make_cleaned_kernel(eval_fn: Callable[[PlanNode, ExecutionContext], Any], op: str):
    def _kernel(node: PlanNode, ctx: ExecutionContext) -> pd.Series:
        ensure_cleaned_loaded()
        canonical = _resolve_canonical(op)

        # Evaluate children first so routing sees the actual workload size rather
        # than a hard-coded global row estimate.
        evaluated: list[Any] = [eval_fn(child, ctx) for child in node.inputs]
        row_count_estimate = _estimate_row_count(evaluated)
        operator, backend = _resolve_operator(
            canonical,
            ctx,
            row_count_estimate=row_count_estimate,
        )
        if operator is None:
            raise NotImplementedError(f"cleaned operator not implemented: {op!r}")
        if backend == "polars":
            _record_polars_op(ctx, op)

        kw = dict(node.attrs)
        # Plan-inference metadata must never become operator kwargs (R5-06).
        # ``dtype`` is stamped by the analyzer from ``OPERATOR_SIGNATURES`` output
        # typing (e.g. Series[Bool] -> 'bool') and is not a declared parameter of
        # any operator — forwarding it makes an undeclared hidden kwarg that the
        # validator rejects.  ``semantic_attrs`` already carries field metadata
        # separately; ``dtype`` is the one inference key that leaked into attrs.
        kw.pop("dtype", None)
        call_args, template, template_panel = _prepare_call_args(
            evaluated, ctx, backend=backend,
            broadcast_scalars=canonical in {"maximum", "minimum"},
        )
        result = _call_cleaned_operator(canonical, operator, call_args, kw)
        # Keep literal-only arithmetic scalar.  Promoting an intermediate such as
        # ``floor(window / 2) + 1`` to a panel makes it an invalid lag/window
        # argument when it is later consumed by a time-series operator.
        if template is None and template_panel is None and not any(
            isinstance(value, (pd.Series, pd.DataFrame))
            for value in evaluated
        ):
            return result
        if template is None:
            template = _ctx_template(ctx)
        if template is None:
            raise ValueError(f"cleaned op {op!r} requires at least one Series input")
        return _normalize_operator_result(
            result,
            backend=backend,
            template=template,
            template_panel=template_panel,
            ctx=ctx,
        )
    return _kernel


def list_cleaned_ops_for_backend(skip: set[str]) -> list[str]:
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry
    names: set[str] = set()
    for canon in OperatorRegistry.list_canonical():
        if canon not in skip and OperatorRegistry.get(canon) is not None:
            names.add(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias not in skip and OperatorRegistry.get(canon) is not None:
            names.add(alias)
    return sorted(names)


def build_cleaned_dsl_allowlist(skip: set[str] | None = None, *, surface: str = "daily") -> dict[str, Any]:
    ensure_cleaned_loaded()
    from cleaned_operators.operator_surface import is_dsl_name_allowed
    from cleaned_operators.registry import OperatorRegistry
    from api.cleaned_ops import make_cleaned_call_factory

    skip = skip or set()
    out: dict[str, Any] = {}
    for canon in OperatorRegistry.list_canonical():
        if canon in skip or canon in out or OperatorRegistry.get(canon) is None:
            continue
        if is_dsl_name_allowed(canon, canon, surface=surface):
            out[canon] = make_cleaned_call_factory(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias in skip or alias in out or OperatorRegistry.get(canon) is None:
            continue
        if is_dsl_name_allowed(alias, canon, surface=surface):
            out[alias] = make_cleaned_call_factory(canon)
    return out


def build_production_dsl_allowlist(skip: set[str] | None = None) -> dict[str, Any]:
    """Expose every production-admitted factor operator, not only daily core."""
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.registry import OperatorRegistry

    full = build_cleaned_dsl_allowlist(skip, surface="all")
    out: dict[str, Any] = {}
    for name, factory in full.items():
        canon = OperatorRegistry._aliases.get(name, name)
        spec = build_operator_spec(canon)
        if spec is not None and spec.allow_in_production:
            out[name] = factory
    return out


build_cleaned_dsl_extensions = build_cleaned_dsl_allowlist
