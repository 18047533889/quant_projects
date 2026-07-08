"""factor_engine × cleaned_operators 执行桥。

职责
----
把引擎内部的 **MultiIndex Series** ``(timestamp, instrument)`` 与 cleaned 算子期望的
**宽表 panel** ``DataFrame(index=time, columns=instrument)`` 互转，并调用
``OperatorRegistry.get(op).calculate(...)``。

数据流::

    PandasBackend._eval(PlanNode)
        → make_cleaned_kernel(op)(node, ctx)
        → series_to_panel(子节点 Series…)
        → operator.calculate(*panels, **attrs)
        → panel_to_series(result, template=…)

``build_cleaned_dsl_allowlist`` 与 ``api.operator_registry.build_dsl_allowlist`` 共用本模块，
确保 **白名单名 = 有 runtime 的算子**。
"""

from __future__ import annotations

from typing import Any, Callable

from .pandas_compat import pd

from planner.logical_plan import PlanNode

from .context import ExecutionContext
from .panel_native import panel_native_enabled, to_panel

_CLEANED_LOADED = False


def ensure_cleaned_loaded() -> None:
    """惰性 import 全部 ``cleaned_operators`` 子模块并注册别名（进程内只执行一次）。"""
    global _CLEANED_LOADED
    if _CLEANED_LOADED:
        return
    from cleaned_operators import load_all

    load_all()
    _CLEANED_LOADED = True


def _series_panel_cache_key(s: pd.Series) -> tuple[Any, ...]:
    """稳定 cache key：Series 身份 + index 身份 + 列名 + 长度。"""
    name = getattr(s, "name", None)
    return (id(s), id(s.index), str(name) if name is not None else "", len(s))


def series_to_panel(s: pd.Series, ctx: ExecutionContext) -> pd.DataFrame:
    """``(timestamp, instrument)`` MultiIndex Series → 宽表（列=标的）。"""
    if not isinstance(s.index, pd.MultiIndex):
        raise TypeError("cleaned_bridge expects MultiIndex (timestamp, instrument) Series")
    cache = ctx.panel_cache
    cache_key = _series_panel_cache_key(s)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
        hit = cache.get(id(s))
        if hit is not None:
            return hit
    tcol = ctx.timestamp_col
    icol = ctx.instrument_col
    if s.index.names != [tcol, icol]:
        s = s.copy()
        s.index = s.index.set_names([tcol, icol])
    panel = s.unstack(level=icol).sort_index()
    if cache is not None:
        cache[cache_key] = panel
        cache[id(s)] = panel
    return panel


def panel_to_series(
    panel: pd.DataFrame,
    ctx: ExecutionContext,
    *,
    template: pd.Series,
) -> pd.Series:
    """宽表 panel → 与 ``template`` index 对齐的 MultiIndex Series。"""
    stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]
    if len(stacked) == len(template.index) and stacked.index.equals(template.index):
        return stacked
    return stacked.reindex(template.index)


def _resolve_canonical(op: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def _remap_d_to_window(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """部分算子参数名是 ``window``，而 planner 节点 attrs 常用 ``d``；在此统一重试。"""
    if "d" not in kwargs or "window" in kwargs:
        return None
    remapped = {k: v for k, v in kwargs.items() if k != "d"}
    remapped["window"] = kwargs["d"]
    return remapped


def _call_cleaned_operator(operator, call_args: list[Any], kw: dict[str, Any]) -> Any:
    """调用 ``calculate``；若因 ``d``/``window`` 关键字不匹配失败则自动重映射一次。"""
    try:
        return operator.calculate(*call_args, **kw)
    except TypeError as exc:
        remapped = _remap_d_to_window(kw)
        if remapped is None:
            raise
        msg = str(exc).lower()
        if "unexpected keyword" not in msg and "got an unexpected" not in msg:
            raise
        try:
            return operator.calculate(*call_args, **remapped)
        except TypeError:
            raise exc from None


def _operator_backend_preference(ctx: ExecutionContext) -> str:
    perf = getattr(ctx, "perf", None)
    if perf is not None and getattr(perf, "operator_backend", None):
        return str(perf.operator_backend)
    return "auto"


def _resolve_operator(canonical: str, ctx: ExecutionContext):
    from cleaned_operators.registry import OperatorRegistry

    prefer = _operator_backend_preference(ctx)
    operator, backend = OperatorRegistry.get_preferred(canonical, prefer=prefer)
    return operator, backend


def _prepare_call_args(
    evaluated: list[Any],
    ctx: ExecutionContext,
    *,
    backend: str,
) -> tuple[list[Any], pd.Series | None, pd.DataFrame | None]:
    call_args: list[Any] = []
    template: pd.Series | None = None
    template_panel: pd.DataFrame | None = None

    for val in evaluated:
        if isinstance(val, (pd.Series, pd.DataFrame)):
            panel = to_panel(val, ctx)
            if template_panel is None:
                template_panel = panel
            if backend == "polars":
                from .panel_polars import panel_to_polars

                call_args.append(panel_to_polars(panel))
            else:
                call_args.append(panel)
            if template is None and isinstance(val, pd.Series):
                template = val
            elif template is None and getattr(ctx, "template_series", None) is not None:
                template = ctx.template_series
        else:
            call_args.append(val)

    return call_args, template, template_panel


def _normalize_operator_result(
    result: Any,
    *,
    backend: str,
    template: pd.Series,
    template_panel: pd.DataFrame | None,
    ctx: ExecutionContext,
) -> Any:
    if backend == "polars":
        from .panel_polars import is_polars_frame, polars_to_panel

        if is_polars_frame(result):
            if template_panel is None:
                raise ValueError("polars operator requires a DataFrame template panel")
            result = polars_to_panel(result, template=template_panel)

    if isinstance(result, pd.DataFrame):
        if panel_native_enabled(ctx):
            return result
        return panel_to_series(result, ctx, template=template)
    if isinstance(result, pd.Series):
        if isinstance(result.index, pd.MultiIndex):
            return result.reindex(template.index)
        return pd.Series(result.values, index=template.index)
    return pd.Series(result, index=template.index)


def make_cleaned_kernel(eval_fn: Callable[[PlanNode, ExecutionContext], Any], op: str):
    """为逻辑计划算子名 ``op`` 生成 PandasBackend 用的 kernel 闭包。"""

    def _kernel(node: PlanNode, ctx: ExecutionContext) -> pd.Series:
        ensure_cleaned_loaded()

        canonical = _resolve_canonical(op)
        operator, backend = _resolve_operator(canonical, ctx)
        if operator is None:
            raise NotImplementedError(f"cleaned operator not implemented: {op!r}")

        evaluated: list[Any] = [eval_fn(child, ctx) for child in node.inputs]
        kw = dict(node.attrs)

        call_args, template, template_panel = _prepare_call_args(
            evaluated, ctx, backend=backend
        )

        result = _call_cleaned_operator(operator, call_args, kw)

        if template is None:
            template = getattr(ctx, "template_series", None)
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
    """列出可挂到 ``KernelRegistry`` 的全部已实现算子名（canonical + 别名）。"""
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    names: set[str] = set()
    for canon in OperatorRegistry.list_canonical():
        if canon in skip:
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        names.add(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias in skip:
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        names.add(alias)
    return sorted(names)


def build_cleaned_dsl_allowlist(skip: set[str] | None = None) -> dict[str, Any]:
    """``{DSL函数名: make_cleaned_call_factory(...)}``，仅含已有 runtime 的条目。"""
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    from api.cleaned_ops import make_cleaned_call_factory

    skip = skip or set()
    out: dict[str, Any] = {}
    for canon in OperatorRegistry.list_canonical():
        if canon in skip or canon in out:
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        out[canon] = make_cleaned_call_factory(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias in skip or alias in out:
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        out[alias] = make_cleaned_call_factory(canon)
    return out


# 兼容旧 import 名
build_cleaned_dsl_extensions = build_cleaned_dsl_allowlist
