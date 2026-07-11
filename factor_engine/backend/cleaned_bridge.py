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
from cache.panel_cache import series_panel_cache_key

_CLEANED_LOADED = False


def ensure_cleaned_loaded() -> None:
    """惰性 import 全部 ``cleaned_operators`` 子模块并注册别名（进程内只执行一次）。"""
    global _CLEANED_LOADED
    if _CLEANED_LOADED:
        return
    from cleaned_operators import load_all

    load_all()
    _CLEANED_LOADED = True



def series_to_panel(s: pd.Series, ctx: ExecutionContext) -> pd.DataFrame:
    """将 ``(timestamp, instrument)`` MultiIndex Series 转为宽表 panel。

    参数
    ----
    s : pd.Series
        输入 Series，index 必须为两级 MultiIndex。
    ctx : ExecutionContext
        执行上下文，提供 ``panel_cache`` 与列名配置。

    返回
    ----
    pd.DataFrame
        宽表 panel，index 为时间，columns 为标的。

    异常
    ----
    TypeError
        输入 Series 的 index 不是 MultiIndex 时抛出。
    """
    if not isinstance(s.index, pd.MultiIndex):
        raise TypeError("cleaned_bridge expects MultiIndex (timestamp, instrument) Series")
    cache = ctx.panel_cache
    cache_key = series_panel_cache_key(s)
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
    """将宽表 panel 转为与 ``template`` index 对齐的 MultiIndex Series。

    参数
    ----
    panel : pd.DataFrame
        宽表 panel，index 为时间，columns 为标的。
    ctx : ExecutionContext
        执行上下文，提供 ``timestamp_col`` / ``instrument_col``。
    template : pd.Series
        对齐模板，决定输出 Series 的 index 形态。

    返回
    ----
    pd.Series
        stack 后与 ``template.index`` 对齐的 MultiIndex Series。
    """
    stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]
    if len(stacked) == len(template.index) and stacked.index.equals(template.index):
        return stacked
    return stacked.reindex(template.index)


def _resolve_canonical(op: str) -> str:
    """将算子名（含别名）解析为 canonical 名称。

    参数
    ----
    op : str
        逻辑计划或 DSL 中的算子名。

    返回
    ----
    str
        ``OperatorRegistry`` 中的 canonical 算子名。
    """
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def _remap_d_to_window(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """将 planner 常用的 ``d`` 参数名重映射为算子期望的 ``window``。

    参数
    ----
    kwargs : dict[str, Any]
        算子调用关键字参数。

    返回
    ----
    dict[str, Any] | None
        重映射后的参数字典；无需重映射时返回 ``None``。
    """
    if "d" not in kwargs or "window" in kwargs:
        return None
    remapped = {k: v for k, v in kwargs.items() if k != "d"}
    remapped["window"] = kwargs["d"]
    return remapped


def _call_cleaned_operator(operator, call_args: list[Any], kw: dict[str, Any]) -> Any:
    """调用算子 ``calculate``；关键字不匹配时自动尝试 ``d`` → ``window`` 重映射。

    参数
    ----
    operator
        cleaned 算子实例，需提供 ``calculate`` 方法。
    call_args : list[Any]
        位置参数列表（panel 或标量）。
    kw : dict[str, Any]
        关键字参数（来自 ``PlanNode.attrs``）。

    返回
    ----
    Any
        算子 ``calculate`` 的返回值。

    异常
    ----
    TypeError
        原始调用与重映射后均失败时抛出。
    """
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
    """从执行上下文读取算子后端偏好。

    参数
    ----
    ctx : ExecutionContext
        执行期上下文。

    返回
    ----
    str
        ``perf.operator_backend`` 值，未配置时返回 ``"auto"``。
    """
    perf = getattr(ctx, "perf", None)
    if perf is not None and getattr(perf, "operator_backend", None):
        return str(perf.operator_backend)
    return "auto"


def _record_polars_op(ctx: ExecutionContext, op: str) -> None:
    """记录一次 Polars 算子热路径调用（写入缓存会话统计）。

    参数
    ----
    ctx : ExecutionContext
        执行期上下文，``runtime_stats.cache`` 可能持有统计会话。
    op : str
        被调用的算子名。
    """
    runtime = getattr(ctx, "runtime_stats", None) or {}
    cache_stats = runtime.get("cache")
    if cache_stats is not None and hasattr(cache_stats, "record_polars_op"):
        cache_stats.record_polars_op(op)


def _resolve_operator(canonical: str, ctx: ExecutionContext):
    """按后端偏好从 ``OperatorRegistry`` 解析算子实例与后端标签。

    参数
    ----
    canonical : str
        canonical 算子名。
    ctx : ExecutionContext
        执行期上下文，决定 ``get_preferred`` 的偏好参数。

    返回
    ----
    tuple
        ``(operator, backend)`` 二元组，``backend`` 为 ``"polars"`` 或 ``"pandas_numpy"``。
    """
    from cleaned_operators.registry import OperatorRegistry

    prefer = _operator_backend_preference(ctx)
    runtime = getattr(ctx, "runtime_stats", None) or {}
    if runtime.get("backend") == "polars" and prefer == "auto":
        # PolarsBackend：与 registry.get_preferred("auto") 一致，尊重 POLARS_PRODUCTION_SAFE
        pass
    operator, backend = OperatorRegistry.get_preferred(canonical, prefer=prefer)
    return operator, backend


def _prepare_call_args(
    evaluated: list[Any],
    ctx: ExecutionContext,
    *,
    backend: str,
) -> tuple[list[Any], pd.Series | None, pd.DataFrame | None]:
    """将子节点求值结果转换为算子 ``calculate`` 所需的位置参数。

    参数
    ----
    evaluated : list[Any]
        子节点递归求值结果列表。
    ctx : ExecutionContext
        执行期上下文。
    backend : str
        目标算子后端（``"polars"`` 或 ``"pandas_numpy"``）。

    返回
    ----
    tuple[list[Any], pd.Series | None, pd.DataFrame | None]
        ``(call_args, template_series, template_panel)`` 三元组。
    """
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
    """将算子返回值规范化为引擎统一的输出形态。

    参数
    ----
    result : Any
        算子 ``calculate`` 的原始返回值。
    backend : str
        实际使用的算子后端。
    template : pd.Series
        对齐模板 Series。
    template_panel : pd.DataFrame | None
        宽表对齐模板（Polars 路径必需）。
    ctx : ExecutionContext
        执行期上下文。

    返回
    ----
    Any
        MultiIndex Series 或 panel-native 模式下的 DataFrame。

    异常
    ----
    ValueError
        Polars 后端返回 DataFrame 但缺少 ``template_panel`` 时抛出。
    """
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
    """为逻辑计划算子名 ``op`` 生成 PandasBackend 用的 kernel 闭包。

    参数
    ----
    eval_fn : Callable
        子节点递归求值函数，签名为 ``(node, ctx) -> Any``。
    op : str
        算子名，用于查找 ``OperatorRegistry`` 中的实现。

    返回
    ----
    Callable
        kernel 闭包，签名为 ``(node: PlanNode, ctx: ExecutionContext) -> pd.Series``。
    """

    def _kernel(node: PlanNode, ctx: ExecutionContext) -> pd.Series:
        """cleaned 算子 kernel：求值子节点、调用算子并规范化输出。

        参数
        ----
        node : PlanNode
            当前逻辑计划节点，``inputs`` 为子节点，``attrs`` 为算子参数。
        ctx : ExecutionContext
            执行期上下文。

        返回
        ----
        pd.Series
            与输入模板对齐的 MultiIndex Series（panel-native 模式下可能为 DataFrame）。

        异常
        ----
        NotImplementedError
            算子在 ``OperatorRegistry`` 中无实现时抛出。
        ValueError
            算子无 Series 输入且 ``template_series`` 未配置时抛出。
        """
        ensure_cleaned_loaded()

        canonical = _resolve_canonical(op)
        operator, backend = _resolve_operator(canonical, ctx)
        if operator is None:
            raise NotImplementedError(f"cleaned operator not implemented: {op!r}")

        from backend.polars_hot_ops import is_polars_backend_ctx
        from runtime.production_policy import record_production_pandas_fallback

        if is_polars_backend_ctx(ctx) and backend == "pandas_numpy":
            record_production_pandas_fallback(
                ctx,
                op=canonical,
                requested_backend="polars",
                actual_backend=backend,
            )

        if backend == "polars":
            _record_polars_op(ctx, op)

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
    """列出可挂到 ``KernelRegistry`` 的全部已实现算子名（canonical + 别名）。

    参数
    ----
    skip : set[str]
        需要排除的算子名集合。

    返回
    ----
    list[str]
        已排序的可用算子名列表。
    """
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
    """构建 DSL 白名单：``{DSL函数名: make_cleaned_call_factory(...)}``。

    仅包含 ``OperatorRegistry`` 中已有 runtime 实现的条目。

    参数
    ----
    skip : set[str] | None
        需要排除的算子名集合，默认空集。

    返回
    ----
    dict[str, Any]
        DSL 函数名到调用工厂的映射。
    """
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


def build_production_dsl_allowlist(skip: set[str] | None = None) -> dict[str, Any]:
    """构建 production 投递白名单：仅 ``PRODUCTION_CORE`` 内且 ``allow_in_production`` 为真的算子。

    composite 虽可在 execution gate 放行，但不得进入 DSL 投递白名单（fail-closed）。
    """
    from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS, build_operator_spec

    full = build_cleaned_dsl_allowlist(skip)
    from cleaned_operators.registry import OperatorRegistry

    out: dict[str, Any] = {}
    for name, factory in full.items():
        canon = OperatorRegistry._aliases.get(name, name)
        if canon not in PRODUCTION_CORE_CANONICALS:
            continue
        spec = build_operator_spec(canon)
        if spec is not None and spec.allow_in_production:
            out[name] = factory
    return out


# 兼容旧 import 名
build_cleaned_dsl_extensions = build_cleaned_dsl_allowlist
