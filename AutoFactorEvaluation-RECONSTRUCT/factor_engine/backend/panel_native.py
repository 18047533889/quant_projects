"""Panel-native 执行：中间结果保持宽表，仅在根节点 stack 一次。"""

from __future__ import annotations

import os
from typing import Any

from .pandas_compat import pd

from .context import ExecutionContext


def panel_native_enabled(ctx: ExecutionContext) -> bool:
    """判断是否启用 panel-native 执行路径（中间结果保持宽表）。

    参数:
        ctx: 执行上下文。

    返回:
        未禁用且 perf 配置允许时为 ``True``。
    """
    # long_table 仅约束数据源形态；算子链仍用 panel-native 避免逐节点 stack/unstack。
    if os.environ.get("FACTOR_ENGINE_DISABLE_PANEL_NATIVE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    perf = getattr(ctx, "perf", None)
    if perf is not None and hasattr(perf, "panel_native"):
        return bool(perf.panel_native)
    return True


def ensure_template_from_series(ctx: ExecutionContext, series: pd.Series) -> None:
    """从 Series 设置输出对齐用的 template_series。

    参数:
        ctx: 执行上下文。
        series: MultiIndex Series 模板。
    """
    if getattr(ctx, "template_series", None) is None:
        ctx.template_series = series


def ensure_template_from_panel(ctx: ExecutionContext, panel: pd.DataFrame) -> None:
    """从宽表 panel 推导并缓存 template_series。

    参数:
        ctx: 执行上下文。
        panel: 宽表 DataFrame（行=时间，列=标的）。
    """
    if getattr(ctx, "template_series", None) is not None:
        return
    stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]
    ctx.template_series = stacked


def load_column_as_panel(data_source: Any, name: str, ctx: ExecutionContext) -> pd.DataFrame:
    """从数据源加载列并转为宽表 panel。

    参数:
        data_source: 数据源实例。
        name: 列名。
        ctx: 执行上下文，用于缓存 template。

    返回:
        宽表 ``DataFrame``。
    """
    loader = getattr(data_source, "load_column_panel", None)
    if callable(loader):
        panel = loader(name)
    else:
        series = data_source.load_column(name)
        ensure_template_from_series(ctx, series)
        from .cleaned_bridge import series_to_panel

        panel = series_to_panel(series, ctx)
    ensure_template_from_panel(ctx, panel)
    return panel


def to_panel(val: Any, ctx: ExecutionContext) -> Any:
    """将 Series 或 DataFrame 规范为宽表 panel。

    参数:
        val: 中间结果（Series 或 DataFrame）。
        ctx: 执行上下文。

    返回:
        宽表 panel 或原值（无法转换时）。
    """
    if isinstance(val, pd.DataFrame):
        ensure_template_from_panel(ctx, val)
        return val
    if isinstance(val, pd.Series):
        ensure_template_from_series(ctx, val)
        from .cleaned_bridge import series_to_panel

        return series_to_panel(val, ctx)
    return val


def finalize_panel_result(val: Any, ctx: ExecutionContext) -> Any:
    """在根节点将 panel 结果 stack 为 MultiIndex Series（仅一次）。

    参数:
        val: 算子链最终结果。
        ctx: 执行上下文。

    返回:
        panel-native 禁用时返回原值，否则将 DataFrame stack 为 Series。
    """
    if not panel_native_enabled(ctx):
        return val
    if isinstance(val, pd.DataFrame):
        from .cleaned_bridge import panel_to_series

        template = getattr(ctx, "template_series", None)
        if template is None:
            ensure_template_from_panel(ctx, val)
            template = ctx.template_series
        return panel_to_series(val, ctx, template=template)
    return val
