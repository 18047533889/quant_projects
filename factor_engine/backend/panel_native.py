"""Panel-native 执行：中间结果保持宽表，仅在根节点 stack 一次。"""

from __future__ import annotations

import os
from typing import Any

from .pandas_compat import pd

from .context import ExecutionContext


def panel_native_enabled(ctx: ExecutionContext) -> bool:
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
    if getattr(ctx, "template_series", None) is None:
        ctx.template_series = series


def ensure_template_from_panel(ctx: ExecutionContext, panel: pd.DataFrame) -> None:
    if getattr(ctx, "template_series", None) is not None:
        return
    stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]
    ctx.template_series = stacked


def load_column_as_panel(data_source: Any, name: str, ctx: ExecutionContext) -> pd.DataFrame:
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
    if isinstance(val, pd.DataFrame):
        ensure_template_from_panel(ctx, val)
        return val
    if isinstance(val, pd.Series):
        ensure_template_from_series(ctx, val)
        from .cleaned_bridge import series_to_panel

        return series_to_panel(val, ctx)
    return val


def finalize_panel_result(val: Any, ctx: ExecutionContext) -> Any:
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
