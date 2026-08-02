"""Panel-native execution: keep intermediate results wide and stack once at the root."""
from __future__ import annotations

import os
from typing import Any

from .pandas_compat import pd
from .context import ExecutionContext


def panel_native_enabled(ctx: ExecutionContext) -> bool:
    perf = getattr(ctx, "perf", None)
    if perf is not None and hasattr(perf, "panel_native"):
        return bool(perf.panel_native)
    return os.environ.get("FACTOR_ENGINE_DISABLE_PANEL_NATIVE", "").lower() not in (
        "1", "true", "yes",
    )


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
    """Load a column as a wide panel while preserving the source's original axis template.

    Logical SourceRef wrappers expose ``load_column_panel`` for specialized
    source-backed values. They also set ``prefer_series_panel_loading`` so that
    panel-native execution first observes the MultiIndex Series and can preserve
    its exact ordering before widening. This keeps native/reference semantics
    unchanged and avoids an instrument-major -> time-major reorder.
    """
    prefer_series = bool(getattr(data_source, "prefer_series_panel_loading", False))
    if prefer_series:
        series = data_source.load_column(name)
        ensure_template_from_series(ctx, series)
        from .cleaned_bridge import series_to_panel
        panel = series_to_panel(series, ctx)
    else:
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
