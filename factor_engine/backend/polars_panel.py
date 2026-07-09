"""Polars 宽表 panel 辅助：在 PolarsBackend 下尽量延迟 collect。"""

from __future__ import annotations

from typing import Any


def polars_panel_enabled(ctx: Any) -> bool:
    if getattr(ctx, "prefer_polars_panel", False):
        return True
    stats = getattr(ctx, "runtime_stats", None) or {}
    return stats.get("backend") == "polars"


def pandas_panel_to_polars(panel: Any) -> Any:
    import polars as pl

    if isinstance(panel, pl.DataFrame):
        return panel
    if hasattr(panel, "reset_index"):
        return pl.from_pandas(panel.reset_index())
    return pl.from_pandas(panel)


def polars_panel_to_pandas(panel: Any) -> Any:
    import pandas as pd

    if isinstance(panel, pd.DataFrame):
        return panel
    return panel.to_pandas()


def rolling_mean_polars_panel(panel: Any, window: int) -> Any:
    """Polars 宽表 rolling mean（列=标的）。"""
    import polars as pl

    frame = pandas_panel_to_polars(panel)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("expected polars DataFrame")
    time_col = frame.columns[0]
    value_cols = [c for c in frame.columns if c != time_col]
    exprs = [
        pl.col(c).rolling_mean(window_size=int(window), min_samples=1).alias(c)
        for c in value_cols
    ]
    return frame.with_columns(exprs)
