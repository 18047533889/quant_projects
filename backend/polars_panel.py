"""Polars 宽表 panel 辅助：在 PolarsBackend 下尽量延迟 collect。"""

from __future__ import annotations

from typing import Any


def polars_panel_enabled(ctx: Any) -> bool:
    """判断当前上下文是否应使用 Polars 宽表 panel 路径。

    参数:
        ctx: 执行上下文。

    返回:
        ``prefer_polars_panel`` 或 runtime backend 为 polars 时为 ``True``。
    """
    if getattr(ctx, "prefer_polars_panel", False):
        return True
    stats = getattr(ctx, "runtime_stats", None) or {}
    return stats.get("backend") == "polars"


def pandas_panel_to_polars(panel: Any) -> Any:
    """将 pandas 宽表转为 Polars DataFrame。

    参数:
        panel: pandas 或 polars DataFrame。

    返回:
        Polars ``DataFrame``。
    """
    import polars as pl

    if isinstance(panel, pl.DataFrame):
        return panel
    if hasattr(panel, "reset_index"):
        return pl.from_pandas(panel.reset_index())
    return pl.from_pandas(panel)


def polars_panel_to_pandas(panel: Any) -> Any:
    """将 Polars 宽表转为 pandas DataFrame。

    R47 P1-05: This is an explicit API conversion function whose purpose is
    to convert Polars to pandas. The .to_pandas() call here is legitimate
    and expected by design.

    参数:
        panel: polars 或 pandas DataFrame。

    返回:
        pandas ``DataFrame``。
    """
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
