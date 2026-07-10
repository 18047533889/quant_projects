# -*- coding: utf-8 -*-
"""宽表 panel 与 Polars DataFrame 互转（列=标的，行=时间顺序）。"""
from __future__ import annotations

from typing import Any

from .pandas_compat import pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_SKIP_COLS = frozenset({"date", "stock_code"})


def panel_to_polars(panel: pd.DataFrame) -> Any:
    """pandas 宽表 → polars（保留列名，不依赖 date/stock_code 列）。"""
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    if is_polars_frame(panel):
        return panel
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"expected DataFrame panel, got {type(panel)!r}")
    return pl.DataFrame({str(c): panel[c].to_numpy() for c in panel.columns})


def polars_to_panel(result: Any, *, template: pd.DataFrame) -> pd.DataFrame:
    """polars 结果 → 与 template 对齐的 pandas 宽表。"""
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"expected polars DataFrame, got {type(result)!r}")

    out_cols = [c for c in template.columns if c in result.columns]
    if not out_cols:
        out_cols = [c for c in result.columns if c not in _SKIP_COLS]
    if not out_cols:
        return pd.DataFrame(index=template.index, columns=template.columns, dtype=float)

    data = result.select([str(c) for c in out_cols]).to_numpy()
    return pd.DataFrame(data, index=template.index, columns=out_cols)


def is_polars_frame(obj: Any) -> bool:
    """判断对象是否为 Polars DataFrame。

    参数:
        obj: 待检测对象。

    返回:
        polars 已安装且对象为 ``pl.DataFrame`` 时为 ``True``。
    """
    return pl is not None and isinstance(obj, pl.DataFrame)
