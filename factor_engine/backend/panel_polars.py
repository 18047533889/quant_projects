# -*- coding: utf-8 -*-
"""宽表 panel 与 Polars DataFrame 互转（列=标的，行=时间顺序）。

WS-B #245: ``panel_to_polars`` 保留 pandas ``DatetimeIndex`` —— 转换为显式
``__fe_time__`` 列（Option A），并由 ``PanelIdentity`` 记录时间轴/标的轴；
``strip_panel_metadata`` 在内核调用前剥离该元数据列，避免数值列提取把时间轴
当作特征。``polars_to_panel`` 始终用 template 的时间轴重建结果，不丢失时间轴。
"""
from __future__ import annotations

from typing import Any

from .pandas_compat import pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_SKIP_COLS = frozenset({"date", "stock_code"})

# Re-export the WS-B PanelSchema/PanelIdentity machinery (single source of truth
# lives in cleaned_operators.common._polars_bridge).
from cleaned_operators.common._polars_bridge import (  # noqa: E402
    FE_TIME_COL,
    PanelIdentity,
    PanelSchema,
    panel_schema,
    strip_panel_metadata,
    verify_frames_share_identity,
)


def panel_to_polars(panel: pd.DataFrame) -> Any:
    """pandas 宽表 → polars（保留列名，不依赖 date/stock_code 列）。

    WS-B #245: 当宽表携带 ``DatetimeIndex`` 时，将其转换为显式 ``__fe_time__``
    列，使时间轴在转换后存活。``__fe_time__`` 是保留元数据列名，由
    ``strip_panel_metadata`` 在内核调用前剥离。
    """
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    if is_polars_frame(panel):
        return panel
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"expected DataFrame panel, got {type(panel)!r}")
    data: dict[str, Any] = {str(c): panel[c].to_numpy() for c in panel.columns}
    if isinstance(panel.index, pd.DatetimeIndex) and not is_polars_frame(panel):
        data[FE_TIME_COL] = panel.index.to_numpy()
    return pl.DataFrame(data)


def polars_to_panel(result: Any, *, template: pd.DataFrame) -> pd.DataFrame:
    """polars 结果 → 与 template 对齐的 pandas 宽表。"""
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"expected polars DataFrame, got {type(result)!r}")

    out_cols = [c for c in template.columns if c in result.columns]
    if not out_cols:
        out_cols = [c for c in result.columns if c not in _SKIP_COLS]
    # The reserved __fe_time__ metadata column is never an output column.
    out_cols = [c for c in out_cols if c != FE_TIME_COL]
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
