# -*- coding: utf-8
"""Long-table 中间结果：Polars / SQL 统一的 ts / inst / value 语义。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pandas_compat import pd


@dataclass(frozen=True)
class LongFrameResult:
    """PlanNode 在 long-table Polars LazyFrame 中的结果列。"""

    frame: Any  # pl.LazyFrame | pl.DataFrame
    value_col: str
    ts_col: str = "ts"
    inst_col: str = "inst"
    is_lazy: bool = True


def polars_long_to_multiindex_series(
    frame: Any,
    *,
    timestamp_col: str,
    instrument_col: str,
    value_col: str,
    template_index: pd.Index | None = None,
) -> pd.Series:
    """Long table → MultiIndex Series（仅在最终输出调用一次）。"""
    from storage.factor_format import long_table_to_series

    pdf = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    rename = {}
    if timestamp_col not in pdf.columns and "ts" in pdf.columns:
        rename["ts"] = timestamp_col
    if instrument_col not in pdf.columns and "inst" in pdf.columns:
        rename["inst"] = instrument_col
    if rename:
        pdf = pdf.rename(columns=rename)
    out = long_table_to_series(
        pdf,
        timestamp_col=timestamp_col,
        asset_col=instrument_col,
        value_col=value_col,
    )
    if template_index is not None:
        return out.reindex(template_index)
    return out
