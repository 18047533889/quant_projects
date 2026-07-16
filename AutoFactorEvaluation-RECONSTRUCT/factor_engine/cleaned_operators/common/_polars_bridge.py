# -*- coding: utf-8 -*-
"""Polars panel ↔ pandas 桥接工具（复杂算子复用 pandas_numpy 实现）。"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

import pandas as pd

SKIP = frozenset({"date", "stock_code"})


def numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SKIP]


def align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


def to_pandas_panel(df: pl.DataFrame) -> pd.DataFrame:
    return df.select(numeric_cols(df)).to_pandas()


def from_pandas_panel(base: pl.DataFrame, out: pd.DataFrame) -> pl.DataFrame:
    cols = [c for c in out.columns if c in base.columns]
    return base.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


def bridge_pandas(
    x: pl.DataFrame,
    compute: Callable[..., pd.DataFrame],
    *args: Any,
    **kwargs: Any,
) -> pl.DataFrame:
    """宽表 Polars → pandas 计算 → 写回 Polars。"""
    pdf = to_pandas_panel(x)
    pargs = []
    for arg in args:
        if isinstance(arg, pl.DataFrame):
            pargs.append(to_pandas_panel(arg))
        else:
            pargs.append(arg)
    out = compute(pdf, *pargs, **kwargs)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"bridge_pandas expects DataFrame result, got {type(out)}")
    return from_pandas_panel(x, out)


def bridge_registry(canonical: str, x: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(canonical, backend="pandas_numpy")
    pargs = []
    for arg in args:
        pargs.append(to_pandas_panel(arg) if isinstance(arg, pl.DataFrame) else arg)
    return bridge_pandas(x, op.calculate, *pargs, **kwargs)


def colwise_numpy_kernel(
    x: pl.DataFrame,
    kernel: Callable[..., np.ndarray],
    *args: Any,
    **kwargs: Any,
) -> pl.DataFrame:
    """逐列应用 numpy 向量核（与 time_series._apply_colwise_kernel 对齐）。"""
    cols = numeric_cols(x)
    pdf = to_pandas_panel(x)
    out = pd.DataFrame(index=pdf.index, columns=cols, dtype=float)
    for col in cols:
        out[col] = kernel(pdf[col].to_numpy(), *args, **kwargs)
    return from_pandas_panel(x, out)
