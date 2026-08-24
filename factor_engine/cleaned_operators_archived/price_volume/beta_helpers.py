# -*- coding: utf-8
"""Beta 类算子：双输入 panel 对齐 + 滚动 beta。"""

from __future__ import annotations

import pandas as pd

from cleaned_operators._rolling_fast import rolling_beta


def align_benchmark_to_ret(ret: pd.DataFrame, benchmark_ret: pd.DataFrame) -> pd.DataFrame:
    """将单列 benchmark 广播到 ret 各列；多列则按列名对齐。"""
    bench = benchmark_ret
    if bench.shape[1] == 1:
        series = bench.iloc[:, 0]
        return pd.concat([series] * ret.shape[1], axis=1).set_axis(ret.columns, axis=1)
    if list(bench.columns) != list(ret.columns):
        bench = bench.reindex(columns=ret.columns)
    return bench


def compute_rolling_beta(
    ret: pd.DataFrame,
    benchmark_ret: pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Cov(ret, benchmark) / Var(benchmark) 按列滚动。"""
    w = max(2, int(window))
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 3)
    bench = align_benchmark_to_ret(ret, benchmark_ret)
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for col in ret.columns:
        out[col] = rolling_beta(
            ret[[col]],
            bench[[col]],
            window=w,
            min_periods=mp,
        )[col]
    return out
