# -*- coding: utf-8
"""Beta 类算子：双输入 panel 对齐 + 滚动 beta。"""

from __future__ import annotations

import pandas as pd

from factor_engine.cleaned_operators._rolling_fast import rolling_beta


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
    # The kernel is already column-wise vectorised: ``DataFrame.rolling().cov``
    # and ``.var`` each run one C loop per column, and the paired stats are
    # column-aligned by construction (``align_benchmark_to_ret`` guarantees
    # ``bench.columns == ret.columns``).  The previous per-column loop therefore
    # bought nothing but pandas object churn -- two single-column frames, a
    # ``bench[[col]]`` copy and a column write per instrument.  Feeding the
    # whole panel once is numerically identical and was measured ~10x faster on
    # the 30 x 972 research panel (report section 1.6).
    return rolling_beta(ret, bench, window=w, min_periods=mp)
