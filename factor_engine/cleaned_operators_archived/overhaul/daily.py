# -*- coding: utf-8 -*-
"""Causal daily-panel and order-statistic overrides."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    Spec,
    aligned_pd,
    finite_pd,
    frame_pd,
    nonnegative_int,
    pl,
    pl_base_with,
    pl_cols,
    pl_finite,
    pl_unary_rolling_map,
    positive_int,
    register_specs,
    window_params,
)


def _assert_condition_bool(condition: Any, name: str = "condition") -> None:
    """R11 #144: the condition input must be a ConditionBool.

    Accepted values: {0, 1} (or boolean True/False); NaN/null = missing and is
    excluded from selection.  Any other finite numeric value (e.g. 5.0, -3.0) is
    neither a probability nor a boolean — silently treating it as "truthy" is a
    hidden semantic the operator contract forbids.  Fail the call (raise) instead
    of guessing.

    Mirrors ``cleaned_operators.common.daily_panel._assert_condition_bool``.
    Accepts both pandas wide panels and polars wide frames (polars backend).
    """
    if isinstance(condition, pd.DataFrame):
        cv = condition.to_numpy()
    else:  # polars wide frame
        columns = pl_cols(condition)
        if not columns:
            return
        cv = np.column_stack(
            [condition[col].cast(pl.Float64, strict=False).to_numpy() for col in columns]
        )
    # NEW-047/253: legal set is strictly {0, 1, NaN}.  The old check used
    # ``np.isfinite`` as the "needs validation" premise, so ±Inf slipped past
    # and different families read it differently (True vs False).  Any value
    # outside {0,1,NaN} — including ±Inf — is now rejected.
    missing = pd.isna(cv)
    valid = (cv == 0.0) | (cv == 1.0)
    bad = ~missing & ~valid
    if np.any(bad):
        bad_values = sorted({str(v) for v in np.unique(cv[bad]).tolist()})[:10]
        raise ValueError(
            f"{name} must be a ConditionBool (values in {{0, 1}} with NaN as "
            f"missing); found {int(bad.sum())} value(s) outside {{0, 1}}: "
            f"{bad_values} (this includes ±Inf, which the previous validator "
            "silently passed)"
        )


def pd_count_if(condition: pd.DataFrame, window: int, min_periods: int = 1, **_: Any) -> pd.DataFrame:
    _assert_condition_bool(condition)
    w, mp = window_params(window, min_periods)
    valid = condition.notna()
    truth = valid & condition.eq(1)
    count = truth.astype(float).rolling(w, min_periods=1).sum()
    valid_count = valid.astype(float).rolling(w, min_periods=1).sum()
    return count.where(valid_count >= mp)


def pd_conditional(x, condition, window, min_periods, op, ddof=1):
    x, condition = aligned_pd(x, condition)
    _assert_condition_bool(condition)
    w, mp = window_params(window, min_periods)
    ddof_i = nonnegative_int(ddof, "ddof")
    if ddof_i not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")
    cond_valid = condition.notna()
    selected = finite_pd(x) & cond_valid & condition.eq(1)
    masked = x.where(selected)
    count = selected.astype(float).rolling(w, min_periods=1).sum()
    if op == "sum":
        result, required = masked.rolling(w, min_periods=1).sum(), mp
    elif op == "mean":
        result, required = masked.rolling(w, min_periods=1).mean(), mp
    else:
        result = masked.rolling(w, min_periods=1).std(ddof=ddof_i)
        required = max(mp, ddof_i + 1)
    return result.where(count >= required).replace([np.inf, -np.inf], np.nan)


def pd_sum_if(x, condition, window, min_periods=1, **_):
    return pd_conditional(x, condition, window, min_periods, "sum")


def pd_mean_if(x, condition, window, min_periods=1, **_):
    return pd_conditional(x, condition, window, min_periods, "mean")


def pd_std_if(x, condition, window, min_periods=2, ddof=1, **_):
    return pd_conditional(x, condition, window, min_periods, "std", ddof)


def pd_last_if(x, condition, window, **_):
    x, condition = aligned_pd(x, condition)
    _assert_condition_bool(condition)
    w, _ = window_params(window, 1)
    xv, cv = x.to_numpy(dtype=float), condition.to_numpy(dtype=float)
    out = np.full_like(xv, np.nan)
    for col in range(xv.shape[1]):
        for row in range(xv.shape[0]):
            start = max(0, row - w + 1)
            vals, cond = xv[start : row + 1, col], cv[start : row + 1, col]
            # ConditionBool: only cond == 1 selects.  A NaN-condition row does
            # not select but the previous known value is still carried forward
            # (last selected value in the window); with no prior known the cell
            # stays NaN (unknown, never 0).
            hits = np.flatnonzero(np.isfinite(vals) & np.isfinite(cond) & (cond == 1.0))
            if hits.size:
                out[row, col] = vals[hits[-1]]
    return frame_pd(x, out)


def pd_days_since(condition, max_lookback=None, **_):
    _assert_condition_bool(condition)
    limit = None if max_lookback is None else positive_int(max_lookback, "max_lookback")
    values = condition.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        last = -1
        for row, value in enumerate(values[:, col]):
            if pd.isna(value):
                # Unknown event state at t censors output[t] AND destroys state
                # certainty: the previously known "last true" is reset until an
                # explicit true (1) observation rebuilds it.
                last = -1
                continue
            if bool(value):
                last = row
            if last >= 0:
                distance = row - last
                if limit is None or distance < limit:
                    out[row, col] = float(distance)
    return frame_pd(condition, out)


def pd_true_streak(condition, **_):
    _assert_condition_bool(condition)
    values = condition.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        streak = 0
        for row, value in enumerate(values[:, col]):
            if pd.isna(value):
                # Unknown event state censors the output AND resets the streak
                # counter (certainty is lost until a true observation restarts
                # a run).  Only confirmed-false rows emit 0.
                streak = 0
                continue
            streak = streak + 1 if bool(value) else 0
            out[row, col] = float(streak)
    return frame_pd(condition, out)


def pl_count_if(condition, window, min_periods=1, **_):
    _assert_condition_bool(condition)
    w, mp = window_params(window, min_periods)
    replacements = {}
    for col in pl_cols(condition):
        temp = pl.DataFrame({"c": condition[col]})
        finite = pl_finite("c")
        truth = finite & (pl.col("c") == 1)
        expr = pl.when(finite.cast(pl.Int64).rolling_sum(w, min_samples=1) >= mp).then(
            truth.cast(pl.Float64).rolling_sum(w, min_samples=1)
        ).otherwise(None)
        replacements[col] = temp.select(expr.alias("v"))["v"]
    return pl_base_with(condition, replacements)


def pl_conditional(x, condition, window, min_periods, op, ddof=1):
    _assert_condition_bool(condition)
    w, mp = window_params(window, min_periods)
    ddof_i = nonnegative_int(ddof, "ddof")
    if ddof_i not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")
    replacements = {}
    for col in [c for c in pl_cols(x) if c in condition.columns]:
        temp = pl.DataFrame({"x": x[col], "c": condition[col]})
        selected = pl_finite("x") & pl_finite("c") & (pl.col("c") == 1)
        masked = pl.when(selected).then(pl.col("x").cast(pl.Float64)).otherwise(None)
        count = selected.cast(pl.Int64).rolling_sum(w, min_samples=1)
        if op == "sum":
            aggregate, required = masked.rolling_sum(w, min_samples=1), mp
        elif op == "mean":
            aggregate, required = masked.rolling_mean(w, min_samples=1), mp
        else:
            aggregate = masked.rolling_std(w, min_samples=1, ddof=ddof_i)
            required = max(mp, ddof_i + 1)
        replacements[col] = temp.select(
            pl.when(count >= required).then(aggregate).otherwise(None).alias("v")
        )["v"]
    return pl_base_with(x, replacements)


def pl_sum_if(x, condition, window, min_periods=1, **_):
    return pl_conditional(x, condition, window, min_periods, "sum")


def pl_mean_if(x, condition, window, min_periods=1, **_):
    return pl_conditional(x, condition, window, min_periods, "mean")


def pl_std_if(x, condition, window, min_periods=2, ddof=1, **_):
    return pl_conditional(x, condition, window, min_periods, "std", ddof)


def pl_last_if(x, condition, window, **_):
    _assert_condition_bool(condition)
    w, _ = window_params(window, 1)
    replacements = {}
    for col in [c for c in pl_cols(x) if c in condition.columns]:
        temp = pl.DataFrame({"x": x[col], "c": condition[col]})
        selected = pl.when(pl_finite("x") & pl_finite("c") & (pl.col("c") == 1)).then(pl.col("x")).otherwise(None)
        def last(values):
            arr = np.asarray(values, dtype=float)
            valid = arr[np.isfinite(arr)]
            return float(valid[-1]) if valid.size else np.nan
        replacements[col] = temp.select(selected.rolling_map(last, w, min_samples=1).alias("v"))["v"]
    return pl_base_with(x, replacements)


def pl_days_since(condition, max_lookback=None, **_):
    _assert_condition_bool(condition)
    limit = None if max_lookback is None else positive_int(max_lookback, "max_lookback")
    replacements = {}
    for col in pl_cols(condition):
        arr = condition[col].cast(pl.Float64, strict=False).to_numpy()
        out, last = np.full(len(arr), np.nan, dtype=float), -1
        for row, value in enumerate(arr):
            if pd.isna(value):
                # Unknown event state censors output AND resets the running
                # count (a later known-true event restarts from 0).
                last = -1
                continue
            if bool(value):
                last = row
            if last >= 0:
                distance = row - last
                if limit is None or distance < limit:
                    out[row] = float(distance)
        replacements[col] = pl.Series(col, out)
    return pl_base_with(condition, replacements)


def pl_true_streak(condition, **_):
    _assert_condition_bool(condition)
    replacements = {}
    for col in pl_cols(condition):
        arr = condition[col].cast(pl.Float64, strict=False).to_numpy()
        out, streak = np.full(len(arr), np.nan, dtype=float), 0
        for i, value in enumerate(arr):
            if pd.isna(value):
                streak = 0
                continue
            streak = streak + 1 if bool(value) else 0
            out[i] = float(streak)
        replacements[col] = pl.Series(col, out)
    return pl_base_with(condition, replacements)


def pd_cs_bucket(x, buckets=10, ascending=True, **_):
    count = positive_int(buckets, "buckets")
    rank = x.rank(axis=1, method="average", ascending=bool(ascending), na_option="keep")
    valid_count = x.notna().sum(axis=1).astype(float)
    rank01 = rank.sub(1).div((valid_count - 1).replace(0, np.nan), axis=0)
    single = valid_count.eq(1)
    if single.any():
        rank01.loc[single] = (x.loc[single].notna().astype(float) * 0.5).where(x.loc[single].notna())
    return (np.floor(rank01 * count) + 1).clip(1, count).where(x.notna())


def pl_cs_bucket(x, buckets=10, ascending=True, **_):
    count = positive_int(buckets, "buckets")
    cols = pl_cols(x)
    long = x.select(cols).with_row_index("row").unpivot(
        index="row", on=cols, variable_name="inst", value_name="value"
    ).with_columns(
        pl.when(pl.col("value").cast(pl.Float64, strict=False).is_finite())
        .then(pl.col("value").cast(pl.Float64)).otherwise(None).alias("value")
    )
    n = pl.col("value").count().over("row")
    rank_input = pl.col("value") if ascending else -pl.col("value")
    rank = rank_input.rank(method="average").over("row")
    rank01 = pl.when(n == 1).then(0.5).otherwise((rank - 1) / (n - 1))
    wide = long.with_columns(
        pl.when(pl.col("value").is_not_null())
        .then((rank01 * count).floor().add(1).clip(1, count))
        .otherwise(None).alias("out")
    ).pivot(on="inst", index="row", values="out", aggregate_function="first").sort("row")
    return pl_base_with(x, {c: wide[c] for c in cols})


def pd_cs_bucket_fixed(x, breaks, **_):
    """固定边界分桶：值 <= breaks[0] → 1；> breaks[-1] → len(breaks)+1；其余按所在区间。

    与等频 ``cs_bucket``（横截面排名分桶）不同，固定边界是明确数值断点。
    """
    b = np.asarray(list(breaks), dtype=float)
    if b.ndim != 1 or b.size == 0:
        raise ValueError("breaks must be a non-empty 1D sequence")
    if np.any(np.diff(b) <= 0):
        raise ValueError("breaks must be strictly increasing")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    idx = np.searchsorted(b, arr, side="right") + 1.0
    out = np.where(np.isfinite(arr), idx, np.nan)
    return frame_pd(x, out)


def pd_cs_bucket_historical(x, window=20, quantiles=(0.2, 0.4, 0.6, 0.8), min_periods=5, **_):
    """基于**历史边界**分桶：对每只股票取 trailing window 内经验分位数断点，
    当前值落入哪个历史分位区间即输出对应桶号（PIT：只用截至当日的 x）。

    与等频 ``cs_bucket``（横截面）和固定边界 ``cs_bucket_fixed`` 语义不同。
    """
    w = positive_int(window, "window")
    q = tuple(float(v) for v in quantiles)
    if not q or any(not 0 < v < 1 for v in q):
        raise ValueError("quantiles must be strictly inside (0, 1)")
    mp = max(1, int(min_periods))
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = arr[start:row, col]  # 历史窗不含当前行
            valid = seg[np.isfinite(seg)]
            if valid.size < mp:
                continue
            breaks = np.quantile(valid, q)
            val = arr[row, col]
            if not np.isfinite(val):
                continue
            out[row, col] = float(np.searchsorted(breaks, val, side="right") + 1)
    return frame_pd(x, out)


def pd_argext(x, window, pick, min_periods=1):
    w, mp = window_params(window, min_periods)
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            values = arr[max(0, row - w + 1): row + 1, col]
            valid = np.isfinite(values)
            if valid.sum() < mp:
                continue
            target = np.nanmax(values) if pick == "max" else np.nanmin(values)
            hit = np.flatnonzero(valid & (values == target))[-1]
            out[row, col] = values.size - 1 - hit
    return frame_pd(x, out)


def pd_argmax(x, window, min_periods=1, **_):
    return pd_argext(x, window, "max", min_periods)


def pd_argmin(x, window, min_periods=1, **_):
    return pd_argext(x, window, "min", min_periods)


def pl_argext(x, window, pick, min_periods=1):
    w, mp = window_params(window, min_periods)
    def fn(values):
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < mp:
            return np.nan
        target = np.nanmax(values) if pick == "max" else np.nanmin(values)
        hit = np.flatnonzero(valid & (values == target))[-1]
        return float(values.size - 1 - hit)
    return pl_unary_rolling_map(x, w, 1, fn)


def pl_argmax(x, window, min_periods=1, **_):
    return pl_argext(x, window, "max", min_periods)


def pl_argmin(x, window, min_periods=1, **_):
    return pl_argext(x, window, "min", min_periods)


def pd_topbottom(x, window, k, min_periods, top, stat):
    w, mp = window_params(window, min_periods, default_mp=1)
    k = positive_int(k, "k")
    if k > w:
        raise ValueError("k must not exceed window")
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            segment = arr[max(0, row - w + 1): row + 1, col]
            valid = np.sort(segment[np.isfinite(segment)])
            if valid.size < max(mp, k):
                continue
            selected = valid[-k:] if top else valid[:k]
            out[row, col] = selected.mean() if stat == "mean" else selected.sum() if stat == "sum" else selected.std(ddof=1)
    return frame_pd(x, out)


def pl_topbottom(x, window, k, min_periods, top, stat):
    w, mp = window_params(window, min_periods, default_mp=1)
    k = positive_int(k, "k")
    if k > w:
        raise ValueError("k must not exceed window")
    def fn(values):
        valid = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
        if valid.size < max(mp, k):
            return np.nan
        selected = valid[-k:] if top else valid[:k]
        return float(selected.mean() if stat == "mean" else selected.sum() if stat == "sum" else selected.std(ddof=1))
    return pl_unary_rolling_map(x, w, 1, fn)


def pd_tail_mean(x, window, q=0.1, side="lower", min_periods=None, **_):
    w, mp = window_params(window, min_periods, default_mp=2)
    q = float(q)
    if not 0 < q <= 0.5:
        raise ValueError("q must satisfy 0 < q <= 0.5")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            valid = arr[max(0, row - w + 1): row + 1, col]
            valid = valid[np.isfinite(valid)]
            if valid.size < mp:
                continue
            threshold = np.quantile(valid, q if side == "lower" else 1 - q)
            tail = valid[valid <= threshold] if side == "lower" else valid[valid >= threshold]
            out[row, col] = tail.mean()
    return frame_pd(x, out)


def pl_tail_mean(x, window, q=0.1, side="lower", min_periods=None, **_):
    w, mp = window_params(window, min_periods, default_mp=2)
    q = float(q)
    if not 0 < q <= 0.5 or side not in {"lower", "upper"}:
        raise ValueError("invalid q or side")
    def fn(values):
        valid = np.asarray(values, dtype=float)
        valid = valid[np.isfinite(valid)]
        if valid.size < mp:
            return np.nan
        threshold = np.quantile(valid, q if side == "lower" else 1 - q)
        tail = valid[valid <= threshold] if side == "lower" else valid[valid >= threshold]
        return float(tail.mean())
    return pl_unary_rolling_map(x, w, 1, fn)


def _top_spec(top, stat):
    return (
        lambda x, window, k, min_periods=None, **kw: pd_topbottom(x, window, k, min_periods, top, stat),
        lambda x, window, k, min_periods=None, **kw: pl_topbottom(x, window, k, min_periods, top, stat),
    )


def register() -> None:
    specs = {
        "ts_count_if": Spec("time_series_condition", ["condition", "window", "min_periods"], "滚动统计有限条件为真的次数", pd_count_if, pl_count_if),
        "ts_sum_if": Spec("time_series_condition", ["x", "condition", "window", "min_periods"], "滚动条件求和", pd_sum_if, pl_sum_if),
        "ts_mean_if": Spec("time_series_condition", ["x", "condition", "window", "min_periods"], "滚动条件均值", pd_mean_if, pl_mean_if),
        "ts_std_if": Spec("time_series_condition", ["x", "condition", "window", "min_periods", "ddof"], "滚动条件标准差", pd_std_if, pl_std_if),
        "ts_last_if": Spec("time_series_event", ["x", "condition", "window"], "窗口内最近一次条件成立时的值", pd_last_if, pl_last_if),
        "ts_days_since": Spec("time_series_event", ["condition", "max_lookback"], "距最近一次条件成立的交易行数", pd_days_since, pl_days_since),
        "ts_true_streak": Spec("time_series_event", ["condition"], "截至当前连续条件成立长度", pd_true_streak, pl_true_streak),
        "cs_bucket": Spec("cross_sectional", ["x", "buckets", "ascending"], "横截面零到一排名分桶", pd_cs_bucket, pl_cs_bucket),
        "cs_bucket_fixed": Spec("cross_sectional", ["x", "breaks"], "固定边界分桶", pd_cs_bucket_fixed),
        "cs_bucket_historical": Spec("cross_sectional", ["x", "window", "quantiles", "min_periods"], "基于历史边界分桶（PIT）", pd_cs_bucket_historical),
        "ts_argmax": Spec("time_series_order", ["x", "window", "min_periods"], "距最近一次窗口最大值的交易行数", pd_argmax, pl_argmax),
        "ts_argmin": Spec("time_series_order", ["x", "window", "min_periods"], "距最近一次窗口最小值的交易行数", pd_argmin, pl_argmin),
        "ts_tail_mean": Spec("time_series_risk", ["x", "window", "q", "side", "min_periods"], "当前窗口统一阈值下的尾部均值", pd_tail_mean, pl_tail_mean),
    }
    for name, top, stat, desc in (
        ("ts_topk_mean", True, "mean", "窗口内最大 K 个值的均值"),
        ("ts_topk_sum", True, "sum", "窗口内最大 K 个值之和"),
        ("ts_topk_std", True, "std", "窗口内最大 K 个值的标准差"),
        ("ts_bottomk_mean", False, "mean", "窗口内最小 K 个值的均值"),
        ("ts_bottomk_sum", False, "sum", "窗口内最小 K 个值之和"),
        ("ts_bottomk_std", False, "std", "窗口内最小 K 个值的标准差"),
    ):
        pfn, plfn = _top_spec(top, stat)
        # ts_topk_sum declares canonical params ``["x","d","k"]`` (R19-050,
        # ``window``/``n`` explicit aliases); its inherited ParamSpec keys must
        # sit inside the overhaul param_names (R6-157 invariant).
        params = ["x", "d", "k", "min_periods"] if name == "ts_topk_sum" else ["x", "window", "k", "min_periods"]
        specs[name] = Spec("time_series_order", params, desc, pfn, plfn)
    register_specs(specs)
