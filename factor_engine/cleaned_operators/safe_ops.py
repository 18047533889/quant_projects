# -*- coding: utf-8 -*-
"""Source-safe building blocks (audit section 19).

These operators do not add alpha surface on their own; they exist so other
factors can be constructed safely: validity/coverage/staleness gates, strict
bounded imputation, gap-limited forward fill, explicit log domains, and
unambiguous extreme-position semantics.

Every operator is causal (only data ``<= t`` is consumed) and preserves the
``timestamp x instrument`` panel shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry

EPS = 1e-12
SOURCE = "safe_ops"


def _register(name: str, category: str, params: list[str], description: str, pandas_fn, polars_fn=None) -> None:
    OperatorRegistry.register(
        PandasFunctionOperator(name, category, params, description, pandas_fn),
        canonical=name,
        backend="pandas_numpy",
        source=SOURCE,
        status="implemented",
        backend_explicit=True,
    )
    if polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(name, category, params, description, polars_fn),
            canonical=name,
            backend="polars",
            source=SOURCE,
            status="implemented",
            backend_explicit=True,
        )
    # Classify as an extended production candidate so the static-surface gate
    # (layer_governance) covers it.  These operators are source-safe building
    # blocks by construction; they are NOT fail-closed by the S2 gate.
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})


# --------------------------------------------------------------------------
# Validity / coverage / staleness gates
# --------------------------------------------------------------------------

def pd_ts_valid_count(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    return x.rolling(w, min_periods=int(min_periods)).count()


def pd_ts_coverage_ratio(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    count = x.rolling(w, min_periods=int(min_periods)).count()
    return count / float(w)


def pd_ts_staleness(x, window, **_):
    """Bars since the last finite value within the causal window.

    A value at bar ``t`` carrying the last finite observation from ``t-k``
    reports ``k``; a window with no finite value reports NaN.  This is the
    "how old is the data I am actually using" gate.
    """
    w = positive_int(window, "window")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        col_arr = arr[:, col]
        last_seen = np.full(w + 1, np.nan, dtype=float)  # ring of recent positions
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = col_arr[start : row + 1]
            finite_idx = np.flatnonzero(np.isfinite(seg))
            if finite_idx.size == 0:
                continue
            out[row, col] = float(row - (start + finite_idx[-1]))
    return frame_pd(x, out)


def _pl_ts_valid_count(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    cols = pl_cols(x)
    return x.with_columns([
        pl.col(c).is_not_null().cast(pl.Int32).rolling_sum(window_size=w, min_samples=int(min_periods)).alias(c)
        for c in cols
    ])


def _pl_ts_coverage_ratio(x, window, min_periods=1, **_):
    w = positive_int(window, "window")
    cols = pl_cols(x)
    return x.with_columns([
        (pl.col(c).is_not_null().cast(pl.Int32).rolling_sum(window_size=w, min_samples=int(min_periods)) / w).alias(c)
        for c in cols
    ])


_register(
    "ts_valid_count",
    "time_series",
    ["x", "window", "min_periods"],
    "窗口内有限值个数。",
    pd_ts_valid_count,
    _pl_ts_valid_count,
)
_register(
    "ts_coverage_ratio",
    "time_series",
    ["x", "window", "min_periods"],
    "窗口内有限值占比（有限值个数 / window）。",
    pd_ts_coverage_ratio,
    _pl_ts_coverage_ratio,
)
_register(
    "ts_staleness",
    "time_series",
    ["x", "window"],
    "距窗口内最近一个有限值的 bar 数。",
    pd_ts_staleness,
)


def pd_cs_valid_count(x, **_):
    arr = x.to_numpy(dtype=float)
    counts = np.sum(np.isfinite(arr), axis=1, keepdims=True)
    out = np.broadcast_to(counts, arr.shape).astype(float)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_cs_coverage_ratio(x, **_):
    arr = x.to_numpy(dtype=float)
    count = np.sum(np.isfinite(arr), axis=1, keepdims=True)
    total = x.shape[1]
    out = np.broadcast_to(count / float(total), arr.shape).astype(float)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


_register(
    "cs_valid_count",
    "cross_sectional",
    ["x"],
    "横截面每行有限值个数（广播到每列）。",
    pd_cs_valid_count,
)
_register(
    "cs_coverage_ratio",
    "cross_sectional",
    ["x"],
    "横截面每行有限值占比（广播到每列）。",
    pd_cs_coverage_ratio,
)


def pd_group_valid_count(x, group, **_):
    x, group = aligned_pd(x, group)
    arr = x.to_numpy(dtype=float)
    garr = group.to_numpy(dtype=object)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            g = garr[row, col]
            if g is None or pd.isna(g):
                continue
            mask = garr[row] == g
            out[row, col] = float(np.sum(np.isfinite(arr[row][mask])))
    return frame_pd(x, out)


_register(
    "group_valid_count",
    "group",
    ["x", "group"],
    "组内有限值个数。",
    pd_group_valid_count,
)


def pd_group_impute_median(x, group, min_group_size=3, **_):
    x, group = aligned_pd(x, group)
    arr = x.to_numpy(dtype=float)
    garr = group.to_numpy(dtype=object)
    out = arr.copy()
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            g = garr[row, col]
            if g is None or pd.isna(g):
                continue
            mask = garr[row] == g
            group_vals = arr[row][mask]
            finite = group_vals[np.isfinite(group_vals)]
            if finite.size < int(min_group_size):
                continue
            median = float(np.median(finite))
            out[row] = np.where(
                np.isfinite(out[row]),
                out[row],
                np.where(garr[row] == g, median, np.nan),
            )
    return frame_pd(x, out)


_register(
    "group_impute_median",
    "group",
    ["x", "group", "min_group_size"],
    "组内中位数填补缺失值（组内有限样本不足 min_group_size 时不填）。",
    pd_group_impute_median,
)


# --------------------------------------------------------------------------
# Strict missing-value handling
# --------------------------------------------------------------------------

def pd_ts_ffill_limited(x, max_gap, **_):
    limit = positive_int(max_gap, "max_gap")
    return x.ffill(limit=limit)


def _pl_ts_ffill_limited(x, max_gap, **_):
    limit = positive_int(max_gap, "max_gap")
    return x.with_columns([pl.col(c).forward_fill(limit=limit).alias(c) for c in pl_cols(x)])


def pd_log_positive_or_nan(x, **_):
    arr = x.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(arr > 0, np.log(arr), np.nan)
    return frame_pd(x, out)


def _pl_log_positive_or_nan(x, **_):
    cols = pl_cols(x)
    return x.with_columns([
        pl.when(pl.col(c) > 0).then(pl.col(c).log()).otherwise(None).alias(c)
        for c in cols
    ])


_register(
    "ts_ffill_limited",
    "data_cleaning",
    ["x", "max_gap"],
    "前向填充，最多跨过 max_gap 根 bar（防长期停牌/断档污染）。",
    pd_ts_ffill_limited,
    _pl_ts_ffill_limited,
)
_register(
    "log_positive_or_nan",
    "elementwise",
    ["x"],
    "对数正域映射：x>0 取 log(x)，否则 NaN（不把非法域夹到 epsilon）。",
    pd_log_positive_or_nan,
    _pl_log_positive_or_nan,
)


# --------------------------------------------------------------------------
# Unambiguous extreme position semantics
# --------------------------------------------------------------------------

def _pd_argext(x, window, pick, min_periods):
    w = positive_int(window, "window")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = arr[start : row + 1, col]
            valid = np.isfinite(seg)
            if valid.sum() < int(min_periods):
                continue
            target = np.nanmax(seg) if pick == "max" else np.nanmin(seg)
            hits = np.flatnonzero(valid & (seg == target))
            age = (seg.size - 1) - hits[-1]          # ties -> most recent
            out[row, col] = float(age)
    return frame_pd(x, out)


def pd_ts_argmax_age(x, window, min_periods=1, **_):
    return _pd_argext(x, window, "max", min_periods)


def pd_ts_argmin_age(x, window, min_periods=1, **_):
    return _pd_argext(x, window, "min", min_periods)


def _pd_argidx(x, window, pick, min_periods):
    """Position of the extreme measured from the window's OLDEST bar."""
    w = positive_int(window, "window")
    arr = x.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            start = max(0, row - w + 1)
            seg = arr[start : row + 1, col]
            valid = np.isfinite(seg)
            if valid.sum() < int(min_periods):
                continue
            target = np.nanmax(seg) if pick == "max" else np.nanmin(seg)
            hits = np.flatnonzero(valid & (seg == target))
            out[row, col] = float(hits[-1])  # 0 = window oldest
    return frame_pd(x, out)


def pd_ts_argmax_index_from_oldest(x, window, min_periods=1, **_):
    return _pd_argidx(x, window, "max", min_periods)


def pd_ts_argmin_index_from_oldest(x, window, min_periods=1, **_):
    return _pd_argidx(x, window, "min", min_periods)


_register(
    "ts_argmax_age",
    "time_series_order",
    ["x", "window", "min_periods"],
    "距窗口内最近一次最大值的 bar 数（0=当前行即极值）。",
    pd_ts_argmax_age,
)
_register(
    "ts_argmin_age",
    "time_series_order",
    ["x", "window", "min_periods"],
    "距窗口内最近一次最小值的 bar 数（0=当前行即极值）。",
    pd_ts_argmin_age,
)
_register(
    "ts_argmax_index_from_oldest",
    "time_series_order",
    ["x", "window", "min_periods"],
    "窗口内最大值位置，0=窗口最旧 bar。",
    pd_ts_argmax_index_from_oldest,
)
_register(
    "ts_argmin_index_from_oldest",
    "time_series_order",
    ["x", "window", "min_periods"],
    "窗口内最小值位置，0=窗口最旧 bar。",
    pd_ts_argmin_index_from_oldest,
)


# --------------------------------------------------------------------------
# Cross-section imputation (explicit broadcasting)
# --------------------------------------------------------------------------

def pd_cs_impute_mean(x, min_finite=1, **_):
    row_mean = x.mean(axis=1, skipna=True)
    return x.mask(x.isna(), row_mean, axis=0)


def pd_cs_impute_median(x, min_finite=1, **_):
    row_median = x.median(axis=1, skipna=True)
    return x.mask(x.isna(), row_median, axis=0)


_register(
    "cs_impute_mean",
    "cross_sectional",
    ["x", "min_finite"],
    "横截面均值填补缺失（逐行广播，显式 axis）。",
    pd_cs_impute_mean,
)
_register(
    "cs_impute_median",
    "cross_sectional",
    ["x", "min_finite"],
    "横截面中位数填补缺失（逐行广播，显式 axis）。",
    pd_cs_impute_median,
)
