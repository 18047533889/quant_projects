# -*- coding: utf-8
"""财报 period 契约：quarter / ttm / yoy 与 fiscal_quarter 对齐。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align_fiscal_quarter(x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None) -> pd.Series | None:
    if fiscal_quarter is None:
        return None
    if fiscal_quarter.shape[1] == 1:
        return fiscal_quarter.iloc[:, 0].reindex(x.index)
    if list(fiscal_quarter.columns) == list(x.columns):
        return None  # 按列处理
    return fiscal_quarter.reindex(columns=x.columns)


def _is_q1_or_rollover(curr_q: int, prev_q: int | None) -> bool:
    if curr_q == 1:
        return True
    if prev_q is not None and prev_q == 4 and curr_q == 1:
        return True
    return False


def _compute_quarter_1d(
    values: np.ndarray,
    fiscal_q: np.ndarray | None,
) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        fq = None if fiscal_q is None else fiscal_q[i]
        if fiscal_q is None or not np.isfinite(fq):
            if i == 0:
                out[i] = v
            elif np.isfinite(values[i - 1]):
                out[i] = v - values[i - 1]
            else:
                out[i] = v
            continue
        curr_q = int(fq)
        prev_q = int(fiscal_q[i - 1]) if i > 0 and np.isfinite(fiscal_q[i - 1]) else None
        if i == 0 or _is_q1_or_rollover(curr_q, prev_q):
            out[i] = v
        elif prev_q is not None and curr_q == prev_q + 1 and np.isfinite(values[i - 1]):
            out[i] = v - values[i - 1]
        else:
            out[i] = v
    return out


def compute_quarter(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """累计值转单季度；传入 fiscal_quarter(1-4) 时按财报期边界处理 Q1/跨年。"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    per_col_fq = _align_fiscal_quarter(x, fiscal_quarter)
    for col in x.columns:
        fq_series = per_col_fq
        if fiscal_quarter is not None and per_col_fq is None:
            fq_series = fiscal_quarter[col]
        fq_arr = None if fq_series is None else fq_series.to_numpy(dtype=float)
        out[col] = _compute_quarter_1d(x[col].to_numpy(dtype=float), fq_arr)
    return out


def _compute_ttm_1d(values: np.ndarray, fiscal_q: np.ndarray | None) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not np.isfinite(values[i]):
            continue
        if fiscal_q is None:
            if i >= 3 and np.all(np.isfinite(values[i - 3 : i + 1])):
                out[i] = np.nansum(values[i - 3 : i + 1])
            else:
                out[i] = values[i]
            continue
        window_vals: list[float] = []
        for j in range(i, -1, -1):
            if not np.isfinite(values[j]):
                break
            window_vals.append(float(values[j]))
            if len(window_vals) >= 4:
                break
        out[i] = float(np.sum(window_vals)) if window_vals else values[i]
    return out


def compute_ttm(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """滚动十二个月（季频 4 期累加）；fiscal_quarter 可选用于校验/report 对齐。"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    per_col_fq = _align_fiscal_quarter(x, fiscal_quarter)
    for col in x.columns:
        fq_series = per_col_fq
        if fiscal_quarter is not None and per_col_fq is None:
            fq_series = fiscal_quarter[col]
        fq_arr = None if fq_series is None else fq_series.to_numpy(dtype=float)
        out[col] = _compute_ttm_1d(x[col].to_numpy(dtype=float), fq_arr)
    return out


def _compute_yoy_1d(values: np.ndarray, fiscal_q: np.ndarray | None) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            if not np.isfinite(values[i]):
                continue
            if fiscal_q is None:
                if i >= 4 and np.isfinite(values[i - 4]):
                    out[i] = values[i] / values[i - 4] - 1.0
                continue
            curr_q = fiscal_q[i]
            if not np.isfinite(curr_q):
                continue
            target = int(curr_q)
            for j in range(i - 1, -1, -1):
                if np.isfinite(fiscal_q[j]) and int(fiscal_q[j]) == target:
                    steps = i - j
                    if steps >= 4 and np.isfinite(values[j]):
                        out[i] = values[i] / values[j] - 1.0
                    break
            if not np.isfinite(out[i]) and i >= 4 and np.isfinite(values[i - 4]):
                out[i] = values[i] / values[i - 4] - 1.0
    return out


def compute_yoy(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """同比增速；默认 lag=4 行，有 fiscal_quarter 时优先找同季度去年同期。"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    per_col_fq = _align_fiscal_quarter(x, fiscal_quarter)
    for col in x.columns:
        fq_series = per_col_fq
        if fiscal_quarter is not None and per_col_fq is None:
            fq_series = fiscal_quarter[col]
        fq_arr = None if fq_series is None else fq_series.to_numpy(dtype=float)
        out[col] = _compute_yoy_1d(x[col].to_numpy(dtype=float), fq_arr)
    return out


def _compute_avg2_1d(values: np.ndarray, fiscal_q: np.ndarray | None) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            continue
        if i == 0 or not np.isfinite(values[i - 1]):
            out[i] = v
            continue
        if fiscal_q is None or not np.isfinite(fiscal_q[i]):
            out[i] = (v + values[i - 1]) / 2.0
            continue
        curr_q = int(fiscal_q[i])
        prev_q = int(fiscal_q[i - 1]) if np.isfinite(fiscal_q[i - 1]) else None
        if prev_q is not None and (curr_q == prev_q + 1 or (prev_q == 4 and curr_q == 1)):
            out[i] = (v + values[i - 1]) / 2.0
        else:
            out[i] = v
    return out


def compute_avg2(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """当期与上期均值；有 fiscal_quarter 时仅在连续报告期间取均值。"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    per_col_fq = _align_fiscal_quarter(x, fiscal_quarter)
    for col in x.columns:
        fq_series = per_col_fq
        if fiscal_quarter is not None and per_col_fq is None:
            fq_series = fiscal_quarter[col]
        fq_arr = None if fq_series is None else fq_series.to_numpy(dtype=float)
        out[col] = _compute_avg2_1d(x[col].to_numpy(dtype=float), fq_arr)
    return out


def quarter_from_cumulative(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """显式语义：输入为**累计值**，输出单季值（等同 ``compute_quarter``）。"""
    return compute_quarter(x, fiscal_quarter)


def compute_ttm_from_quarterly(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """显式语义：输入为**单季值**，滚动累加最近 4 个有效单季为 TTM。"""
    return compute_ttm(x, fiscal_quarter)


def ttm_from_quarterly(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return compute_ttm_from_quarterly(x, fiscal_quarter)


def ttm_from_cumulative(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """累计值 → 单季 → 滚动 4 季 TTM。"""
    quarterly = quarter_from_cumulative(x, fiscal_quarter)
    return compute_ttm_from_quarterly(quarterly, fiscal_quarter)


def yoy_by_period(
    x: pd.DataFrame,
    fiscal_quarter: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """显式语义：按报告期/行 lag 计算同比增速（等同 ``compute_yoy``）。"""
    return compute_yoy(x, fiscal_quarter)
