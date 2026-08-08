# -*- coding: utf-8 -*-
"""Causal daily-panel operators that remain shape preserving.

The operators in this module deliberately exclude whole-sample hypothesis
tests and performance summaries.  Every function consumes one or more
``timestamp x instrument`` panels and returns a panel with the same shape,
using only the current row and historical rows.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import Operator as PandasOperator
from cleaned_operators.base import OperatorMetadata as PandasMetadata
from cleaned_operators.base_polars import Operator as PolarsOperator
from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore


def _positive_int(value: Any, name: str) -> int:
    out = int(value)
    if out < 1 or float(value) != float(out):
        raise ValueError(f"{name} must be a positive integer")
    return out


def _nonnegative_int(value: Any, name: str) -> int:
    out = int(value)
    if out < 0 or float(value) != float(out):
        raise ValueError(f"{name} must be a non-negative integer")
    return out


def _aligned(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Strict multi-panel alignment (round-6 P0-25).

    Two panels with different instrument columns or a shifted index must never
    be paired positionally — silently reindexing hides a missing symbol / a
    one-day date shift as NaN.  Production defaults to fail-closed; the central
    ``validate_operator_call`` gate already rejects misaligned panels before
    this helper is reached, so this strictness is defense-in-depth for direct
    kernel calls.
    """
    if not frames:
        return ()
    from cleaned_operators.alignment import align_panel_inputs

    return align_panel_inputs(*frames, strict_axes=True)


def _rolling_apply_2d(
    values: np.ndarray,
    window: int,
    fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            out[row, col] = fn(values[start : row + 1, col])
    return out


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _trailing_contiguous(chunk: np.ndarray) -> np.ndarray:
    """Longest trailing contiguous finite run ending at the current row.

    A missing value breaks the time axis — rows on either side of a NaN are
    never re-paired as adjacent (no drop-finite/reconnect).  A NaN at the
    current row yields an empty block so the caller emits NaN instead of
    silently reusing the last valid history (review P1-125).
    """
    n = chunk.shape[0]
    if n == 0 or not np.isfinite(chunk[-1]):
        return chunk[:0]
    end = n
    while end > 0 and np.isfinite(chunk[end - 1]):
        end -= 1
    return chunk[end:]


def ts_count_if(
    condition: pd.DataFrame,
    window: int,
    min_periods: int = 1,
    **_: Any,
) -> pd.DataFrame:
    w = _positive_int(window, "window")
    mp = _positive_int(min_periods, "min_periods")
    valid = condition.notna()
    truth = valid & condition.ne(0)
    count = truth.astype(float).rolling(w, min_periods=1).sum()
    valid_count = valid.astype(float).rolling(w, min_periods=1).sum()
    return count.where(valid_count >= mp)


def _conditional_rolling(
    x: pd.DataFrame,
    condition: pd.DataFrame,
    window: int,
    min_periods: int,
    op: str,
    ddof: int = 1,
) -> pd.DataFrame:
    x, condition = _aligned(x, condition)
    w = _positive_int(window, "window")
    mp = _positive_int(min_periods, "min_periods")
    ddof_i = _nonnegative_int(ddof, "ddof")
    selected = condition.notna() & condition.ne(0) & x.notna()
    masked = x.where(selected)
    count = selected.astype(float).rolling(w, min_periods=1).sum()
    if op == "sum":
        result = masked.rolling(w, min_periods=1).sum()
        required = mp
    elif op == "mean":
        result = masked.rolling(w, min_periods=1).mean()
        required = mp
    elif op == "std":
        result = masked.rolling(w, min_periods=1).std(ddof=ddof_i)
        required = max(mp, ddof_i + 1)
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unsupported conditional rolling op: {op}")
    return result.where(count >= required)


def ts_sum_if(
    x: pd.DataFrame,
    condition: pd.DataFrame,
    window: int,
    min_periods: int = 1,
    **_: Any,
) -> pd.DataFrame:
    return _conditional_rolling(x, condition, window, min_periods, "sum")


def ts_mean_if(
    x: pd.DataFrame,
    condition: pd.DataFrame,
    window: int,
    min_periods: int = 1,
    **_: Any,
) -> pd.DataFrame:
    return _conditional_rolling(x, condition, window, min_periods, "mean")


def ts_std_if(
    x: pd.DataFrame,
    condition: pd.DataFrame,
    window: int,
    min_periods: int = 2,
    ddof: int = 1,
    **_: Any,
) -> pd.DataFrame:
    return _conditional_rolling(x, condition, window, min_periods, "std", ddof)


def ts_last_if(
    x: pd.DataFrame,
    condition: pd.DataFrame,
    window: int,
    **_: Any,
) -> pd.DataFrame:
    x, condition = _aligned(x, condition)
    w = _positive_int(window, "window")
    xv = x.to_numpy(dtype=float)
    cv = condition.to_numpy()

    def last_selected(pair: np.ndarray) -> float:
        vals = pair[:, 0]
        cond = pair[:, 1]
        mask = np.isfinite(vals) & np.isfinite(cond) & (cond != 0)
        hits = np.flatnonzero(mask)
        return float(vals[hits[-1]]) if hits.size else np.nan

    rows, cols = xv.shape
    out = np.full_like(xv, np.nan)
    for col in range(cols):
        pair = np.column_stack((xv[:, col], np.asarray(cv[:, col], dtype=float)))
        for row in range(rows):
            out[row, col] = last_selected(pair[max(0, row - w + 1) : row + 1])
    return _frame_like(x, out)


def ts_days_since(
    condition: pd.DataFrame,
    max_lookback: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    limit = None if max_lookback is None else _positive_int(max_lookback, "max_lookback")
    cv = condition.to_numpy()
    rows, cols = cv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        last_true = -1
        for row in range(rows):
            value = cv[row, col]
            if pd.notna(value) and bool(value):
                last_true = row
            if last_true >= 0:
                distance = row - last_true
                if limit is None or distance < limit:
                    out[row, col] = float(distance)
    return _frame_like(condition, out)


def ts_true_streak(condition: pd.DataFrame, **_: Any) -> pd.DataFrame:
    cv = condition.to_numpy()
    rows, cols = cv.shape
    out = np.zeros((rows, cols), dtype=float)
    for col in range(cols):
        streak = 0
        for row in range(rows):
            value = cv[row, col]
            streak = streak + 1 if pd.notna(value) and bool(value) else 0
            out[row, col] = float(streak)
    return _frame_like(condition, out)


def cs_bucket(
    x: pd.DataFrame,
    buckets: int = 10,
    ascending: bool = True,
    **_: Any,
) -> pd.DataFrame:
    count = _positive_int(buckets, "buckets")
    pct = x.rank(axis=1, method="average", pct=True, ascending=bool(ascending))
    result = np.floor(pct * count) + 1.0
    return result.clip(upper=float(count)).where(x.notna())


def _ols_residual(
    y: np.ndarray,
    features: list[np.ndarray],
    *,
    add_intercept: bool,
    min_obs: int | None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    out = np.full(y.shape, np.nan, dtype=float)
    if not features:
        return out
    mask = np.isfinite(y)
    for feature in features:
        mask &= np.isfinite(feature)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights > 0)
    k = len(features)
    required = max(k + 2, 2) if min_obs is None else _positive_int(min_obs, "min_obs")
    if int(mask.sum()) < required:
        return out
    design = np.column_stack([feature[mask] for feature in features])
    if add_intercept:
        design = np.column_stack((np.ones(design.shape[0]), design))
    if design.shape[0] <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return out
    if np.linalg.cond(design) > 1e12:
        return out
    target = y[mask]
    if weights is not None:
        root_w = np.sqrt(weights[mask])
        fit_design = design * root_w[:, None]
        fit_target = target * root_w
    else:
        fit_design = design
        fit_target = target
    beta, *_ = np.linalg.lstsq(fit_design, fit_target, rcond=None)
    out[mask] = target - design @ beta
    return out


def cs_multi_resid(
    y: pd.DataFrame,
    *features: pd.DataFrame,
    add_intercept: bool = True,
    min_obs: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    if not features:
        raise ValueError("cs_multi_resid requires at least one exposure")
    aligned = _aligned(y, *features)
    y = aligned[0]
    features = aligned[1:]
    out = np.full(y.shape, np.nan, dtype=float)
    for row in range(len(y.index)):
        out[row] = _ols_residual(
            y.iloc[row].to_numpy(dtype=float),
            [feature.iloc[row].to_numpy(dtype=float) for feature in features],
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
        )
    return _frame_like(y, out)


def cs_wls_resid(
    y: pd.DataFrame,
    x: pd.DataFrame,
    weight: pd.DataFrame,
    add_intercept: bool = True,
    min_obs: int = 5,
    **_: Any,
) -> pd.DataFrame:
    y, x, weight = _aligned(y, x, weight)
    out = np.full(y.shape, np.nan, dtype=float)
    for row in range(len(y.index)):
        out[row] = _ols_residual(
            y.iloc[row].to_numpy(dtype=float),
            [x.iloc[row].to_numpy(dtype=float)],
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
            weights=weight.iloc[row].to_numpy(dtype=float),
        )
    return _frame_like(y, out)


def _period_ordinal(value: Any):
    # Imported lazily: ``fiscal_strict`` transitively imports
    # ``cleaned_operators.overhaul.base`` (which registers ``period_lag`` under
    # its own source).  A top-level import here would make daily_panel load
    # after the overhaul layer and collide with this module's own registration.
    from cleaned_operators.fiscal_strict import period_ordinal

    return period_ordinal(value)


def _fiscal_ordered_insert(order: list[Any], key: Any) -> None:
    """Insert ``key`` into ``order`` keeping fiscal-ordinal sorted order.

    Mirrors ``transforms_v2._period_insert``: report periods advance by fiscal
    ordinal (``year*4 + quarter``), not first-appearance order.  A late-disclosed
    or back-filled older period (e.g. restated 2025Q2 arriving after 2025Q3)
    must not reorder the sequence that ``period_lag`` walks (review P0-02).
    Unparseable period ids keep first-seen order at the end.
    """
    target = _period_ordinal(key)
    if target is None:
        order.append(key)
        return
    for position, existing in enumerate(order):
        existing_ord = _period_ordinal(existing)
        if existing_ord is not None and existing_ord > target:
            order.insert(position, key)
            return
    order.append(key)


def period_lag(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    periods: int = 1,
    **_: Any,
) -> pd.DataFrame:
    x, period_id = _aligned(x, period_id)
    lag = _nonnegative_int(periods, "periods")
    xv = x.to_numpy()
    pv = period_id.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        order: list[Any] = []
        values: dict[Any, float] = {}
        for row in range(rows):
            period = pv[row, col]
            if pd.isna(period):
                continue
            try:
                key = period.item()
            except AttributeError:
                key = period
            if key not in order:
                _fiscal_ordered_insert(order, key)
            value = xv[row, col]
            if pd.notna(value):
                values[key] = float(value)
            cur_ord = _period_ordinal(key)
            if lag == 0:
                out[row, col] = float(value) if pd.notna(value) else np.nan
            elif cur_ord is not None:
                # Exact fiscal-ordinal lag: a skipped/missing fiscal period
                # yields NaN, never the value of a non-adjacent report
                # (same contract as transforms_v2._lag_value).
                target = cur_ord - lag
                hit: Any = None
                for k in reversed(order):
                    if _period_ordinal(k) == target:
                        hit = k
                        break
                out[row, col] = values.get(hit, np.nan) if hit is not None else np.nan
            else:
                # Unparseable period id: position-based lag in fiscal/appearance
                # sorted order (legacy behaviour preserved for these keys).
                if key in order:
                    target_pos = order.index(key) - lag
                    if target_pos >= 0:
                        out[row, col] = values.get(order[target_pos], np.nan)
    return _frame_like(x, out)


def _slope_tstat(y: np.ndarray, x: np.ndarray, add_intercept: bool) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    yv = y[mask]
    xv = x[mask]
    if yv.size < (3 if add_intercept else 2) or np.var(xv) <= 0:
        return np.nan
    design = xv[:, None]
    if add_intercept:
        design = np.column_stack((np.ones(xv.size), xv))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return np.nan
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    residual = yv - design @ beta
    dof = yv.size - design.shape[1]
    if dof <= 0:
        return np.nan
    sigma2 = float(residual @ residual) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    slope_var = sigma2 * xtx_inv[-1, -1]
    if not np.isfinite(slope_var) or slope_var <= 0:
        return np.nan
    return float(beta[-1] / np.sqrt(slope_var))


def ts_regression_tstat(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    add_intercept: bool = True,
    **_: Any,
) -> pd.DataFrame:
    y, x = _aligned(y, x)
    w = _positive_int(window, "window")
    mp = w if min_periods is None else _positive_int(min_periods, "min_periods")
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    rows, cols = yv.shape
    out = np.full_like(yv, np.nan)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            mask = np.isfinite(yv[start : row + 1, col]) & np.isfinite(
                xv[start : row + 1, col]
            )
            if int(mask.sum()) >= mp:
                out[row, col] = _slope_tstat(
                    yv[start : row + 1, col],
                    xv[start : row + 1, col],
                    bool(add_intercept),
                )
    return _frame_like(y, out)


def ts_trend_tstat(
    x: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    w = _positive_int(window, "window")
    mp = w if min_periods is None else _positive_int(min_periods, "min_periods")
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full_like(xv, np.nan)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            vals = xv[start : row + 1, col]
            valid = np.isfinite(vals)
            if int(valid.sum()) >= mp:
                time = np.arange(vals.size, dtype=float)
                out[row, col] = _slope_tstat(vals, time, True)
    return _frame_like(x, out)


def ts_max_drawdown(
    x: pd.DataFrame,
    window: int,
    min_periods: int = 2,
    **_: Any,
) -> pd.DataFrame:
    w = _positive_int(window, "window")
    mp = max(_positive_int(min_periods, "min_periods"), 2)

    def drawdown(values: np.ndarray) -> float:
        # Trailing contiguous valid price run — a NaN must not re-pair values
        # that were not temporally adjacent, and a NaN current row yields an
        # empty block (fail-closed) rather than the last valid history
        # (review P1-125).
        valid = _trailing_contiguous(values)
        if valid.size < mp or np.any(valid <= 0):
            return np.nan
        peaks = np.maximum.accumulate(valid)
        return float(np.min(valid / peaks - 1.0))

    return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, drawdown))


def ts_partial_corr(
    x: pd.DataFrame,
    y: pd.DataFrame,
    z: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    x, y, z = _aligned(x, y, z)
    w = _positive_int(window, "window")
    mp = w if min_periods is None else _positive_int(min_periods, "min_periods")
    xa, ya, za = (frame.to_numpy(dtype=float) for frame in (x, y, z))
    rows, cols = xa.shape
    out = np.full_like(xa, np.nan)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            xv, yv, zv = (
                arr[start : row + 1, col] for arr in (xa, ya, za)
            )
            mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv)
            if int(mask.sum()) < mp or np.var(zv[mask]) <= 0:
                continue
            design = np.column_stack((np.ones(mask.sum()), zv[mask]))
            rx = xv[mask] - design @ np.linalg.lstsq(design, xv[mask], rcond=None)[0]
            ry = yv[mask] - design @ np.linalg.lstsq(design, yv[mask], rcond=None)[0]
            if np.std(rx) > 0 and np.std(ry) > 0:
                out[row, col] = float(np.corrcoef(rx, ry)[0, 1])
    return _frame_like(x, out)


def ts_nth_value(
    x: pd.DataFrame,
    window: int,
    n: int = 1,
    order: str = "largest",
    min_periods: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    w = _positive_int(window, "window")
    nth = _positive_int(n, "n")
    mp = nth if min_periods is None else _positive_int(min_periods, "min_periods")
    direction = str(order).lower()
    if direction not in {"largest", "smallest"}:
        raise ValueError("order must be 'largest' or 'smallest'")

    def pick(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if valid.size < max(mp, nth):
            return np.nan
        valid.sort()
        return float(valid[-nth] if direction == "largest" else valid[nth - 1])

    return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, pick))


_OPERATORS: dict[str, tuple[str, list[str], Callable[..., pd.DataFrame], str]] = {
    "ts_count_if": ("time_series_condition", ["condition", "window", "min_periods"], ts_count_if, "滚动统计条件为真的次数"),
    "ts_sum_if": ("time_series_condition", ["x", "condition", "window", "min_periods"], ts_sum_if, "滚动条件求和"),
    "ts_mean_if": ("time_series_condition", ["x", "condition", "window", "min_periods"], ts_mean_if, "滚动条件均值"),
    "ts_std_if": ("time_series_condition", ["x", "condition", "window", "min_periods", "ddof"], ts_std_if, "滚动条件样本标准差"),
    "ts_last_if": ("time_series_event", ["x", "condition", "window"], ts_last_if, "窗口内最近一次条件成立时的值"),
    "ts_days_since": ("time_series_event", ["condition", "max_lookback"], ts_days_since, "距最近一次条件成立的交易行数"),
    "ts_true_streak": ("time_series_event", ["condition"], ts_true_streak, "截至当前连续条件成立长度"),
    "cs_bucket": ("cross_sectional", ["x", "buckets", "ascending"], cs_bucket, "按交易日横截面平均排名分桶"),
    "cs_multi_resid": ("cross_sectional_regression", ["y", "x1", "x2", "...", "add_intercept", "min_obs"], cs_multi_resid, "多变量横截面 OLS 残差"),
    "cs_wls_resid": ("cross_sectional_regression", ["y", "x", "weight", "add_intercept", "min_obs"], cs_wls_resid, "加权横截面回归残差"),
    # R4-22: ``period_lag`` is NOT registered here.  The canonical ``period_lag``
    # must resolve to the fiscal_strict implementation (exact fiscal-ordinal lag
    # with revision policy) registered by ``cleaned_operators.fiscal_strict``.
    # This module's legacy occurrence-order ``period_lag`` function (below) is
    # retained for reference only and must not act as a second canonical
    # semantic; the legacy helper ``_period_ordinal`` / ``_fiscal_ordered_insert``
    # back it and the fiscal-strict canonical.
    "ts_regression_tstat": ("time_series_regression", ["y", "x", "window", "min_periods", "add_intercept"], ts_regression_tstat, "滚动回归斜率 t 统计量"),
    "ts_trend_tstat": ("time_series_regression", ["x", "window", "min_periods"], ts_trend_tstat, "滚动时间趋势斜率 t 统计量"),
    "ts_max_drawdown": ("time_series_risk", ["x", "window", "min_periods"], ts_max_drawdown, "滚动窗口最大回撤"),
    "ts_partial_corr": ("time_series_regression", ["x", "y", "z", "window", "min_periods"], ts_partial_corr, "控制单一变量后的滚动偏相关"),
    "ts_nth_value": ("time_series_order", ["x", "window", "n", "order", "min_periods"], ts_nth_value, "滚动窗口第 N 大或第 N 小有效值"),
}


class _PandasDailyPanelOperator(PandasOperator):
    # R5-02: direct ``calculate`` that routes through validate_operator_call.
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, name: str, category: str, params: list[str], fn: Callable[..., pd.DataFrame], description: str):
        self._fn = fn
        self.metadata = PandasMetadata(
            name=name,
            category=category,
            description=description,
            examples=[],
            param_names=params,
            return_type="series",
            tags=["daily", "panel", "pit_safe", category],
        )

    def calculate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        from cleaned_operators.base import validate_operator_call

        processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
        return self._fn(*processed_args, **processed_kwargs)


def _pl_to_pandas(frame: Any) -> pd.DataFrame:
    return pd.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


class _PolarsDailyPanelOperator(PolarsOperator):
    # R5-02: direct ``calculate`` that routes through validate_operator_call.
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, name: str, category: str, params: list[str], fn: Callable[..., pd.DataFrame], description: str):
        self._fn = fn
        self.metadata = PolarsMetadata(
            name=name,
            category=category,
            description=description,
            examples=[],
            param_names=params,
            return_type="series",
            tags=["daily", "panel", "pit_safe", "polars", category],
        )

    def calculate(self, *args: Any, **kwargs: Any) -> Any:
        from cleaned_operators.base import validate_operator_call

        validate_operator_call(self, args, kwargs)
        if pl is None:
            raise ImportError("polars is required for the Polars daily-panel backend")
        pandas_args = [_pl_to_pandas(arg) if isinstance(arg, pl.DataFrame) else arg for arg in args]
        result = self._fn(*pandas_args, **kwargs)
        return pl.DataFrame({str(column): result[column].to_numpy() for column in result.columns})


for _name, (_category, _params, _fn, _description) in _OPERATORS.items():
    OperatorRegistry.register(
        _PandasDailyPanelOperator(_name, _category, _params, _fn, _description),
        canonical=_name,
        backend="pandas_numpy",
        source="daily_panel",
        backend_explicit=True,
    )
    OperatorRegistry.register(
        _PolarsDailyPanelOperator(_name, _category, _params, _fn, _description),
        canonical=_name,
        backend="polars",
        source="daily_panel_polars",
        backend_explicit=True,
    )
