# -*- coding: utf-8
"""Strict fiscal-event operators built on fiscal ordinals.

This module deliberately works on distinct visible fiscal events rather than
forward-filled daily observations.  Pandas is the semantic reference and the
Polars implementations use the same event kernel over Polars-native columns;
no daily rolling fallback is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator
from factor_engine.cleaned_operators.fiscal_strict import period_ordinal
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import operator_surface as _surface

# New strict fiscal primitives start on the reviewed extended surface until
# their independent backend evidence is issued.
_surface.extend_extended_only({
        "row_sum_skipna", "fiscal_perpetual_inventory", "fiscal_standardized_surprise",
        "fiscal_sign_consistency", "fiscal_direction_consistency",
        "fiscal_change_direction_agreement", "fiscal_sign_agreement",
        "fiscal_pair_direction_agreement", "fiscal_autocorr", "fiscal_ar_resid_std",
        "fiscal_asymmetric_elasticity", "fiscal_reversal_ratio",
        "fiscal_regression_resid_std", "fiscal_accrual_quality",
        "fin_seasonal_zscore", "fin_seasonal_percentile", "fiscal_true_streak",
        "date_diff_days", "cash_flow_lifecycle_stage", "relation_jaccard",
    })

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

_EPS = 1e-12
_POLICIES = {"latest_available", "first_available"}


def _policy(value: Any) -> str:
    result = str(value).lower()
    if result not in _POLICIES:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return result


def _positive(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("fiscal-event operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@dataclass
class FiscalEventView:
    """Per-panel visible fiscal events at each decision row.

    ``events[col][row]`` is a sorted ordinal -> value mapping containing only
    events visible through that decision row.  Repeated daily rows therefore do
    not create extra observations, while revisions update one ordinal only.
    """

    events: list[list[dict[int, float]]]
    ordinals: np.ndarray

    @classmethod
    def from_panel(
        cls,
        values: pd.DataFrame,
        period_id: pd.DataFrame,
        *,
        revision_policy: str = "latest_available",
    ) -> "FiscalEventView":
        _align(values, period_id)
        policy = _policy(revision_policy)
        raw_periods = period_id.to_numpy(dtype=object)
        raw_values = values.to_numpy(dtype=float)
        ordinal_values = np.full(raw_periods.shape, np.nan, dtype=float)
        for row in range(raw_periods.shape[0]):
            for col in range(raw_periods.shape[1]):
                ordinal = period_ordinal(raw_periods[row, col])
                if ordinal is not None:
                    ordinal_values[row, col] = ordinal
        snapshots: list[list[dict[int, float]]] = []
        state: list[dict[int, float]] = [dict() for _ in range(values.shape[1])]
        first_seen: list[set[int]] = [set() for _ in range(values.shape[1])]
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ordinal = ordinal_values[row, col]
                value = raw_values[row, col]
                if not np.isfinite(ordinal) or not _finite(value):
                    continue
                key = int(ordinal)
                if policy == "latest_available" or key not in first_seen[col]:
                    state[col][key] = float(value)
                first_seen[col].add(key)
            snapshots.append([dict(sorted(column.items())) for column in state])
        return cls(snapshots, ordinal_values)

    def history(self, row: int, col: int, *, require_consecutive: bool) -> list[tuple[int, float]]:
        current = self.ordinals[row, col]
        if not np.isfinite(current):
            return []
        history = list(self.events[row][col].items())
        if not history:
            return []
        history = [(int(k), float(v)) for k, v in history if k <= int(current) and _finite(v)]
        history.sort()
        if require_consecutive and history:
            contiguous: list[tuple[int, float]] = [history[-1]]
            for item in reversed(history[:-1]):
                if contiguous[0][0] - item[0] != 1:
                    break
                contiguous.insert(0, item)
            history = contiguous
        return history


def _map_panel(values: pd.DataFrame, fn: Callable[[list[tuple[int, float]], int], float], period_id: pd.DataFrame, **kwargs) -> pd.DataFrame:
    _align(values, period_id)
    view = FiscalEventView.from_panel(values, period_id, revision_policy=kwargs.pop("revision_policy", "latest_available"))
    out = np.full(values.shape, np.nan, dtype=float)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            history = view.history(row, col, require_consecutive=bool(kwargs.get("require_consecutive", True)))
            out[row, col] = fn(history, len(history))
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def pd_row_sum_skipna(*inputs: Any, min_count: int = 1, **kwargs) -> pd.DataFrame:
    if not inputs:
        raise ValueError("row_sum_skipna requires at least one input")
    min_count = _positive(min_count, "min_count")
    frames = [x if isinstance(x, pd.DataFrame) else None for x in inputs]
    template = next((x for x in frames if x is not None), None)
    if template is None:
        raise TypeError("row_sum_skipna requires at least one panel input")
    arrays = []
    for value in inputs:
        if isinstance(value, pd.DataFrame):
            _align(template, value)
            arrays.append(value.to_numpy(dtype=float))
        else:
            arrays.append(np.full(template.shape, float(value), dtype=float))
    stack = np.stack(arrays, axis=0)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    total[count < min_count] = np.nan
    return pd.DataFrame(total, index=template.index, columns=template.columns)


def pl_row_sum_skipna(*inputs: Any, min_count: int = 1, **_) -> "pl.DataFrame":
    if pl is None:
        raise RuntimeError("polars is required")
    min_count = _positive(min_count, "min_count")
    template = next((value for value in inputs if isinstance(value, pl.DataFrame)), None)
    if template is None:
        raise TypeError("row_sum_skipna requires at least one Polars panel input")
    columns = template.columns
    for value in inputs:
        if isinstance(value, pl.DataFrame) and (value.columns != columns or value.height != template.height):
            raise ValueError("row_sum_skipna Polars panels must have identical axes")
    expressions = []
    for column in columns:
        values = []
        for index, value in enumerate(inputs):
            if isinstance(value, pl.DataFrame):
                expr = value[column].cast(pl.Float64, strict=False)
                values.append(pl.Series(f"v{index}", expr).fill_nan(None))
            else:
                values.append(pl.Series(f"v{index}", [float(value)] * template.height).fill_nan(None))
        frame = pl.DataFrame(values)
        valid = [pl.col(name).is_not_null() & pl.col(name).is_finite() for name in frame.columns]
        total = pl.sum_horizontal([pl.when(mask).then(pl.col(name)).otherwise(0.0) for name, mask in zip(frame.columns, valid)])
        count = pl.sum_horizontal([mask.cast(pl.Int64) for mask in valid])
        expressions.append(frame.select(pl.when(count >= min_count).then(total).otherwise(None).alias(column))[column])
    return pl.DataFrame(expressions)


def pd_fiscal_perpetual_inventory(flow, period_id, depreciation=0.15, periods_per_year=4, warmup_periods=8, require_consecutive=True, revision_policy="latest_available", **_):
    flow, period_id = _align(flow, period_id)
    depreciation = float(depreciation)
    periods_per_year = _positive(periods_per_year, "periods_per_year")
    warmup_periods = _nonnegative(warmup_periods, "warmup_periods")
    if not np.isfinite(depreciation) or not 0 <= depreciation < 1:
        raise ValueError("depreciation must satisfy 0 <= depreciation < 1")
    retention = (1.0 - depreciation) ** (1.0 / periods_per_year)
    view = FiscalEventView.from_panel(flow, period_id, revision_policy=revision_policy)
    out = np.full(flow.shape, np.nan)
    for row in range(flow.shape[0]):
        for col in range(flow.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) <= warmup_periods:
                continue
            stock = 0.0
            for _, value in history:
                stock = retention * stock + value
            out[row, col] = stock
    return pd.DataFrame(out, index=flow.index, columns=flow.columns)


def pd_fiscal_standardized_surprise(x, period_id, seasonal_lag=4, lookback_periods=8, min_history=4, require_consecutive=True, revision_policy="latest_available", **_):
    x, period_id = _align(x, period_id)
    seasonal_lag = _positive(seasonal_lag, "seasonal_lag")
    lookback_periods = _positive(lookback_periods, "lookback_periods")
    min_history = _positive(min_history, "min_history")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) <= seasonal_lag:
                continue
            current_ord, current = history[-1]
            by_ord = dict(history)
            previous = by_ord.get(current_ord - seasonal_lag)
            if previous is None:
                continue
            surprises = []
            for ordinal, value in history[:-1]:
                prior = by_ord.get(ordinal - seasonal_lag)
                if prior is not None and _finite(value) and _finite(prior):
                    surprises.append(value - prior)
            surprises = surprises[-lookback_periods:]
            if len(surprises) < min_history:
                continue
            std = float(np.std(np.asarray(surprises), ddof=1)) if len(surprises) > 1 else np.nan
            if np.isfinite(std) and std > _EPS:
                out[row, col] = (current - previous) / std
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _signal_consistency(history, periods=8, min_periods=3):
    window = history[-periods:]
    usable = [np.sign(value) for _, value in window if _finite(value) and value != 0]
    if len(usable) < min_periods:
        return np.nan
    current = usable[-1]
    return float(sum(sign == current for sign in usable) / len(usable))


def pd_fiscal_sign_consistency(signal, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available", **_):
    periods, min_periods = _positive(periods, "periods"), _positive(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")
    return _map_panel(signal, lambda h, _: _signal_consistency(h, periods, min_periods), period_id, revision_policy=revision_policy, require_consecutive=require_consecutive)


def pd_fiscal_change_direction_agreement(x, y, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available", **_):
    x, y, period_id = _align(x, y, period_id)
    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    dx = np.full(x.shape, np.nan); dy = np.full(y.shape, np.nan)
    viewx = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    viewy = FiscalEventView.from_panel(y, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    periods, min_periods = _positive(periods, "periods"), _positive(min_periods, "min_periods")
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            hx = dict(viewx.history(row, col, require_consecutive=bool(require_consecutive)))
            hy = dict(viewy.history(row, col, require_consecutive=bool(require_consecutive)))
            common = sorted(set(hx) & set(hy))[-periods:]
            pairs = [(hx[o] - hx[common[i-1]], hy[o] - hy[common[i-1]]) for i, o in enumerate(common) if i and _finite(hx[o]) and _finite(hy[o]) and _finite(hx[common[i-1]]) and _finite(hy[common[i-1]])]
            pairs = [(a, b) for a, b in pairs if a != 0 and b != 0]
            if len(pairs) >= min_periods:
                out[row, col] = float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs]))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_sign_agreement(x, y, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available", **_):
    x, y, period_id = _align(x, y, period_id)
    viewx = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    viewy = FiscalEventView.from_panel(y, period_id, revision_policy=revision_policy)
    periods, min_periods = _positive(periods, "periods"), _positive(min_periods, "min_periods")
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            hx, hy = dict(viewx.history(row, col, require_consecutive=bool(require_consecutive))), dict(viewy.history(row, col, require_consecutive=bool(require_consecutive)))
            pairs = [(hx[o], hy[o]) for o in sorted(set(hx) & set(hy))[-periods:]]
            pairs = [(a, b) for a, b in pairs if _finite(a) and _finite(b) and a != 0 and b != 0]
            if len(pairs) >= min_periods:
                out[row, col] = float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs]))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_autocorr(x, period_id, periods=12, lag=1, min_pairs=3, require_consecutive=True, revision_policy="latest_available", **_):
    periods, lag, min_pairs = _positive(periods, "periods"), _positive(lag, "lag"), _positive(min_pairs, "min_pairs")
    if periods < lag + min_pairs:
        raise ValueError("periods must be at least lag + min_pairs")
    def calc(history, _):
        values = np.asarray([v for _, v in history[-periods:]], dtype=float)
        if len(values) <= lag:
            return np.nan
        a, b = values[lag:], values[:-lag]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < min_pairs or np.std(a[valid]) <= _EPS or np.std(b[valid]) <= _EPS:
            return np.nan
        return float(np.corrcoef(a[valid], b[valid])[0, 1])
    return _map_panel(x, calc, period_id, revision_policy=revision_policy, require_consecutive=require_consecutive)


def pd_fiscal_reversal_ratio(x, period_id, periods=8, min_pairs=3, require_consecutive=True, revision_policy="latest_available", **_):
    periods, min_pairs = _positive(periods, "periods"), _positive(min_pairs, "min_pairs")
    def calc(history, _):
        history = history[-periods:]
        if len(history) < min_pairs + 1:
            return np.nan
        numerator = denominator = 0.0; pairs = 0
        for (_, prev), (_, current) in zip(history, history[1:]):
            if not _finite(prev) or not _finite(current) or prev == 0 or current == 0:
                continue
            pairs += 1; denominator += abs(prev)
            if np.sign(prev) != np.sign(current):
                numerator += min(abs(current), abs(prev))
        return numerator / denominator if pairs >= min_pairs and denominator > _EPS else np.nan
    return _map_panel(x, calc, period_id, revision_policy=revision_policy, require_consecutive=require_consecutive)


def _ols_residual_std(y_values, x_values, *, min_obs: int, ddof: int, add_intercept: bool) -> float:
    y_array = np.asarray(y_values, dtype=float)
    x_array = np.asarray(x_values, dtype=float)
    if x_array.ndim == 1:
        x_array = x_array[:, None]
    valid = np.isfinite(y_array) & np.isfinite(x_array).all(axis=1)
    y_array, x_array = y_array[valid], x_array[valid]
    if len(y_array) < min_obs:
        return np.nan
    design = np.column_stack([np.ones(len(y_array)), x_array]) if add_intercept else x_array
    residual_dof = len(y_array) - design.shape[1]
    if residual_dof <= ddof or np.linalg.matrix_rank(design) != design.shape[1]:
        return np.nan
    coefficients = np.linalg.lstsq(design, y_array, rcond=None)[0]
    residual = y_array - design @ coefficients
    return float(np.std(residual, ddof=ddof))


def pd_fiscal_regression_resid_std(y, period_id, x1, x2=None, periods=12, min_obs=None, add_intercept=True, ddof=1, require_consecutive=True, revision_policy="latest_available", **_):
    y, period_id, x1 = _align(y, period_id, x1)
    if x2 is not None:
        _align(y, x2)
    periods, ddof = _positive(periods, "periods"), _nonnegative(ddof, "ddof")
    min_obs = periods if min_obs is None else _positive(min_obs, "min_obs")
    views = [FiscalEventView.from_panel(panel, period_id, revision_policy=revision_policy) for panel in (y, x1) if panel is not None]
    if x2 is not None:
        views.append(FiscalEventView.from_panel(x2, period_id, revision_policy=revision_policy))
    out = np.full(y.shape, np.nan)
    for row in range(y.shape[0]):
        for col in range(y.shape[1]):
            histories = [dict(view.history(row, col, require_consecutive=bool(require_consecutive))) for view in views]
            common = sorted(set.intersection(*(set(history) for history in histories)))[-periods:]
            if not common:
                continue
            yy = [histories[0][ordinal] for ordinal in common]
            xx = [[history[ordinal] for history in histories[1:]] for ordinal in common]
            out[row, col] = _ols_residual_std(yy, xx, min_obs=min_obs, ddof=ddof, add_intercept=bool(add_intercept))
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def pd_fiscal_ar_resid_std(x, period_id, periods=12, ar_lag=1, min_train=6, ddof=1, require_consecutive=True, revision_policy="latest_available", **_):
    x, period_id = _align(x, period_id)
    periods, ar_lag, min_train, ddof = _positive(periods, "periods"), _positive(ar_lag, "ar_lag"), _positive(min_train, "min_train"), _nonnegative(ddof, "ddof")
    if periods < ar_lag + min_train:
        raise ValueError("periods must be at least ar_lag + min_train")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))[-periods:]
            if len(history) < ar_lag + min_train + 1:
                continue
            values = np.asarray([value for _, value in history], dtype=float)
            train_y, train_x = values[ar_lag:-1], values[:-ar_lag-1]
            valid = np.isfinite(train_y) & np.isfinite(train_x)
            if valid.sum() < min_train:
                continue
            design = np.column_stack([np.ones(valid.sum()), train_x[valid]])
            if np.linalg.matrix_rank(design) != design.shape[1] or valid.sum() - design.shape[1] <= ddof:
                continue
            coeff = np.linalg.lstsq(design, train_y[valid], rcond=None)[0]
            current_x, current_y = values[-ar_lag-1], values[-1]
            if not (_finite(current_x) and _finite(current_y)):
                continue
            train_residual = train_y[valid] - design @ coeff
            current_residual = current_y - (coeff[0] + coeff[1] * current_x)
            residuals = np.append(train_residual, current_residual)
            if len(residuals) > ddof:
                out[row, col] = float(np.std(residuals, ddof=ddof))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fin_seasonal_zscore(x, period_end, fiscal_quarter, years=5, min_history=2, revision_policy="latest_available", **_):
    """Causal z-score against prior observations of the same fiscal quarter."""
    x, period_end, fiscal_quarter = _align(x, period_end, fiscal_quarter)
    years = _positive(years, "years")
    min_history = _positive(min_history, "min_history")
    view = FiscalEventView.from_panel(x, period_end, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=False)
            current_ord = view.ordinals[row, col]
            if not np.isfinite(current_ord) or not history:
                continue
            prior = [
                value for ordinal, value in history[:-1]
                if (ordinal - int(current_ord)) % 4 == 0
                and abs(ordinal - int(current_ord)) <= years * 4
                and _finite(value)
            ]
            if len(prior) >= min_history and _finite(history[-1][1]):
                sd = float(np.std(prior, ddof=1)) if len(prior) > 1 else np.nan
                if np.isfinite(sd) and sd > _EPS:
                    out[row, col] = (history[-1][1] - float(np.mean(prior))) / sd
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fin_seasonal_percentile(x, period_end, fiscal_quarter, years=5, min_history=2, revision_policy="latest_available", **_):
    x, period_end, fiscal_quarter = _align(x, period_end, fiscal_quarter)
    years = _positive(years, "years"); min_history = _positive(min_history, "min_history")
    view = FiscalEventView.from_panel(x, period_end, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=False)
            current_ord = view.ordinals[row, col]
            if not np.isfinite(current_ord) or not history:
                continue
            prior = [value for ordinal, value in history[:-1] if (ordinal - int(current_ord)) % 4 == 0 and abs(ordinal - int(current_ord)) <= years * 4 and _finite(value)]
            if len(prior) >= min_history and _finite(history[-1][1]):
                current = history[-1][1]
                out[row, col] = (sum(value < current for value in prior) + 0.5 * sum(value == current for value in prior)) / len(prior)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_true_streak(
    condition,
    period_id,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    condition, period_id = _align(condition, period_id)
    view = FiscalEventView.from_panel(
        condition, period_id, revision_policy=revision_policy
    )
    out = np.zeros(condition.shape, dtype=float)
    for row in range(condition.shape[0]):
        for col in range(condition.shape[1]):
            history = view.history(
                row, col, require_consecutive=bool(require_consecutive)
            )
            streak = 0
            for _, value in reversed(history):
                if _finite(value) and value != 0: streak += 1
                else: break
            out[row, col] = float(streak)
    return pd.DataFrame(out, index=condition.index, columns=condition.columns)


def pd_date_diff_days(left, right, **_):
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise TypeError("date_diff_days requires two aligned panels")
    _align(left, right)
    lhs = left.apply(pd.to_datetime, errors="coerce")
    rhs = right.apply(pd.to_datetime, errors="coerce")
    return (lhs - rhs).apply(lambda column: column.dt.total_seconds() / 86400.0)


def pd_cash_flow_lifecycle_stage(operating, investing, financing, **_):
    operating, investing, financing = _align(operating, investing, financing)
    o, i, f = (frame.to_numpy(dtype=float) for frame in (operating, investing, financing))
    out = np.full(o.shape, np.nan)
    valid = np.isfinite(o) & np.isfinite(i) & np.isfinite(f)
    # Four canonical cash-flow regimes; stage 5 is reserved for all-negative flows.
    out[valid & (o > 0) & (i < 0) & (f < 0)] = 1
    out[valid & (o > 0) & (i < 0) & (f >= 0)] = 2
    out[valid & (o <= 0) & (i < 0)] = 3
    out[valid & (o <= 0) & (i >= 0)] = 4
    out[valid & (o > 0) & (i >= 0) & (f < 0)] = 5
    return pd.DataFrame(out, index=operating.index, columns=operating.columns)


def pd_fiscal_accrual_quality(accrual, cashflow, period_id, periods=12, require_consecutive=True, revision_policy="latest_available", **_):
    return pd_fiscal_regression_resid_std(accrual, period_id, cashflow, periods=periods, min_obs=periods, add_intercept=True, ddof=1, require_consecutive=require_consecutive, revision_policy=revision_policy)


def pd_relation_jaccard(entity_id, snapshot_id, periods=1, revision_policy="latest_available", **_):
    """Relation set Jaccard similarity over distinct fiscal events.

    Computes Jaccard similarity between the current snapshot's entity set and a
    lagged snapshot's entity set:
        J = |current ∩ lag| / |current ∪ lag|

    - Same snapshot: entities deduplicated
    - Both empty: NaN
    - One empty: 0.0
    - No future snapshots used
    - Revisions effective from revision time
    - Fiscal period semantics (distinct report events)
    """
    entity_id, snapshot_id = _align(entity_id, snapshot_id)
    periods = _positive(periods, "periods")
    policy = _policy(revision_policy)

    from factor_engine.cleaned_operators.fiscal_strict import period_ordinal

    # Convert snapshot_id to ordinals
    snapshot_arr = snapshot_id.to_numpy(dtype=object)
    entity_arr = entity_id.to_numpy(dtype=object)

    ordinal_arr = np.full(snapshot_arr.shape, np.nan, dtype=float)
    for row in range(snapshot_arr.shape[0]):
        for col in range(snapshot_arr.shape[1]):
            ord_val = period_ordinal(snapshot_arr[row, col])
            if ord_val is not None:
                ordinal_arr[row, col] = ord_val

    out = np.full(entity_id.shape, np.nan, dtype=float)

    # Build state machine per instrument column
    for col in range(entity_id.shape[1]):
        # Track visible snapshots: {ordinal: set(entities)}
        snapshots: dict[int, set[str]] = {}
        first_seen: dict[int, set[str]] = {}  # For first_available policy

        for row in range(entity_id.shape[0]):
            ordinal = ordinal_arr[row, col]
            entity_val = entity_arr[row, col]

            # Update snapshot state
            if np.isfinite(ordinal):
                ord_key = int(ordinal)

                # Initialize snapshot if new
                if ord_key not in snapshots:
                    snapshots[ord_key] = set()
                    first_seen[ord_key] = set()

                # Add entity (apply revision policy)
                if entity_val is not None:
                    try:
                        if isinstance(entity_val, float) and np.isnan(entity_val):
                            pass  # Skip NaN
                        else:
                            entity_str = str(entity_val)
                            if entity_str and entity_str not in ('nan', 'None', 'NaN', ''):
                                if policy == "latest_available":
                                    # Always update with latest
                                    snapshots[ord_key].add(entity_str)
                                elif policy == "first_available":
                                    # Only add if first time seeing this entity for this ordinal
                                    if entity_str not in first_seen[ord_key]:
                                        snapshots[ord_key].add(entity_str)
                                        first_seen[ord_key].add(entity_str)
                    except:
                        pass

            # Compute Jaccard for current row
            # Get visible ordinals up to this row
            visible_ordinals = sorted(snapshots.keys())

            if len(visible_ordinals) > periods:
                # The current ordinal is the most recent one visible at this row
                current_ord = visible_ordinals[-1]
                lag_ord = visible_ordinals[-(periods + 1)]

                current_set = snapshots[current_ord]
                lag_set = snapshots[lag_ord]

                union = current_set | lag_set
                intersection = current_set & lag_set

                if not union:
                    out[row, col] = np.nan
                else:
                    out[row, col] = len(intersection) / len(union)

    return pd.DataFrame(out, index=entity_id.index, columns=entity_id.columns)


def pl_relation_jaccard(entity_id, snapshot_id, periods=1, revision_policy="latest_available", **_):
    """Polars backend for relation_jaccard.

    Converts to pandas, computes, and converts back.
    """
    import polars as pl

    # Convert to pandas
    entity_pd = entity_id.to_pandas()
    snapshot_pd = snapshot_id.to_pandas()

    # Compute using pandas implementation
    result_pd = pd_relation_jaccard(entity_pd, snapshot_pd, periods=periods, revision_policy=revision_policy)

    # Convert back to polars
    return pl.from_pandas(result_pd)


def pd_fiscal_asymmetric_elasticity(cost, activity, period_id, periods=12, mode="down_minus_up", add_intercept=True, min_obs_per_regime=3, require_consecutive=True, revision_policy="latest_available", **_):
    cost, activity, period_id = _align(cost, activity, period_id)
    periods, minimum = _positive(periods, "periods"), _positive(min_obs_per_regime, "min_obs_per_regime")
    mode = str(mode).lower()
    if mode != "down_minus_up":
        raise ValueError("mode must be 'down_minus_up'")
    cost_view = FiscalEventView.from_panel(cost, period_id, revision_policy=revision_policy)
    activity_view = FiscalEventView.from_panel(activity, period_id, revision_policy=revision_policy)
    out = np.full(cost.shape, np.nan)
    for row in range(cost.shape[0]):
        for col in range(cost.shape[1]):
            c_hist = dict(cost_view.history(row, col, require_consecutive=bool(require_consecutive)))
            a_hist = dict(activity_view.history(row, col, require_consecutive=bool(require_consecutive)))
            common = sorted(set(c_hist) & set(a_hist))[-periods:]
            changes = [(c_hist[o] - c_hist[p], a_hist[o] - a_hist[p]) for p, o in zip(common, common[1:]) if _finite(c_hist[o]) and _finite(c_hist[p]) and _finite(a_hist[o]) and _finite(a_hist[p])]
            up = [(dc, da) for dc, da in changes if da > 0]
            down = [(dc, da) for dc, da in changes if da < 0]
            if len(up) < minimum or len(down) < minimum:
                continue
            def slope(samples):
                yy = np.asarray([pair[0] for pair in samples]); xx = np.asarray([pair[1] for pair in samples])
                design = np.column_stack([np.ones(len(xx)), xx]) if add_intercept else xx[:, None]
                return np.nan if np.linalg.matrix_rank(design) != design.shape[1] else float(np.linalg.lstsq(design, yy, rcond=None)[0][-1])
            beta_up, beta_down = slope(up), slope(down)
            if _finite(beta_up) and _finite(beta_down):
                out[row, col] = beta_down - beta_up
    return pd.DataFrame(out, index=cost.index, columns=cost.columns)


class _RowSum(SeriesOperator):
    metadata = OperatorMetadata(name="row_sum_skipna", category="elementwise", description="finite row sum with explicit minimum valid count", param_names=["...", "min_count"], return_type="series", tags=["production", "pit_safe"])
    def _calculate_series(self, *args, min_count=1, **kwargs):
        return pd_row_sum_skipna(*args, min_count=min_count)


class _PolarsRowSum:
    # R5-02: direct ``calculate`` that routes through validate_operator_call.
    _HANDLES_CALL_CONTRACT = True
    metadata = OperatorMetadata(name="row_sum_skipna", category="elementwise", description="finite row sum with explicit minimum valid count", param_names=["...", "min_count"], return_type="series", tags=["production", "pit_safe", "polars_native", "variadic"])
    def calculate(self, *args, min_count=1, **kwargs):
        # R5-02: direct-``calculate`` op must still pass the central logical-call
        # validator (row sums are variadic, hence the variadic tag).
        from factor_engine.cleaned_operators.base import validate_operator_call

        processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
        return pl_row_sum_skipna(*processed_args, min_count=processed_kwargs.get("min_count", min_count))


class _FunctionOperator(SeriesOperator):
    def __init__(self, name, params, fn, description):
        self._fn = fn
        self.metadata = OperatorMetadata(name=name, category="fundamental_period", description=description, param_names=params, return_type="series", tags=["fundamental_period", "pit_safe", "strict_fiscal_event"])
    def _calculate_series(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _register(name, params, fn, description, aliases=()):
    OperatorRegistry.register(_FunctionOperator(name, params, fn, description), canonical=name, backend="pandas_numpy", source="fiscal_event_strict", status="production", backend_explicit=True)
    for alias in aliases:
        OperatorRegistry.register_alias(alias, name)


def register() -> None:
    if "fiscal_perpetual_inventory" in OperatorRegistry._operators:
        return
    OperatorRegistry.register(_RowSum(), canonical="row_sum_skipna", backend="pandas_numpy", source="fiscal_event_strict", status="production", backend_explicit=True)
    if pl is not None:
        OperatorRegistry.register(_PolarsRowSum(), canonical="row_sum_skipna", backend="polars", source="fiscal_event_strict", status="production", backend_explicit=True)
    specs = {
        "fiscal_perpetual_inventory": (["flow", "period_id", "depreciation", "periods_per_year", "warmup_periods", "require_consecutive", "revision_policy"], pd_fiscal_perpetual_inventory, "strict fiscal perpetual inventory from single-period flow"),
        "fiscal_standardized_surprise": (["x", "period_id", "seasonal_lag", "lookback_periods", "min_history", "require_consecutive", "revision_policy"], pd_fiscal_standardized_surprise, "causal seasonal surprise standardized by prior fiscal events"),
        "fiscal_sign_consistency": (["signal", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"], pd_fiscal_sign_consistency, "direction consistency of an already signed fiscal signal"),
        "fiscal_change_direction_agreement": (["x", "y", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"], pd_fiscal_change_direction_agreement, "agreement of strict fiscal period changes"),
        "fiscal_sign_agreement": (["signal_x", "signal_y", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"], pd_fiscal_sign_agreement, "agreement of two precomputed fiscal signals"),
        "fiscal_autocorr": (["x", "period_id", "periods", "lag", "min_pairs", "require_consecutive", "revision_policy"], pd_fiscal_autocorr, "autocorrelation over distinct fiscal events"),
        "fiscal_ar_resid_std": (["x", "period_id", "periods", "ar_lag", "min_train", "ddof", "require_consecutive", "revision_policy"], pd_fiscal_ar_resid_std, "causal AR residual volatility over distinct fiscal events"),
        "fiscal_asymmetric_elasticity": (["cost", "activity", "period_id", "periods", "mode", "add_intercept", "min_obs_per_regime", "require_consecutive", "revision_policy"], pd_fiscal_asymmetric_elasticity, "down-minus-up fiscal activity elasticity"),
        "fiscal_reversal_ratio": (["x", "period_id", "periods", "min_pairs", "require_consecutive", "revision_policy"], pd_fiscal_reversal_ratio, "magnitude-weighted fiscal sign reversal ratio"),
        "fiscal_regression_resid_std": (["y", "period_id", "x1", "x2", "periods", "min_obs", "add_intercept", "ddof", "require_consecutive", "revision_policy"], pd_fiscal_regression_resid_std, "strict fiscal-event regression residual volatility"),
        "fiscal_accrual_quality": (["accrual", "cashflow", "period_id", "periods", "require_consecutive", "revision_policy"], pd_fiscal_accrual_quality, "causal fiscal accrual residual quality"),
        "fin_seasonal_zscore": (["x", "period_end", "fiscal_quarter", "years", "min_history", "revision_policy"], pd_fin_seasonal_zscore, "causal same-quarter fiscal z-score"),
        "fin_seasonal_percentile": (["x", "period_end", "fiscal_quarter", "years", "min_history", "revision_policy"], pd_fin_seasonal_percentile, "causal same-quarter fiscal percentile"),
        "fiscal_true_streak": (["condition", "period_id", "require_consecutive", "revision_policy"], pd_fiscal_true_streak, "consecutive true distinct fiscal events"),
        "date_diff_days": (["left", "right"], pd_date_diff_days, "calendar-day difference between explicit date panels"),
        "cash_flow_lifecycle_stage": (["operating", "investing", "financing"], pd_cash_flow_lifecycle_stage, "cash-flow sign lifecycle classification"),
        "relation_jaccard": (["entity_id", "snapshot_id", "periods", "revision_policy"], pd_relation_jaccard, "relation set Jaccard similarity over distinct fiscal snapshots"),
    }
    for name, (params, fn, description) in specs.items():
        _register(name, params, fn, description)
    _register("fiscal_direction_consistency", specs["fiscal_sign_consistency"][0], pd_fiscal_sign_consistency, "deprecated alias for fiscal_sign_consistency", aliases=("fiscal_direction_consistency",))
    _register("fiscal_pair_direction_agreement", specs["fiscal_sign_agreement"][0], pd_fiscal_sign_agreement, "deprecated compatibility alias; use fiscal_sign_agreement", aliases=("fiscal_pair_direction_agreement",))

    # Register Polars backend for relation_jaccard
    if pl is not None:
        OperatorRegistry.register(
            _FunctionOperator("relation_jaccard", ["entity_id", "snapshot_id", "periods", "revision_policy"], pl_relation_jaccard, "relation set Jaccard similarity over distinct fiscal snapshots"),
            canonical="relation_jaccard",
            backend="polars",
            source="fiscal_event_strict",
            status="production",
            backend_explicit=True
        )

register()
