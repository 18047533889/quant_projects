# -*- coding: utf-8 -*-
"""Reviewed production replacements for legacy compatibility kernels.

These implementations intentionally re-register the same canonical/backend late
in registry bootstrap. ``OperatorRegistry.register`` replaces only the physical
Pandas/Numpy implementation while preserving the established canonical metadata
and aliases. Keeping the repairs in a dedicated source file makes evidence and
future code review explicit instead of hiding production semantics in a monkey
patch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _aligned_float_frames(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    if not frames:
        return []
    index = frames[0].index
    columns = frames[0].columns
    out: list[pd.DataFrame] = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("production repair expects DataFrame panel inputs")
        out.append(frame.reindex(index=index, columns=columns).astype(float))
    return out


@register_operator(
    name="ts_max_buildup",
    canonical="ts_max_buildup",
    category="time_series",
    source="production_repairs",
    backend="pandas_numpy",
    status="production",
)
class ProductionTSMaxBuildup(SeriesOperator):
    """Trailing-window count of record highs, with no full-history dependence."""

    metadata = OperatorMetadata(
        name="ts_max_buildup",
        category="time_series",
        description=(
            "Count record-high observations independently inside each trailing window. "
            "The record maximum resets at every output timestamp, so appending future "
            "rows can never alter past outputs."
        ),
        param_names=["x", "d"],
        tags=["pit_safe", "causal", "rolling", "production_repair"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        window = int(d)
        if window <= 0:
            raise ValueError("ts_max_buildup d must be a positive integer")
        (frame,) = _aligned_float_frames(x)
        values = frame.to_numpy(dtype=float)
        result = np.full(values.shape, np.nan, dtype=float)

        # O(T*W*N), bounded by the declared rolling window. The legacy kernel was
        # O(T*N) but semantically wrong because it accumulated over the complete
        # input and used total input length to decide which historical rows to
        # zero. Correctness is preferred; a later Numba/Polars optimization can
        # preserve this exact golden semantics.
        rows, cols = values.shape
        for col in range(cols):
            series = values[:, col]
            for end in range(rows):
                start = max(0, end - window + 1)
                segment = series[start:end + 1]
                current_max = -np.inf
                count = 0
                has_finite = False
                for value in segment:
                    if not np.isfinite(value):
                        continue
                    has_finite = True
                    if value >= current_max:
                        current_max = value
                        count += 1
                if has_finite:
                    result[end, col] = float(count)
        return pd.DataFrame(result, index=frame.index, columns=frame.columns)


@register_operator(
    name="digital_count",
    canonical="digital_count",
    category="time_series",
    source="production_repairs",
    backend="pandas_numpy",
    status="production",
)
class ProductionDigitalCount(SeriesOperator):
    """Causal rolling count of qualifying small-change runs."""

    metadata = OperatorMetadata(
        name="digital_count",
        category="time_series",
        description=(
            "Length of the current consecutive run whose absolute one-step change is "
            "within threshold, capped by d; values below the requested run length are 0."
        ),
        param_names=["x", "d", "threshold", "run"],
        tags=["pit_safe", "causal", "stateful", "production_repair"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        d: int,
        threshold: float,
        run: int,
    ) -> pd.DataFrame:
        lookback = int(d)
        minimum_run = int(run)
        threshold = float(threshold)
        if lookback <= 0:
            raise ValueError("digital_count d must be positive")
        if minimum_run <= 0:
            raise ValueError("digital_count run must be positive")
        if threshold < 0 or not np.isfinite(threshold):
            raise ValueError("digital_count threshold must be finite and non-negative")

        (frame,) = _aligned_float_frames(x)
        values = frame.to_numpy(dtype=float)
        out = np.zeros(values.shape, dtype=float)
        rows, cols = values.shape
        for col in range(cols):
            streak = 0
            previous = np.nan
            for i in range(rows):
                value = values[i, col]
                if i == 0 or not np.isfinite(value) or not np.isfinite(previous) or previous == 0:
                    streak = 0
                else:
                    change = abs(value / previous - 1.0)
                    streak = min(lookback, streak + 1) if change <= threshold else 0
                out[i, col] = float(streak if streak >= minimum_run else 0)
                previous = value
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)


@register_operator(
    name="ts_regression_slope",
    canonical="ts_regression_slope",
    category="time_series",
    source="production_repairs",
    backend="pandas_numpy",
    status="production",
)
class ProductionTSRegressionSlope(SeriesOperator):
    """Rolling OLS slope with an explicit intercept contract."""

    metadata = OperatorMetadata(
        name="ts_regression_slope",
        category="time_series",
        description=(
            "Rolling OLS slope of y on x. add_intercept=True estimates the standard "
            "intercept+slope regression; False fits through the origin."
        ),
        param_names=["x", "y", "window", "add_intercept"],
        tags=["pit_safe", "causal", "rolling_regression", "production_repair"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        window: int,
        add_intercept: bool = True,
    ) -> pd.DataFrame:
        lookback = int(window)
        if lookback <= 0:
            raise ValueError("ts_regression_slope window must be positive")
        if not isinstance(add_intercept, (bool, np.bool_)):
            raise TypeError("ts_regression_slope add_intercept must be boolean")
        x_frame, y_frame = _aligned_float_frames(x, y)
        xv = x_frame.to_numpy(dtype=float)
        yv = y_frame.to_numpy(dtype=float)
        result = np.full(xv.shape, np.nan, dtype=float)
        rows, cols = xv.shape

        for col in range(cols):
            for end in range(lookback - 1, rows):
                start = end - lookback + 1
                xs = xv[start:end + 1, col]
                ys = yv[start:end + 1, col]
                valid = np.isfinite(xs) & np.isfinite(ys)
                if valid.sum() < 3:
                    continue
                xs = xs[valid]
                ys = ys[valid]
                if bool(add_intercept):
                    centered = xs - xs.mean()
                    denom = float(np.dot(centered, centered))
                    if denom <= 0:
                        continue
                    slope = float(np.dot(centered, ys - ys.mean()) / denom)
                else:
                    denom = float(np.dot(xs, xs))
                    if denom <= 0:
                        continue
                    slope = float(np.dot(xs, ys) / denom)
                result[end, col] = slope
        return pd.DataFrame(result, index=x_frame.index, columns=x_frame.columns)
