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
    """Strict alignment + float coercion (round-6 P0-25): never silently reindex."""
    if not frames:
        return []
    from cleaned_operators.alignment import align_panel_inputs

    aligned = align_panel_inputs(*frames, strict_axes=True)
    out: list[pd.DataFrame] = []
    for frame in aligned:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("production repair expects DataFrame panel inputs")
        out.append(frame.astype(float))
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
    replace=True,
    replacement_reason="canonical rolling-OLS contract overrides the overhaul compat layer (round-7 P0 chain pinning)",
    expected_old_source="operator_overhaul_compat",
)
class ProductionTSRegressionSlope(SeriesOperator):
    """Rolling OLS with the canonical and historical GTJA call contracts."""

    metadata = OperatorMetadata(
        name="ts_regression_slope",
        category="time_series",
        description=(
            "Pairwise-finite rolling OLS. The canonical keeps the historical "
            "ts_regression(y, x, window, lag, retval, min_periods=...) contract."
        ),
        param_names=[
            "y",
            "x",
            "window",
            "add_intercept",
        ],
        tags=["pit_safe", "causal", "rolling_regression", "production_repair"],
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame,
        window: int,
        *legacy_args,
        lag: int | None = None,
        retval: str | None = None,
        min_periods: int | None = None,
        add_intercept: bool = True,
        **_,
    ) -> pd.DataFrame:
        from cleaned_operators.overhaul.compat import ts_regression_compat

        return ts_regression_compat(
            y,
            x,
            window,
            *legacy_args,
            lag=lag,
            retval=retval,
            min_periods=min_periods,
            add_intercept=add_intercept,
        )
