# -*- coding: utf-8 -*-
"""Final native-Polars parity fixes and strict COS/fiscal layers."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Imported here because this module is deliberately the last runtime layer in
# cleaned_operators.load_all. Their registrations therefore override historical
# implementations without widening the bootstrap surface.
from cleaned_operators import fiscal_strict as _fiscal_strict  # noqa: F401
from cleaned_operators import layer_cos_fundamental as _cos_fundamental  # noqa: F401
from cleaned_operators import layer_weighted as _weighted  # noqa: F401
from cleaned_operators import layer_topk_compat as _topk_compat  # noqa: F401
from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    pl_finite,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry


def _assert_condition_bool(condition: pd.DataFrame, name: str = "condition") -> None:
    """R11 #144: the condition input must be a ConditionBool.

    Accepted values: {0, 1} (or boolean True/False); NaN/null = missing and is
    excluded from selection.  Any other finite numeric value (e.g. 5.0, -3.0) is
    neither a probability nor a boolean — silently treating it as "truthy" is a
    hidden semantic the operator contract forbids.  Fail the call (raise) instead
    of guessing.

    Mirrors ``cleaned_operators.common.daily_panel._assert_condition_bool``.
    """
    cv = condition.to_numpy()
    finite = np.isfinite(cv)
    bad = finite & (cv != 0.0) & (cv != 1.0)
    if np.any(bad):
        raise ValueError(
            f"{name} must be a ConditionBool (values in {{0, 1}} with NaN as "
            f"missing); found {int(bad.sum())} finite value(s) outside {{0, 1}}"
        )


def pl_adx_strict(high, low, close, window=14, **_):
    """Wilder ADX with the same missing previous-close seed as pandas."""
    w = positive_int(window, "window")
    replacements = {}
    for col in [c for c in pl_cols(high) if c in low.columns and c in close.columns]:
        temp = pl.DataFrame({"h": high[col], "l": low[col], "c": close[col]})
        previous_close = pl.col("c").shift(1)
        raw_plus = pl.col("h") - pl.col("h").shift(1)
        raw_minus = pl.col("l").shift(1) - pl.col("l")
        true_range = pl.when(previous_close.is_null()).then(None).otherwise(
            pl.max_horizontal(
                pl.col("h") - pl.col("l"),
                (pl.col("h") - previous_close).abs(),
                (pl.col("l") - previous_close).abs(),
            )
        )
        plus = pl.when((raw_plus > raw_minus) & (raw_plus > 0)).then(raw_plus).otherwise(0.0)
        minus = pl.when((raw_minus > raw_plus) & (raw_minus > 0)).then(raw_minus).otherwise(0.0)
        staged = temp.with_columns(
            true_range.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("atr"),
            plus.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("pdm"),
            minus.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).alias("mdm"),
        ).with_columns(
            (
                100.0
                * pl.col("pdm")
                / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)
            ).alias("pdi"),
            (
                100.0
                * pl.col("mdm")
                / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)
            ).alias("mdi"),
        ).with_columns(
            (
                100.0
                * (pl.col("pdi") - pl.col("mdi")).abs()
                / pl.when((pl.col("pdi") + pl.col("mdi")).abs() > EPS)
                .then(pl.col("pdi") + pl.col("mdi"))
                .otherwise(None)
            ).alias("dx")
        )
        replacements[col] = staged.select(
            pl.col("dx")
            .ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w)
            .alias("value")
        )["value"]
    return pl_base_with(high, replacements)


def pd_days_since_inclusive(condition: pd.DataFrame, max_lookback=None, **_):
    """Distance to latest true observation; max_lookback is an inclusive distance.

    Condition is a ConditionBool ({0, 1} with NaN = unknown).  A NaN (unknown)
    event state at row ``t`` censors ``output[t]`` to NaN AND resets the running
    count — the previously known "last true" is no longer trustworthy, so it is
    reset until an explicit true (1) observation rebuilds it (daily_panel.ts_days_since
    semantics, kept here for the inclusive max_lookback variant).
    """
    _assert_condition_bool(condition)
    limit = None if max_lookback is None else positive_int(max_lookback, "max_lookback")
    values = condition.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        last = -1
        for row, value in enumerate(values[:, col]):
            if pd.isna(value):
                last = -1
                continue
            if bool(value):
                last = row
            if last >= 0:
                distance = row - last
                if limit is None or distance <= limit:
                    out[row, col] = float(distance)
    return frame_pd(condition, out)


def pl_days_since_inclusive(condition, max_lookback=None, **_):
    """Distance to latest true observation; max_lookback is an inclusive distance.

    ConditionBool semantics (see ``pd_days_since_inclusive``): a NaN condition
    censors output to NaN and resets the running count so a later known-true
    event restarts from 0.
    """
    _assert_condition_bool(condition)
    limit = None if max_lookback is None else positive_int(max_lookback, "max_lookback")
    replacements = {}
    for col in pl_cols(condition):
        arr = condition[col].cast(pl.Float64, strict=False).to_numpy()
        out, last = np.full(len(arr), np.nan, dtype=float), -1
        for row, value in enumerate(arr):
            if pd.isna(value):
                last = -1
                continue
            if bool(value):
                last = row
            if last >= 0:
                distance = row - last
                if limit is None or distance <= limit:
                    out[row] = float(distance)
        replacements[col] = pl.Series(col, out)
    return pl_base_with(condition, replacements)


OperatorRegistry.register(
    PandasFunctionOperator(
        "ts_days_since",
        "time_series_condition",
        ["condition", "max_lookback"],
        "distance to latest true observation; max_lookback is inclusive",
        pd_days_since_inclusive,
    ),
    canonical="ts_days_since",
    backend="pandas_numpy",
    source="layer_composite_fixes",
    status="production",
    backend_explicit=True,
    replace=True,
    replacement_reason="final layer fixes ts_days_since semantics (inclusive max_lookback)",
    expected_old_source="operator_overhaul_audited",
)

if pl is not None:
    OperatorRegistry.register(
        PolarsFunctionOperator(
            "ADX",
            "technical_signal",
            ["high", "low", "close", "window"],
            "Wilder ADX with strict pandas-compatible missing seed",
            pl_adx_strict,
        ),
        canonical="ADX",
        backend="polars",
        source="layer_governance_native_polars",
        status="production",
        backend_explicit=True,
        replace=True,
        replacement_reason="final layer fixes ADX polars parity (strict missing-seed)",
        expected_old_source="operator_overhaul_native_polars",
    )
    OperatorRegistry.register(
        PolarsFunctionOperator(
            "ts_days_since",
            "time_series_condition",
            ["condition", "max_lookback"],
            "distance to latest true observation; max_lookback is inclusive",
            pl_days_since_inclusive,
        ),
        canonical="ts_days_since",
        backend="polars",
        source="layer_governance_native_polars",
        status="production",
        backend_explicit=True,
        replace=True,
        replacement_reason="final layer fixes ts_days_since polars semantics (inclusive max_lookback)",
        expected_old_source="operator_overhaul_native_polars",
    )
