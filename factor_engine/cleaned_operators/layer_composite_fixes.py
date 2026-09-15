# -*- coding: utf-8 -*-
"""Final native-Polars parity fixes and strict COS/fiscal layers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base import ParamSpec, ParamRole
from factor_engine.cleaned_operators.common.strict_params import strict_int

# Imported here because this module is deliberately the last runtime layer in
# cleaned_operators.load_all. Their registrations therefore override historical
# implementations without widening the bootstrap surface.
from factor_engine.cleaned_operators import fiscal_strict as _fiscal_strict  # noqa: F401
from factor_engine.cleaned_operators import layer_cos_fundamental as _cos_fundamental  # noqa: F401
from factor_engine.cleaned_operators import layer_weighted as _weighted  # noqa: F401
from factor_engine.cleaned_operators import layer_topk_compat as _topk_compat  # noqa: F401
from factor_engine.cleaned_operators.overhaul.base import (
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
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _assert_condition_bool(condition, name="condition"):
    from factor_engine.cleaned_operators.common.strict_params import strict_condition_bool
    values = condition.select(pl_cols(condition)).to_numpy() if pl is not None and isinstance(condition, pl.DataFrame) else condition
    strict_condition_bool(values, name)


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


def _days_since_op(cls, fn):
    op = cls("ts_days_since", "time_series_condition", ["condition", "max_lookback"],
        "distance to latest true observation; max_lookback is inclusive", fn,
        panel_params=("condition",), scalar_params=("max_lookback",),
        param_specs={"max_lookback": ParamSpec(dtype=int, min=1, default=None,
            param_role=ParamRole.HORIZON, history_semantics="max_rows",
            history_formula="max_lookback + 1")})
    if cls is PolarsFunctionOperator:
        from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
        op._physical_spec = PhysicalImplementationSpec(canonical="ts_days_since",backend="polars",
            execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
            stateful=True, requires_sorted=True,
            supports_nulls=True,supports_nan=True,supports_inf=True,
            implementation_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            kernel_identity="layer_composite_fixes.pl_days_since_inclusive",
            notes="CPU event-age state, explicit unknown reset; no Pandas panel conversion.")
    return op


def pd_days_since_inclusive(condition: pd.DataFrame, max_lookback=None, **_):
    """Distance to latest true observation; max_lookback is an inclusive distance.

    Condition is a ConditionBool ({0, 1} with NaN = unknown).  A NaN (unknown)
    event state at row ``t`` censors ``output[t]`` to NaN AND resets the running
    count — the previously known "last true" is no longer trustworthy, so it is
    reset until an explicit true (1) observation rebuilds it (daily_panel.ts_days_since
    semantics, kept here for the inclusive max_lookback variant).
    """
    _assert_condition_bool(condition)
    limit = None if max_lookback is None else strict_int(max_lookback, "max_lookback", minimum=1)
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
    limit = None if max_lookback is None else strict_int(max_lookback, "max_lookback", minimum=1)
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
    _days_since_op(PandasFunctionOperator, pd_days_since_inclusive),
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
        _days_since_op(PolarsFunctionOperator, pl_days_since_inclusive),
        canonical="ts_days_since",
        backend="polars",
        source="layer_governance_native_polars",
        status="production",
        backend_explicit=True,
        replace=True,
        replacement_reason="final layer fixes ts_days_since polars semantics (inclusive max_lookback)",
        expected_old_source="operator_overhaul_native_polars",
    )
