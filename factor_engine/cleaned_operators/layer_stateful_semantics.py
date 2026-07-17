# -*- coding: utf-8 -*-
"""Align full-history recursive operators with checkpoint missing policy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    PandasFunctionOperator,
    PolarsFunctionOperator,
    pl,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry

_APPLIED = False


def pd_ema_carry_emit_null(x: pd.DataFrame, span=12, window=None, **_) -> pd.DataFrame:
    period = positive_int(span if span is not None else window, "span")
    finite = pd.DataFrame(
        np.isfinite(x.to_numpy(dtype=float)), index=x.index, columns=x.columns
    )
    clean = x.where(finite)
    # ignore_na=True means a missing observation does not decay the state;
    # masking restores the contract's emit-null behavior on that row.
    return clean.ewm(span=period, adjust=False, ignore_na=True).mean().where(finite)


def pl_ema_carry_emit_null(x, span=12, window=None, **_):
    period = positive_int(span if span is not None else window, "span")
    expressions = []
    for column in pl_cols(x):
        value = pl.col(column).cast(pl.Float64, strict=False)
        finite = value.is_not_null() & value.is_finite()
        smoothed = (
            pl.when(finite).then(value).otherwise(None)
            .ewm_mean(span=period, adjust=False, ignore_nulls=True)
        )
        expressions.append(pl.when(finite).then(smoothed).otherwise(None).alias(column))
    return x.with_columns(expressions)


def install_stateful_full_history_semantics() -> None:
    global _APPLIED
    if _APPLIED:
        return
    exists_pd = "pandas_numpy" in OperatorRegistry.backends_for("ts_ema")
    OperatorRegistry.register(
        PandasFunctionOperator(
            "ts_ema", "time_series_stateful", ["x", "span"],
            "EMA with carry-state/emit-null missing semantics", pd_ema_carry_emit_null,
        ),
        canonical="ts_ema",
        backend="pandas_numpy",
        source="stateful_full_segment_parity",
        status="production",
        backend_explicit=True,
        replace=exists_pd,
        replacement_reason="align full-history EMA with checkpoint missing policy",
        semantic_version="3.0",
    )
    if pl is not None:
        exists_pl = "polars" in OperatorRegistry.backends_for("ts_ema")
        OperatorRegistry.register(
            PolarsFunctionOperator(
                "ts_ema", "time_series_stateful", ["x", "span"],
                "expression-native EMA with carry-state/emit-null missing semantics",
                pl_ema_carry_emit_null,
            ),
            canonical="ts_ema",
            backend="polars",
            source="final_expression_native_polars",
            status="production",
            backend_explicit=True,
            replace=exists_pl,
            replacement_reason="align Polars EMA with checkpoint missing policy",
            semantic_version="3.0",
        )
    _APPLIED = True


__all__ = ["install_stateful_full_history_semantics"]
