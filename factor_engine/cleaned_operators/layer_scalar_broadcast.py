# -*- coding: utf-8 -*-
"""Scalar-aware elementwise primitives required by recipe execution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_cols,
)
from cleaned_operators.registry import OperatorRegistry

_APPLIED = False


def _as_panel(value, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if np.isscalar(value):
        return pd.DataFrame(
            np.full(template.shape, float(value), dtype=float),
            index=template.index,
            columns=template.columns,
        )
    raise TypeError(f"safe_div_null expects a panel or scalar, got {type(value)!r}")


def pd_safe_div_broadcast(x, y, epsilon=EPS, **_):
    if isinstance(x, pd.DataFrame):
        template = x
    elif isinstance(y, pd.DataFrame):
        template = y
    else:
        denominator = float(y)
        epsilon = float(epsilon)
        if not np.isfinite(float(x)) or not np.isfinite(denominator) or abs(denominator) <= epsilon:
            return np.nan
        return float(x) / denominator

    x_panel = _as_panel(x, template)
    y_panel = _as_panel(y, template)
    x_panel, y_panel = aligned_pd(x_panel, y_panel)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    xv, yv = x_panel.to_numpy(dtype=float), y_panel.to_numpy(dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv) & (np.abs(yv) > epsilon)
    out = np.full(x_panel.shape, np.nan)
    out[valid] = xv[valid] / yv[valid]
    return frame_pd(x_panel, out)


def pl_safe_div_broadcast(x, y, epsilon=EPS, **_):
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    x_is_frame = pl is not None and isinstance(x, pl.DataFrame)
    y_is_frame = pl is not None and isinstance(y, pl.DataFrame)
    if not x_is_frame and not y_is_frame:
        denominator = float(y)
        if not np.isfinite(float(x)) or not np.isfinite(denominator) or abs(denominator) <= epsilon:
            return None
        return float(x) / denominator
    template = x if x_is_frame else y
    columns = pl_cols(template)
    expressions = []
    for column in columns:
        xv = pl.col(column).cast(pl.Float64, strict=False) if x_is_frame else pl.lit(float(x))
        yv = (
            y[column].cast(pl.Float64, strict=False)
            if y_is_frame and column in y.columns
            else pl.lit(float(y))
        )
        expressions.append(
            pl.when(xv.is_finite() & yv.is_finite() & (yv.abs() > epsilon))
            .then(xv / yv)
            .otherwise(None)
            .alias(column)
        )
    return template.with_columns(expressions)


def install_scalar_broadcast_primitives() -> None:
    global _APPLIED
    if _APPLIED:
        return
    OperatorRegistry.register(
        PandasFunctionOperator(
            "safe_div_null", "elementwise", ["x", "y", "epsilon"],
            "finite safe division with scalar or panel broadcasting",
            pd_safe_div_broadcast,
        ),
        canonical="safe_div_null",
        backend="pandas_numpy",
        source="scalar_broadcast_primitives",
        status="production",
        backend_explicit=True,
        replace="pandas_numpy" in OperatorRegistry.backends_for("safe_div_null"),
        replacement_reason="support recipe scalar broadcasting without weakening null semantics",
        semantic_version="3.0",
    )
    if pl is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(
                "safe_div_null", "elementwise", ["x", "y", "epsilon"],
                "expression-native safe division with scalar or panel broadcasting",
                pl_safe_div_broadcast,
            ),
            canonical="safe_div_null",
            backend="polars",
            source="final_expression_native_polars",
            status="production",
            backend_explicit=True,
            replace="polars" in OperatorRegistry.backends_for("safe_div_null"),
            replacement_reason="support scalar broadcasting in native Polars",
            semantic_version="3.0",
        )
    _APPLIED = True


__all__ = ["install_scalar_broadcast_primitives"]
