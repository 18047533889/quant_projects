# -*- coding: utf-8 -*-
"""Pandas ``where`` runtime with symmetric scalar/panel broadcasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import Operator, OperatorMetadata, register_operator


def _panel(value, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=template.index, columns=template.columns)
    if isinstance(value, pd.Series):
        if value.index.equals(template.index):
            arr = np.broadcast_to(value.to_numpy()[:, None], template.shape)
            return pd.DataFrame(arr, index=template.index, columns=template.columns)
        raise ValueError("where Series branch is not aligned with the panel index")
    return pd.DataFrame(value, index=template.index, columns=template.columns)


@register_operator(
    name="where",
    category="elementwise_math",
    business_category="elementwise_math",
    canonical="where",
    source="scalar_where",
    backend="pandas_numpy",
)
class ScalarBroadcastWhere(Operator):
    """Conditional selection supporting scalar/scalar and scalar/panel branches."""

    metadata = OperatorMetadata(
        name="where",
        category="elementwise_math",
        description="conditional selection with symmetric scalar broadcasting",
        examples=["where(close > open, 1, 0)", "where(close > open, close, 0)"],
        param_names=["condition", "x", "y"],
        return_type="series",
        tags=["pit_safe", "causal", "scalar_broadcast"],
    )

    # R5-02: direct ``calculate`` that routes through validate_operator_call.
    _HANDLES_CALL_CONTRACT = True

    def calculate(self, condition, x, y, **kwargs) -> pd.DataFrame:
        # R5-02: this operator overrides ``calculate`` directly (its
        # scalar/panel symmetric broadcast does not fit the SeriesOperator
        # kernel shape), but it must still pass the central logical-call
        # validator so integer / panel-axis / unknown-kwarg checks apply.
        from factor_engine.cleaned_operators.base import validate_operator_call

        processed, processed_kwargs = validate_operator_call(
            self, (condition, x, y), kwargs
        )
        condition, x, y = processed
        template = next(
            (v for v in (condition, x, y) if isinstance(v, pd.DataFrame)),
            None,
        )
        if template is None:
            # 2026-08-29: all-scalar ``where(1, 1.0, 0.5)`` — the LQTP platform
            # treats a constant truthy condition as a constant selector.  Return
            # the selected scalar (condition != 0 → x else y); the surrounding
            # arithmetic broadcasts it to a constant panel.
            try:
                cond_val = float(condition)
            except (TypeError, ValueError):
                raise TypeError("where requires at least one panel input")
            return x if (np.isfinite(cond_val) and cond_val != 0) else y
        cond = _panel(condition, template)
        left = _panel(x, template)
        right = _panel(y, template)

        cond_arr = cond.to_numpy(dtype=float, copy=False)
        valid_cond = np.isfinite(cond_arr)
        selected = cond_arr != 0
        out = np.where(
            selected,
            left.to_numpy(copy=False),
            right.to_numpy(copy=False),
        )
        # Unknown condition remains unknown instead of silently taking y.
        if np.issubdtype(np.asarray(out).dtype, np.number):
            out = np.asarray(out, dtype=float)
            out[~valid_cond] = np.nan
        return pd.DataFrame(out, index=template.index, columns=template.columns)
