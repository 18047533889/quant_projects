# -*- coding: utf-8 -*-
"""Pandas elementwise comparison with symmetric scalar broadcasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, TwoVarOperator, register_operator


def _broadcast_pair(left, right) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = left if isinstance(left, pd.DataFrame) else right if isinstance(right, pd.DataFrame) else None
    if template is None:
        raise TypeError("comparison requires at least one DataFrame input")

    def frame(value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.reindex(index=template.index, columns=template.columns)
        if isinstance(value, pd.Series):
            if value.index.equals(template.index):
                arr = np.broadcast_to(value.to_numpy()[:, None], template.shape)
                return pd.DataFrame(arr, index=template.index, columns=template.columns)
            raise ValueError("Series comparison input is not aligned with panel index")
        return pd.DataFrame(value, index=template.index, columns=template.columns)

    return frame(left), frame(right)


def _compare(op: str, left, right) -> pd.DataFrame:
    x, y = _broadcast_pair(left, right)
    xa = x.to_numpy(dtype=float, copy=False)
    ya = y.to_numpy(dtype=float, copy=False)
    valid = np.isfinite(xa) & np.isfinite(ya)
    operations = {
        "lt": xa < ya,
        "le": xa <= ya,
        "eq": xa == ya,
        "gt": xa > ya,
        "ge": xa >= ya,
        "ne": xa != ya,
    }
    out = np.where(valid, operations[op].astype(float), np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


class _Comparison(TwoVarOperator):
    op = ""

    def _calculate_series(self, x, y, **kwargs):
        return _compare(self.op, x, y)


def _meta(name: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="elementwise_math",
        description=f"{name} comparison with symmetric scalar broadcasting",
        param_names=["x", "y"],
        return_type="series",
        tags=["comparison", "scalar_broadcast", "pit_safe"],
    )


@register_operator(name="lt", category="elementwise_math", business_category="elementwise_math", canonical="lt", source="scalar_compare", backend="pandas_numpy")
class ScalarLt(_Comparison):
    op = "lt"
    metadata = _meta("lt")


@register_operator(name="le", category="elementwise_math", business_category="elementwise_math", canonical="le", source="scalar_compare", backend="pandas_numpy")
class ScalarLe(_Comparison):
    op = "le"
    metadata = _meta("le")


@register_operator(name="eq", category="elementwise_math", business_category="elementwise_math", canonical="eq", source="scalar_compare", backend="pandas_numpy")
class ScalarEq(_Comparison):
    op = "eq"
    metadata = _meta("eq")


@register_operator(name="gt", category="elementwise_math", business_category="elementwise_math", canonical="gt", source="scalar_compare", backend="pandas_numpy")
class ScalarGt(_Comparison):
    op = "gt"
    metadata = _meta("gt")


@register_operator(name="ge", category="elementwise_math", business_category="elementwise_math", canonical="ge", source="scalar_compare", backend="pandas_numpy")
class ScalarGe(_Comparison):
    op = "ge"
    metadata = _meta("ge")


@register_operator(name="ne", category="elementwise_math", business_category="elementwise_math", canonical="ne", source="scalar_compare", backend="pandas_numpy")
class ScalarNe(_Comparison):
    op = "ne"
    metadata = _meta("ne")
