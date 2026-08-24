# -*- coding: utf-8 -*-
"""Polars-only canonical 的 pandas_numpy 回退（企业级双 backend 对称）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry


def _has_pandas(canonical: str) -> bool:
    return "pandas_numpy" in OperatorRegistry.backends_for(canonical)


if not _has_pandas("maximum"):

    @register_operator(
        name="maximum",
        category="math",
        business_category="elementwise_math",
        canonical="maximum",
        source="factor_dsl_np_parity",
    )
    class MaximumNp(SeriesOperator):
        """逐元素 max(x,y)"""
        metadata = OperatorMetadata(
            name="maximum",
            category="math",
            description="逐元素 max(x,y)",
            param_names=["x", "y"],
            return_type="series",
            tags=["math", "parity"],
        )

        def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
            from backend.elementwise_semantics import max_horizontal_pandas

            cols = [c for c in x.columns if c in y.columns]
            return pd.DataFrame(
                {c: max_horizontal_pandas(x[c], y[c]) for c in cols},
                index=x.index,
            )


if not _has_pandas("minimum"):

    @register_operator(
        name="minimum",
        category="math",
        business_category="elementwise_math",
        canonical="minimum",
        source="factor_dsl_np_parity",
    )
    class MinimumNp(SeriesOperator):
        """逐元素 min(x,y)"""
        metadata = OperatorMetadata(
            name="minimum",
            category="math",
            description="逐元素 min(x,y)",
            param_names=["x", "y"],
            return_type="series",
            tags=["math", "parity"],
        )

        def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
            from backend.elementwise_semantics import min_horizontal_pandas

            cols = [c for c in x.columns if c in y.columns]
            return pd.DataFrame(
                {c: min_horizontal_pandas(x[c], y[c]) for c in cols},
                index=x.index,
            )
