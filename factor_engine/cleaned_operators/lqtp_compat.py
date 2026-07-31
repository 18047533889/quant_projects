# -*- coding: utf-8 -*-
"""LQTP-specific compatibility primitives that cannot be represented as aliases/macros.

The three-argument Chinese/JQ-style SMA is intentionally a separate canonical
operator. It must never be aliased to ``ts_mean`` because its recurrence is
stateful::

    y_t = (m * x_t + (n - m) * y_{t-1}) / n

The first finite observation seeds the recurrence. A missing input produces a
missing output and leaves the previous state unchanged. This is causal but
remains research-only until a checkpoint contract proves segmented/incremental
execution parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


@register_operator(
    name="ts_sma_cn",
    category="time_series",
    business_category="time_series",
    canonical="ts_sma_cn",
    source="lqtp_compat",
    backend="pandas_numpy",
    status="research",
)
class ChineseRecursiveSMA(SeriesOperator):
    """Chinese/JQ-style three-argument recursive SMA."""

    metadata = OperatorMetadata(
        name="ts_sma_cn",
        category="time_series",
        description="recursive SMA: y=(m*x+(n-m)*prev)/n",
        examples=["sma(close, 7, 2)"],
        param_names=["x", "n", "m"],
        return_type="series",
        tags=["time_series", "stateful", "pit_safe", "causal", "lqtp_compat"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        n: int = 7,
        m: int = 2,
        **kwargs,
    ) -> pd.DataFrame:
        n_i = int(n)
        m_i = int(m)
        if n_i <= 0 or float(n) != float(n_i):
            raise ValueError("sma n must be a positive integer")
        if m_i <= 0 or float(m) != float(m_i) or m_i > n_i:
            raise ValueError("sma m must be an integer satisfying 1 <= m <= n")

        values = x.to_numpy(dtype=float)
        out = np.full(values.shape, np.nan, dtype=float)
        alpha = float(m_i) / float(n_i)
        for col in range(values.shape[1]):
            state = np.nan
            for row in range(values.shape[0]):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                if not np.isfinite(state):
                    state = float(value)
                else:
                    state = alpha * float(value) + (1.0 - alpha) * state
                out[row, col] = state
        return pd.DataFrame(out, index=x.index, columns=x.columns)
