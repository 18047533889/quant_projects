# -*- coding: utf-8 -*-
"""LQTP compatibility primitives that require dedicated numerical kernels."""
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
    status="research")
class ChineseRecursiveSMA(SeriesOperator):
    """Chinese/JQ recursive SMA: y=(m*x+(n-m)*prev)/n."""
    metadata = OperatorMetadata(
        name="ts_sma_cn", category="time_series",
        description="recursive SMA: y=(m*x+(n-m)*prev)/n",
        examples=["sma(close, 7, 2)"],
        param_names=["x", "n", "m"], return_type="series",
        tags=["time_series", "stateful", "pit_safe", "causal", "lqtp_compat"],
    )

    def _calculate_series(self, x: pd.DataFrame, n: int = 7, m: int = 2, **kwargs) -> pd.DataFrame:
        n_i, m_i = int(n), int(m)
        if n_i <= 0 or float(n) != float(n_i):
            raise ValueError("sma n must be a positive integer")
        if m_i <= 0 or float(m) != float(m_i) or m_i > n_i:
            raise ValueError("sma m must satisfy 1 <= m <= n")
        values = x.to_numpy(dtype=float)
        out = np.full(values.shape, np.nan, dtype=float)
        alpha = float(m_i) / float(n_i)
        for col_idx in range(values.shape[1]):
            state = np.nan
            for row_idx in range(values.shape[0]):
                value = values[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                state = float(value) if not np.isfinite(state) else alpha * float(value) + (1.0-alpha) * state
                out[row_idx, col_idx] = state
        return pd.DataFrame(out, index=x.index, columns=x.columns)


@register_operator(
    name="lqtp_historical_cvar",
    category="time_series",
    business_category="risk",
    canonical="lqtp_historical_cvar",
    source="lqtp_compat",
    backend="pandas_numpy",
    status="research")
class LQTPHistoricalCVaR(SeriesOperator):
    """Historical Expected Shortfall: negative mean of observations <= rolling q-quantile."""
    metadata = OperatorMetadata(
        name="lqtp_historical_cvar", category="time_series",
        description="historical CVaR/Expected Shortfall: -mean(lower-tail returns)",
        examples=["historical_cvar(ret, 252, 0.05)"],
        param_names=["x", "window", "q"], return_type="series",
        tags=["time_series", "risk", "pit_safe", "causal", "lqtp_compat"],
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 252, q: float = 0.05, **kwargs) -> pd.DataFrame:
        w = int(window)
        qf = float(q)
        if w <= 0 or float(window) != float(w):
            raise ValueError("historical_cvar window must be a positive integer")
        if not 0.0 < qf <= 1.0:
            raise ValueError("historical_cvar q must be in (0,1]")

        def expected_shortfall(values: np.ndarray) -> float:
            vals = values[np.isfinite(values)]
            if vals.size == 0:
                return np.nan
            cutoff = float(np.quantile(vals, qf))
            tail = vals[vals <= cutoff]
            return float(-np.mean(tail)) if tail.size else np.nan

        return x.rolling(window=w, min_periods=1).apply(expected_shortfall, raw=True)
