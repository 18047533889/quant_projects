# -*- coding: utf-8 -*-
"""Backward-compatible call contracts for deduplicated operators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import PandasFunctionOperator
from cleaned_operators.overhaul.regression import _rolling_regression
from cleaned_operators.registry import OperatorRegistry


def ts_regression_compat(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    *legacy_args,
    min_periods: int | None = None,
    add_intercept: bool = True,
    lag: int | None = None,
    retval: str | None = None,
    **_,
) -> pd.DataFrame:
    """Audited regression with the historical GTJA positional contract.

    Historical calls use ``ts_regression(y, x, window, lag, retval,
    min_periods=...)``.  The canonical is now ``ts_regression_slope`` but the
    alias must remain executable without weakening the no-future-data rule.
    """
    if len(legacy_args) > 2:
        raise TypeError("ts_regression accepts at most legacy lag and retval arguments")
    if legacy_args:
        lag = int(legacy_args[0])
    if len(legacy_args) == 2:
        retval = str(legacy_args[1])
    lag_i = 0 if lag is None else int(lag)
    if lag_i < 0:
        return pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    if lag_i:
        x = x.shift(lag_i)
    requested = str(retval or "slope").lower()
    output = {
        "slope": "slope",
        "beta": "slope",
        "intercept": "intercept",
        "residual": "resid",
        "resid": "resid",
        "r_squared": "r2",
        "r2": "r2",
        "tstat": "tstat",
        "t_stat": "tstat",
    }.get(requested)
    if output is None:
        raise ValueError(f"unsupported regression retval: {retval!r}")
    return _rolling_regression(
        y,
        x,
        window,
        min_periods,
        bool(add_intercept),
        output,
    )


def register() -> None:
    OperatorRegistry.register(
        PandasFunctionOperator(
            "ts_regression_slope",
            "time_series_regression",
            ["y", "x", "window", "lag", "retval", "min_periods", "add_intercept"],
            "滚动 OLS；兼容历史 ts_regression(lag, retval) 调用契约",
            ts_regression_compat,
        ),
        canonical="ts_regression_slope",
        backend="pandas_numpy",
        source="operator_overhaul_compat",
        status="production",
        backend_explicit=True,
    )
