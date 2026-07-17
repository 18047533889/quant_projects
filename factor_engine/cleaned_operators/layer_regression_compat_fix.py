# -*- coding: utf-8 -*-
"""Final call-contract adapter for the fused regression slope canonical."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.layer_regression_fusion import _compute_all
from cleaned_operators.overhaul.base import PandasFunctionOperator
from cleaned_operators.registry import OperatorRegistry

_APPLIED = False


def _calculate(y, x, window, *args, min_periods=None, add_intercept=True, lag=None, retval=None, **_):
    if len(args) > 2:
        raise TypeError("rolling regression accepts at most two extra positional arguments")
    if len(args) == 1:
        if lag is None and retval is None and min_periods is None:
            min_periods = int(args[0])
        else:
            lag = int(args[0])
    elif len(args) == 2:
        lag, retval = int(args[0]), str(args[1])

    lag_i = 0 if lag is None else int(lag)
    if lag_i < 0:
        return pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    if lag_i:
        x = x.shift(lag_i)

    output = {
        "slope": "slope", "beta": "slope", "intercept": "intercept",
        "residual": "resid", "resid": "resid", "r_squared": "r2",
        "r2": "r2", "tstat": "tstat", "t_stat": "tstat",
    }.get(str(retval or "slope").lower())
    if output is None:
        raise ValueError(f"unsupported regression retval: {retval!r}")
    return _compute_all(y, x, window, min_periods, add_intercept)[output]


def install_regression_call_compatibility() -> None:
    global _APPLIED
    if _APPLIED:
        return
    OperatorRegistry.register(
        PandasFunctionOperator(
            "ts_regression_slope", "time_series_regression",
            ["y", "x", "window", "min_periods", "add_intercept", "lag", "retval"],
            "fused rolling OLS slope with direct and historical call contracts",
            _calculate,
        ),
        canonical="ts_regression_slope",
        backend="pandas_numpy",
        source="rolling_ols_fused_cache",
        status="production",
        backend_explicit=True,
        replace=True,
        replacement_reason="disambiguate direct min_periods and legacy lag/retval positional calls",
        semantic_version="3.1",
    )
    _APPLIED = True


__all__ = ["install_regression_call_compatibility"]
