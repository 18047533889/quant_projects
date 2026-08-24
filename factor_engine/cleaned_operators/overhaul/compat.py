# -*- coding: utf-8 -*-
"""Backward-compatible call contracts for deduplicated operators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.overhaul.base import PandasFunctionOperator
from factor_engine.cleaned_operators.overhaul.regression import _rolling_regression
from factor_engine.cleaned_operators.registry import OperatorRegistry


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
    """Audited regression with canonical and historical positional contracts.

    Historical GTJA calls use ``ts_regression(y, x, window, lag, retval,
    min_periods=...)``.  Registry-driven callers may supply the complete public
    parameter sequence positionally.  Both forms are normalized before the
    rolling kernel is invoked, while negative lag remains fail-closed.
    """
    if len(legacy_args) > 4:
        raise TypeError(
            "ts_regression accepts at most lag, retval, min_periods and "
            "add_intercept positional arguments"
        )
    if legacy_args:
        # The repaired public contract is (y, x, window, add_intercept).
        # Preserve the historical lag-first form for non-boolean arguments.
        if len(legacy_args) == 1 and isinstance(legacy_args[0], (bool, np.bool_)):
            add_intercept = bool(legacy_args[0])
            legacy_args = ()
        else:
            lag = int(legacy_args[0])
    if legacy_args and len(legacy_args) >= 2:
        retval = str(legacy_args[1])
    if len(legacy_args) >= 3:
        min_periods = int(legacy_args[2])
    if len(legacy_args) == 4:
        add_intercept = bool(legacy_args[3])

    lag_i = 0 if lag is None else int(lag)
    if lag_i < 0:
        from factor_engine.backend.operator_errors import FutureReferenceError

        raise FutureReferenceError(f"ts_regression_compat: negative lag {lag_i} references future data")
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
    from factor_engine.cleaned_operators.overhaul.base import _inherit_canonical_logical_contract

    op = PandasFunctionOperator(
        "ts_regression_slope",
        "time_series_regression",
        ["y", "x", "window", "lag", "retval", "min_periods", "add_intercept"],
        "滚动 OLS；兼容历史 ts_regression(lag, retval) 调用契约",
        ts_regression_compat,
    )
    # R13 NEW-P0-04: a replacement may change only the implementation — the
    # canonical logical contract is inherited / verified, never rebuilt.
    _inherit_canonical_logical_contract("ts_regression_slope", op)
    OperatorRegistry.register(
        op,
        canonical="ts_regression_slope",
        backend="pandas_numpy",
        source="operator_overhaul_compat",
        status="production",
        backend_explicit=True,
    )
