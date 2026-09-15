# -*- coding: utf-8 -*-
"""LQTP compatibility primitives that require dedicated numerical kernels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, RelationalParamSpec, SeriesOperator, register_operator


@register_operator(
    name="ts_sma_cn",
    category="time_series",
    business_category="time_series",
    canonical="ts_sma_cn",
    source="lqtp_compat",
    backend="pandas_numpy",
    status="implemented")
class ChineseRecursiveSMA(SeriesOperator):
    """Chinese/JQ recursive SMA: y=(m*x+(n-m)*prev)/n."""
    metadata = OperatorMetadata(
        name="ts_sma_cn", category="time_series",
        description="recursive SMA: y=(m*x+(n-m)*prev)/n",
        examples=["sma(close, 7, 2)"],
        param_names=["x", "n", "m"], return_type="series",
        panel_params=("x",), scalar_params=("n", "m"),
        param_specs={k: ParamSpec(dtype=int, min=1, default=d, param_role=ParamRole.HORIZON) for k,d in (("n",7),("m",2))},
        relational_specs=[RelationalParamSpec("m <= n")],
        tags=["time_series", "stateful", "pit_safe", "causal", "lqtp_compat"],
    )

    def _calculate_series(self, x: pd.DataFrame, n: int = 7, m: int = 2, **kwargs) -> pd.DataFrame:
        out = sma_array(x.to_numpy(dtype=float), n, m)
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
        panel_params=("x",), scalar_params=("window", "q"), window_semantics="exact_rows",
        param_specs={"window": ParamSpec(dtype=int,min=1,default=252,param_role=ParamRole.HORIZON),
            "q": ParamSpec(dtype=float,min=0,max=1,default=.05,param_role=ParamRole.ECONOMIC)},
        relational_specs=[RelationalParamSpec("q > 0")],
        tags=["time_series", "risk", "pit_safe", "causal", "lqtp_compat"],
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 252, q: float = 0.05, **kwargs) -> pd.DataFrame:
        out = cvar_array(x.to_numpy(dtype=float), window, q)
        return pd.DataFrame(out, index=x.index, columns=x.columns)


def sma_array(values, n=7, m=2):
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    n = strict_int(n, "n", minimum=1)
    m = strict_int(m, "m", minimum=1, maximum=n)
    out = np.full(values.shape, np.nan)
    alpha = np.longdouble(m) / n
    for col in range(values.shape[1]):
        state = np.longdouble(np.nan)
        for row in range(values.shape[0]):
            value = values[row,col]
            if not np.isfinite(value):
                continue
            state = np.longdouble(value) if not np.isfinite(state) else alpha*value+(1-alpha)*state
            out[row,col] = float(state)
    return out


def cvar_array(values, window=252, q=.05):
    from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_float
    window = strict_int(window, "window", minimum=1)
    q = strict_float(q, "q", minimum=0, maximum=1)
    if q <= 0:
        raise ValueError("q must be in (0,1]")
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        for row in range(values.shape[0]):
            segment = values[max(0,row-window+1):row+1,col]
            finite = np.sort(segment[np.isfinite(segment)].astype(np.longdouble))
            if not len(finite):
                continue
            position = np.longdouble(len(finite)-1)*q
            lo = int(np.floor(position)); hi = int(np.ceil(position))
            fraction = position-lo
            cutoff = (1-fraction)*finite[lo]+fraction*finite[hi]
            tail = finite[finite <= cutoff]
            out[row,col] = float(-np.mean(tail, dtype=np.longdouble))
    return out


def polars_calculate(name, x, **params):
    import polars as pl
    from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
    kernel = sma_array if name == "ts_sma_cn" else cvar_array
    cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
    values = x.select(cols).to_numpy().astype(float)
    out = kernel(values, **params)
    return x.with_columns([pl.Series(c,out[:,i],dtype=pl.Float64) for i,c in enumerate(cols)])


def polars_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        stateful=name == "ts_sma_cn", requires_sorted=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_identity="lqtp_compat."+name,
        notes="CPU NumPy kernels; SMA preserves state across missing input, CVaR finite support min 1.")


def register_polars(name):
    import copy
    from factor_engine.cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    reference = ChineseRecursiveSMA if name == "ts_sma_cn" else LQTPHistoricalCVaR
    class NativeLQTP(PolarsSeriesOperator):
        metadata = copy.deepcopy(reference.metadata)
        @property
        def _contract_callable(self):
            return reference()._calculate_series
        def _calculate_series(self,x,*args,**kwargs):
            import inspect
            bound = inspect.signature(reference()._calculate_series).bind(x,*args,**kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            values.pop("x")
            values.update(values.pop("kwargs", {}))
            return polars_calculate(name,x,**values)
        def physical_spec(self):
            return polars_spec(name)
    OperatorRegistry.register(NativeLQTP(),canonical=name,backend="polars",
        source="lqtp_numpy",backend_explicit=True,status="implemented")


register_polars("ts_sma_cn")
register_polars("lqtp_historical_cvar")
