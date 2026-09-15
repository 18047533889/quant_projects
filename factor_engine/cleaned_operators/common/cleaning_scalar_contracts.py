"""Scalar contracts and genuine Polars cleaning/EWM kernels."""
import numpy as np
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar
def contract(name):
    key="span" if name in ("ewm_std","ewm_var","ts_ewm_std","ts_ewm_var") else "value" if name=="fillna_const" else "num"
    spec=ParamSpec(dtype=float,min=np.nextafter(0.,1.) if key=="span" else None,
                   default=20. if key=="span" else 0.,param_role=ParamRole.HORIZON if key=="span" else ParamRole.POLICY)
    return dict(param_names=["x",key],panel_params=("x",),scalar_params=(key,),param_specs={key:spec})
def alpha(span):
    value=strict_finite_scalar(span,"span")
    if value<=0:
        raise ValueError("span must be positive; (0,1) means alpha, >=1 means span")
    return value if value<1 else 2/(value+1)
def _cols(x):
    from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
    return [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
def polars_ewm(x,span=20.,*,kind):
    import polars as pl
    a=alpha(span)
    if a==1.:
        # Unbiased variance needs at least two positive weights. Alpha=1
        # retains just the latest observation, so variance/std are undefined.
        return x.with_columns([pl.lit(None,dtype=pl.Float64).alias(c) for c in _cols(x)])
    exprs=[]
    for c in _cols(x):
        finite=pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None)
        estimate=getattr(finite,"ewm_"+kind)(alpha=a,adjust=False,bias=False,min_samples=1,ignore_nulls=False)
        exprs.append(pl.when(finite.is_not_null().cum_sum()<2).then(None)
                     .otherwise(estimate.forward_fill()).alias(c))
    return x.with_columns(exprs)
def ewm_std(x,span=20.):
    return polars_ewm(x,span,kind="std")
def ewm_var(x,span=20.):
    return polars_ewm(x,span,kind="var")
def fillna_const(x,value=0.):
    import polars as pl
    value=strict_finite_scalar(value,"value")
    return x.with_columns([pl.col(c).fill_nan(None).fill_null(value).alias(c) for c in _cols(x)])
def nonfinite_to_num(x,num=0.):
    import polars as pl
    num=strict_finite_scalar(num,"num")
    return x.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(num).alias(c) for c in _cols(x)])
def physical_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,materializes_full_panel=True,
        stateful="ewm" in name,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_identity="cleaning_scalar_contracts."+name,
        notes="Polars expression kernel; explicit finite scalar parameters, no Pandas panel conversion.")
