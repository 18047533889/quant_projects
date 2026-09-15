"""Typed source-safe contracts and per-column NumPy kernels."""
import numpy as np
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.parameter_validation import strict_integer

EXTREMES=("ts_argmax_age","ts_argmin_age","ts_argmax_index_from_oldest","ts_argmin_index_from_oldest")
def contract(name):
    if name in EXTREMES:
        return dict(param_names=["x","window","min_periods"],panel_params=("x",),
                    scalar_params=("window","min_periods"),param_specs={
            "window":ParamSpec(dtype=int,min=1,param_role=ParamRole.HORIZON),
            "min_periods":ParamSpec(dtype=int,min=1,default=1,param_role=ParamRole.POLICY)})
    if name=="group_impute_median":
        return dict(param_names=["x","group","min_group_size"],panel_params=("x","group"),
                    scalar_params=("min_group_size",),param_specs={
            "min_group_size":ParamSpec(dtype=int,min=1,default=3,param_role=ParamRole.POLICY)})
    return dict(param_names=["x","max_gap","lineage"],panel_params=("x",),
                scalar_params=("max_gap","lineage"),param_specs={
        "max_gap":ParamSpec(dtype=int,min=1,param_role=ParamRole.POLICY),
        "lineage":ParamSpec(dtype=str,choices=("price",),default=None,searchable=False,param_role=ParamRole.POLICY)})

def arg_values(values,window,min_periods,pick,age):
    w=strict_integer(window,"window",minimum=1)
    mp=strict_integer(min_periods,"min_periods",minimum=1)
    if mp>w:
        raise ValueError("min_periods must be <= window")
    output=np.full(values.shape,np.nan)
    for c in range(values.shape[1]):
        for row in range(values.shape[0]):
            segment=values[max(0,row-w+1):row+1,c]
            finite=np.isfinite(segment)
            if finite.sum()<mp:
                continue
            target=np.max(segment[finite]) if pick=="max" else np.min(segment[finite])
            latest=np.flatnonzero(finite & (segment==target))[-1]
            output[row,c]=len(segment)-1-latest if age else latest
    return output

def group_values(values,groups,min_group_size):
    from pandas import isna
    minimum=strict_integer(min_group_size,"min_group_size",minimum=1)
    output=values.copy()
    for row in range(values.shape[0]):
        partitions={}
        for c,g in enumerate(groups[row]):
            if g is None or isna(g) or (isinstance(g,str) and not g.strip()):
                continue
            if isinstance(g,(int,float,np.number)) and not np.isfinite(g):
                continue
            partitions.setdefault(g,[]).append(c)
        for positions in partitions.values():
            sample=values[row,positions]
            finite=sample[np.isfinite(sample)]
            if len(finite)<minimum:
                continue
            # Scaled median avoids overflow in even-cardinality averaging.
            scale=np.max(np.abs(finite))
            median=float(np.median(finite/scale)*scale) if scale else 0.
            for c in positions:
                if not np.isfinite(output[row,c]):
                    output[row,c]=median
    return output

def polars_group_impute_median(x,group,min_group_size=3):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import align_cols
    cols=align_cols(x,group)
    values=x.select(cols).to_numpy().astype(float)
    groups=group.select(cols).to_numpy()
    result=group_values(values,groups,min_group_size)
    return x.with_columns([pl.Series(c,result[:,i]) for i,c in enumerate(cols)])

def polars_arg(x,window,min_periods=1,*,pick,age):
    import polars as pl
    from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
    cols=[c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
    result=arg_values(x.select(cols).to_numpy().astype(float),window,min_periods,pick,age)
    return x.with_columns([pl.Series(c,result[:,i]) for i,c in enumerate(cols)])

def polars_ffill_limited(x,max_gap,lineage=None):
    import polars as pl
    from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
    limit=strict_integer(max_gap,"max_gap",minimum=1)
    if lineage not in (None,"price"):
        raise ValueError("ts_ffill_limited: forward-fill is not permitted for this lineage")
    return x.with_columns([pl.col(c).fill_nan(None).forward_fill(limit=limit).alias(c)
                           for c in x.columns if c not in PANEL_SKIP_COLUMNS])

def physical_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    return PhysicalImplementationSpec(
        canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR if name=="ts_ffill_limited" else ExecutionKind.POLARS_NUMPY_KERNEL,
        materializes_full_panel=True,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_identity="safe_kernels."+name,
        notes="Polars expressions or NumPy panel kernels; no Pandas panel conversion and no GPU claim.")
