"""CPU despike kernels on Polars columns, preserving the actual time axis."""
import copy
import hashlib
import inspect
from pathlib import Path
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator,PANEL_SKIP_COLUMNS
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

NAMES=("ts_hampel_filter_causal","ts_median3_causal","ts_rolling_median_causal")

def reference(name):
    from factor_engine.cleaned_operators import filter_despike as source
    return dict(zip(NAMES,(source.TSHampelFilterCausal,source.TSMedian3Causal,source.TSRollingMedianCausal)))[name]()

def metadata(name):
    out=copy.deepcopy(reference(name).metadata)
    out.tags=[*out.tags,"numpy_kernel"]
    return out

def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators import filter_despike as source
    bound=inspect.signature(reference(name)._calculate_series).bind(*args,**kwargs)
    bound.apply_defaults()
    values=bound.arguments
    x=values["x"]
    if name==NAMES[0]:
        params=source._hampel_parameters(*(values[p] for p in ("window","n_sigma","replacement","scale_floor")))
        kernel=source._hampel
    elif name==NAMES[1]:
        params=()
        kernel=source._median3
    else:
        params=source._median_parameters(values["window"],values["min_periods"])
        kernel=source._rolling_median
    out=[]
    for c in x.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        result=kernel(x[c].to_numpy().astype(float).reshape(-1,1),*params)
        out.append(pl.Series(c,result[:,0],dtype=pl.Float64))
    return x.with_columns(out)

def physical_spec(name):
    from factor_engine.cleaned_operators import filter_despike
    digest=hashlib.sha256(Path(filter_despike.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="filter_despike."+name,
        notes="Exact CPU despike with long-double intermediate arithmetic; no Pandas conversion or GPU claim.")

def make(name):
    class DespikeNative(SeriesOperator):
        metadata=globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    return DespikeNative
