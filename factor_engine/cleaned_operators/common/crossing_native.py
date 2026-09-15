"""Exact crossing geometry on per-column NumPy CPU buffers."""
import copy
import hashlib
from pathlib import Path
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator,PANEL_SKIP_COLUMNS
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import crossing
    return (crossing.TsCrossingSpeed if name=="ts_crossing_speed" else crossing.TsCrossingAcceleration)()
def metadata(name):
    out=copy.deepcopy(reference(name).metadata)
    out.tags=[*out.tags,"numpy_kernel"]
    return out
def calculate(name,x,y,window=20,**kwargs):
    from factor_engine.cleaned_operators import crossing
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    verify_frames_share_identity(name,x,y)
    w=strict_int(window,"window",minimum=2)
    kernel=crossing._crossing_speed_series if name=="ts_crossing_speed" else crossing._crossing_acceleration_series
    result=[]
    for c in x.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        a=x[c].to_numpy().astype(float).reshape(-1,1)
        b=y[c].to_numpy().astype(float).reshape(-1,1)
        result.append(pl.Series(c,kernel(a,b,w)[:,0],dtype=pl.Float64))
    return x.with_columns(result)
def physical_spec(name):
    from factor_engine.cleaned_operators import crossing
    digest=hashlib.sha256(Path(crossing.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="crossing."+name,
        notes="Two real panels; CPU wide-precision O(N) scratch per column; no Pandas or GPU claim.")
def make(name):
    class CrossingNative(SeriesOperator):
        metadata=globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,x,y,window=20,**kwargs):
            return calculate(name,x,y,window,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    return CrossingNative
