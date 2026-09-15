"""Exact expectile CPU kernels on Polars columns; no Pandas panel conversion."""
import copy
import hashlib
from pathlib import Path
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator, PANEL_SKIP_COLUMNS
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import advanced_expectile as source
    return (source.TsExpectileBeta if name == "ts_expectile_beta" else source.TsExpectile)()

def metadata(name):
    out=copy.deepcopy(reference(name).metadata)
    out.tags=[*out.tags,"numpy_kernel"]
    return out

def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators import advanced_expectile as source
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    import inspect
    bound=inspect.signature(reference(name)._calculate_series).bind(*args,**kwargs)
    bound.apply_defaults()
    values=bound.arguments
    pair=name=="ts_expectile_beta"
    panels=[values[p] for p in (("y","x") if pair else ("x",))]
    verify_frames_share_identity(name,*panels)
    w,tau,nmin=source._parameters(values["window"],values["tau"],values["n_min"],4 if pair else 3)
    base=panels[0]
    output=[]
    for c in base.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        a=base[c].to_numpy().astype(float).reshape(-1,1)
        if pair:
            b=panels[1][c].to_numpy().astype(float).reshape(-1,1)
            result=source.map_pair_rolling(a,b,w,lambda x,y:source._expectile_slope(x,y,tau,nmin))
        else:
            result=source.map_rolling(a,w,lambda x:source._expectile(x,tau,nmin))
        output.append(pl.Series(c,result[:,0],dtype=pl.Float64))
    return base.with_columns(output)

def physical_spec(name):
    from factor_engine.cleaned_operators import advanced_expectile, rolling_pack
    digest=hashlib.sha256(Path(advanced_expectile.__file__).read_bytes()
                         +Path(rolling_pack.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="advanced_expectile."+name,
        notes="Exact per-column CPU asymmetric least squares with convergence/support gates; no GPU claim.")

def make(name):
    class ExpectileNative(SeriesOperator):
        metadata=globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    return ExpectileNative
