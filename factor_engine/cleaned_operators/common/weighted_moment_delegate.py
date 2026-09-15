"""Explicit reference delegates for conditional covariance and weighted moments."""
import copy
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base_polars import SeriesOperator
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import weighted_moment_ext as source
    return source._KERNELS[name]
def metadata(name):
    return copy.deepcopy(OperatorRegistry.get(name,"pandas_numpy",mode="research").metadata)
def calculate(name,*args,**kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import to_pandas_panel,from_pandas_panel,verify_frames_share_identity
    panels=[p for p in (*args,*kwargs.values()) if isinstance(p,pl.DataFrame)]
    if not panels:
        raise TypeError(name+": a Polars input panel is required")
    verify_frames_share_identity(name,*panels)
    def convert(v):
        return to_pandas_panel(v) if isinstance(v,pl.DataFrame) else v
    out=reference(name)(*(convert(v) for v in args),**{k:convert(v) for k,v in kwargs.items()})
    return from_pandas_panel(panels[0],out)
def physical_spec(name):
    from factor_engine.cleaned_operators import weighted_moment_ext
    digest=hashlib.sha256(Path(weighted_moment_ext.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="weighted_moment_ext."+name,
        notes="Explicit Pandas reference conversion, not native/GPU acceleration.")
def register(name):
    class WeightedMomentPolars(SeriesOperator):
        metadata=globals()["metadata"](name)
        _contract_callable=staticmethod(reference(name))
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    OperatorRegistry.register(WeightedMomentPolars(),canonical=name,backend="polars",
        source="weighted_moment_reference",status="implemented",backend_explicit=True)
