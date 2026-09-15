"""Explicit three-feature energy / copula reference delegates."""
import copy
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base_polars import SeriesOperator
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec
NAMES=("ts_joint_energy_shift","ts_energy_break_score","ts_copula_central_asymmetry")
def reference(name):
    from factor_engine.cleaned_operators import distribution_break as source
    classes=(source.TsJointEnergyShift,source.TsEnergyBreakScore,source.TsCopulaCentralAsymmetry)
    return dict(zip(NAMES,classes))[name]()
def metadata(name):
    return copy.deepcopy(reference(name).metadata)
def calculate(name,*args,**kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import to_pandas_panel,from_pandas_panel,verify_frames_share_identity
    panels=[v for v in (*args,*kwargs.values()) if isinstance(v,pl.DataFrame)]
    if not panels:
        raise TypeError(name+": input panels are required")
    verify_frames_share_identity(name,*panels)
    convert=lambda v:to_pandas_panel(v) if isinstance(v,pl.DataFrame) else v
    out=reference(name).calculate(*(convert(v) for v in args),**{k:convert(v) for k,v in kwargs.items()})
    return from_pandas_panel(panels[0],out)
def physical_spec(name):
    from factor_engine.cleaned_operators import distribution_break
    digest=hashlib.sha256(Path(distribution_break.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="distribution_break."+name,
        notes="Explicit Pandas CPU conversion; energy uses wide-precision quadratic pair buffers. No GPU/native claim.")
def register(name):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    class DistributionPolars(SeriesOperator):
        metadata=globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    OperatorRegistry.register(DistributionPolars(),canonical=name,backend="polars",
        source="distribution_reference_polars",backend_explicit=True,
        status="experimental" if name=="ts_copula_central_asymmetry" else "implemented")
