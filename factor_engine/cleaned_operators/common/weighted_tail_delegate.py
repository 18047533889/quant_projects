"""Weighted-tail Polars interfaces delegate to the canonical causal reference."""
from factor_engine.cleaned_operators.base_polars import OperatorMetadata
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import weighted_tail as s
    return {c.metadata.name:c for c in (
        s.TsStratifiedMeanSpread,s.TsWeightedSemivariance,s.TsWeightedDownsideDeviation,
        s.TsWeightedExpectedShortfall,s.TsWeightedDrawdownArea)}[name]()

def metadata(name):
    m=reference(name).metadata
    return OperatorMetadata(name=name,category=m.category,description=m.description,
        param_names=list(m.param_names),param_specs=dict(m.param_specs),
        panel_params=m.panel_params,scalar_params=m.scalar_params,tags=[*m.tags,"pandas_delegate"])

def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
    return _call_pandas_delegate(name,args,kwargs)

def physical_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.cleaned_operators import weighted_tail
    digest=hashlib.sha256(Path(weighted_tail.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="weighted_tail."+name,
        notes="Full-panel conversion to actual weighted-tail reference; not native or GPU acceleration.")
