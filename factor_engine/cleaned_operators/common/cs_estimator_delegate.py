"""Explicit reference delegates for cross-sectional estimators; no native claim."""
from factor_engine.cleaned_operators.base_polars import OperatorMetadata
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import cs_batch1 as source
    return {c.metadata.name:c for c in (
        source.CsIsolationForestScore,source.CsFactorBucketReturn,
        source.CsEmpiricalBayesShrinkage,source.CsShrinkToGroupMean)}[name]()

def metadata(name):
    meta=reference(name).metadata
    return OperatorMetadata(name=name,category=meta.category,description=meta.description,
        param_names=list(meta.param_names),panel_params=meta.panel_params,
        scalar_params=meta.scalar_params,param_specs=dict(meta.param_specs),
        tags=[*meta.tags,"pandas_delegate"])

def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
    return _call_pandas_delegate(name,args,kwargs)

def physical_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.cleaned_operators import cs_batch1
    digest=hashlib.sha256(Path(cs_batch1.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="cs_batch1."+name,
        notes="Explicit full-panel conversion to cross-sectional reference; not native acceleration.")
