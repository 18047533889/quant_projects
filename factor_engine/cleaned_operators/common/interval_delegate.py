"""Explicit pandas delegates for interval geometry; no native/GPU claim."""
from factor_engine.cleaned_operators.base_polars import OperatorMetadata
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec
def reference(name):
    from factor_engine.cleaned_operators import interval_geometry as source
    owners={c.metadata.name:c for c in (
        source.TsIntervalUnionCoverage,source.TsIntervalOccupancyEntropy,
        source.TsIntervalOccupancyModeDistance,source.TsIntervalNestingDepth,
        source.TsIntervalExplorationEfficiency,source.TsIntervalOverlapComponentRatio)}
    return owners[name]()
def metadata(name):
    meta=reference(name).metadata
    return OperatorMetadata(name=name,category="interval_geometry",description=meta.description,
        param_names=list(meta.param_names),panel_params=meta.panel_params,scalar_params=meta.scalar_params,
        param_specs=dict(meta.param_specs),tags=["pandas_delegate"])
def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
    return _call_pandas_delegate(name,args,kwargs)
def physical_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.cleaned_operators import interval_geometry
    digest=hashlib.sha256(Path(interval_geometry.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="interval_geometry."+name,
        notes="Explicit Pandas panel conversion and reference kernel; correct semantics, not native acceleration.")
