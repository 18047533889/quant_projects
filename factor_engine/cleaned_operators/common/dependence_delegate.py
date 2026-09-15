"""Explicit dependence reference delegates. These are not native/GPU kernels."""
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def reference(name):
    if name == "ts_conditional_transfer_entropy":
        from factor_engine.cleaned_operators.conditional_dependence import TsConditionalTransferEntropy
        return TsConditionalTransferEntropy()
    from factor_engine.cleaned_operators import dependence_ext as source
    return {c.metadata.name:c for c in (
        source.TsChatterjeeXi, source.TsHsic,
        source.TsConditionalMutualInformation, source.TsPartialDistanceCorrelation)}[name]()

def metadata(name):
    m = reference(name).metadata
    return OperatorMetadata(name=name,category=m.category,description=m.description,
        param_names=list(m.param_names),panel_params=m.panel_params,scalar_params=m.scalar_params,
        param_specs=dict(m.param_specs),relational_specs=list(m.relational_specs),
        tags=[*m.tags,"pandas_delegate"])

def calculate(name,*args,**kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import to_pandas_panel, from_pandas_panel
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    panels = [v for v in (*args,*kwargs.values()) if isinstance(v,pl.DataFrame)]
    if not panels:
        raise TypeError(f"{name}: a Polars panel is required")
    verify_frames_share_identity(name,*panels)
    def convert(v):
        return to_pandas_panel(v) if isinstance(v,pl.DataFrame) else v
    out=reference(name).calculate(*(convert(v) for v in args),**{k:convert(v) for k,v in kwargs.items()})
    return from_pandas_panel(panels[0],out)

def physical_spec(name):
    from factor_engine.cleaned_operators import dependence_ext
    kernel_source = Path(dependence_ext.__file__)
    if name == "ts_conditional_transfer_entropy":
        from factor_engine.cleaned_operators import conditional_dependence
        kernel_source = Path(conditional_dependence.__file__)
    digest=hashlib.sha256(kernel_source.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="dependence_ext."+name,
        notes="Explicit full-panel conversion to dependence reference, not native acceleration.")

def register(name):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    class DependencePolars(SeriesOperator):
        metadata = globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    OperatorRegistry.register(DependencePolars(),canonical=name,backend="polars",
        source="dependence_reference_polars",backend_explicit=True,
        status="research_only" if name=="ts_conditional_transfer_entropy" else "implemented")
