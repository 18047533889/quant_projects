"""Exact Markov reference adapters; explicit full-panel conversion, not native acceleration."""
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base_polars import OperatorMetadata
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import markov_dynamics as source
    return {c.metadata.name:c for c in (
        source.TsKramersMoyalLocalStability,source.TsMarkovEntropyProduction)}[name]()

def metadata(name):
    m=reference(name).metadata
    return OperatorMetadata(name=name,category=m.category,description=m.description,
        param_names=list(m.param_names),panel_params=m.panel_params,scalar_params=m.scalar_params,
        param_specs=dict(m.param_specs),relational_specs=list(m.relational_specs),
        tags=[*m.tags,"pandas_delegate"])

def calculate(name,*args,**kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import (
        to_pandas_panel,from_pandas_panel,verify_frames_share_identity)
    panels=[v for v in (*args,*kwargs.values()) if isinstance(v,pl.DataFrame)]
    if not panels: raise TypeError(f"{name}: a Polars panel is required")
    verify_frames_share_identity(name,*panels)
    def convert(v):
        return to_pandas_panel(v) if isinstance(v,pl.DataFrame) else v
    out=reference(name).calculate(*(convert(v) for v in args),
                                  **{k:convert(v) for k,v in kwargs.items()})
    return from_pandas_panel(panels[0],out)

def physical_spec(name):
    from factor_engine.cleaned_operators import markov_dynamics
    digest=hashlib.sha256(Path(markov_dynamics.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="markov_dynamics."+name,
        notes="Exact causal Markov reference with explicit Polars/Pandas conversion; not native/GPU.")
