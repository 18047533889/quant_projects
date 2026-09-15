"""Exact Polars-to-pandas delegates for drawdown-path estimators."""
from __future__ import annotations
import copy, hashlib
from pathlib import Path
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import SeriesOperator

def reference(name):
    from factor_engine.cleaned_operators.stateful import drawdown_path as source
    classes=(source.TsRecoveryFraction,source.TsCurrentDrawdownArea)
    return {c.metadata.name:c for c in classes}[name]()

def calculate(name,*args,**kwargs):
    import polars as pl
    from factor_engine.cleaned_operators.common._polars_bridge import to_pandas_panel,from_pandas_panel
    panels=[v for v in (*args,*kwargs.values()) if isinstance(v,pl.DataFrame)]
    if not panels: raise TypeError(f"{name}: a Polars panel is required")
    cv=lambda v:to_pandas_panel(v) if isinstance(v,pl.DataFrame) else v
    out=reference(name).calculate(*(cv(v) for v in args),**{k:cv(v) for k,v in kwargs.items()})
    return from_pandas_panel(panels[0],out)

def physical_spec(name):
    from factor_engine.cleaned_operators.stateful import drawdown_path
    digest=hashlib.sha256(Path(drawdown_path.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,supports_lazy=False,
        supports_streaming=False,materializes_full_panel=True,supports_nulls=True,
        supports_nan=True,supports_inf=True,implementation_source_hash=digest,
        kernel_identity="stateful.drawdown_path."+name,
        notes="Exact full-panel pandas authority delegate; CPU only, not native acceleration.")

def make(name):
    class DrawdownPathPolars(SeriesOperator):
        metadata=copy.deepcopy(reference(name).metadata)
        _physical_spec=physical_spec(name)
        @property
        def _contract_callable(self): return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs): return calculate(name,*args,**kwargs)
    return DrawdownPathPolars
