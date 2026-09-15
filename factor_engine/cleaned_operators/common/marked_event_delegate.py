"""Explicit marked-event reference delegates; no native or GPU claim."""
import copy
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base_polars import SeriesOperator
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

NAMES = ("event_mark_autocorr", "event_interval_mark_coupling")

def reference(name):
    from factor_engine.cleaned_operators.marked_event import EventMarkAutocorr, EventIntervalMarkCoupling
    return dict(zip(NAMES, (EventMarkAutocorr, EventIntervalMarkCoupling)))[name]()

def register(name):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    class MarkedEventPolars(SeriesOperator):
        metadata = copy.deepcopy(reference(name).metadata)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self, *args, **kwargs):
            import polars as pl
            from factor_engine.cleaned_operators.common._polars_bridge import (
                to_pandas_panel, from_pandas_panel, verify_frames_share_identity)
            panels = [v for v in (*args, *kwargs.values()) if isinstance(v, pl.DataFrame)]
            if not panels:
                raise TypeError(name + ": input panels required")
            verify_frames_share_identity(name, *panels)
            convert = lambda v: to_pandas_panel(v) if isinstance(v, pl.DataFrame) else v
            result = reference(name).calculate(*(convert(v) for v in args),
                **{k: convert(v) for k, v in kwargs.items()})
            return from_pandas_panel(panels[0], result)
        def physical_spec(self):
            from factor_engine.cleaned_operators import marked_event
            digest = hashlib.sha256(Path(marked_event.__file__).read_bytes() +
                Path(__file__).read_bytes()).hexdigest()
            return PhysicalImplementationSpec(canonical=name, backend="polars",
                execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
                materializes_full_panel=True, supports_nulls=True, supports_nan=True,
                supports_inf=True, implementation_source_hash=digest,
                kernel_identity="marked_event." + name,
                notes="Explicit Pandas CPU reference conversion; no native/GPU claim.")
    OperatorRegistry.register(MarkedEventPolars(), canonical=name, backend="polars",
        source="marked_event_reference", status="implemented", backend_explicit=True)
