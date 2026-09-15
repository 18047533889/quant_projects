"""Real return-decomposition NumPy kernels; no Pandas-panel conversion."""
import copy
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.base_polars import SeriesOperator
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import return_decomp as source
    return {c.metadata.name:c for c in (
        source.OvernightReturn,source.OpenCloseReturn,source.OpenToVwapReturn,source.VwapToCloseReturn)}[name]()

def metadata(name):
    m=copy.deepcopy(reference(name).metadata)
    m.tags=[*m.tags,"numpy_kernel","preserve_panel_time_coordinate"]
    return m

def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators.common import polars_state_event
    return getattr(polars_state_event,name)(*args,**kwargs)

def physical_spec(name):
    from factor_engine.cleaned_operators import return_decomp
    digest=hashlib.sha256(Path(return_decomp.__file__).read_bytes()+Path(__file__).read_bytes()+
                         Path(__file__).with_name("polars_state_event.py").read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="polars_state_event."+name,
        notes="Supplied same-basis positive prices; no synthetic VWAP or extra preclose shift, no Pandas-panel conversion.")

def make(name):
    class ReturnDecompNative(SeriesOperator):
        metadata = globals()["metadata"](name)
        _physical_spec = physical_spec(name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
    return ReturnDecompNative
