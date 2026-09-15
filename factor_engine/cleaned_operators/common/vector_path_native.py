"""Exact two-panel vector geometry on Polars columns, using NumPy CPU kernels."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator, PANEL_SKIP_COLUMNS
from factor_engine.cleaned_operators.common.strict_params import strict_int
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import vector_path as source
    return {c.metadata.name:c for c in (
        source.TsVectorPathEfficiency, source.TsVectorTurningCoherence,
        source.TsVectorPathCurvature, source.TsVectorSelfIntersectionRate)}[name]()

def metadata(name):
    out=copy.deepcopy(reference(name).metadata)
    out.tags=[*out.tags, "numpy_kernel", "preserve_panel_time_coordinate"]
    return out

def calculate(name, f1, f2, window=60, **kwargs):
    from factor_engine.cleaned_operators import vector_path as source
    w=strict_int(window, "window", minimum=2)
    if f1.shape != f2.shape or f1.columns != f2.columns:
        raise ValueError("vector path inputs must have identical axes")
    for c in f1.columns:
        if c in PANEL_SKIP_COLUMNS and not f1[c].equals(f2[c]):
            raise ValueError("vector path input time coordinates must agree")
    kernel={
        "ts_vector_path_efficiency":source._path_efficiency,
        "ts_vector_turning_coherence":source._turning_coherence,
        "ts_vector_path_curvature":source._path_curvature,
        "ts_vector_self_intersection_rate":source._self_intersection_rate,
    }[name]
    out=[]
    for c in f1.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        a=f1[c].to_numpy().astype(float).reshape(-1,1)
        b=f2[c].to_numpy().astype(float).reshape(-1,1)
        values=source._pair_series(a,b,w,kernel)[:,0]
        out.append(pl.Series(c,values,dtype=pl.Float64))
    return f1.with_columns(out)

def physical_spec(name):
    from factor_engine.cleaned_operators import vector_path
    digest=hashlib.sha256(Path(vector_path.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="vector_path."+name,
        notes="Per-column NumPy CPU geometry; two real panels; full contiguous support. No GPU claim.")

def make(name):
    class VectorNative(SeriesOperator):
        metadata=globals()["metadata"](name)
        _physical_spec=physical_spec(name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,f1,f2,window=60,**kwargs):
            return calculate(name,f1,f2,window,**kwargs)
    return VectorNative
