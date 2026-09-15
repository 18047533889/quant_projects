"""Real per-column NumPy directional-change state machine for Polars panels."""
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import OperatorMetadata,SeriesOperator,PANEL_SKIP_COLUMNS
from factor_engine.cleaned_operators.parameter_validation import strict_integer,strict_finite_scalar
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

NAMES={ "ts_dc_overshoot_ratio":"overshoot_ratio","ts_dc_event_rate":"event_rate",
    "ts_dc_duration_asymmetry":"duration_asymmetry","ts_dc_overshoot_asymmetry":"overshoot_asymmetry" }

def metadata(name):
    from factor_engine.cleaned_operators import directional_change as s
    refs={c.metadata.name:c for c in (s.TsDcOvershootRatio,s.TsDcEventRate,s.TsDcDurationAsymmetry,s.TsDcOvershootAsymmetry)}
    m=refs[name].metadata
    return OperatorMetadata(name=name,category=m.category,description=m.description,param_names=list(m.param_names),
        param_specs=dict(m.param_specs),panel_params=m.panel_params,scalar_params=m.scalar_params,
        input_units=dict(m.input_units),compatible_units=dict(m.compatible_units),tags=[*m.tags,"numpy_kernel"])

def calculate(name,x,scale,threshold=1.,window=120,threshold_mode="adaptive",scale_mode="absolute"):
    from factor_engine.cleaned_operators.directional_change import _dc_column
    w=strict_integer(window,"window",minimum=3)
    threshold=strict_finite_scalar(threshold,"threshold")
    if threshold<=0:
        raise ValueError("threshold must be positive")
    if threshold_mode not in ("adaptive","fixed_absolute") or scale_mode not in ("absolute","relative"):
        raise ValueError("unknown directional-change threshold/scale mode")
    outputs=[]
    for col in x.columns:
        if col in PANEL_SKIP_COLUMNS:
            continue
        prices=x[col].to_numpy().astype(float)
        scales=scale[col].to_numpy().astype(float)
        values=_dc_column(prices,scales,threshold,w,threshold_mode,scale_mode,requested=NAMES[name])[NAMES[name]]
        outputs.append(pl.Series(col,values,dtype=pl.Float64))
    return x.with_columns(outputs)

def physical_spec(name):
    from factor_engine.cleaned_operators import directional_change
    digest=hashlib.sha256(Path(directional_change.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
        materializes_full_panel=False,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="directional_change."+name,
        notes="Full-history replay per column, NumPy arrays only; no Pandas-panel conversion, not GPU.")

def make(name):
    def _calculate_series(self,*args,**kwargs):
        return calculate(name,*args,**kwargs)
    return type("DirectionalChange_"+name,(SeriesOperator,),dict(metadata=metadata(name),
        _physical_spec=physical_spec(name),_calculate_series=_calculate_series,__module__=__name__))()
