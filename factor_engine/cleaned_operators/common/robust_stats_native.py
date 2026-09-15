"""Robust rolling statistics on Polars columns using shared NumPy kernels."""
import copy
import hashlib
import inspect
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator,PANEL_SKIP_COLUMNS
from factor_engine.cleaned_operators.common.strict_params import strict_int
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

def reference(name):
    from factor_engine.cleaned_operators import robust_stats as source
    return {c.metadata.name:c for c in (source.TsQuantileRange,source.TsTrimmedMean,
        source.TsRobustZscore,source.TsRobustZscorePrior)}[name]()

def metadata(name):
    m=copy.deepcopy(reference(name).metadata)
    m.tags=[*m.tags,"numpy_kernel","preserve_panel_time_coordinate"]
    return m

def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators import robust_stats as source
    bound=inspect.signature(reference(name)._calculate_series).bind(*args,**kwargs)
    bound.apply_defaults()
    v=bound.arguments
    frame=v["x"]
    w=strict_int(v["window"],"window",minimum=2)
    prior=name.endswith("_prior")
    mp=1 if name=="ts_robust_zscore_inclusive" else source._auto_min_periods(w,v.get("min_periods"))
    lo,hi=v.get("q_low",.25),v.get("q_high",.75)
    trim=v.get("trim_ratio",.1)
    center=str(v.get("center","median")).lower()
    scale=str(v.get("scale","mad")).lower()
    clip=v.get("clip")
    if not 0.<lo<hi<1.:
        raise ValueError("ts_quantile_range requires 0 < q_low < q_high < 1")
    if not 0.<=trim<.5:
        raise ValueError("trim_ratio must be in [0,.5)")
    if center not in ("median","mean") or scale not in ("mad","std"):
        raise ValueError("invalid center or scale")
    outputs=[]
    for c in [c for c in frame.columns if c not in PANEL_SKIP_COLUMNS]:
        arr=frame[c].to_numpy().astype(float)
        out=np.full(len(arr),np.nan)
        for t in range(len(arr)):
            end=t if prior else t+1
            chunk=arr[max(0,end-w):end]
            valid=chunk[np.isfinite(chunk)]
            if valid.size<mp:
                continue
            if name=="ts_quantile_range":
                out[t]=source._quantile_spread(valid,lo,hi)
            elif name=="ts_trimmed_mean":
                ordered=np.sort(valid)
                cut=int(np.floor(trim*len(ordered)))
                selected=ordered[cut:len(ordered)-cut]
                magnitude=float(np.max(np.abs(selected)))
                out[t]=float(np.mean(selected/magnitude))*magnitude if magnitude else 0.
            else:
                out[t]=source._robust_score(valid,float(arr[t]),center,scale,clip)
        outputs.append(pl.Series(c,out,dtype=pl.Float64))
    return frame.with_columns(outputs)

def physical_spec(name):
    from factor_engine.cleaned_operators import robust_stats
    digest=hashlib.sha256(Path(robust_stats.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="robust_stats."+name,
        notes="Per-column rolling NumPy CPU; exact prior/inclusive windows, no Pandas-panel conversion.")

def make(name):
    class RobustNative(SeriesOperator):
        metadata=globals()["metadata"](name)
        _physical_spec=physical_spec(name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
    return RobustNative
