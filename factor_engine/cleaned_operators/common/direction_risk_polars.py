"""Polars/NumPy kernels for direction and downside-risk contracts (no Pandas panels)."""
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec, strict_bool_param
from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

DIRECTION = {
 "ts_positive_ratio": ("x","window","threshold","min_periods"),
 "ts_negative_ratio": ("x","window","threshold","min_periods"),
 "ts_zero_ratio": ("x","window","tolerance","min_periods"),
 "ts_abs_concentration": ("x","window","min_periods"),
 "ts_abs_entropy": ("x","window","normalize","min_periods"),
 "ts_abs_entropy_normalized": ("x","window","min_periods"),
 "ts_abs_entropy_nats": ("x","window","min_periods"),
}
RISK = {
 "ts_downside_deviation": ("x","window","target","min_periods"),
 "ts_upside_deviation": ("x","window","target","min_periods"),
 "ts_current_drawdown_duration": ("x","window"),
 "ts_time_under_water": ("x","window"),
 "ts_best_lag_corr_raw": ("y","x","window","max_lag"),
 "ts_best_lag_corr_excess": ("y","x","window","max_lag"),
 "ts_price_delay": ("stock_return","benchmark_return","window","max_lag","min_periods"),
}
PARAMS = {**DIRECTION, **RISK}

def contract(name):
    names = PARAMS[name]
    panels = tuple(p for p in names if p in {"x","y","stock_return","benchmark_return"})
    minimum = 1 if name in DIRECTION else 2
    mp = 1 if name in DIRECTION else 3 if name=="ts_price_delay" else 2
    specs = {
        "window": ParamSpec(dtype=int,min=minimum,default=20,param_role=ParamRole.HORIZON),
        "min_periods": ParamSpec(dtype=int,min=mp,default=mp,param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "threshold": ParamSpec(dtype=float,min=0.,default=0.,param_role=ParamRole.STATE_THRESHOLD),
        "tolerance": ParamSpec(dtype=float,min=0.,default=0.,param_role=ParamRole.STATE_THRESHOLD),
        "target": ParamSpec(dtype=float,default=0.,param_role=ParamRole.STATE_THRESHOLD),
        "normalize": ParamSpec(dtype=bool,choices=(True,),default=True,searchable=False,param_role=ParamRole.NUMERICAL),
        "max_lag": ParamSpec(dtype=int,min=1 if name=="ts_price_delay" else 0,default=5,param_role=ParamRole.HORIZON),
    }
    return dict(param_names=list(names),panel_params=panels,tags=["preserve_panel_time_coordinate"],
                scalar_params=tuple(p for p in names if p not in panels),
                param_specs={p:specs[p] for p in names if p in specs})

def _window(window,min_periods,minimum=1):
    w=strict_integer(window,"window",minimum=minimum)
    mp=strict_integer(min_periods,"min_periods",minimum=minimum)
    if mp>w:
        raise ValueError("min_periods must be <= window")
    return w,mp

def _cols(x):
    return [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]

def _ratio(x,window,threshold,min_periods,sign):
    w,mp=_window(window,min_periods)
    threshold=strict_finite_scalar(threshold,"tolerance" if sign==0 else "threshold",minimum=0.)
    expressions=[]
    for c in _cols(x):
        value=pl.col(c)
        valid=value.is_finite().fill_null(False)
        hits=value>threshold if sign>0 else value<threshold if sign<0 else value.abs()<=threshold
        count=valid.cast(pl.Float64).rolling_sum(w,min_samples=1)
        total=(hits.fill_null(False)&valid).cast(pl.Float64).rolling_sum(w,min_samples=1)
        expressions.append(pl.when(count>=mp).then(total/count).otherwise(None).alias(c))
    return x.with_columns(expressions)

def ts_positive_ratio(x,window=20,threshold=0.,min_periods=1):
    return _ratio(x,window,threshold,min_periods,1)
def ts_negative_ratio(x,window=20,threshold=0.,min_periods=1):
    return _ratio(x,window,threshold,min_periods,-1)
def ts_zero_ratio(x,window=20,tolerance=0.,min_periods=1):
    return _ratio(x,window,tolerance,min_periods,0)

def _rolling_numpy(x,w,fn):
    columns=[]
    for c in _cols(x):
        values=x[c].cast(pl.Float64).to_numpy()
        output=np.asarray([fn(values[max(0,t-w+1):t+1]) for t in range(len(values))],dtype=float)
        columns.append(pl.Series(c,output))
    return x.with_columns(columns)

def _mass(x,window,min_periods,entropy,normalized):
    from factor_engine.cleaned_operators.direction_concentration import _abs_concentration, _abs_entropy
    w,mp=_window(window,min_periods)
    return _rolling_numpy(x,w,lambda a:_abs_entropy(a,mp,normalized) if entropy else _abs_concentration(a,mp))
def ts_abs_concentration(x,window=20,min_periods=1):
    return _mass(x,window,min_periods,False,False)
def ts_abs_entropy(x,window=20,normalize=True,min_periods=1):
    if not strict_bool_param(normalize,"normalize"):
        raise ValueError("Use ts_abs_entropy_nats for nats output")
    return _mass(x,window,min_periods,True,True)
def ts_abs_entropy_normalized(x,window=20,min_periods=1):
    return _mass(x,window,min_periods,True,True)
def ts_abs_entropy_nats(x,window=20,min_periods=1):
    return _mass(x,window,min_periods,True,False)

def _deviation(x,window,target,min_periods,up):
    from factor_engine.cleaned_operators.downside_risk import _stable_deviation_rms
    w,mp=_window(window,min_periods,2)
    target=strict_finite_scalar(target,"target")
    def kernel(a):
        finite=a[np.isfinite(a)]
        if len(finite)<mp:
            return np.nan
        with np.errstate(over="ignore",invalid="ignore"):
            d=finite-target
        return _stable_deviation_rms(np.maximum(d,0.) if up else np.minimum(d,0.))
    return _rolling_numpy(x,w,kernel)
def ts_downside_deviation(x,window=20,target=0.,min_periods=2):
    return _deviation(x,window,target,min_periods,False)
def ts_upside_deviation(x,window=20,target=0.,min_periods=2):
    return _deviation(x,window,target,min_periods,True)

def _drawdown(x,window,duration):
    w=strict_integer(window,"window",minimum=2)
    def kernel(a):
        if not np.isfinite(a[-1]):
            return np.nan
        peak=-np.inf;streak=0;under=0;count=0
        for value in a:
            if not np.isfinite(value):
                peak=-np.inf;streak=0
                continue
            count+=1;peak=max(peak,value)
            if value<peak:
                under+=1;streak+=1
            else:
                streak=0
        return float(streak) if duration else under/count
    return _rolling_numpy(x,w,kernel)
def ts_current_drawdown_duration(x,window=20):
    return _drawdown(x,window,True)
def ts_time_under_water(x,window=20):
    return _drawdown(x,window,False)

def _models(y,x,window,max_lag,min_periods,kind):
    from factor_engine.cleaned_operators.common._polars_bridge import align_cols
    from factor_engine.cleaned_operators.downside_risk import (
        _best_lag_corr_raw,_best_lag_corr_excess,_stable_surrogate_seed,_price_delay_model)
    from factor_engine.backend.operator_semantic_version import versioned_name
    w=strict_integer(window,"window",minimum=2)
    ml=strict_integer(max_lag,"max_lag",minimum=1 if kind=="delay" else 0)
    if ml>=w:
        raise ValueError("max_lag must be < window")
    mp=strict_integer(min_periods,"min_periods",minimum=3) if kind=="delay" else 1
    if mp>w or (kind=="delay" and w<2*(ml+2)):
        raise ValueError("STATIC_DOMAIN_INFEASIBLE: insufficient window")
    columns=align_cols(y,x)
    time_col=next((c for c in ("__fe_time__","date","timestamp","trade_date","datetime") if c in y.columns),None)
    if kind=="excess" and time_col is None:
        raise ValueError("ts_best_lag_corr_excess requires an explicit date coordinate")
    times=list(range(y.height))
    if time_col is not None:
        series=y[time_col]
        if isinstance(series.dtype,pl.Datetime):
            # Python datetime drops sub-microsecond precision; keep scalar identity exact.
            from pandas import Timestamp
            times=[None if v is None else Timestamp(v,unit=series.dtype.time_unit,tz=series.dtype.time_zone)
                   for v in series.cast(pl.Int64).to_list()]
        else:
            times=series.to_list()
    observed=[t for t in times if t is not None]
    if len(set(observed))!=len(observed):
        raise ValueError("absolute time coordinate must be unique")
    result=[]
    for c in columns:
        ya,xa=y[c].cast(pl.Float64).to_numpy(),x[c].cast(pl.Float64).to_numpy()
        output=np.full(y.height,np.nan)
        for t,time in enumerate(times):
            if kind=="raw":
                output[t]=_best_lag_corr_raw(xa,ya,t,w,ml)
            elif kind=="delay":
                output[t]=_price_delay_model(ya,xa,t,w,ml,mp)
            elif time is not None:
                seed=_stable_surrogate_seed(42,time,c,versioned_name("ts_best_lag_corr_excess"))
                output[t]=_best_lag_corr_excess(xa,ya,t,w,ml,identity_seed=seed)
        result.append(pl.Series(c,output))
    return y.with_columns(result)
def ts_best_lag_corr_raw(y,x,window=20,max_lag=5):
    return _models(y,x,window,max_lag,1,"raw")
def ts_best_lag_corr_excess(y,x,window=20,max_lag=5):
    return _models(y,x,window,max_lag,1,"excess")
def ts_price_delay(stock_return,benchmark_return,window=20,max_lag=5,min_periods=3):
    return _models(stock_return,benchmark_return,window,max_lag,min_periods,"delay")


def physical_spec(name):
    """Execution/provenance declaration, not numerical or production certification."""
    import hashlib
    from pathlib import Path
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    folder=Path(__file__).resolve().parent.parent
    files=[Path(__file__).resolve(),folder/"direction_concentration.py",folder/"downside_risk.py"]
    digest=hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    return PhysicalImplementationSpec(
        canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR if name in {"ts_positive_ratio","ts_negative_ratio","ts_zero_ratio"} else ExecutionKind.POLARS_NUMPY_KERNEL,
        materializes_full_panel=True,requires_sorted=True,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="direction_risk_polars."+name,
        emitter_identity="direction_risk_polars:v2",
        parameter_domain_hash=name+":authored-contract:v2",semantic_contract_hash=name+":finite-support:causal:v2",
        notes="Polars expressions or per-column NumPy kernels; no Pandas panel conversion, no GPU claim.")
