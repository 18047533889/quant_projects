"""Actual Polars/NumPy rolling estimators; no Pandas panel conversion."""
import numpy as np
from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
PARAMS={
 "ts_quantile_regression_slope":("y","x","window","q","min_periods"),
 "ts_variance_ratio_proxy":("x","window","q","min_periods"),
 "ts_lo_mackinlay_vr":("x","window","q","min_periods"),
 "ts_lo_mackinlay_z":("x","window","q","min_periods"),
 "ts_cumulative_deviation_score":("x","window","min_periods"),
 "ts_level_shift_score":("x","window","min_periods"),
 "ts_vol_shift_score":("x","window","min_periods")}
def contract(name):
    from factor_engine.cleaned_operators.regression_models import _remaining_specs
    names=PARAMS[name]
    return dict(param_names=list(names),panel_params=tuple(k for k in names if k in {"x","y"}),
                scalar_params=tuple(k for k in names if k not in {"x","y"}),param_specs=_remaining_specs(name))
def _single(x,window,min_periods,name,q=None):
    import polars as pl
    from factor_engine.cleaned_operators import regression_models as m
    if name=="ts_level_shift_score":
        window=m.strict_integer(window,"window",minimum=20)
    floor=6 if q is not None else 3 if name=="ts_cumulative_deviation_score" else 4
    w,mp=m._remaining_window(window,min_periods,floor,levels_extra=int(name=="ts_variance_ratio_proxy"))
    if q is not None:
        q=m.strict_integer(q,"q",minimum=2)
        if q>w-2:
            raise ValueError("q must be <= window - 2")
    outputs=[]
    for c in x.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        values=x[c].cast(pl.Float64).to_numpy()
        out=np.full(len(values),np.nan)
        for row in range(len(values)):
            segment=m._scale_finite(values[max(0,row-w+1):row+1])
            vals=m._trailing_contiguous(segment)
            if len(vals)<mp:
                continue
            if q is not None:
                rets=np.diff(vals)
                if name=="ts_variance_ratio_proxy":
                    if len(rets)<mp or len(rets)-q+1<2 or np.var(rets)<=0:
                        continue
                    qrets=np.array([vals[i+q]-vals[i] for i in range(len(rets)-q+1)])
                    out[row]=np.var(qrets)/(q*np.var(rets))-1.
                elif len(rets)>=q+1:
                    out[row]=(m._lo_mackinlay_z if name=="ts_lo_mackinlay_z" else m._lo_mackinlay_vr)(rets,q)
            elif name=="ts_cumulative_deviation_score":
                sd=vals.std()
                if sd>0:
                    out[row]=np.max(np.abs(np.cumsum((vals-vals.mean())/sd)))
            else:
                first,second=m._trailing_run_halves(segment)
                if not len(first) or not len(second):
                    continue
                if name=="ts_level_shift_score":
                    sd=vals.std()
                    if sd>0:
                        out[row]=(second.mean()-first.mean())/sd
                elif len(first)>=2 and len(second)>=2:
                    a,b=first.std(),second.std()
                    if a>0 and b>0:
                        out[row]=np.log(b)-np.log(a)
        outputs.append(pl.Series(c,out))
    return x.with_columns(outputs)
def ts_variance_ratio_proxy(x,window=60,q=5,min_periods=10):
    return _single(x,window,min_periods,"ts_variance_ratio_proxy",q)
def ts_lo_mackinlay_vr(x,window=60,q=5,min_periods=10):
    return _single(x,window,min_periods,"ts_lo_mackinlay_vr",q)
def ts_lo_mackinlay_z(x,window=60,q=5,min_periods=10):
    return _single(x,window,min_periods,"ts_lo_mackinlay_z",q)
def ts_cumulative_deviation_score(x,window=20,min_periods=5):
    return _single(x,window,min_periods,"ts_cumulative_deviation_score")
def ts_level_shift_score(x,window=20,min_periods=5):
    return _single(x,window,min_periods,"ts_level_shift_score")
def ts_vol_shift_score(x,window=20,min_periods=5):
    return _single(x,window,min_periods,"ts_vol_shift_score")
def ts_quantile_regression_slope(y,x,window=20,q=.5,min_periods=5):
    import polars as pl
    from factor_engine.cleaned_operators import regression_models as m
    from factor_engine.cleaned_operators.common._polars_bridge import align_cols
    w,mp=m._remaining_window(window,min_periods,3)
    q=m.strict_finite_scalar(q,"q")
    if not 0<q<1:
        raise ValueError("q must be in (0,1)")
    output=[]
    for c in align_cols(y,x):
        ya,xa=y[c].cast(pl.Float64).to_numpy(),x[c].cast(pl.Float64).to_numpy()
        vals=[m._quantile_slope(ya[max(0,row-w+1):row+1],xa[max(0,row-w+1):row+1],q,mp) for row in range(len(ya))]
        output.append(pl.Series(c,vals))
    return y.with_columns(output)
def physical_spec(name):
    import hashlib
    from pathlib import Path
    from factor_engine.cleaned_operators import regression_models as m
    from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec
    digest=hashlib.sha256(Path(__file__).read_bytes()+Path(m.__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
        materializes_full_panel=True,supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="regression_model_polars."+name,
        notes="NumPy per-column estimator; pinball LP for quantile slope. No Pandas panels or GPU claim.")
