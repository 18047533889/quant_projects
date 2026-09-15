"""Exact OHLC/Roll CPU kernels with real price panels, not proxy formulas."""
import copy
import hashlib
import inspect
from pathlib import Path
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator,PANEL_SKIP_COLUMNS
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

NAMES=("ohlc_corwin_schultz_spread","ts_roll_effective_spread")
def reference(name):
    from factor_engine.cleaned_operators import spread_estimators as source
    return (source.OhlcCorwinSchultzSpread if name==NAMES[0] else source.TsRollEffectiveSpread)()
def metadata(name):
    out=copy.deepcopy(reference(name).metadata)
    out.tags=[*out.tags,"numpy_kernel"]
    return out
def calculate(name,*args,**kwargs):
    from factor_engine.cleaned_operators import spread_estimators as source
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    bound=inspect.signature(reference(name)._calculate_series).bind(*args,**kwargs)
    bound.apply_defaults();values=bound.arguments
    panels=[values[p] for p in (("high","low") if name==NAMES[0] else ("price",))]
    verify_frames_share_identity(name,*panels)
    if name==NAMES[0]:
        w=strict_int(values["smooth_window"],"smooth_window",minimum=1)
    else:
        w=strict_int(values["window"],"window",minimum=5)
        mp=max(4,w//2) if values["min_periods"] is None else strict_int(values["min_periods"],"min_periods",minimum=4,maximum=w-1)
    base=panels[0];out=[]
    for c in base.columns:
        if c in PANEL_SKIP_COLUMNS:
            continue
        a=base[c].to_numpy().astype(float).reshape(-1,1)
        if name==NAMES[0]:
            b=panels[1][c].to_numpy().astype(float).reshape(-1,1)
            result=source._cs_spread_series(a,b,w)
        else:
            result=source._roll_spread_series(a,w,mp)
        out.append(pl.Series(c,result[:,0],dtype=pl.Float64))
    return base.with_columns(out)
def physical_spec(name):
    from factor_engine.cleaned_operators import spread_estimators
    digest=hashlib.sha256(Path(spread_estimators.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
    return PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=True,
        implementation_source_hash=digest,kernel_identity="spread_estimators."+name,
        notes="CPU shared log-price spread kernels; atomic positive-price column gate preserved for Roll. No GPU claim.")
def make(name):
    class SpreadNative(SeriesOperator):
        metadata=globals()["metadata"](name)
        @property
        def _contract_callable(self):
            return reference(name)._calculate_series
        def _calculate_series(self,*args,**kwargs):
            return calculate(name,*args,**kwargs)
        def physical_spec(self):
            return physical_spec(name)
    return SpreadNative
