"""Bounded synthetic weighted-tail conversion ledger, not 100k-factor throughput."""
import json,time,statistics
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
load_all()
rng=np.random.default_rng(57)
rows,cols=160,4
x=pd.DataFrame(rng.normal(size=(rows,cols)),index=pd.date_range("2025-01-01",periods=rows),columns=[f"S{i}" for i in range(cols)])
weights=x.abs()+.5
def convert(v):
    return pl.from_pandas(v.rename_axis("date").reset_index()) if isinstance(v,pd.DataFrame) else v
def measure(f):
    vals=[]
    answer=None
    for _ in range(3):
        start=time.perf_counter()
        answer=f()
        vals.append(time.perf_counter()-start)
    return statistics.median(vals),answer
cases={
    "ts_stratified_mean_spread":dict(target=x,sorter=weights,window=40,quantile=.2,min_periods=5),
    "ts_weighted_semivariance":dict(x=x,weight=weights,window=40),
    "ts_weighted_downside_deviation":dict(x=x,weight=weights,window=40),
    "ts_weighted_expected_shortfall":dict(x=x,weight=weights,window=40,quantile=.2,min_tail_count=2),
    "ts_weighted_drawdown_area":dict(x=x.abs()+1,weight=weights,window=40),
}
ledger=[]
for name,kw in cases.items():
    ref=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    op=OperatorRegistry.get(name,"polars",mode="research")
    conversion,inputs=measure(lambda:{k:convert(v) for k,v in kw.items()})
    pt,pout=measure(lambda:ref.calculate(**kw))
    nt,nout=measure(lambda:op.calculate(**inputs))
    actual=nout.select(x.columns.tolist()).to_numpy()
    expected=pout.to_numpy()
    np.testing.assert_allclose(actual,expected,equal_nan=True,atol=1e-10,rtol=1e-10)
    mask=np.isfinite(expected)
    ledger.append(dict(operator=name,pandas_seconds=pt,polars_seconds=nt,input_conversion_seconds=conversion,
        speedup_kernel=pt/nt,speedup_including_input_conversion=pt/(nt+conversion),
        max_abs_difference=float(np.max(np.abs(actual[mask]-expected[mask]))) if mask.any() else None,
        final_polars_owner=type(op).__module__+"."+type(op).__qualname__))
print(json.dumps(dict(scope="Synthetic 160x4 panels, median of 3; no 100k-factor throughput or GPU claim.",
    rows=rows,columns=cols,output_conversion_timed=False,operators=ledger),indent=2))
