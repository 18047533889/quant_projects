"""Small-panel timing/conversion ledger; not production/GPU/100k evidence."""
import json
from time import perf_counter
from statistics import median
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()
rng=np.random.default_rng(720)
index=pd.date_range("2025-01-01",periods=96)
x=pd.DataFrame(rng.normal(size=(96,4)),index=index,columns=list("ABCD"))
score=pd.DataFrame(rng.normal(size=(96,4)),index=index,columns=list("ABCD"))
records=[]
for name,inputs,params in [
    ("ts_kurt",[x],{}),
    ("ts_sma_cn",[x],{}),
    ("lqtp_historical_cvar",[x],{}),
    ("cs_rank_gaussian",[x],{}),
    ("ts_days_since",[(x>0).astype(float)],{}),
]:
    reference=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    native=OperatorRegistry.get(name,"polars",mode="research")
    expected=reference.calculate(*inputs,**params)
    timings=[]
    for repeat in range(3):
        t0=perf_counter()
        reference.calculate(*inputs,**params)
        t1=perf_counter()
        converted=[pl.from_pandas(p.rename_axis("date").reset_index()) for p in inputs]
        t2=perf_counter()
        result=native.calculate(*converted,**params)
        t3=perf_counter()
        result=result.to_pandas().set_index("date").rename_axis(None)
        t4=perf_counter()
        np.testing.assert_allclose(result,expected,equal_nan=True,atol=1e-12,rtol=1e-12)
        timings.append([t1-t0,t2-t1,t3-t2,t4-t3])
    pd_s,in_s,kernel_s,out_s=[median(v[i] for v in timings) for i in range(4)]
    valid=np.isfinite(expected.to_numpy()) & np.isfinite(result.to_numpy())
    difference=float(np.max(abs(expected.to_numpy()[valid]-result.to_numpy()[valid]))) if valid.any() else None
    spec = native.physical_spec() if callable(getattr(native,"physical_spec",None)) else native._physical_spec
    records.append(dict(canonical=name,pandas_seconds=pd_s,input_conversion_seconds=in_s,
        polars_kernel_seconds=kernel_s,output_conversion_seconds=out_s,
        end_to_end_speedup=pd_s/(in_s+kernel_s+out_s),max_absolute_difference=difference,
        execution_kind=spec.execution_kind.value))
print(json.dumps(dict(scope="Synthetic 96x4 CPU, median of 3; no 100k-factor/GPU/production claim.",
    shape=[96,4],repeats=3,records=records),indent=2))
