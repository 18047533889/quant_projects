"""Small-shape reproducible CPU conversion ledger, not 100k-factor or GPU evidence."""
import json
from statistics import median
from time import perf_counter
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
load_all()
rng=np.random.default_rng(61)
frame=pd.DataFrame(10+np.cumsum(rng.normal(0,.1,(128,4)),axis=0),
                   index=pd.date_range("2025-01-01",periods=128),columns=list("ABCD"))
cases={
    "ts_expectile":dict(x=frame,window=60),
    "ts_expectile_beta":dict(y=frame,x=frame.diff().fillna(0.)+1.,window=60),
    "ts_hampel_filter_causal":dict(x=frame,window=20),
    "ts_median3_causal":dict(x=frame),
    "ts_rolling_median_causal":dict(x=frame,window=5),
}

rows=[]
for name,kwargs in cases.items():
    ref=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    native=OperatorRegistry.get(name,"polars",mode="research")
    times=[]
    for iteration in range(4):
        t=perf_counter(); expected=ref.calculate(**kwargs); pandas_time=perf_counter()-t
        t=perf_counter()
        inputs={k:pl.from_pandas(v.rename_axis("date").reset_index()) if isinstance(v,pd.DataFrame) else v for k,v in kwargs.items()}
        input_time=perf_counter()-t
        t=perf_counter(); result=native.calculate(**inputs); kernel_time=perf_counter()-t
        t=perf_counter(); got=result.to_pandas().set_index("date").rename_axis(None); output_time=perf_counter()-t
        np.testing.assert_allclose(got,expected,atol=1e-12,rtol=1e-12,equal_nan=True)
        if iteration:
            times.append((pandas_time,input_time,kernel_time,output_time))
    values=[median(x[i] for x in times) for i in range(4)]
    error=np.abs(got.to_numpy()-expected.to_numpy()); finite=error[np.isfinite(error)]
    rows.append(dict(canonical=name,pandas_seconds=values[0],input_conversion_seconds=values[1],
                     native_kernel_seconds=values[2],output_conversion_seconds=values[3],
                     including_both_conversions_speedup=values[0]/sum(values[1:]),
                     max_abs_difference=float(finite.max()) if len(finite) else 0.))
print(json.dumps(dict(shape=[128,4],repetitions=3,warmup=1,
    scope="Synthetic CPU microbenchmark; does not establish 100k-factor/GPU performance.",
    conversions="Both Pandas-to-Polars inputs and Polars-to-Pandas output included in end-to-end speedup.",
    results=rows),indent=2))
