"""Bounded synthetic benchmark; no production data and no certification claim."""
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_semantic_version import semantic_version
from factor_engine.cleaned_operators import envelope

load_all()
rng=np.random.default_rng(601)
shape=(512,8)
lower=pd.DataFrame(90+rng.normal(size=shape))
upper=lower+pd.DataFrame(5+rng.random(shape)*10)
mid=(upper+lower)/2
x=lower+(upper-lower)*pd.DataFrame(rng.random(shape))
cases={
    "ts_envelope_compression": [upper,lower,mid],
    "ts_envelope_pressure": [x,upper,lower],
    "ts_envelope_boundary_dwell": [x,upper,lower],
}
rows=[]
for name,panels in cases.items():
    reference=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    native=OperatorRegistry.get(name,"polars",mode="research")
    t=time.perf_counter()
    pp=[pl.from_pandas(p) for p in panels]
    conversion=time.perf_counter()-t
    expected=reference.calculate(*panels,window=20)
    actual=native.calculate(*pp,window=20).to_pandas()
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10,equal_nan=True)
    timings={}
    for backend,operator,inputs in (("pandas_numpy",reference,panels),("polars",native,pp)):
        samples=[]
        for _ in range(3):
            t=time.perf_counter()
            operator.calculate(*inputs,window=20)
            samples.append(time.perf_counter()-t)
        timings[backend]=statistics.median(samples)
    rows.append({
        "canonical":name,"semantic_version":semantic_version(name),
        "median_seconds":timings,"pandas_to_polars_input_seconds":conversion,
        "kernel_speedup":timings["pandas_numpy"]/timings["polars"],
        "end_to_end_speedup_including_input_conversion":timings["pandas_numpy"]/(timings["polars"]+conversion),
        "finite_output_cells":int(np.isfinite(actual.to_numpy()).sum()),
        "max_abs_difference":float(np.nanmax(np.abs(actual.to_numpy()-expected.to_numpy()))),
        "internal_pandas_numpy_conversions":0,
        "conversion_evidence":"test_r6_envelope_contracts forbids pandas/numpy conversion during final native calls",
    })
print(json.dumps({
    "scope":"Synthetic microbenchmark, not 100k-factor throughput or production certification",
    "rows":shape[0],"instruments":shape[1],"window":20,"repetitions":3,
    "python":platform.python_version(),"pandas":pd.__version__,"polars":pl.__version__,
    "polars_threads":pl.thread_pool_size(),
    "kernel_source_sha256":hashlib.sha256(Path(envelope.__file__).read_bytes()).hexdigest(),
    "results":rows,
},indent=2))
