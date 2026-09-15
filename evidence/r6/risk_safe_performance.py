"""Small synthetic performance/conversion ledger; not a 100k-factor benchmark."""
import json
import time
import statistics
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
load_all()
rng=np.random.default_rng(779)
x=pd.DataFrame(rng.normal(size=(128,4)),index=pd.date_range("2025-01-01",periods=128),columns=list("ABCD"))
def measured(fn):
    elapsed=[]
    for _ in range(3):
        start=time.perf_counter()
        value=fn()
        elapsed.append(time.perf_counter()-start)
    return value,statistics.median(elapsed)
px,conversion=measured(lambda:pl.from_pandas(x.rename_axis("date").reset_index()))
records=[]
for name,args in (
    ("ts_positive_ratio",{"window":20}),
    ("ts_abs_concentration",{"window":20}),
    ("ts_downside_deviation",{"window":20}),
    ("ts_argmax_age",{"window":20}),
    ("ts_ewm_std",{"span":.2}),
    ("fillna_const",{"value":0.}),
):
    pdop=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    plop=OperatorRegistry.get(name,"polars",mode="research")
    pdval,pdseconds=measured(lambda:pdop.calculate(x,**args))
    original=pl.DataFrame.to_pandas
    def forbidden(*a,**kw):
        raise AssertionError("Internal Pandas panel conversion")
    pl.DataFrame.to_pandas=forbidden
    try:
        plval,plseconds=measured(lambda:plop.calculate(px,**args))
    finally:
        pl.DataFrame.to_pandas=original
    actual=plval.select(x.columns).to_numpy()
    expected=pdval.to_numpy()
    np.testing.assert_allclose(actual,expected,equal_nan=True,rtol=1e-10,atol=1e-10)
    delta=np.abs(actual-expected)
    records.append({"canonical":name,"pandas_seconds":pdseconds,"polars_seconds":plseconds,
                    "input_conversion_seconds":conversion,"speedup_kernel":pdseconds/plseconds,
                    "speedup_including_input_conversion":pdseconds/(plseconds+conversion),
                    "max_abs_difference":float(np.nanmax(delta)),
                    "internal_pandas_panel_conversions":0})
print(json.dumps({"scope":"synthetic 128 rows x 4 securities, median 3; not 100k factors",
                  "default_bootstrap":True,"records":records},indent=2))
