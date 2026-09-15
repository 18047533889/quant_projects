"""Fresh default bootstrap acceptance of four scalar cleaning canonicals."""
import json
import runpy
import pytest
from factor_engine.cleaned_operators import load_all
load_all()
suite=runpy.run_path("factor_engine/tests/runtime/test_r6_cleaning_scalars.py")
for backend in ("pandas_numpy","polars"):
    for name in ("ts_ewm_std","ts_ewm_var"):
        for span in (.2,1.,2.5,20.):
            with pytest.MonkeyPatch.context() as patch:
                suite["test_ewm_fractional_decay_missing_and_prefix"](name,span,backend,patch)
    for name in ("fillna_const","nonfinite_to_num"):
        suite["test_constant_missing_vs_nonfinite"](name,backend)
print(json.dumps({"status":"PASS","bootstrap":"fresh default load_all","operators":4,"checks":20}))
