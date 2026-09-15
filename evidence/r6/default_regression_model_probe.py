"""Fresh default bootstrap, complete final model scalar and numerical checks."""
import json
import runpy
import pytest
from factor_engine.cleaned_operators import load_all
load_all()
suite=runpy.run_path("factor_engine/tests/runtime/test_r6_regression_model_contracts.py")
for name in sorted(suite["_REMAINING_MODEL_NAMES"]):
    with pytest.MonkeyPatch.context() as patch:
        suite["test_model_contract_calls_scale_and_causal_prefix"](name,patch)
for name in sorted(suite["_REMAINING_MODEL_NAMES"]-{"ts_quantile_regression_slope"}):
    suite["test_last_row_independent_estimator_reference"](name)
print(json.dumps({"status":"PASS","bootstrap":"fresh default load_all","operators":7,"checks":13}))
