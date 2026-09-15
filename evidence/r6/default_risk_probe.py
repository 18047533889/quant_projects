"""Default load acceptance for risk kernels, without pytest bootstrap hooks."""
import json
import runpy
import pytest
from factor_engine.cleaned_operators import load_all
load_all()
suite=runpy.run_path("factor_engine/tests/runtime/test_r6_direction_risk_contracts.py")
for name in sorted(suite["PARAMS"]):
    with pytest.MonkeyPatch.context() as patch:
        suite["test_final_risk_call_contract_prefix_and_native_parity"](name,patch)
coords=runpy.run_path("factor_engine/tests/runtime/test_r6_direction_risk_coordinates.py")
for timezone in (None,"Asia/Hong_Kong"):
    coords["test_excess_nanosecond_identity_and_slice"](timezone)
print(json.dumps({"status":"PASS","bootstrap":"fresh default load_all","risk_operators":14,"coordinate_checks":2}))
