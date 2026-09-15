"""Default-bootstrap acceptance of the six actual interval-geometry operators."""
import json
import runpy
from factor_engine.cleaned_operators import load_all
load_all()
suite=runpy.run_path("factor_engine/tests/runtime/test_r6_interval_contracts.py")
for name in suite["_NEW_CANONICALS"]:
    suite["test_final_interval_scalar_binding_numerics_prefix"](name)
for backend in ("pandas_numpy","polars"):
    suite["test_mode_distance_uses_exactly_window_prior_intervals"](backend)
print(json.dumps({"status":"PASS","bootstrap":"fresh default load_all","operators":6,"checks":8}))
