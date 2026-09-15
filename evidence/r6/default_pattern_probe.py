"""Exercise pattern regressions in a fresh default bootstrap, without pytest hooks."""
import json
import runpy
from factor_engine.cleaned_operators import load_all

load_all()
suite = runpy.run_path("factor_engine/tests/runtime/test_r6_extra_pattern_contracts.py")
checked = 0
for name in sorted(suite["EXTRA_PATTERNS"]):
    suite["test_final_pattern_native_parity_keywords_prefix_and_scalar_bounds"](name)
    checked += 1
for backend in ("pandas_numpy", "polars"):
    suite["test_rounding_independent_quadratic_reference_and_unknown_warmup"](backend)
    suite["test_triple_top_requires_fifth_pivot_confirmation"](backend)
    suite["test_123_signal_requires_trailing_low_high_low_order"](backend)
    checked += 3
print(json.dumps({"bootstrap": "fresh default load_all without pytest hooks", "checks": checked, "status": "PASS"}))
