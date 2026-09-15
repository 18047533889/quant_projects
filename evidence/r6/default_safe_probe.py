"""Actual default-bootstrap safe kernels and conversion accounting assertions."""
import json
import runpy
import pytest
import polars as pl
from factor_engine.cleaned_operators import load_all
load_all()
suite=runpy.run_path("factor_engine/tests/runtime/test_r6_safe_contracts.py")
for backend in ("pandas_numpy","polars"):
    for name in suite["EXTREMES"]:
        suite["test_extreme_finite_ties_and_real_bar_positions"](name,backend)
    suite["test_group_imputation_is_cross_sectional_bounded_and_finite"](backend)
    suite["test_ffill_nan_limit_and_lineage_are_enforced"](backend)
print(json.dumps({"status":"PASS","bootstrap":"fresh default load_all","operators":6,"backend_checks":12}))
