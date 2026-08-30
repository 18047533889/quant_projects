import io, sys, warnings
from contextlib import redirect_stderr, redirect_stdout
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_engine")
buf_out, buf_err = io.StringIO(), io.StringIO()
with redirect_stdout(buf_out), redirect_stderr(buf_err):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from factor_engine.cleaned_operators import load_all
        from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
        load_all()
        register_sql_backends()
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.evidence_provenance import evidence_artifact_valid
from factor_engine.backend.primitive_evidence import PRIMITIVE_BACKEND_EXECUTION_CERTIFIED

print("evidence_artifact_valid:", evidence_artifact_valid())
print("abs in PRIMITIVE_CERTIFIED:", "abs" in PRIMITIVE_BACKEND_EXECUTION_CERTIFIED)
meta = (OperatorRegistry._catalog.get("abs", {}).get("backend_meta") or {})
pm = meta.get("pandas_numpy") or {}
print("abs pandas backend_meta:", {k: pm.get(k) for k in ("production_certified", "certification_source", "status")})
print("abs backends:", OperatorRegistry.backends_for("abs"))
