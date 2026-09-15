"""Counterexample isolation uses the existing exact evidence identity."""
import pytest
from factor_engine.runtime.parameter_domain_store import CertificationKey, ParameterDomainCertificationStore
from factor_engine.runtime.default_execution_policy import ExecutionPurpose
from factor_engine.runtime.exceptions import BackendBug, KnownBadImplementation
from factor_engine.runtime.production_policy import assert_production_fastpath_runtime
from types import SimpleNamespace


def test_missing_certificate_is_not_known_failure():
    store = ParameterDomainCertificationStore()
    assert not store.exact_call_has_known_failure("ts_mean", {"window": 4})


def test_failure_is_scoped_to_exact_backend_implementation_input_and_parameters():
    store = ParameterDomainCertificationStore()
    values = {"semantic_version": "v3", "backend": "sql", "execution_variant": "current",
              "source_context": "data_access", "dtype": "float64", "grain": "daily"}
    store.certify_point(CertificationKey("ts_kurt", parameter_point=(("window", 4),), **values),
                        False, source="independent-counterexample")
    assert store.exact_call_has_known_failure("ts_kurt", {"window": 4}, **values)
    assert not store.exact_call_has_known_failure("ts_kurt", {"window": 5}, **values)
    for field, different in {"semantic_version": "v4", "backend": "pandas_numpy",
                             "execution_variant": "fixed", "source_context": "other",
                             "dtype": "float32", "grain": "intraday"}.items():
        assert not store.exact_call_has_known_failure(
            "ts_kurt", {"window": 4}, **{**values, field: different})
    assert issubclass(KnownBadImplementation, BackendBug)
    assert KnownBadImplementation.reason_code == "KNOWN_BAD_IMPLEMENTATION"


def test_research_purpose_skips_only_fastpath_certificate_audit(monkeypatch):
    import factor_engine.backend.production_fastpath_gate as gate
    monkeypatch.setattr("factor_engine.runtime.production_policy._fastpath_gate_enabled",
                        lambda: True)
    seen = []
    monkeypatch.setattr(gate, "audit_runtime_fastpath_violations", lambda stats: seen.append(stats) or [])
    ctx = SimpleNamespace(run_mode="production", runtime_stats={}, execution_purpose=ExecutionPurpose())
    assert_production_fastpath_runtime(ctx, mode="production")
    assert not seen
    ctx.execution_purpose = ExecutionPurpose("production_compute")
    assert_production_fastpath_runtime(ctx, mode="production")
    assert len(seen) == 1
    assert ctx.run_mode == "production"
