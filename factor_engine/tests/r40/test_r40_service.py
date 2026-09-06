"""R40 #84/#85/#88/#92/#94/#95/#149/#150: service.app remediation tests."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

os.environ.setdefault("FACTOR_ENGINE_SERVICE_ROOT", "/tmp/r40_service_root")

import factor_engine.api.dsl_parser  # noqa: E402,F401  (ensure submodule attr set before monkeypatch)
import factor_engine.api.mining_integration  # noqa: E402,F401

from factor_engine.runtime.endpoint_policy import EndpointExecutionPolicy  # noqa: E402
from factor_engine.service.app import (  # noqa: E402
    _redact_preview_for_access,
    _summarize_result,
    _validate_and_build_request,
    _validate_config_path_sources,
    _execute_inline,
)
from factor_engine.service.errors import ServiceError  # noqa: E402
from factor_engine.service.jobstore import JobRecord  # noqa: E402
from factor_engine.service.security import ANONYMOUS_PRINCIPAL  # noqa: E402


def _build(payload):
    return _validate_and_build_request(
        payload,
        endpoint_policy=EndpointExecutionPolicy.RESEARCH,
        principal=ANONYMOUS_PRINCIPAL,
    )


# ---------------------------------------------------------------------------
# #84: config path source validation fails closed on import error
# ---------------------------------------------------------------------------


def test_validate_config_path_sources_fails_closed_on_import_error(monkeypatch):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "factor_engine.runtime.config_runtime", None)
    with pytest.raises(ServiceError) as ei:
        _validate_config_path_sources("/tmp/whatever.yaml", production=True)
    assert ei.value.status == 500
    assert "cannot validate config path sources" in ei.value.message


# ---------------------------------------------------------------------------
# #85: validate_spec production path passes market to the validator
# ---------------------------------------------------------------------------


def test_validate_spec_passes_market_to_production_validator(monkeypatch):
    captured: dict[str, object] = {}

    def fake_validate_production_dsl(formula, *, market=None):
        captured["formula"] = formula
        captured["market"] = market
        return True, "OK"

    monkeypatch.setattr(
        "factor_engine.api.mining_integration.validate_production_dsl", fake_validate_production_dsl
    )
    # parse_expr in the concurrent session is mid-edit (NameError) — stub it out.
    monkeypatch.setattr("factor_engine.api.dsl_parser.parse_expr", lambda *a, **k: None)
    monkeypatch.setattr(
        "factor_engine.api.dsl_parser.DSLParseError", type("DSLParseError", (Exception,), {})
    )
    from factor_engine.service.app import validate_spec

    result = validate_spec(
        {"formula": "close", "run_mode": "production", "market": "us"}
    )
    assert result["ok"] is True
    assert captured["market"] == "us"


def test_validate_spec_production_requires_explicit_market(monkeypatch):
    captured: dict[str, object] = {}

    def fake_validate_production_dsl(formula, *, market=None):
        captured["market"] = market
        return True, "OK"

    monkeypatch.setattr(
        "factor_engine.api.mining_integration.validate_production_dsl", fake_validate_production_dsl
    )
    monkeypatch.setattr("factor_engine.api.dsl_parser.parse_expr", lambda *a, **k: None)
    monkeypatch.setattr(
        "factor_engine.api.dsl_parser.DSLParseError", type("DSLParseError", (Exception,), {})
    )
    from factor_engine.service.app import validate_spec

    result = validate_spec({"formula": "close", "run_mode": "production"})
    assert not result["ok"]
    assert "explicit market" in " ".join(result["errors"])
    assert captured == {}


# ---------------------------------------------------------------------------
# #88: build_data_source receives DataSourceBuildContext
# ---------------------------------------------------------------------------


def test_execute_inline_passes_build_context(monkeypatch):
    execution, digest = _build(
        {
            "formula": "close",
            "name": "f",
            "market": "ashare",
            "calendar": "SSE",
            "data_source": {"type": "data_access", "dataset": "d"},
        }
    )
    captured: dict[str, object] = {}

    def fake_bds(cfg, build_context=None):
        captured["bc"] = build_context
        return object()

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def run(self, *a, **k):
            return {"result": None}

    monkeypatch.setattr("factor_engine.storage.factory.build_data_source", fake_bds)
    monkeypatch.setattr("factor_engine.backend.factory.build_backend", lambda *a, **k: object())
    monkeypatch.setattr("factor_engine.runtime.engine.FactorEngine", FakeEngine)
    from factor_engine.api.factor import Factor
    monkeypatch.setattr("factor_engine.api.dsl_parser.parse_factor", lambda *a, **k: Factor(name=k["name"], expr=None))

    job = JobRecord(run_id="bc_job", request={"execution": execution}, request_digest=digest)
    _execute_inline(job, execution)
    bc = captured["bc"]
    assert bc is not None
    assert bc.market == "ashare"
    assert bc.calendar_id == "SSE"
    assert bc.run_mode == "research"


def test_execution_dict_contains_build_context():
    execution, _ = _build({"formula": "close", "market": "ashare", "calendar": "SSE"})
    bc = execution["build_context"]
    assert bc["market"] == "ashare"
    assert bc["calendar_id"] == "SSE"
    assert bc["run_mode"] == "research"


# ---------------------------------------------------------------------------
# #92: execution dict preserves the factor name
# ---------------------------------------------------------------------------


def test_execution_dict_keeps_factor_name():
    execution, _ = _build({"formula": "close", "name": "keep_me"})
    assert execution["name"] == "keep_me"


def test_execute_inline_parse_factor_gets_name(monkeypatch):
    execution, digest = _build(
        {"formula": "close", "name": "parse_me", "data_source": {"type": "data_access", "dataset": "d"}}
    )
    captured: dict[str, object] = {}

    def fake_parse_factor(formula, *, name, **kwargs):
        captured["name"] = name
        from factor_engine.api.factor import Factor
        return Factor(name=name, expr=None)

    monkeypatch.setattr("factor_engine.storage.factory.build_data_source", lambda *a, **k: object())
    monkeypatch.setattr("factor_engine.backend.factory.build_backend", lambda *a, **k: object())

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def run(self, *a, **k):
            return {"result": None}

    monkeypatch.setattr("factor_engine.runtime.engine.FactorEngine", FakeEngine)
    monkeypatch.setattr("factor_engine.api.dsl_parser.parse_factor", fake_parse_factor)
    job = JobRecord(run_id="nm_job", request={"execution": execution}, request_digest=digest)
    _execute_inline(job, execution)
    assert captured["name"] == "parse_me"


# ---------------------------------------------------------------------------
# #94: import service.app has no side effects (threads / dirs)
# ---------------------------------------------------------------------------


def test_import_service_app_has_no_side_effects():
    code = (
        "import os, threading\n"
        'os.environ["FACTOR_ENGINE_SERVICE_ROOT"] = "/tmp/r40_import_side_effect"\n'
        "import factor_engine.service.app\n"
        "bad = [t.name for t in threading.enumerate()\n"
        "       if t.name.startswith(('fe-job-worker', 'fe-heartbeat', 'factor-engine-job'))]\n"
        "print('THREADS', bad)\n"
        "print('DIR', os.path.exists('/tmp/r40_import_side_effect'))\n"
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "THREADS []" in out.stdout
    assert "DIR False" in out.stdout


# ---------------------------------------------------------------------------
# #95: reload_policies does not affect in-flight job snapshots
# ---------------------------------------------------------------------------


def test_policy_reload_does_not_affect_inflight_jobs(tmp_path, monkeypatch):
    import factor_engine.service.app as app

    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    app._ensure_runtime(start=True)
    old_digest = app.FEATURE_POLICY.digest()
    job = JobRecord(
        run_id="inflight",
        status="running",
        policy_id="runtime_feature_policy",
        policy_version=app._POLICY_VERSION,
        policy_digest=old_digest,
    )
    from factor_engine.service.policies import RuntimeFeaturePolicy

    fake = RuntimeFeaturePolicy(flags={"FASTPATH": True}, production=False)
    monkeypatch.setattr(
        RuntimeFeaturePolicy,
        "from_env",
        classmethod(lambda cls, *a, **k: fake),
    )
    result = app.reload_policies()
    assert result["changed"] is True
    assert result["policy_version"] == job.policy_version + 1
    # in-flight job retains the OLD snapshot
    assert job.policy_digest == old_digest
    assert job.policy_version < result["policy_version"]


def test_submit_job_snapshots_policy(monkeypatch, tmp_path):
    import factor_engine.service.app as app

    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(tmp_path))
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_PER_PRINCIPAL_JOBS", "100")
    app._ensure_runtime(start=True)
    job = JobRecord(
        run_id="snap",
        status="running",
        policy_id="runtime_feature_policy",
        policy_version=app._POLICY_VERSION,
        policy_digest=app.FEATURE_POLICY.digest(),
    )
    assert job.policy_id == "runtime_feature_policy"
    assert job.policy_digest == app.FEATURE_POLICY.digest()


# ---------------------------------------------------------------------------
# #149: catalog_generations includes all 7 dimensions
# ---------------------------------------------------------------------------


def test_catalog_generations_includes_all_dimensions():
    execution, _ = _build({"formula": "close", "backend": "pandas"})
    gens = execution["validated"]["catalog_generations"]
    assert len(gens) == 7
    assert all(isinstance(g, str) and g for g in gens)


def test_catalog_generations_backend_change_changes_digest():
    exec_a, dig_a = _build({"formula": "close", "backend": "pandas"})
    from dataclasses import replace
    from factor_engine.service.models import ValidatedFactorRequest
    from factor_engine.service.errors import ServiceError
    bound = ValidatedFactorRequest(**exec_a["validated"])
    assert dig_a != replace(bound, backend="polars", resolved_backend_policy="polars").digest()
    with pytest.raises(ServiceError, match="physical-plan admission"):
        _build({"formula": "close", "backend": "polars"})


def gens(execution):
    return execution["validated"]["catalog_generations"]


# ---------------------------------------------------------------------------
# #150: result preview inherits source access classification
# ---------------------------------------------------------------------------


def test_result_preview_inherits_source_access_classification():
    premium = {"type": "data_access", "dataset": "p", "access_tags": ["alt.premium"]}
    job_read = JobRecord(run_id="r1", request_metadata={"roles": ["READ"]})
    redacted = _redact_preview_for_access(premium, job_read)
    assert redacted is not None and "redacted" in redacted
    # higher-privilege caller can read the raw preview
    job_admin = JobRecord(run_id="r2", request_metadata={"roles": ["ADMIN"]})
    assert _redact_preview_for_access(premium, job_admin) is None
    # public source is never redacted
    public = {"type": "data_access", "dataset": "d", "access_tags": ["public"]}
    assert _redact_preview_for_access(public, job_read) is None


def test_summarize_result_redacts_premium_preview():
    class FakeResult:
        shape = (5, 2)

        def head(self, n):
            return self

        def to_string(self):
            return "col0 col1\n1 2"

    job = JobRecord(
        run_id="r3",
        request_metadata={"roles": ["READ"]},
        request={"execution": {"data_source": {"access_tags": ["alt.premium"]}}},
    )
    _summarize_result(
        {"mode": "inline_dsl", "out": {"result": FakeResult()}}, job
    )
    assert "redacted" in job.artifacts["result_preview"]


def test_summarize_result_keeps_public_preview():
    class FakeResult:
        shape = (5, 2)

        def head(self, n):
            return self

        def to_string(self):
            return "col0 col1\n1 2"

    job = JobRecord(
        run_id="r4",
        request_metadata={"roles": ["READ"]},
        request={"execution": {"data_source": {"access_tags": ["public"]}}},
    )
    _summarize_result(
        {"mode": "inline_dsl", "out": {"result": FakeResult()}}, job
    )
    assert job.artifacts["result_preview"].startswith("col0 col1")
