"""Default facade contract only; not a native execution or real-data certificate."""
import json

from factor_engine import get_engine
from factor_engine.runtime import bounded_pipeline
from factor_engine.tests.runtime.test_v6_default_engine import profile


def test_default_public_run_many_needs_no_performance_arguments(tmp_path, monkeypatch):
    config = tmp_path / "approved-test-profile.json"
    config.write_text(json.dumps(profile(tmp_path)))
    monkeypatch.setenv("FACTOR_ENGINE_V2_PROFILE", str(config))
    observed = {}

    def capture(core, factors, **kwargs):
        observed.update(kwargs)
        observed["factors"] = list(factors)
        observed["core"] = core
        return {"status": "CONTRACT_PROBE_ONLY",
                "deployment_digest": kwargs["run_identity"]["deployment_digest"]}

    monkeypatch.setattr(bounded_pipeline, "execute_run_many_durable", capture)
    factors = ["fixture-a", "fixture-b"]
    cancellation_token = object()
    with get_engine() as engine:
        result = engine.run_many(
            iter(factors), resume_run_id="a" * 32,
            cancellation_token=cancellation_token,
        )
    assert result["status"] == "CONTRACT_PROBE_ONLY"
    assert observed["factors"] == factors
    assert observed["policy"].backend == "auto"
    assert observed["policy"].memory_fraction == 0.80
    assert observed["core"].run_mode == "production"
    assert observed["run_kwargs"]["pit_enforce"] is True
    assert observed["run_kwargs"]["input_dq_strict"] is True
    assert observed["engine_factory_config"].deployment.digest == result["deployment_digest"]
    assert observed["resume_run_id"] == "a" * 32
    assert observed["cancellation_token"] is cancellation_token
