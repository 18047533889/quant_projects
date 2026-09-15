"""The default facade must pass actual optimization settings, not trust legacy env."""
import json
from factor_engine import get_engine
from factor_engine.runtime import bounded_pipeline
from factor_engine.tests.runtime.test_v6_default_engine import profile


def test_default_call_pins_dag_fusion_and_automatic_workers(tmp_path, monkeypatch):
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(profile(tmp_path)))
    monkeypatch.setenv("FACTOR_ENGINE_V2_PROFILE", str(config))
    for key, value in {
        "FACTOR_ENGINE_DISABLE_CSE": "1",
        "FACTOR_ENGINE_DISABLE_PANEL_NATIVE": "1",
        "FACTOR_ENGINE_NATIVE_FUSION": "0",
        "FACTOR_ENGINE_SCHEDULER": "legacy",
        "FACTOR_ENGINE_MAX_WORKERS": "1",
        "FACTOR_ENGINE_OPERATOR_BACKEND": "pandas",
    }.items():
        monkeypatch.setenv(key, value)
    observed = {}
    def capture(core, factors, **kwargs):
        observed.update(kwargs)
        return {"status": "CONTRACT_PROBE_ONLY"}
    monkeypatch.setattr(bounded_pipeline, "execute_run_many_durable", capture)
    with get_engine() as engine:
        engine.run_many(iter(()))
    options = observed["run_kwargs"]
    assert observed["automatic_dag"] is True
    assert options["enable_cse"] is True
    perf = options["perf"]
    assert perf.enable_cse and perf.panel_native and perf.native_fusion
    assert perf.scheduler == "adaptive"
    assert perf.operator_backend == "auto"
    assert perf.max_workers is None
    assert options["pit_enforce"] and options["input_dq_strict"]
    assert observed["policy"].memory_fraction == 0.80
