"""Public run_many controls must select the real bounded streaming runner."""
from types import SimpleNamespace

import pytest

from factor_engine.runtime.engine import FactorEngine


@pytest.fixture
def dispatch(monkeypatch):
    from factor_engine.runtime import engine, batch_service, streaming_batch_service
    from factor_engine.planner import physical_lowerer
    calls = []
    monkeypatch.setattr(engine, "_assert_public_batch_authority", lambda *a, **k: None)
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_dag_width_limit", lambda: 128)
    def record(kind):
        def run(*args, **kwargs):
            calls.append((kind, kwargs))
            return {"runner": kind}
        return run
    monkeypatch.setattr(batch_service, "execute_run_many", record("batch"))
    monkeypatch.setattr(streaming_batch_service, "execute_run_many_stream", record("stream"))
    return SimpleNamespace(), calls


def test_default_small_batch_keeps_compatibility(dispatch):
    engine, calls = dispatch
    assert FactorEngine.run_many(engine, []) == {"runner": "batch"}


@pytest.mark.parametrize("controls", [{"wave_size": 16}, {"sink_queue_bytes": 4096}])
def test_explicit_controls_reach_stream_runner(dispatch, controls):
    engine, calls = dispatch
    sink = lambda *a: None
    result = FactorEngine.run_many(engine, [], result_policy="sink", sink=sink, **controls)
    assert result == {"runner": "stream"}
    kind, kwargs = calls[-1]
    assert kwargs["_wave_runner"] == "run_many"
    assert kwargs["sink"] is sink
    for name, value in controls.items():
        assert kwargs[name] == value


@pytest.mark.parametrize("policy", ["return", "yield", "materialize"])
@pytest.mark.parametrize("controls", [{"wave_size": 16}, {"sink_queue_bytes": 4096}])
def test_controls_never_silently_ignored(dispatch, policy, controls):
    engine, calls = dispatch
    with pytest.raises(ValueError, match="require result_policy"):
        FactorEngine.run_many(engine, [], result_policy=policy, **controls)
    assert calls == []


def test_streaming_rejects_precompiled_dag_instead_of_silently_discarding(dispatch):
    engine, calls = dispatch
    with pytest.raises(ValueError, match="precompiled"):
        FactorEngine.run_many(engine, [], result_policy="sink", sink=lambda *a: None,
                              wave_size=16, _compiled=(object(), {}))
    assert calls == []


def test_large_default_sink_still_auto_streams(dispatch):
    engine, calls = dispatch
    FactorEngine.run_many(engine, [object()] * 129, result_policy="sink", sink=lambda *a: None)
    assert calls[-1][0] == "stream"
    assert calls[-1][1]["wave_size"] is None


def test_public_controls_execute_real_waves(monkeypatch):
    from factor_engine.runtime import engine as engine_module
    from factor_engine.planner import physical_lowerer
    monkeypatch.setattr(engine_module, "_assert_public_batch_authority", lambda *a, **k: None)
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_dag_width_limit", lambda: 128)

    class Engine:
        def __init__(self):
            self.waves = []
        def run_many(self, factors, *, sink, **kwargs):
            self.waves.append([f.name for f in factors])
            for factor in factors:
                sink(factor.name, 1)
            return {"results": {}}

    engine = Engine()
    written = []
    result = FactorEngine.run_many(
        engine, [SimpleNamespace(name=str(i)) for i in range(7)],
        result_policy="sink", sink=lambda name, value: written.append(name),
        wave_size=3, sink_queue_bytes=1024,
    )
    assert list(map(len, engine.waves)) == [3, 3, 1]
    assert written == list(map(str, range(7)))
    assert result["completed_factors"] == 7
    assert result["results"] == {}
