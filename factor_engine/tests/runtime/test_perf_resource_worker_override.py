"""Worker selection must reach the resource plan, without mutating caller config."""
import pytest

from factor_engine.runtime.perf_config import PerfConfig


def test_explicit_worker_limit_reaches_resource_plan(monkeypatch):
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan
    seen = {}
    def capture(cls, cfg):
        seen.update(cfg)
        return cfg
    monkeypatch.setattr(ExecutionResourcePlan, "from_dict", classmethod(capture))
    config = {"concurrency": {"max_workers": 8, "io_concurrency": 2}}
    PerfConfig(max_workers=1).build_resource_plan(config)
    assert seen["concurrency"] == {"max_workers": 1, "io_concurrency": 2}
    assert config == {"concurrency": {"max_workers": 8, "io_concurrency": 2}}


def test_no_override_preserves_config(monkeypatch):
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan
    monkeypatch.setattr(ExecutionResourcePlan, "from_dict", classmethod(lambda cls, cfg: cfg))
    result = PerfConfig().build_resource_plan({"concurrency": {"max_workers": 3}})
    assert result["concurrency"]["max_workers"] == 3


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
def test_bad_worker_limit_rejected(value):
    with pytest.raises(ValueError, match="max_workers"):
        PerfConfig(max_workers=value)


def test_real_plan_obeys_explicit_serial_override(monkeypatch):
    from factor_engine.runtime import resource_governor
    monkeypatch.setattr(resource_governor, "effective_cpu_slots", lambda: 8)
    monkeypatch.setattr(resource_governor, "effective_memory_limit_bytes", lambda: 64 * 1024**3)
    plan = PerfConfig(max_workers=1).build_resource_plan()
    assert plan.max_workers == 1
