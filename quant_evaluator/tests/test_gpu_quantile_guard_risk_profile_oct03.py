"""Mocked contracts for the opt-in GPU quantile risk profiler."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_gpu_quantile_guard_risk_oct03.py"
SPEC = importlib.util.spec_from_file_location("gpu_quantile_guard_risk_profile", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(profile)
class _FakeKernel:
    attributes = {"local_size_bytes": 12}
    def __init__(self, name): self.name = name
    def compile(self): pass
    def __call__(self, grid, block, args):
        if self.name in ("risk_guard", "finance_risk_guard"): args[4][:] = [1, 0, 1, 1, 0, 0]
class _FakeCupy:
    def __init__(self):
        self.sync_calls = 0
        self.RawKernel = self._raw_kernel
        self.cuda = SimpleNamespace(Stream=SimpleNamespace(null=SimpleNamespace(
            synchronize=lambda: setattr(self, "sync_calls", self.sync_calls + 1))))
    def _raw_kernel(self, source, name, **kwargs): return _FakeKernel(name)
    @staticmethod
    def asnumpy(value): return np.asarray(value)
    @staticmethod
    def sum(value, axis=None): return np.sum(value, axis=axis)
@pytest.mark.parametrize("guard_name", ["risk_guard", "finance_risk_guard"])
def test_recorder_counts_risk_per_quantile_and_fixed_mean_ids(guard_name):
    cp = _FakeCupy(); original = cp.RawKernel; recorder = profile.KernelLaunchRecorder(cp)
    with recorder.installed():
        risk_kernel = cp.RawKernel("source", guard_name); risk_kernel.compile()
        risk = np.zeros(6, dtype=np.uint8)
        risk_kernel((1,), (1,), (None, None, None, None, risk, None, 2, 3, 3))
        cp.RawKernel("source", "fixed_mean")((1,), (32,), (None, None, None, None, None, None, 17))
    assert cp.RawKernel is original
    assert cp.sync_calls == 1
    assert recorder.snapshot() == {"risk_guard_calls": 1, "risk_count_total": 3,
        "risk_count_by_q": [2, 0, 1], "fixed_mean_launches": 1, "fixed_mean_nids_total": 17}
def test_factory_restores_after_exception():
    cp = _FakeCupy(); original = cp.RawKernel; recorder = profile.KernelLaunchRecorder(cp)
    with pytest.raises(RuntimeError):
        with recorder.installed(): raise RuntimeError("sentinel")

def test_cli_requires_explicit_opt_in(monkeypatch, capsys):
    def forbidden():
        raise AssertionError("real profile must not start")

    monkeypatch.setattr(profile, "run_profile", forbidden)
    with pytest.raises(SystemExit) as error:
        profile.main([])
    assert error.value.code == 2
    assert "--run" in capsys.readouterr().err
    with pytest.raises(SystemExit) as help_exit:
        profile.main(["--help"])
    assert help_exit.value.code == 0

def test_reads_linux_memavailable_as_bytes(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 65536000 kB\nMemFree: 19922944 kB\nMemAvailable: 41943040 kB\n")
    assert profile._host_available_bytes(meminfo) == 40 * 1024**3


def test_missing_memavailable_fails_closed(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 65536000 kB\nMemFree: 41943040 kB\n")
    with pytest.raises(RuntimeError, match="MemAvailable"):
        profile._host_available_bytes(meminfo)


def test_below_32_gib_is_rejected_before_input_generation():
    calls = []
    benchmark = SimpleNamespace(
        DEFAULT_SHAPE=(128, 5461, 48),
        _check_shape=lambda shape: 100,
        _CPU_DRIVER=SimpleNamespace(_make_inputs=lambda *args: calls.append(args)),
    )
    with pytest.raises(RuntimeError, match="32 GiB"):
        profile.run_profile(
            benchmark_module=benchmark,
            host_available_bytes=32 * 1024**3 - 1,
        )
    assert calls == []


def test_exact_32_gib_available_meets_host_admission_floor():
    floor = 32 * 1024**3
    assert profile._admit_host_memory(floor) == floor
