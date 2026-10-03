"""Small mocked checks for the opt-in GPU quantile guard benchmark."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_gpu_quantile_numeric_guard_oct03.py"
SPEC = importlib.util.spec_from_file_location("gpu_quantile_guard_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(benchmark)


class _Clock:
    def __init__(self):
        self.value = 0.0
        self.step = 0.25

    def __call__(self):
        self.value += self.step
        return self.value


class _FakeCupy:
    def __init__(self, free_bytes=16 * 1024**3, total_bytes=16 * 1024**3):
        self.sync_calls = 0
        self.asarray_calls = 0

        class _Stream:
            def synchronize(inner_self):
                self.sync_calls += 1

        self.cuda = SimpleNamespace(
            Stream=SimpleNamespace(null=_Stream()),
            runtime=SimpleNamespace(memGetInfo=lambda: (free_bytes, total_bytes)),
        )

    def asarray(self, value):
        self.asarray_calls += 1
        return np.asarray(value)

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)


def _mocked_run(*, runtime_fingerprint=None, gpu_callable=None):
    cp = _FakeCupy()
    shape = (2, 32, 2)
    references = {}
    calls = []
    original_repair = benchmark.gpu_quantile.repair_quantile_means_gpu
    clock = _Clock()

    def cpu_reference(batch, bundle, *, n_quantiles, min_assets):
        result = benchmark.cpu_quantile.compute_quantile_returns(
            batch, bundle, n_quantiles=n_quantiles, min_assets=min_assets,
        )
        references[n_quantiles] = result
        return result

    def fake_gpu(x, labels, *, n_quantiles, min_assets, return_counts,
                 method, workspace_bytes):
        assert x.shape == (shape[0], shape[2], shape[1])
        assert x.flags.c_contiguous and labels.flags.c_contiguous
        mode = (benchmark.BASELINE
                if benchmark.gpu_quantile.repair_quantile_means_gpu is not original_repair
                else benchmark.GUARDED)
        calls.append((n_quantiles, mode, return_counts, method, workspace_bytes))
        clock.step = len(calls) / 100.0
        result = references[n_quantiles]
        return tuple(part.copy() for part in result)

    identity_calls = []

    def identity(_cp):
        identity_calls.append(True)
        return {"cpu": "mock-cpu", "gpu": "mock-device"}

    result = benchmark.run_benchmark(
        shape=shape,
        rounds=3,
        warmup_cycles=1,
        workspace_bytes=4096,
        cupy_module=cp,
        gpu_callable=gpu_callable or fake_gpu,
        cpu_reference=cpu_reference,
        runtime_fingerprint=runtime_fingerprint or identity,
        clock=clock,
    )
    return result, calls, cp, identity_calls, original_repair


def test_small_mock_checks_abba_parity_hashes_sync_and_no_environment_changes():
    before_env = dict(os.environ)
    result, calls, cp, identity_calls, original_repair = _mocked_run()
    estimate = result["device_memory_estimate"]
    assert result["host_memory_estimate_bytes"] > 0
    assert estimate["factor_input_bytes"] == 8 * 2 * 2 * 32
    assert estimate["label_input_bytes"] == 8 * 2 * 32
    assert estimate["output_bytes"] == 16 * 2 * 2 * 20
    assert estimate["workspace_bytes"] == 4096
    assert estimate["safety_allowance_bytes"] == benchmark.DEVICE_SAFETY_BYTES
    assert cp.asarray_calls == 2

    expected_modes = [
        benchmark.BASELINE, benchmark.GUARDED,
        benchmark.BASELINE, benchmark.GUARDED,
        benchmark.GUARDED, benchmark.BASELINE,
        benchmark.BASELINE, benchmark.GUARDED,
        benchmark.GUARDED, benchmark.BASELINE,
        benchmark.BASELINE, benchmark.GUARDED,
        benchmark.GUARDED, benchmark.BASELINE,
        benchmark.BASELINE, benchmark.GUARDED,
        benchmark.GUARDED, benchmark.BASELINE,
    ]
    for q in benchmark.QUANTILE_COUNTS:
        modes = [mode for call_q, mode, *_ in calls if call_q == q]
        assert modes == expected_modes
    assert all(call[2:] == (True, "max", 4096) for call in calls)
    assert result["warm_order_per_round"] == "ABBA"
    assert result["warm_rounds"] == 3
    assert result["warmup_order_per_cycle"] == "ABBA"
    assert result["parity_calls_checked_per_quantile"] == {"5": 18, "20": 18}
    assert result["source_sha256_before"] == result["source_sha256_after"]
    assert result["input_sha256_before"] == result["input_sha256_after"]
    assert result["device_input_sha256_before"] == result["device_input_sha256_after"]
    assert result["runtime_identity_before"] == result["runtime_identity_after"]
    assert identity_calls == [True, True]
    assert cp.sync_calls >= 74
    assert benchmark.gpu_quantile.repair_quantile_means_gpu is original_repair
    assert dict(os.environ) == before_env
    baseline_samples = result["timing_seconds"]["warm_samples_ABBA"]["5"][benchmark.BASELINE]
    assert len(baseline_samples) == 6 and len(set(baseline_samples)) == 6


def test_temporary_baseline_patch_restores_if_gpu_callable_raises():
    cp = _FakeCupy()
    original = benchmark.gpu_quantile.repair_quantile_means_gpu

    def fail(*args, **kwargs):
        raise RuntimeError("sentinel GPU failure")

    with pytest.raises(RuntimeError, match="sentinel"):
        benchmark.run_benchmark(
            shape=(2, 32, 2), rounds=1, warmup_cycles=0,
            cupy_module=cp, gpu_callable=fail,
            cpu_reference=lambda *args, **kwargs: (
                np.empty((2, 5, 2)), np.empty((2, 5, 2), dtype=np.int32),
            ),
            runtime_fingerprint=lambda _cp: {"mock": "stable"},
            clock=_Clock(),
        )
    assert benchmark.gpu_quantile.repair_quantile_means_gpu is original


def test_rejects_runtime_identity_drift():
    identities = iter(({"mock": "before"}, {"mock": "after"}))
    with pytest.raises(benchmark.BenchmarkError, match="runtime identity changed"):
        _mocked_run(runtime_fingerprint=lambda _cp: next(identities))


def test_cli_is_opt_in_and_help_does_not_run_benchmark(monkeypatch, capsys):
    def forbidden():
        raise AssertionError("wide benchmark must not start")

    monkeypatch.setattr(benchmark, "run_benchmark", forbidden)
    with pytest.raises(SystemExit) as exc:
        benchmark.main([])
    assert exc.value.code == 2
    assert "--run" in capsys.readouterr().err
    with pytest.raises(SystemExit) as help_exit:
        benchmark.main(["--help"])
    assert help_exit.value.code == 0
    assert "--run" in capsys.readouterr().out


def test_three_gib_memory_guard_rejects_oversized_input():
    with pytest.raises(benchmark.BenchmarkError, match="memory estimate"):
        benchmark._check_shape((1, 1000, 70000))


def test_device_admission_rejects_before_any_input_allocation():
    cp = _FakeCupy(free_bytes=1)
    with pytest.raises(benchmark.BenchmarkError, match="before input allocation"):
        benchmark.run_benchmark(
            shape=(2, 32, 2), rounds=1, warmup_cycles=0,
            workspace_bytes=4096, cupy_module=cp,
            gpu_callable=lambda *a, **k: pytest.fail("GPU call must not start"),
            cpu_reference=lambda *a, **k: pytest.fail("reference must not start"),
            runtime_fingerprint=lambda *a: pytest.fail("identity must not start"),
        )
    assert cp.asarray_calls == 0
