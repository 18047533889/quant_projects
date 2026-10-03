from __future__ import annotations

import pytest

from quant_evaluator.kernels.gpu import quantile_numeric
from quant_evaluator.scripts import benchmark_gpu_finance_guard_ab_oct04 as bench


class _FakeKernel:
    def __init__(self, source, name, options):
        self.source = source
        self.name = name
        self.options = options


class _FakeCuPy:
    def __init__(self):
        self.calls = []

    def RawKernel(self, source, name, *args, **kwargs):
        self.calls.append((source, name, args, kwargs))
        return _FakeKernel(source, name, kwargs.get("options"))


def test_factory_maps_only_guard_name_to_prototype():
    cp = _FakeCuPy()
    proxy = bench._RawKernelProxy(cp)

    guard = proxy.RawKernel("generic-source", "risk_guard", options=("--std=c++11",))
    other = proxy.RawKernel("other-source", "fixed_mean", options=("--std=c++11",))

    assert guard.name == "finance_risk_guard"
    assert guard.source == bench.finance_guard.FINANCE_RISK_GUARD_SOURCE
    assert other.name == "fixed_mean"
    assert other.source == "other-source"


def test_prototype_mode_restores_cupy_provider_on_exception(monkeypatch):
    cp = _FakeCuPy()
    original = lambda: cp
    monkeypatch.setattr(quantile_numeric, "_cupy", original)

    with pytest.raises(RuntimeError):
        with bench._guard_mode(bench.PROTOTYPE):
            assert quantile_numeric._cupy() is not cp
            raise RuntimeError("exercise finally restoration")

    assert quantile_numeric._cupy is original


def test_benchmark_source_hashes_include_wrapper_and_finance_source(monkeypatch):
    def fake_run_benchmark(*, root, **kwargs):
        hashes = bench.base._source_hashes(root)
        return {
            "source_sha256_before": hashes,
            "source_sha256_after": hashes,
            "timing_seconds": {},
        }

    monkeypatch.setattr(bench.base, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(bench, "_mem_available_bytes",
                        lambda: bench.MIN_HOST_AVAILABLE_BYTES)
    result = bench.run_benchmark()

    hashes = result["source_sha256_before"]
    assert bench.OWN_SOURCE in result["source_files"]
    assert bench.FINANCE_SOURCE in result["source_files"]
    assert bench.OWN_SOURCE in hashes
    assert bench.FINANCE_SOURCE in hashes
    assert result["guard_policy"][bench.PROTOTYPE].endswith("enabled; no bypass")


def test_t256_requires_ram_floor_before_base_allocation(monkeypatch):
    called = False

    def forbidden_gpu_run(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("must reject before benchmark allocation")

    monkeypatch.setattr(bench, "_mem_available_bytes", lambda: 31 * 1024**3)
    monkeypatch.setattr(bench.base, "run_benchmark", forbidden_gpu_run)

    with pytest.raises(bench.base.BenchmarkError, match="requires 32 GiB"):
        bench.run_benchmark(shape=(256, 5461, 48))

    assert not called
    assert bench.base._guard_mode is not bench._guard_mode


def test_t256_shape_and_memory_receipt(monkeypatch):
    def fake_gpu_run(*, root, **kwargs):
        hashes = bench.base._source_hashes(root)
        return {
            "shape_TNF": list(kwargs["shape"]),
            "source_sha256_before": hashes,
            "source_sha256_after": hashes,
            "timing_seconds": {},
        }

    monkeypatch.setattr(bench, "_mem_available_bytes",
                        lambda: bench.MIN_HOST_AVAILABLE_BYTES)
    monkeypatch.setattr(bench.base, "run_benchmark", fake_gpu_run)

    result = bench.run_benchmark(shape=(256, 5461, 48))
