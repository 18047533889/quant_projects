from __future__ import annotations

import pytest

from quant_evaluator.kernels.gpu import quantile_finance_guard as finance_guard
from quant_evaluator.kernels.gpu import quantile_numeric
from quant_evaluator.scripts import benchmark_gpu_finance_specialized_q5_oct04 as bench


class _Kernel:
    def __init__(self, source, name, options):
        self.source = source
        self.name = name
        self.options = options
        self.attributes = {"local_size_bytes": 224}
        self.compile_calls = 0

    def compile(self):
        self.compile_calls += 1


class _CuPy:
    def __init__(self):
        self.calls = []

    def RawKernel(self, source, name, *args, **kwargs):
        kernel = _Kernel(source, name, kwargs.get("options"))
        self.calls.append(kernel)
        return kernel


def test_proxy_routes_only_risk_guard_through_validated_q5_factory():
    cp = _CuPy()
    compiled = []
    proxy = bench._RawKernelProxy(cp, compiled)

    guard = proxy.RawKernel("generic-source", "risk_guard", options=("--std=c++11",))
    other = proxy.RawKernel("other-source", "fixed_mean", options=("--std=c++11",))

    assert guard.name == "finance_risk_guard"
    assert guard.options == ("--std=c++11", "-DFINANCE_Q_CAPACITY=8")
    assert guard.compile_calls == 1
    assert compiled == [guard]
    assert other.name == "fixed_mean"
    assert other.source == "other-source"


def test_specialized_context_restores_cupy_provider_after_exception(monkeypatch):
    cp = _CuPy()
    original = lambda: cp
    monkeypatch.setattr(quantile_numeric, "_cupy", original)

    with pytest.raises(RuntimeError):
        with bench._guard_mode(bench.SPECIALIZED, []):
            assert quantile_numeric._cupy() is not cp
            raise RuntimeError("exercise finally restoration")

    assert quantile_numeric._cupy is original


def test_q5_only_and_provenance_scope_are_restored_after_run(monkeypatch):
    old_counts = bench.common.base.QUANTILE_COUNTS
    old_sources = bench.common.base.SOURCE_FILES
    old_mode = bench.common._guard_mode
    old_labels = (bench.common.CURRENT, bench.common.PROTOTYPE, bench.common.ABBA)

    monkeypatch.setattr(bench.common, "_mem_available_bytes",
                        lambda: bench.common.MIN_HOST_AVAILABLE_BYTES)

    def fake_gpu_run(*, root, **kwargs):
        assert bench.common.base.QUANTILE_COUNTS == (5,)
        assert kwargs["shape"] == bench.SHAPE
        hashes = bench.common.base._source_hashes(root)
        return {
            "shape_TNF": list(kwargs["shape"]),
            "source_sha256_before": hashes,
            "source_sha256_after": hashes,
            "timing_seconds": {},
        }

    monkeypatch.setattr(bench.common.base, "run_benchmark", fake_gpu_run)

    result = bench.run_benchmark()
    hashes = result["source_sha256_before"]

    assert result["quantile_counts"] == [5]
    assert result["compile_capacity"] == 8
    assert result["source_files"] == list(old_sources) + [bench.OWN_SOURCE, bench.common.FINANCE_SOURCE, bench.common.OWN_SOURCE]
    assert bench.OWN_SOURCE in hashes
    assert result["candidate_wrapper_sha256"] == hashes[bench.OWN_SOURCE]
    assert result["parity_call_interpretation"]["total_calls_per_quantile"] == 18

    assert bench.common.base.QUANTILE_COUNTS == old_counts
    assert bench.common.base.SOURCE_FILES == old_sources
    assert bench.common._guard_mode is old_mode
    assert (bench.common.CURRENT, bench.common.PROTOTYPE, bench.common.ABBA) == old_labels


def test_compiled_local_attributes_are_deduplicated_and_reported():
    kernels = [_Kernel("", "finance_risk_guard", ()), _Kernel("", "finance_risk_guard", ())]
    reports = bench._compiled_local_attributes(kernels)

    assert reports == [{
        "local_size_bytes_per_thread": 224,
        "compile_options": ["--std=c++11", "-DFINANCE_Q_CAPACITY=8"],
    }]
