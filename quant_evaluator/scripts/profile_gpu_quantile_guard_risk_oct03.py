"""Opt-in GPU quantile numeric-guard risk distribution probe; no timing."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_PATH = ROOT / "scripts" / "benchmark_gpu_quantile_numeric_guard_oct03.py"
MIN_HOST_AVAILABLE_BYTES = 32 * 1024**3


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("_gpu_quantile_guard_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the established GPU quantile benchmark helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelLaunchRecorder:
    """Temporarily wrap RawKernel launches and retain only small risk metadata."""

    def __init__(self, cp):
        self.cp = cp
        self._stats = self._empty_stats()

    @staticmethod
    def _empty_stats():
        return {
            "risk_guard_calls": 0,
            "risk_count_total": 0,
            "risk_count_by_q": [],
            "fixed_mean_launches": 0,
            "fixed_mean_nids_total": 0,
        }

    def snapshot(self):
        return {
            "risk_guard_calls": self._stats["risk_guard_calls"],
            "risk_count_total": self._stats["risk_count_total"],
            "risk_count_by_q": list(self._stats["risk_count_by_q"]),
            "fixed_mean_launches": self._stats["fixed_mean_launches"],
            "fixed_mean_nids_total": self._stats["fixed_mean_nids_total"],
        }

    @contextmanager
    def installed(self):
        cp = self.cp
        original_factory = cp.RawKernel
        recorder = self

        class KernelProxy:
            def __init__(self, kernel, name):
                self._kernel = kernel
                self._name = name

            def __getattr__(self, name):
                return getattr(self._kernel, name)

            def __call__(self, grid, block, args):
                result = self._kernel(grid, block, args)
                if self._name in ("risk_guard", "finance_risk_guard"):
                    cp.cuda.Stream.null.synchronize()
                    risk = args[4]
                    rows, nq = int(args[6]), int(args[8])
                    per_q = cp.sum(risk.reshape((rows, nq)), axis=0)
                    counts = np.asarray(cp.asnumpy(per_q), dtype=np.int64).reshape(-1)
                    recorder._stats["risk_guard_calls"] += 1
                    recorder._stats["risk_count_total"] += int(counts.sum())
                    if not recorder._stats["risk_count_by_q"]:
                        recorder._stats["risk_count_by_q"] = [0] * nq
                    if len(recorder._stats["risk_count_by_q"]) != nq:
                        raise RuntimeError("quantile count changed during one profiling run")
                    for index, value in enumerate(counts):
                        recorder._stats["risk_count_by_q"][index] += int(value)
                elif self._name == "fixed_mean":
                    recorder._stats["fixed_mean_launches"] += 1
                    recorder._stats["fixed_mean_nids_total"] += int(args[-1])
                return result

        def wrapped_factory(source, name, *args, **kwargs):
            kernel = original_factory(source, name, *args, **kwargs)
            if name in ("risk_guard", "finance_risk_guard", "fixed_mean"):
                return KernelProxy(kernel, name)
            return kernel

        cp.RawKernel = wrapped_factory
        try:
            yield self
        finally:
            cp.RawKernel = original_factory
def _host_available_bytes(meminfo_path=Path("/proc/meminfo")):
    try:
        with Path(meminfo_path).open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        raise ValueError("invalid MemAvailable entry")
                    available = int(fields[1]) * 1024
                    if available < 0:
                        raise ValueError("negative MemAvailable value")
                    return available
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("cannot determine Linux MemAvailable for host admission") from exc
    raise RuntimeError("cannot determine Linux MemAvailable: entry is missing")


def _admit_host_memory(available):
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise RuntimeError("cannot determine valid Linux MemAvailable for host admission")
    if available < MIN_HOST_AVAILABLE_BYTES:
        raise RuntimeError(
            f"host admission requires 32 GiB available; found {available} bytes"
        )
    return available


def run_profile(*, benchmark_module=None, cupy_module=None, host_available_bytes=None):
    benchmark = benchmark_module if benchmark_module is not None else _load_benchmark()
    shape = benchmark.DEFAULT_SHAPE
    host_estimate = benchmark._check_shape(shape)
    host_available = _admit_host_memory(
        _host_available_bytes() if host_available_bytes is None else host_available_bytes
    )
    cp = cupy_module if cupy_module is not None else benchmark._cupy()
    workspace_bytes = benchmark.DEFAULT_WORKSPACE_BYTES
    device_estimate = benchmark._device_memory_estimate(shape, workspace_bytes)
    device_available = benchmark._device_memory_preflight(cp, device_estimate)

    batch, bundle = benchmark._CPU_DRIVER._make_inputs(shape, benchmark.SEED)
    x_host = np.ascontiguousarray(batch.values.transpose(0, 2, 1))
    r_host = np.ascontiguousarray(bundle.values)
    input_before = benchmark._input_hashes(batch, bundle, x_host)
    source_before = benchmark._source_hashes()
    benchmark_wrapper_before = benchmark._wrapper_hash()
    profiler_wrapper_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime_before = benchmark._runtime_identity(cp)

    by_quantile = {}
    for nq in benchmark.QUANTILE_COUNTS:
        recorder = KernelLaunchRecorder(cp)
        with recorder.installed():
            result = benchmark.gpu_quantile.batched_quantile_returns(
                x_host, r_host, n_quantiles=nq, method="max",
                min_assets=benchmark.MIN_ASSETS, return_counts=True,
                workspace_bytes=workspace_bytes,
            )
            cp.cuda.Stream.null.synchronize()
        del result
        by_quantile[str(nq)] = recorder.snapshot()

    input_after = benchmark._input_hashes(batch, bundle, x_host)
    source_after = benchmark._source_hashes()
    benchmark_wrapper_after = benchmark._wrapper_hash()
    profiler_wrapper_after = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime_after = benchmark._runtime_identity(cp)
    if input_before != input_after:
        raise RuntimeError("host input hashes changed during risk profiling")
    if source_before != source_after:
        raise RuntimeError("quantile source hashes changed during risk profiling")
    if benchmark_wrapper_before != benchmark_wrapper_after:
        raise RuntimeError("benchmark wrapper changed during risk profiling")
    if profiler_wrapper_before != profiler_wrapper_after:
        raise RuntimeError("risk profiler changed during profiling")
    if runtime_before != runtime_after:
        raise RuntimeError("actual CPU/GPU runtime identity changed during profiling")

    return {
        "scope": "synthetic current GPU quantile numeric-guard risk counts only; no performance timing or route registration",
        "shape_TNF": list(shape),
        "seed": benchmark.SEED,
        "label_scale": benchmark.LABEL_SCALE,
        "quantile_counts": list(benchmark.QUANTILE_COUNTS),
        "min_assets": benchmark.MIN_ASSETS,
        "host_memory_estimate_bytes": host_estimate,
        "host_available_bytes_before_inputs": host_available,
        "device_memory_estimate": device_estimate,
        "device_memory_available_before_inputs": device_available,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "benchmark_wrapper_sha256": benchmark_wrapper_before,
        "risk_profiler_sha256": profiler_wrapper_before,
        "runtime_identity_before": runtime_before,
        "runtime_identity_after": runtime_after,
        "risk_profile_by_quantile": by_quantile,
        "risk_metadata_policy": "risk map reduced on device; only one Q-length integer vector copied to host per risk_guard launch; fixed_mean records launched nids only",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="run the admitted full-size synthetic CUDA risk profile")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("risk profiling is opt-in; pass --run only in an approved GPU slot")
    try:
        report = run_profile()
    except Exception as exc:
        print(f"GPU quantile guard risk profile failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
