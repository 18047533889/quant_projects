"""Opt-in synthetic CUDA benchmark for quantile numeric-guard overhead.

This compares the current GPU quantile callable with only its exact numeric
repair bypassed versus enabled. It is not a full QE, COS, or fastest-backend
comparison. Host preparation, input transfer, and cold calls are reported
separately from synchronized warm GPU-call samples.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_evaluator.kernels.gpu import quantile as gpu_quantile
from quant_evaluator.metrics import quantile as cpu_quantile

CPU_DRIVER_PATH = ROOT / "quant_evaluator" / "scripts" / "benchmark_quantile_numeric_guard_oct03.py"
GPU_SOURCE_FILES = (
    "quant_evaluator/kernels/gpu/quantile.py",
    "quant_evaluator/kernels/gpu/rank.py",
    "quant_evaluator/kernels/gpu/quantile_numeric.py",
)
SOURCE_FILES = GPU_SOURCE_FILES + (
    "quant_evaluator/metrics/quantile.py",
    "quant_evaluator/metrics/quantile_numeric.py",
    "quant_evaluator/scripts/benchmark_quantile_numeric_guard_oct03.py",
)
DEFAULT_SHAPE = (128, 5461, 48)
SEED = 81033
LABEL_SCALE = 0.02
QUANTILE_COUNTS = (5, 20)
MIN_ASSETS = 10
MAX_ESTIMATED_BYTES = 3 * 1024**3
DEFAULT_WORKSPACE_BYTES = 1 << 30
DEVICE_SAFETY_BYTES = 256 * 1024**2
BASELINE = "guard_bypassed"
GUARDED = "guard_enabled"
ABBA = (BASELINE, GUARDED, GUARDED, BASELINE)


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot preserve its parity/provenance contract."""


def _load_cpu_driver():
    spec = importlib.util.spec_from_file_location("_quantile_guard_cpu_input_driver", CPU_DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise BenchmarkError("could not load the existing CPU input generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CPU_DRIVER = _load_cpu_driver()


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise BenchmarkError("CuPy is required; CPU fallback is not supported") from exc
    return cp


def _check_shape(shape: tuple[int, int, int]) -> int:
    if (not isinstance(shape, tuple) or len(shape) != 3
            or any(type(value) is not int or value < 1 for value in shape)):
        raise BenchmarkError("shape must be a tuple of three positive builtin integers")
    try:
        estimate = _CPU_DRIVER._check_shape(*shape)
    except Exception as exc:
        raise BenchmarkError(str(exc)) from exc
    if estimate > MAX_ESTIMATED_BYTES:
        raise BenchmarkError("conservative peak memory estimate exceeds the 3 GiB guard")
    return estimate


def _device_memory_estimate(shape: tuple[int, int, int], workspace_bytes: int) -> dict[str, int]:
    time_count, asset_count, factor_count = shape
    max_quantiles = max(QUANTILE_COUNTS)
    factor_input_bytes = 8 * time_count * factor_count * asset_count
    label_input_bytes = 8 * time_count * asset_count
    output_bytes = 16 * time_count * factor_count * max_quantiles
    total = (factor_input_bytes + label_input_bytes + output_bytes
             + workspace_bytes + DEVICE_SAFETY_BYTES)
    if total > MAX_ESTIMATED_BYTES:
        raise BenchmarkError("GPU device staging/workspace estimate exceeds the 3 GiB guard")
    return {
        "factor_input_bytes": factor_input_bytes,
        "label_input_bytes": label_input_bytes,
        "output_bytes": output_bytes,
        "workspace_bytes": workspace_bytes,
        "safety_allowance_bytes": DEVICE_SAFETY_BYTES,
        "total_estimated_bytes": total,
    }


def _device_memory_preflight(cp, estimate: dict[str, int]) -> dict[str, int]:
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        free_bytes, total_bytes = int(free_bytes), int(total_bytes)
    except Exception as exc:
        raise BenchmarkError("could not query CUDA free/total memory before device allocation") from exc
    required = estimate["total_estimated_bytes"]
    if free_bytes < required or total_bytes < required:
        raise BenchmarkError(
            f"device memory admission failed before input allocation: "
            f"need {required} bytes, free {free_bytes}, total {total_bytes}"
        )
    return {"free_bytes_before_allocation": free_bytes,
            "total_bytes": total_bytes}


def _array_digest(array) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _input_hashes(batch, bundle, x_host) -> dict[str, str]:
    parts = {
        "factor_values_TNF": _array_digest(batch.values),
        "factor_validity_TNF": _array_digest(batch.validity),
        "gpu_layout_factor_values_TFN": _array_digest(x_host),
        "labels_TN": _array_digest(bundle.values),
        "label_validity_TN": _array_digest(bundle.validity),
    }
    combined = hashlib.sha256()
    for name, value in sorted(parts.items()):
        combined.update(name.encode("ascii"))
        combined.update(value.encode("ascii"))
    parts["combined"] = combined.hexdigest()
    return parts


def _source_hashes(root: Path = ROOT) -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        try:
            result[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkError(f"could not read benchmark source {relative}") from exc
    return result


def _wrapper_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as exc:
        raise BenchmarkError("could not read benchmark wrapper for provenance") from exc


def _sync(cp) -> None:
    cp.cuda.Stream.null.synchronize()


def _runtime_identity(cp) -> dict:
    # Importing threadpoolctl only queries actual pools; it does not change any
    # environment variable or thread count.
    try:
        import threadpoolctl  # noqa: F401
        from quant_evaluator.runtime.source_runtime_identity import (
            capture_source_runtime_identity,
            is_qualified_source_runtime_identity,
        )
        snapshot = capture_source_runtime_identity()
    except Exception as exc:
        raise BenchmarkError("could not capture actual CPU/GPU runtime identity") from exc
    if not is_qualified_source_runtime_identity(snapshot):
        raise BenchmarkError("actual CPU/GPU runtime identity is incomplete")
    runtime = cp.cuda.runtime
    try:
        return {
            "cpu": {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "logical_cpu_count": os.cpu_count(),
            },
            "source_runtime_identity": snapshot,
            "cuda_driver_version": int(runtime.driverGetVersion()),
            "cuda_runtime_version": int(runtime.runtimeGetVersion()),
        }
    except Exception as exc:
        raise BenchmarkError("could not capture CUDA driver/runtime identity") from exc


def _device_input_hashes(cp, x_device, r_device) -> dict[str, str]:
    _sync(cp)
    return {
        "factor_values_TFN": _array_digest(cp.asnumpy(x_device)),
        "labels_TN": _array_digest(cp.asnumpy(r_device)),
    }


@contextmanager
def _guard_mode(mode: str):
    if mode == GUARDED:
        yield
        return
    if mode != BASELINE:
        raise BenchmarkError(f"unknown benchmark mode: {mode}")
    original = gpu_quantile.repair_quantile_means_gpu
    try:
        gpu_quantile.repair_quantile_means_gpu = lambda *args, **kwargs: args[2]
        yield
    finally:
        gpu_quantile.repair_quantile_means_gpu = original


def _assert_result_equal(actual, expected, *, context: str) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise BenchmarkError(f"{context} did not return (returns, counts)")
    try:
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(
            actual[0], expected[0], rtol=1e-9, atol=1e-12, equal_nan=True,
        )
    except AssertionError:
        raise BenchmarkError(f"{context} quantile values/counts parity failed") from None


def _host_result(cp, result, shape: tuple[int, int, int]):
    if not isinstance(result, tuple) or len(result) != 2:
        raise BenchmarkError("GPU callable must return (returns, counts)")
    host = (np.asarray(cp.asnumpy(result[0])), np.asarray(cp.asnumpy(result[1])))
    expected = shape
    if host[0].shape != expected or host[1].shape != expected:
        raise BenchmarkError("GPU callable returned an invalid output shape")
    return host


def _timed_call(*, cp, mode: str, q: int, x_device, r_device,
                gpu_callable: Callable, reference, shape, workspace_bytes: int,
                clock: Callable[[], float]):
    with _guard_mode(mode):
        _sync(cp)
        started = clock()
        result = gpu_callable(
            x_device, r_device, n_quantiles=q, method="max",
            min_assets=MIN_ASSETS, return_counts=True,
            workspace_bytes=workspace_bytes,
        )
        _sync(cp)
        elapsed = clock() - started
    host = _host_result(cp, result, (shape[0], q, shape[2]))
    _assert_result_equal(host, reference, context=f"{mode} Q={q}")
    return host, elapsed


def _run_pair(*, order, cp, q, x_device, r_device, gpu_callable, reference,
              shape, workspace_bytes, clock):
    outputs = {}
    elapsed = {mode: [] for mode in order}
    for mode in order:
        outputs[mode], seconds = _timed_call(
            cp=cp, mode=mode, q=q, x_device=x_device, r_device=r_device,
            gpu_callable=gpu_callable, reference=reference, shape=shape,
            workspace_bytes=workspace_bytes, clock=clock,
        )
        elapsed[mode].append(seconds)
    _assert_result_equal(outputs[order[0]], outputs[order[1]],
                         context=f"paired Q={q}")
    return elapsed


def run_benchmark(*, shape: tuple[int, int, int] = DEFAULT_SHAPE,
                  rounds: int = 3, warmup_cycles: int = 1, seed: int = SEED,
                  workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
                  root: Path = ROOT, cupy_module=None,
                  gpu_callable: Callable | None = None,
                  cpu_reference: Callable | None = None,
                  runtime_fingerprint: Callable | None = None,
                  clock: Callable[[], float] | None = None) -> dict:
    """Run synchronized CUDA A/B timing; this is never invoked without --run."""
    if type(rounds) is not int or rounds < 1 or rounds > 20:
        raise BenchmarkError("rounds must be an integer in [1, 20]")
    if type(warmup_cycles) is not int or warmup_cycles < 0 or warmup_cycles > 5:
        raise BenchmarkError("warmup_cycles must be an integer in [0, 5]")
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int) or workspace_bytes < 1:
        raise BenchmarkError("workspace_bytes must be a positive integer")
    host_estimate = _check_shape(shape)
    device_estimate = _device_memory_estimate(shape, workspace_bytes)
    cp = cupy_module if cupy_module is not None else _cupy()
    if gpu_callable is None:
        gpu_callable = gpu_quantile.batched_quantile_returns
    if cpu_reference is None:
        cpu_reference = cpu_quantile.compute_quantile_returns
    if runtime_fingerprint is None:
        runtime_fingerprint = _runtime_identity
    if clock is None:
        clock = time.perf_counter

    batch, bundle = _CPU_DRIVER._make_inputs(shape, seed)
    host_started = clock()
    x_host = np.ascontiguousarray(batch.values.transpose(0, 2, 1))
    r_host = np.ascontiguousarray(bundle.values)
    host_layout_seconds = clock() - host_started
    if not x_host.flags.c_contiguous or not r_host.flags.c_contiguous:
        raise BenchmarkError("GPU input arrays must be C-contiguous")

    device_memory_before = _device_memory_preflight(cp, device_estimate)
    _sync(cp)
    transfer_started = clock()
    x_device = cp.asarray(x_host)
    r_device = cp.asarray(r_host)
    _sync(cp)
    transfer_seconds = clock() - transfer_started

    input_before = _input_hashes(batch, bundle, x_host)
    device_hash_before = _device_input_hashes(cp, x_device, r_device)
    source_before = _source_hashes(root)
    wrapper_before = _wrapper_hash()
    runtime_before = runtime_fingerprint(cp)

    cpu_references = {}
    cpu_reference_seconds = {}
    for q in QUANTILE_COUNTS:
        started = clock()
        reference = cpu_reference(batch, bundle, n_quantiles=q, min_assets=MIN_ASSETS)
        cpu_reference_seconds[str(q)] = clock() - started
        cpu_references[q] = tuple(np.asarray(part) for part in reference)

    cold_seconds = {str(q): {} for q in QUANTILE_COUNTS}
    warmup_seconds = {str(q): [] for q in QUANTILE_COUNTS}
    warm_samples = {str(q): {BASELINE: [], GUARDED: []} for q in QUANTILE_COUNTS}
    parity_calls = {str(q): 0 for q in QUANTILE_COUNTS}
    # The GPU API returns (T,Q,F); the input shape is (T,N,F).
    for q in QUANTILE_COUNTS:
        key = str(q)
        cold = _run_pair(
            order=(BASELINE, GUARDED), cp=cp, q=q, x_device=x_device,
            r_device=r_device, gpu_callable=gpu_callable,
            reference=cpu_references[q], shape=shape,
            workspace_bytes=workspace_bytes, clock=clock,
        )
        cold_seconds[key] = {mode: values[0] for mode, values in cold.items()}
        parity_calls[key] += 2
        for _ in range(warmup_cycles):
            pair = _run_pair(
                order=ABBA, cp=cp, q=q, x_device=x_device, r_device=r_device,
                gpu_callable=gpu_callable, reference=cpu_references[q],
                shape=shape, workspace_bytes=workspace_bytes, clock=clock,
            )
            warmup_seconds[key].append(pair)
            parity_calls[key] += len(ABBA)
        for _ in range(rounds):
            pair = _run_pair(
                order=ABBA, cp=cp, q=q, x_device=x_device, r_device=r_device,
                gpu_callable=gpu_callable, reference=cpu_references[q],
                shape=shape, workspace_bytes=workspace_bytes, clock=clock,
            )
            for mode in (BASELINE, GUARDED):
                warm_samples[key][mode].extend(pair[mode])
            parity_calls[key] += len(ABBA)

    device_hash_after = _device_input_hashes(cp, x_device, r_device)
    input_after = _input_hashes(batch, bundle, x_host)
    source_after = _source_hashes(root)
    wrapper_after = _wrapper_hash()
    runtime_after = runtime_fingerprint(cp)
    if input_before != input_after or device_hash_before != device_hash_after:
        raise BenchmarkError("host or device inputs changed during measurement")
    if source_before != source_after or wrapper_before != wrapper_after:
        raise BenchmarkError("GPU metric sources or benchmark wrapper changed during measurement")
    if runtime_before != runtime_after:
        raise BenchmarkError("actual CPU/GPU runtime identity changed during measurement")

    medians = {
        q: {mode: statistics.median(samples) for mode, samples in modes.items()}
        for q, modes in warm_samples.items()
    }
    return {
        "scope": "synthetic current GPU quantile call; numeric exact-repair toggle only; not full QE/COS or fastest-backend comparison",
        "shape_TNF": list(shape),
        "gpu_factor_shape_TFN": list(x_host.shape),
        "seed": seed,
        "label_scale": LABEL_SCALE,
        "scenario": "CPU guard-driver rounded Gaussian factor ties and finance-scale Gaussian labels with deterministic NaN/Inf masks",
        "quantile_counts": list(QUANTILE_COUNTS),
        "min_assets": MIN_ASSETS,
        "host_memory_estimate_bytes": host_estimate,
        "host_memory_guard_bytes": MAX_ESTIMATED_BYTES,
        "device_memory_estimate": device_estimate,
        "device_memory_available_before_allocation": device_memory_before,
        "workspace_bytes_per_GPU_call": workspace_bytes,
        "source_files": list(SOURCE_FILES),
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "wrapper_sha256_before": wrapper_before,
        "wrapper_sha256_after": wrapper_after,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "device_input_sha256_before": device_hash_before,
        "device_input_sha256_after": device_hash_after,
        "runtime_identity_before": runtime_before,
        "runtime_identity_after": runtime_after,
        "timing_seconds": {
            "host_layout": host_layout_seconds,
            "input_transfer_HtoD": transfer_seconds,
            "CPU_reference_untimed": cpu_reference_seconds,
            "cold_calls": cold_seconds,
            "warmup_ABBA_not_summarized": warmup_seconds,
            "warm_samples_ABBA": warm_samples,
            "warm_medians": medians,
        },
        "cold_order": [BASELINE, GUARDED],
        "warmup_cycles": warmup_cycles,
        "warmup_order_per_cycle": "ABBA",
        "warm_rounds": rounds,
        "warm_order_per_round": "ABBA",
        "sync_policy": "CUDA null stream synchronized immediately before and after each measured GPU callable; input transfer and output copies excluded from call timings",
        "modes": {
            BASELINE: "temporarily replace only gpu.quantile.repair_quantile_means_gpu with a no-op, restored in try/finally",
            GUARDED: "current GPU numeric repair helper",
        },
        "parity_calls_checked_per_quantile": parity_calls,
        "parity_policy": "each cold, warmup, and timed output compared with the genuine current NumPy CPU reference; counts exact and returns allclose(rtol=1e-9, atol=1e-12, equal_nan=True)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="authorize the full-size synthetic CUDA timing run")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("timing is opt-in; pass --run only in an approved GPU timing slot")
    try:
        result = run_benchmark()
    except BenchmarkError as exc:
        print(f"GPU quantile numeric guard benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
