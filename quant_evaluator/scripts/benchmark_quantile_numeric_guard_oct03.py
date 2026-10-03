"""Opt-in synthetic CPU benchmark for quantile numeric-guard overhead.

This compares the current NumPy and Numba callables with numeric guard helpers
enabled versus locally bypassed. It is not an old/new repository comparison,
whole-evaluator benchmark, CUDA test, or COS workload. First-call diagnostics
are separate from warm ABBA samples; no fastest-backend claim is made.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics import quantile, quantile_numba, quantile_numeric

SOURCE_FILES = (
    "quant_evaluator/metrics/quantile.py",
    "quant_evaluator/metrics/quantile_numba.py",
    "quant_evaluator/metrics/quantile_numeric.py",
)
DEFAULT_SHAPE = (128, 5461, 48)
SEED = 81033
LABEL_SCALE = 0.02
QUANTILE_COUNTS = (5, 20)
MIN_ASSETS = 10
MAX_FACTOR_CELLS = 70_000_000
ESTIMATED_BYTES_PER_CELL = 40
FIXED_MEMORY_ALLOWANCE = 512 * 1024**2
MAX_ESTIMATED_BYTES = 3 * 1024**3
GUARDS_BYPASSED = "guards_bypassed"
GUARDS_ENABLED = "guards_enabled"
ABBA = (GUARDS_BYPASSED, GUARDS_ENABLED, GUARDS_ENABLED, GUARDS_BYPASSED)

PATCH_TARGETS = (
    (quantile, "label_sum_error_bounds"),
    (quantile, "repair_bucket_means"),
    (quantile_numba, "repair_quantile_panel"),
    (quantile_numeric, "label_sum_error_bounds"),
    (quantile_numeric, "repair_bucket_means"),
    (quantile_numeric, "repair_quantile_panel"),
)


class BenchmarkError(RuntimeError):
    """Raised when the synthetic benchmark cannot preserve its contract."""


def estimate_peak_bytes(time_count: int, asset_count: int, factor_count: int) -> int:
    """Conservative input/output and NumPy temporary memory allowance."""
    return FIXED_MEMORY_ALLOWANCE + ESTIMATED_BYTES_PER_CELL * time_count * asset_count * factor_count


def _check_shape(time_count: int, asset_count: int, factor_count: int) -> int:
    for name, value in (("time_count", time_count), ("asset_count", asset_count), ("factor_count", factor_count)):
        if type(value) is not int or value < 1:
            raise BenchmarkError(f"{name} must be a positive builtin integer")
    cells = time_count * asset_count * factor_count
    if cells > MAX_FACTOR_CELLS:
        raise BenchmarkError(f"factor tensor exceeds {MAX_FACTOR_CELLS:,} cell admission limit")
    estimate = estimate_peak_bytes(time_count, asset_count, factor_count)
    if estimate > MAX_ESTIMATED_BYTES:
        raise BenchmarkError("conservative peak memory estimate exceeds the 3 GiB admission limit")
    return estimate


def _make_inputs(shape: tuple[int, int, int], seed: int = SEED):
    t_count, n_count, f_count = shape
    _check_shape(*shape)
    rng = np.random.default_rng(seed)
    values = np.ascontiguousarray(np.round(rng.standard_normal(shape), 1), dtype=np.float64)
    values[::11, ::7, :] = np.nan
    values[::17, ::13, ::3] = np.inf
    factor_validity = np.ascontiguousarray(np.isfinite(values), dtype=np.bool_)

    labels = np.ascontiguousarray(LABEL_SCALE * rng.standard_normal((t_count, n_count)), dtype=np.float64)
    labels[::9, ::19] = np.nan
    labels[::23, ::31] = -np.inf
    label_validity = np.ascontiguousarray(np.isfinite(labels), dtype=np.bool_)

    time_axis = AxisRef("time", "int64", t_count, np.arange(t_count, dtype=np.int64))
    asset_axis = AxisRef("asset", "int64", n_count, np.arange(n_count, dtype=np.int64))
    batch = FactorBatch(tuple(f"factor_{i}" for i in range(f_count)),
                        time_axis, asset_axis, values, validity=factor_validity)
    bundle = LabelBundle(
        "synthetic_gaussian_002", labels, 1,
        decision_time=tuple(range(t_count)),
        label_start_time=tuple(range(1, t_count + 1)),
        label_end_time=tuple(range(2, t_count + 2)),
        validity=label_validity,
        asset_axis=asset_axis,
    )
    return batch, bundle


def _array_digest(array) -> str:
    digest = hashlib.sha256()
    if array is None:
        digest.update(b"<none>")
        return digest.hexdigest()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _input_hashes(batch: FactorBatch, bundle: LabelBundle) -> dict[str, str]:
    parts = {
        "factor_values": _array_digest(batch.values),
        "factor_validity": _array_digest(batch.validity),
        "labels": _array_digest(bundle.values),
        "label_validity": _array_digest(bundle.validity),
    }
    combined = hashlib.sha256()
    for name, value in sorted(parts.items()):
        combined.update(name.encode("ascii"))
        combined.update(value.encode("ascii"))
    parts["combined"] = combined.hexdigest()
    return parts


def _source_hashes(root: Path = ROOT) -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        try:
            hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkError("could not read a metric source for provenance") from exc
    return hashes


def _wrapper_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as exc:
        raise BenchmarkError("could not read benchmark wrapper for provenance") from exc


def _guard_bindings() -> dict[tuple[object, str], object]:
    return {(module, name): getattr(module, name) for module, name in PATCH_TARGETS}


def _noop_error_bounds(labels):
    return np.zeros(labels.shape[0], dtype=np.float64)


def _noop(*args, **kwargs):
    return None


@contextmanager
def _without_numeric_guards():
    """Bypass only these helpers for the current call; restore every binding."""
    originals = _guard_bindings()
    noops = {
        "label_sum_error_bounds": _noop_error_bounds,
        "repair_bucket_means": _noop,
        "repair_quantile_panel": _noop,
    }
    try:
        for module, name in PATCH_TARGETS:
            setattr(module, name, noops[name])
        yield
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)


def _assert_result_shape(result, batch: FactorBatch, n_quantiles: int) -> None:
    if not isinstance(result, tuple) or len(result) != 2:
        raise BenchmarkError("CPU callable must return (returns, counts)")
    returns, counts = result
    expected = (batch.num_times, n_quantiles, batch.num_factors)
    if not isinstance(returns, np.ndarray) or returns.shape != expected or returns.dtype != np.float64:
        raise BenchmarkError("CPU callable returned an invalid quantile-return panel")
    if not isinstance(counts, np.ndarray) or counts.shape != expected or counts.dtype != np.int32:
        raise BenchmarkError("CPU callable returned an invalid quantile-count panel")


def _assert_result_equal(left, right, *, context: str) -> None:
    try:
        np.testing.assert_array_equal(left[1], right[1])
        np.testing.assert_allclose(left[0], right[0], rtol=1e-9, atol=1e-12, equal_nan=True)
    except AssertionError:
        raise BenchmarkError(f"{context} quantile values/counts parity failed") from None


def _call_once(callable_, mode: str, batch: FactorBatch, bundle: LabelBundle,
               n_quantiles: int) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    manager = _without_numeric_guards() if mode == GUARDS_BYPASSED else nullcontext()
    with manager:
        started = time.perf_counter()
        result = callable_(batch, bundle, n_quantiles=n_quantiles, min_assets=MIN_ASSETS)
        elapsed = time.perf_counter() - started
    _assert_result_shape(result, batch, n_quantiles)
    return result, elapsed


def _run_pair(callable_, order: tuple[str, str], batch: FactorBatch, bundle: LabelBundle,
              n_quantiles: int, reference=None) -> tuple[dict, dict]:
    outputs = {}
    elapsed = {}
    for mode in order:
        outputs[mode], elapsed[mode] = _call_once(callable_, mode, batch, bundle, n_quantiles)
        if reference is not None:
            _assert_result_equal(outputs[mode], reference, context=f"{mode} Q={n_quantiles}")
    _assert_result_equal(outputs[order[0]], outputs[order[1]], context=f"paired Q={n_quantiles}")
    return outputs, elapsed


def _run_order(callable_, batch, bundle, n_quantiles, order, reference, samples=None):
    pair_checks = 0
    for index in range(0, len(order), 2):
        outputs, elapsed = _run_pair(callable_, tuple(order[index:index + 2]),
                                     batch, bundle, n_quantiles, reference)
        pair_checks += 1
        if samples is not None:
            for mode in order[index:index + 2]:
                samples[mode].append(elapsed[mode])
        del outputs
    return pair_checks


def _route_callable(backend: str) -> Callable:
    if backend == "numpy":
        return quantile.compute_quantile_returns
    if backend == "numba":
        return quantile_numba.compute_quantile_returns_numba
    raise ValueError(f"unsupported CPU route: {backend}")


def _run_route_quantile(backend: str, batch: FactorBatch, bundle: LabelBundle,
                        n_quantiles: int, callable_, *, rounds: int, warmup_cycles: int,
                        cross_backend_reference=None) -> tuple[dict, dict, object, int]:
    first_outputs, first_times = _run_pair(
        callable_, (GUARDS_BYPASSED, GUARDS_ENABLED), batch, bundle, n_quantiles,
        reference=cross_backend_reference,
    )
    route_reference = first_outputs[GUARDS_ENABLED]
    samples = {GUARDS_BYPASSED: [], GUARDS_ENABLED: []}
    pair_checks = 1

    for _ in range(warmup_cycles):
        pair_checks += _run_order(callable_, batch, bundle, n_quantiles, ABBA, route_reference)

    for _ in range(rounds):
        pair_checks += _run_order(callable_, batch, bundle, n_quantiles, ABBA,
                                  route_reference, samples=samples)
    return first_times, samples, route_reference, pair_checks


def run_benchmark(*, shape: tuple[int, int, int] = DEFAULT_SHAPE, rounds: int = 3,
                  warmup_cycles: int = 1, seed: int = SEED, root: Path = ROOT,
                  callables: Mapping[str, Callable] | None = None,
                  runtime_fingerprint: Callable | None = None) -> dict:
    """Execute the isolated NumPy/Numba guard toggle under strict parity checks."""
    if type(rounds) is not int or rounds < 1 or rounds > 20:
        raise BenchmarkError("rounds must be an integer in [1, 20]")
    if type(warmup_cycles) is not int or warmup_cycles < 0 or warmup_cycles > 5:
        raise BenchmarkError("warmup_cycles must be an integer in [0, 5]")
    estimate = _check_shape(*shape)
    batch, bundle = _make_inputs(shape, seed)
    if not batch.values.flags.c_contiguous or not bundle.values.flags.c_contiguous:
        raise BenchmarkError("synthetic inputs must be C-contiguous wide arrays")

    if callables is None:
        if not quantile_numba.is_numba_available():
            raise BenchmarkError("Numba is required for the two-route CPU guard benchmark")
        selected_callables = {name: _route_callable(name) for name in ("numpy", "numba")}
    else:
        if set(callables) != {"numpy", "numba"}:
            raise BenchmarkError("callables must provide exactly numpy and numba routes")
        selected_callables = dict(callables)

    if runtime_fingerprint is None:
        from quant_evaluator.runtime.backend_calibration import _runtime_fingerprint
        runtime_fingerprint = _runtime_fingerprint

    source_before = _source_hashes(root)
    wrapper_before = _wrapper_hash()
    input_before = _input_hashes(batch, bundle)
    runtime_before = runtime_fingerprint(None)

    first_call_seconds = {}
    warm_samples_seconds = {}
    parity_pairs_checked = {}
    references = {}
    for q in QUANTILE_COUNTS:
        q_key = str(q)
        first_call_seconds[q_key] = {}
        warm_samples_seconds[q_key] = {}
        parity_pairs_checked[q_key] = {}
        numpy_first, numpy_samples, numpy_reference, numpy_checks = _run_route_quantile(
            "numpy", batch, bundle, q, selected_callables["numpy"],
            rounds=rounds, warmup_cycles=warmup_cycles,
        )
        first_call_seconds[q_key]["numpy"] = {
            "guards_bypassed": numpy_first[GUARDS_BYPASSED],
            "guards_enabled": numpy_first[GUARDS_ENABLED],
        }
        warm_samples_seconds[q_key]["numpy"] = {
            "guards_bypassed": numpy_samples[GUARDS_BYPASSED],
            "guards_enabled": numpy_samples[GUARDS_ENABLED],
        }
        parity_pairs_checked[q_key]["numpy"] = numpy_checks
        references[q] = numpy_reference

        numba_first, numba_samples, _, numba_checks = _run_route_quantile(
            "numba", batch, bundle, q, selected_callables["numba"],
            rounds=rounds, warmup_cycles=warmup_cycles,
            cross_backend_reference=references[q],
        )
        first_call_seconds[q_key]["numba"] = {
            "guards_bypassed": numba_first[GUARDS_BYPASSED],
            "guards_enabled": numba_first[GUARDS_ENABLED],
        }
        warm_samples_seconds[q_key]["numba"] = {
            "guards_bypassed": numba_samples[GUARDS_BYPASSED],
            "guards_enabled": numba_samples[GUARDS_ENABLED],
        }
        parity_pairs_checked[q_key]["numba"] = numba_checks

    source_after = _source_hashes(root)
    wrapper_after = _wrapper_hash()
    input_after = _input_hashes(batch, bundle)
    runtime_after = runtime_fingerprint(None)
    if source_before != source_after or wrapper_before != wrapper_after:
        raise BenchmarkError("metric sources or benchmark wrapper changed during measurement")
    if input_before != input_after:
        raise BenchmarkError("synthetic input arrays or masks changed during measurement")
    if runtime_before != runtime_after:
        raise BenchmarkError("runtime fingerprint changed during measurement")

    medians = {
        q: {
            backend: {
                mode: statistics.median(samples)
                for mode, samples in warm_samples_seconds[q][backend].items()
            }
            for backend in ("numpy", "numba")
        }
        for q in warm_samples_seconds
    }
    overhead = {
        q: {
            backend: medians[q][backend]["guards_enabled"] - medians[q][backend]["guards_bypassed"]
            for backend in ("numpy", "numba")
        }
        for q in medians
    }
    return {
        "scope": "synthetic CPU-only current-callable guard overhead; not full QE/CUDA/COS",
        "shape": list(shape),
        "wrapper_sha256": wrapper_before,
        "seed": seed,
        "label_scale": LABEL_SCALE,
        "scenario": "rounded gaussian ties with deterministic missing and infinite inputs; finance-scale Gaussian labels",
        "layout": "C-contiguous wide",
        "quantile_counts": list(QUANTILE_COUNTS),
        "min_assets": MIN_ASSETS,
        "estimated_peak_bytes": estimate,
        "memory_guard_bytes": MAX_ESTIMATED_BYTES,
        "source_files": list(SOURCE_FILES),
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "wrapper_sha256_before": wrapper_before,
        "wrapper_sha256_after": wrapper_after,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "runtime_fingerprint_before": runtime_before,
        "runtime_fingerprint_after": runtime_after,
        "guard_modes": {
            "guards_bypassed": "only label_sum_error_bounds/repair_bucket_means/repair_quantile_panel replaced with no-ops",
            "guards_enabled": "current genuine numeric guard path",
        },
        "first_call_seconds": first_call_seconds,
        "first_call_note": "Separate diagnostics; Numba's first call may include compilation and is not used in warm summaries.",
        "warmup_cycles": warmup_cycles,
        "warmup_order_per_cycle": "ABBA",
        "warm_rounds": rounds,
        "warm_order_per_round": "ABBA",
        "warm_samples_seconds": warm_samples_seconds,
        "warm_medians_seconds": medians,
        "median_guard_overhead_seconds": overhead,
        "parity_pairs_checked": parity_pairs_checked,
        "parity_policy": "every cold, warmup, and timed call is paired; counts exact and returns allclose(rtol=1e-9, atol=1e-12, equal_nan=True), plus cross-route comparison",
        "interpretation_note": "Timing deltas isolate these numeric guard helpers only; they are not fastest-backend claims.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="authorize the full-size synthetic CPU timing run")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("timing is opt-in; pass --run only in an approved timing slot")
    try:
        result = run_benchmark()
    except BenchmarkError as exc:
        print(f"numeric guard benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
