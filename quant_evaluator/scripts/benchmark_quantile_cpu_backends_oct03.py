"""Bounded synthetic CPU A/B of QE's NumPy and Numba quantile paths.

The full matrix is opt-in; --smoke performs tiny parity checks only. This is
not a whole-evaluator or COS/GPU benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import numpy as np
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quantile import assign_quantiles_batch, compute_quantile_returns
from quant_evaluator.metrics.quantile_numba import assign_quantiles_numba, compute_quantile_returns_numba, is_numba_available

MAX_FACTOR_CELLS = 70_000_000
ESTIMATED_BYTES_PER_CELL = 40
FIXED_MEMORY_ALLOWANCE = 512 * 1024**2
MAX_ESTIMATED_BYTES = 3 * 1024**3
DEFAULT_SHAPE = (128, 5461, 48)
QUANTILE_COUNTS = (5, 20)

def estimate_peak_bytes(time_count: int, asset_count: int, factor_count: int) -> int:
    """Conservative allowance for inputs, assignments and NumPy sort temporaries."""
    return FIXED_MEMORY_ALLOWANCE + ESTIMATED_BYTES_PER_CELL * time_count * asset_count * factor_count

def _check_rounds(rounds: int) -> None:
    if type(rounds) is not int or not 3 <= rounds <= 20:
        raise ValueError("rounds must be a positive builtin integer between 3 and 20")

def _check_shape(time_count: int, asset_count: int, factor_count: int, max_estimated_bytes: int = MAX_ESTIMATED_BYTES) -> None:
    for name, value in (("time_count", time_count), ("asset_count", asset_count), ("factor_count", factor_count)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive builtin integer")
    if type(max_estimated_bytes) is not int or max_estimated_bytes < 1:
        raise ValueError("max_estimated_bytes must be a positive builtin integer")
    cells = time_count * asset_count * factor_count
    if cells > MAX_FACTOR_CELLS:
        raise ValueError(f"factor tensor is capped at {MAX_FACTOR_CELLS:,} cells")
    estimate = estimate_peak_bytes(time_count, asset_count, factor_count)
    if estimate > max_estimated_bytes:
        raise MemoryError(f"estimated peak {estimate:,} bytes exceeds configured guard {max_estimated_bytes:,}")

def _inputs(shape: tuple[int, int, int], seed: int, scenario: str = "ties_missing"):
    t_count, n_count, f_count = shape
    if scenario not in ("dense", "ties_missing"):
        raise ValueError("scenario must be 'dense' or 'ties_missing'")
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape)
    labels = rng.standard_normal((t_count, n_count))
    if scenario == "ties_missing":
        values = np.round(values, 1)
        values[::11, ::7, :] = np.nan
        values[::17, ::13, ::3] = np.inf
        labels[::9, ::19] = np.nan
        labels[::23, ::31] = -np.inf
    fingerprint = hashlib.sha256()
    fingerprint.update(np.ascontiguousarray(values).view(np.uint8))
    fingerprint.update(np.ascontiguousarray(labels).view(np.uint8))
    ta = AxisRef("time", "int64", t_count, np.arange(t_count, dtype=np.int64))
    aa = AxisRef("asset", "int64", n_count, np.arange(n_count, dtype=np.int64))
    batch = FactorBatch(tuple(f"f{i}" for i in range(f_count)), ta, aa, values)
    bundle = LabelBundle("synthetic", labels, 1, decision_time=tuple(range(t_count)),
        label_start_time=tuple(range(1, t_count + 1)), label_end_time=tuple(range(2, t_count + 2)), asset_axis=aa)
    return batch, bundle, fingerprint.hexdigest()

def _parity(batch, bundle, q: int) -> dict[str, str]:
    values = batch.values
    nq = assign_quantiles_batch(values, q)
    jq = assign_quantiles_numba(values, q)
    np.testing.assert_array_equal(jq, nq)
    np.testing.assert_array_equal(nq >= 0, _assignment_valid_mask(values, q))
    nr = compute_quantile_returns(batch, bundle, q, 2)
    jr = compute_quantile_returns_numba(batch, bundle, q, 2)
    _assert_result_equal(nr, jr)
    np.testing.assert_array_equal(jr[1] > 0, nr[1] > 0)
    return {"assignments": "exact", "counts": "exact", "assignment_mask": "exact",
            "bucket_nonempty_mask": "exact", "returns": "allclose(rtol=1e-9, atol=1e-12)"}

def _assignment_valid_mask(values: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Values assigned under the contract; underfilled time/factor slices are invalid."""
    finite = np.isfinite(values)
    enough = finite.sum(axis=1, keepdims=True) >= n_quantiles
    return finite & enough

def _assert_result_equal(reference, candidate) -> None:
    np.testing.assert_array_equal(candidate[1], reference[1])
    np.testing.assert_allclose(candidate[0], reference[0], rtol=1e-9, atol=1e-12, equal_nan=True)

def smoke_check(scenario: str = "ties_missing") -> dict[str, object]:
    """Tiny correctness-only invocation, with no timing matrix."""
    if not is_numba_available():
        raise RuntimeError("Numba is required for backend smoke check")
    shape = (3, 64, 4)
    _check_shape(*shape)
    batch, bundle, fingerprint = _inputs(shape, 81033, scenario)
    return {"shape": list(shape), "scenario": scenario, "input_fingerprint": fingerprint,
            "validity_path": "finite-value masks; no explicit validity arrays",
            "quantiles": list(QUANTILE_COUNTS),
            "correctness": {str(q): _parity(batch, bundle, q) for q in QUANTILE_COUNTS}}

def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)

def benchmark(time_count: int = DEFAULT_SHAPE[0], asset_count: int = DEFAULT_SHAPE[1],
              factor_count: int = DEFAULT_SHAPE[2], seed: int = 81033, rounds: int = 3,
              scenario: str = "ties_missing") -> dict[str, object]:
    """Correctness-gated ABBA timing with separately reported first invocations."""
    if not is_numba_available():
        raise RuntimeError("Numba is unavailable; comparison cannot run")
    _check_rounds(rounds)
    _check_shape(time_count, asset_count, factor_count)
    shape = (time_count, asset_count, factor_count)
    batch, bundle, fingerprint = _inputs(shape, seed, scenario)
    first, samples, checks = {}, {}, {}
    for q in QUANTILE_COUNTS:
        funcs = {"numpy": lambda: compute_quantile_returns(batch, bundle, q, 2),
                 "numba": lambda: compute_quantile_returns_numba(batch, bundle, q, 2)}
        first[str(q)] = {}
        reference = None
        for name, fn in funcs.items():
            start = time.perf_counter()
            result = fn()
            first[str(q)][name] = time.perf_counter() - start
            if name == "numpy":
                reference = result
            else:
                _assert_result_equal(reference, result)
            del result
        checks[str(q)] = _parity(batch, bundle, q)
        samples[str(q)] = {"numpy": [], "numba": []}
        for _ in range(rounds):
            for name in ("numpy", "numba", "numba", "numpy"):
                start = time.perf_counter()
                result = funcs[name]()
                elapsed = time.perf_counter() - start
                samples[str(q)][name].append(elapsed)
                _assert_result_equal(reference, result)
                del result
        del reference
    root = Path(__file__).resolve().parents[2]
    sources = [root / "quant_evaluator/metrics/quantile.py",
               root / "quant_evaluator/metrics/quantile_numba.py", Path(__file__).resolve()]
    import numba
    try:
        from numba import get_num_threads
        threads = int(get_num_threads())
    except Exception:
        threads = None
    return {"shape": list(shape), "seed": seed, "quantile_counts": list(QUANTILE_COUNTS),
        "scenario": scenario, "input_fingerprint": fingerprint,
        "validity_path": "finite-value masks; no explicit validity arrays",
        "dtype": str(batch.values.dtype), "layout": "C-contiguous wide", "numba_threads": threads,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "numba": numba.__version__},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "estimated_peak_bytes": estimate_peak_bytes(*shape), "memory_guard_bytes": MAX_ESTIMATED_BYTES,
        "process_peak_rss_bytes": _peak_rss_bytes(), "first_invocation_seconds": first,
        "first_invocation_note": "First Numba invocation includes JIT compilation or cache load; Q=20 reuses the same compiled specialization.",
        "warm_rounds": rounds, "warm_order_per_round": "ABBA", "warm_samples_seconds": samples,
        "warm_medians_seconds": {q: {name: statistics.median(vals) for name, vals in methods.items()} for q, methods in samples.items()},
        "correctness": checks}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny correctness check only")
    parser.add_argument("--times", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--assets", type=int, default=DEFAULT_SHAPE[1])
    parser.add_argument("--factors", type=int, default=DEFAULT_SHAPE[2])
    parser.add_argument("--seed", type=int, default=81033)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scenario", choices=("dense", "ties_missing"), default="ties_missing")
    args = parser.parse_args()
    report = smoke_check(args.scenario) if args.smoke else benchmark(args.times, args.assets, args.factors, args.seed, args.rounds, args.scenario)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
