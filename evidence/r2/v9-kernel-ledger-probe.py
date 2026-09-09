#!/usr/bin/env python3
"""Small reproducible CPU/RSS ledger for current research kernel helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _workspace(operator: str, n: int) -> dict:
    item_bytes = 8
    if operator == "granger":
        lag = 2
        n_available = n - lag
        train_rows = int(0.7 * n_available)
        square = train_rows * train_rows * item_bytes
        return {
            "basis": {"n": n, "lag": lag, "train_rows": train_rows},
            "principal_2d_arrays": {
                "one_train_square_float64_bytes": square,
                "distance_accumulator_float64_bytes": square,
                "one_train_kernel_float64_bytes": square,
                "one_test_train_float64_bytes": (n_available - train_rows) * train_rows * item_bytes,
            },
            "interpretation": "Analytical individual-array sizes, not an allocation trace or peak-live sum.",
        }
    purge = 3
    split = n // 2
    residual_rows = n - 2 * purge
    return {
        "basis": {"n": n, "purge_gap": purge, "largest_train_rows": max(split, n - split),
                  "pooled_residual_rows": residual_rows},
        "principal_2d_arrays": {
            "one_fold_train_square_float64_bytes": max(split, n - split) ** 2 * item_bytes,
            "one_fold_test_train_float64_bytes": max(split - purge, n - split - purge) * max(split, n - split) * item_bytes,
            "one_residual_square_float64_bytes": residual_rows ** 2 * item_bytes,
        },
        "interpretation": "Analytical individual-array sizes, not an allocation trace or peak-live sum.",
    }


def _worker(operator: str, n: int, seed: int) -> dict:
    import numpy as np
    from factor_engine.cleaned_operators.research_spectral import (
        _kernel_granger_score,
        _residualized_hsic,
    )

    rng = np.random.default_rng(seed)
    x, y, z = rng.normal(size=(3, n))
    started = time.perf_counter()
    if operator == "granger":
        value = _kernel_granger_score(y, x, 2)
    else:
        value = _residualized_hsic(x, y, z, 3)
    elapsed = time.perf_counter() - started
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss_raw * 1024) if sys.platform.startswith("linux") else int(rss_raw)
    return {"wall_seconds": elapsed, "ru_maxrss_raw": rss_raw,
            "ru_maxrss_platform_unit": "KiB" if sys.platform.startswith("linux") else "bytes",
            "ru_maxrss_bytes": rss_bytes, "finite_output": bool(np.isfinite(value))}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--operator", choices=("granger", "hsic"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.operator, args.n, args.seed), sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required")

    source = Path(__file__).resolve()
    kernel_source = Path(os.environ["FACTOR_ENGINE_RESEARCH_SPECTRAL"])
    records = []
    for operator in ("granger", "hsic"):
        for n in (120, 240):
            samples = []
            for repeat in range(args.repeats):
                command = [sys.executable, str(source), "--worker", "--operator", operator,
                           "--n", str(n), "--seed", str(93500 + n + repeat)]
                env = dict(os.environ)
                env.update(THREAD_ENV)
                completed = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
                samples.append(json.loads(completed.stdout))
            wall = [sample["wall_seconds"] for sample in samples]
            rss = [sample["ru_maxrss_bytes"] for sample in samples]
            records.append({
                "operator": operator, "n": n, "repeats": args.repeats,
                "wall_seconds": {"samples": wall, "min": min(wall), "median": statistics.median(wall),
                                 "mean": statistics.mean(wall), "max": max(wall),
                                 "pstdev": statistics.pstdev(wall)},
                "self_process_ru_maxrss_bytes": {"samples": rss, "max": max(rss)},
                "all_outputs_finite": all(sample["finite_output"] for sample in samples),
                "workspace_analysis": _workspace(operator, n),
            })
    evidence = {
        "schema": "factor_engine.v9.kernel_cpu_ledger.v1",
        "probe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "research_spectral_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
        "thread_environment": THREAD_ENV,
        "records": records,
        "conversion_bytes": "NOT_INSTRUMENTED",
        "scope": [
            "Current helper-kernel CPU measurements on synthetic fixed-seed arrays only.",
            "Not an automatic backend or broker-admission benchmark.",
            "Not GPU evidence and not a 100k-scale certification.",
            "ru_maxrss is the worker self-process high-water mark including interpreter/imports; it is not process-family RSS and not kernel-only incremental memory.",
            "Analytical workspace bytes describe principal 2D float64 arrays, not measured peak-live allocations.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    _main()
