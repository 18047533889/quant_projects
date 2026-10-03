"""Bounded fresh-process A/B for FactorBatch output assembly only.

This does not call optimize_factor_batch and makes no full optimizer speed
claim. Run from the project root with a new report path, for example:
  .venv/bin/python factor_optimizer/scripts/benchmark_output_buffer_oct04.py \\
      --report factor_optimizer/docs/benchmarks/output_buffer_oct04_run1.json
The default 16x500x5000 fixture is limited to 40M cells. Smaller dimensions
are selectable with --factors, --times, and --assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch


MAX_INPUT_CELLS = 40_000_000
MAX_REPORT_BYTES = 1 << 20
MIN_AVAILABLE_BYTES = 4 * 1024**3
DEFAULT_FACTORS, DEFAULT_TIMES, DEFAULT_ASSETS = 16, 500, 5_000
DEFAULT_SEED = 20261004
PAIRED_ORDERS = (
    ("old_list_stack", "factor_major"),
    ("factor_major", "old_list_stack"),
    ("old_list_stack", "factor_major"),
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "factor_optimizer/scripts/benchmark_output_buffer_oct04.py",
    "factor_optimizer/factor_optimizer/research_batch.py",
    "quant_evaluator/contracts/factor_batch.py",
    "quant_evaluator/contracts/metric_artifacts.py",
)


def validate_case(factors: int, times: int, assets: int,
                  available_bytes: int | None = None) -> dict:
    dims = (factors, times, assets)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in dims):
        raise ValueError("factors, times, and assets must be positive integers")
    cells = factors * times * assets
    if cells > MAX_INPUT_CELLS:
        raise ValueError(f"fixture has {cells:,} cells; limit is {MAX_INPUT_CELLS:,}")
    panel_bytes = cells * np.dtype(np.float64).itemsize
    required = MIN_AVAILABLE_BYTES + 4 * panel_bytes
    if available_bytes is not None and available_bytes < required:
        raise MemoryError(f"available RAM {available_bytes:,} below required {required:,} bytes")
    return {"factors": factors, "times": times, "assets": assets,
            "cells": cells, "float64_panel_bytes": panel_bytes,
            "required_available_bytes": required}


def available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    raise RuntimeError("cannot determine available RAM from /proc/meminfo")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _fixture_values(factors: int, times: int, assets: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(factors):
        out = rng.standard_normal((times, assets), dtype=np.float64)
        out[::29, ::31] = np.nan
        yield out


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def run_child(mode: str, factors: int, times: int, assets: int, seed: int) -> dict:
    if mode not in {"old_list_stack", "factor_major"}:
        raise ValueError(f"unsupported mode: {mode}")
    case = validate_case(factors, times, assets, available_memory_bytes())
    ids = tuple(f"factor_{i:02d}" for i in range(factors))
    ta = AxisRef("time", "int", times, np.arange(times, dtype=np.int64))
    aa = AxisRef("asset", "str", assets,
                 np.asarray([f"asset_{i:05d}" for i in range(assets)]))
    started = time.perf_counter()
    if mode == "old_list_stack":
        outputs = list(_fixture_values(factors, times, assets, seed))
        values = np.stack(outputs, axis=-1)
    else:
        factor_outputs = np.empty((factors, times, assets), dtype=np.float64)
        for i, out in enumerate(_fixture_values(factors, times, assets, seed)):
            factor_outputs[i] = out
        values = factor_outputs.transpose(1, 2, 0)
    validity = np.isfinite(values)
    batch = FactorBatch(ids, ta, aa, values, validity=validity,
                        context_refs={"benchmark": "output_assembly_only"})
    elapsed = time.perf_counter() - started
    if batch.validity is None:
        raise RuntimeError("FactorBatch did not retain validity")
    return {
        "mode": mode, "fixture": {**case, "seed": seed},
        "elapsed_seconds": elapsed, "process_peak_rss_bytes": _rss_bytes(),
        "factor_ids": list(batch.factor_ids), "shape": list(batch.values.shape),
        "dtype": str(batch.values.dtype), "validity_dtype": str(batch.validity.dtype),
        "values_sha256": _digest(batch.values),
        "validity_sha256": _digest(batch.validity),
        "values_readonly": not batch.values.flags.writeable,
        "validity_readonly": not batch.validity.flags.writeable,
    }


def compare_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no child records to compare")
    fields = ("fixture", "factor_ids", "shape", "dtype", "validity_dtype",
              "values_sha256", "validity_sha256", "values_readonly", "validity_readonly")
    if any(not isinstance(record, dict) or any(field not in record for field in fields)
           for record in records):
        raise ValueError("child record is missing required equivalence fields")
    if {record.get("mode") for record in records} != {"old_list_stack", "factor_major"}:
        raise ValueError("equivalence requires both old_list_stack and factor_major modes")
    json.dumps(records, allow_nan=False)
    reference = records[0]
    mismatches = [{"record": i, "fields": [f for f in fields
                  if record.get(f) != reference.get(f)]}
                  for i, record in enumerate(records[1:], 1)]
    mismatches = [item for item in mismatches if item["fields"]]
    if mismatches:
        raise RuntimeError(f"old/new output equivalence failed: {mismatches}")
    if not all(r["values_readonly"] and r["validity_readonly"] for r in records):
        raise RuntimeError("FactorBatch outputs are unexpectedly writable")
    return {"equivalent": True, "records_compared": len(records)}


def runtime_context() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "executable": sys.executable, "host": platform.node(),
            "thread_environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS")}}


def source_hashes() -> dict:
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def _invoke_child(mode: str, factors: int, times: int, assets: int, seed: int) -> dict:
    validate_case(factors, times, assets, available_memory_bytes())
    command = [sys.executable, str(Path(__file__).resolve()), "--child", mode,
               "--factors", str(factors), "--times", str(times),
               "--assets", str(assets), "--seed", str(seed)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True,
                               capture_output=True, text=True, timeout=300)
    if len(completed.stdout.encode("utf-8")) >= MAX_REPORT_BYTES:
        raise RuntimeError("child output reaches 1 MiB limit")
    return json.loads(completed.stdout)


def write_report_exclusive(path: str | Path, payload: dict) -> Path:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) >= MAX_REPORT_BYTES:
        raise ValueError("report must be strictly smaller than 1 MiB")
    destination = Path(path)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as report:
            report.write(data)
            report.flush()
            os.fsync(report.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return destination


def run_benchmark(*, report: str | Path, factors: int = DEFAULT_FACTORS,
                  times: int = DEFAULT_TIMES, assets: int = DEFAULT_ASSETS,
                  seed: int = DEFAULT_SEED, pairs: int = 3) -> dict:
    if isinstance(pairs, bool) or not isinstance(pairs, int) or not 1 <= pairs <= 10:
        raise ValueError("pairs must be an integer in 1..10")
    destination = Path(report)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    case = validate_case(factors, times, assets, available_memory_bytes())
    before = source_hashes()
    runtime_before = runtime_context()
    all_records, pair_records = [], []
    for pair_index in range(pairs):
        order = PAIRED_ORDERS[pair_index % len(PAIRED_ORDERS)]
        pair = [_invoke_child(mode, factors, times, assets, seed) for mode in order]
        compare_records(pair)
        all_records.extend(pair)
        pair_records.append({
            "pair": pair_index + 1, "order": list(order),
            "elapsed_seconds": {r["mode"]: r["elapsed_seconds"] for r in pair},
            "process_peak_rss_bytes": {r["mode"]: r["process_peak_rss_bytes"] for r in pair},
        })
    after = source_hashes()
    runtime_after = runtime_context()
    if before != after or runtime_before != runtime_after:
        raise RuntimeError("benchmark source or runtime changed during run")
    payload = {
        "schema": "factor_optimizer.output_buffer_benchmark.v1",
        "scope": "output_assembly_only", "full_optimizer_speed_claim": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": case | {"seed": seed}, "paired_runs": pair_records,
        "equivalence": compare_records(all_records),
        "source_sha256_before": before, "source_sha256_after": after,
        "runtime_before": runtime_before, "runtime_after": runtime_after,
        "process_model": "one fresh child process per mode per pair",
        "report_limit_bytes_strictly_less_than": MAX_REPORT_BYTES,
        "report_overwrite": "refused via O_EXCL",
    }
    written = write_report_exclusive(report, payload)
    return {"report": str(written), **payload}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="new report path; existing files are refused")
    parser.add_argument("--factors", type=int, default=DEFAULT_FACTORS)
    parser.add_argument("--times", type=int, default=DEFAULT_TIMES)
    parser.add_argument("--assets", type=int, default=DEFAULT_ASSETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--child", choices=("old_list_stack", "factor_major"))
    args = parser.parse_args(argv)
    if args.child is None and args.report is None:
        parser.error("--report is required unless --child is specified")
    return args


def main(argv=None):
    args = _parse_args(argv)
    if args.child is not None:
        result = run_child(args.child, args.factors, args.times, args.assets, args.seed)
    else:
        result = run_benchmark(report=args.report, factors=args.factors,
                               times=args.times, assets=args.assets,
                               seed=args.seed, pairs=args.pairs)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
