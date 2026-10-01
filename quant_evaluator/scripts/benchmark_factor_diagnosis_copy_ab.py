"""Bounded paired real-COS A/B for flatten copies in factor diagnostics.

The reference intentionally retains the old flatten-based array traversal;
its scalar semantics match the current implementation, including input-dtype
constant comparison and the existing overflow-safe mean fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np


EXPECTED_SHAPE = (2586, 5461, 32)
EXPECTED_DTYPE = np.dtype("float64")

from quant_evaluator.api.requests import FactorDiagnosis
from quant_evaluator.diagnosis.factor import diagnose_all_factors


def legacy_diagnose_all_factors(batch):
    """Independent flatten-based reference with current scalar semantics."""
    result = {}
    for index, factor_id in enumerate(batch.factor_ids):
        values = batch.values[:, :, index].flatten()
        finite = np.isfinite(values)
        has_nans = bool(np.any(np.isnan(values)))
        has_infs = bool(np.any(np.isinf(values)))
        valid = (finite & batch.validity[:, :, index].flatten()
                 if batch.validity is not None else finite)
        count = int(np.sum(valid))
        valid_values = values[valid]
        total = values.size
        coverage = count / total if total else 0.0
        if count:
            raw_min, raw_max = np.min(valid_values), np.max(valid_values)
            min_value, max_value = float(raw_min), float(raw_max)
            is_constant = bool(raw_min == raw_max)
            with np.errstate(over="ignore", invalid="ignore"):
                mean_value = float(np.mean(valid_values))
            if not np.isfinite(mean_value):
                scale = max(abs(min_value), abs(max_value))
                mean_value = float(np.mean(valid_values / scale) * scale)
        else:
            min_value = max_value = mean_value = None
            is_constant = True
        warnings = []
        if coverage < 0.5:
            warnings.append(f"Low coverage: {coverage:.2%}")
        if is_constant:
            warnings.append("Factor is constant")
        if has_nans:
            warnings.append("Contains NaN values")
        if has_infs:
            warnings.append("Contains Inf values")
        result[factor_id] = FactorDiagnosis(
            factor_id=factor_id, num_valid_observations=count,
            num_missing=total-count, coverage=coverage,
            is_constant=is_constant, has_nans=has_nans, has_infs=has_infs,
            min_value=min_value, max_value=max_value, mean_value=mean_value,
            warnings=tuple(warnings),
        )
    return result


def _rss_bytes():
    """Current resident set size where procfs is available."""
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _diagnosis_payload(result):
    return {key: asdict(value) for key, value in result.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_real_cos_ab(*, run=False, output=None, repeats=3):
    """Preflight by default; require explicit run and output to load real data."""
    if not isinstance(run, bool):
        raise ValueError("run must be a bool")
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("repeats must be 1..10")
    output_path = None if output is None else Path(output)
    if run and output_path is None:
        raise ValueError("--output is required with --run")
    if output_path is not None and (output_path.exists() or output_path.is_symlink()):
        raise FileExistsError(f"refusing to overwrite evidence path: {output_path}")
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as bench
    from quant_evaluator.scripts import benchmark_real_cos_metric_batch as full
    from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

    source_paths = {
        "diagnosis_module": Path(__file__).parents[1] / "diagnosis" / "factor.py",
        "benchmark_module": Path(__file__),
    }
    source_sha256_before = {name: _sha256(path) for name, path in source_paths.items()}
    preflight = bench.preflight_factor_count_profile(SimpleNamespace(
        manifest_sha256=full.MANIFEST_SHA256,
        profile_max_object_mib=128, profile_max_total_mib=2048,
        max_working_gib=50, metric_ids=("coverage",),
    ))
    if preflight.get("status") != "ready":
        return {"status": "preflight_rejected", "preflight": preflight}
    if not run:
        report = {"status": "preflight_only", "preflight": preflight,
                  "manifest_sha256": full.MANIFEST_SHA256}
        if output_path is not None:
            _write_report(output_path, report)
        return report

    load_started = time.perf_counter()
    batch, labels, provenance = load_real_batch(
        factors=32, days=0, assets=5500,
        max_object_mib=128, max_total_mib=2048,
        manifest_sha256=full.MANIFEST_SHA256,
    )
    load_seconds = time.perf_counter() - load_started
    if batch.values.shape != EXPECTED_SHAPE or batch.values.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"loaded batch differs from certified workload: shape={batch.values.shape}, "
            f"dtype={batch.values.dtype}; expected {EXPECTED_SHAPE}, {EXPECTED_DTYPE}"
        )
    source_sha256_after_load = {name: _sha256(path) for name, path in source_paths.items()}
    if source_sha256_after_load != source_sha256_before:
        raise RuntimeError("QE diagnosis or benchmark source changed during data load")

    # Alternate which implementation goes first in each matched pair.
    methods = {"flatten_reference": legacy_diagnose_all_factors,
               "strided_view": diagnose_all_factors}
    observations = []
    for pair in range(repeats):
        order = (("flatten_reference", "strided_view") if pair % 2 == 0
                 else ("strided_view", "flatten_reference"))
        pair_result = {}
        for name in order:
            started = time.perf_counter()
            result = methods[name](batch)
            elapsed = time.perf_counter() - started
            pair_result[name] = result
            observations.append({"pair": pair, "order": name,
                                 "seconds": elapsed, "rss_bytes": _rss_bytes()})
        if _diagnosis_payload(pair_result["flatten_reference"]) != _diagnosis_payload(
                pair_result["strided_view"]):
            raise AssertionError(f"diagnostic field mismatch in pair {pair}")
        current_sha = {name: _sha256(path) for name, path in source_paths.items()}
        if current_sha != source_sha256_before:
            raise RuntimeError("QE diagnosis or benchmark source changed during A/B")
    report = {
        "status": "complete", "manifest_sha256": full.MANIFEST_SHA256,
        "shape": list(batch.values.shape), "dtype": str(batch.values.dtype),
        "validity_dtype": None if batch.validity is None else str(batch.validity.dtype),
        "factor_count": batch.num_factors, "load_seconds": load_seconds,
        "preflight": preflight,
        "source_sha256": {
            "before": source_sha256_before,
            "after": current_sha,
        },
        "observations": observations,
        "exact_field_parity": True,
    }
    _write_report(output_path, report)
    return report


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, separators=(",", ":"),
                  allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true",
                        help="load and profile only after a ready preflight")
    parser.add_argument("--output", type=Path, default=None,
                        help="new JSON evidence path; required with --run")
    args = parser.parse_args()
    report = run_real_cos_ab(run=args.run, output=args.output, repeats=args.repeats)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False), flush=True)
    if report["status"] == "preflight_rejected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
