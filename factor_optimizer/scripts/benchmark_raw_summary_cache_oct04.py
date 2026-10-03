"""Bounded ABBA cache A/B on one strict, manifest-bound real F16 cohort.

This compares only one 500-date by 256-asset research cohort. It is not a
full-market, multiyear, production-admission, or statistical-significance claim.
No report or copied dataset is created until all four runs pass parity checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping

import numpy as np


ROOT = Path("/home/sunhaiwei/quant_projects")
SCRIPTS = ROOT / "factor_optimizer/scripts"
EXAMPLES = ROOT / "factor_optimizer/examples"
FACTOR_COUNT, ASSET_COUNT, DATE_COUNT = 16, 256, 500
MAX_REPORT_BYTES = 1 << 20
MIN_MEMORY_BYTES = 16 * 1024**3
MIN_DISK_BYTES = 2 * 1024**3
SOURCE_PATHS = (
    "factor_optimizer/scripts/benchmark_raw_summary_cache_oct04.py",
    "factor_optimizer/scripts/audit_real_scaling_cohort_oct03.py",
    "factor_optimizer/examples/cos_batch_audit.py",
    "factor_optimizer/examples/real_batch_audit.py",
    "factor_optimizer/scripts/audit_real_training_methods_oct03.py",
    "factor_optimizer/scripts/audit_real_automatic_oct03.py",
    "factor_optimizer/factor_optimizer/research_batch.py",
    "factor_optimizer/factor_optimizer/research_fitness.py",
    "factor_optimizer/factor_optimizer/research_summary_cache.py",
)
ABBA_MODES = ("uncached", "cached", "cached", "uncached")


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": value.dtype.str,
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported result value in benchmark signature: {type(value).__name__}")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_identity(array):
    if array is None:
        return {"present": False, "shape": None, "dtype": None, "sha256": None}
    value = np.asarray(array)
    contiguous = np.ascontiguousarray(value)
    return {"present": True, "shape": list(value.shape),
            "dtype": value.dtype.str, "sha256": _digest(contiguous.tobytes())}


def _load_audit_modules():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import audit_real_automatic_oct03 as automatic_audit
    import audit_real_scaling_cohort_oct03 as scaling_audit
    import audit_real_training_methods_oct03 as source_audit
    return automatic_audit, scaling_audit, source_audit


def check_environment():
    _, _, source_audit = _load_audit_modules()
    source_audit.check_environment()


def check_resource_headroom():
    _, scaling_audit, source_audit = _load_audit_modules()
    source_audit.check_resource_headroom(
        minimum_memory_bytes=scaling_audit.MIN_MEMORY_BYTES,
        minimum_disk_bytes=scaling_audit.MIN_DISK_BYTES,
    )


def _default_source_loader(**kwargs):
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    from cos_batch_audit import load_cos_sample
    return load_cos_sample(**kwargs)

def source_hashes():
    return {name: _digest((ROOT / name).read_bytes()) for name in SOURCE_PATHS}


def runtime_context():
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    packages = {}
    for name in ("scipy", "pandas", "polars", "numba", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "package_versions": packages,
        "git_head": head,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "source_closure": "selected files only; not complete imported-code closure",
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS",
            )
        },
    }


def _peak_rss_bytes():
    # ru_maxrss is process-cumulative; it is deliberately not presented as a
    # per-run measurement for these four in-process calls.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_and_validate(source_loader, config):
    from factor_optimizer.research_batch import automatic_time_split
    automatic_audit, scaling_audit, source_audit = _load_audit_modules()

    batch, labels, provenance, lineages = source_loader(
        n_factors=FACTOR_COUNT, n_assets=ASSET_COUNT, include_lineages=True,
        max_factor_bytes=scaling_audit.MAX_FACTOR_BYTES,
        max_batch_factor_bytes=scaling_audit.MAX_BATCH_BYTES,
        coverage_policy="strict",
    )
    loaded_bytes = scaling_audit._validate_cohort(batch, provenance, lineages)
    if tuple(batch.values.shape) != (DATE_COUNT, ASSET_COUNT, FACTOR_COUNT):
        raise ValueError("loaded source is not the declared 500x256 strict F16 cohort")
    split = automatic_time_split(labels, config)
    if split.identity != provenance.get("asset_selection_split"):
        raise ValueError("optimizer split differs from TRAIN asset-selection split")
    for factor_id in batch.factor_ids:
        factor_index = batch.factor_ids.index(factor_id)
        one = __import__(
            "quant_evaluator.contracts.factor_batch",
            fromlist=["FactorBatch"],
        ).FactorBatch(
            (factor_id,), batch.time_axis, batch.asset_axis,
            batch.values[:, :, factor_index:factor_index + 1],
            validity=(None if batch.validity is None else
                      batch.validity[:, :, factor_index:factor_index + 1]),
        )
        source_audit.validate_source(
            one, labels, scaling_audit._leaf_source(provenance, factor_id),
            max_factor_bytes=scaling_audit.MAX_FACTOR_BYTES,
            max_batch_factor_bytes=scaling_audit.MAX_BATCH_BYTES,
        )
        del one
    return (batch, labels, provenance, lineages, split, loaded_bytes,
            automatic_audit, scaling_audit, source_audit)


def _run_with_cache_mode(mode, batch, labels, *, config, lineages, runner):
    from factor_optimizer import research_fitness, research_summary_cache

    if mode not in ("cached", "uncached"):
        raise ValueError(f"unknown cache mode: {mode}")
    original_cache = research_summary_cache.RawMetricSummaryCache
    original_summarize = research_fitness.summarize
    counts = {
        "raw_summary_requests": 0, "raw_summary_computes": 0,
        "raw_summary_cache_hits": 0, "raw_summary_failures": 0,
        "all_summarize_computes": 0,
    }

    def tracked_summarize(series, *args, **kwargs):
        counts["all_summarize_computes"] += 1
        return original_summarize(series, *args, **kwargs)

    research_fitness.summarize = tracked_summarize
    if mode == "uncached":
        class UncachedSummaryCache:
            def summarize(self, series, *, periods_per_year=252):
                counts["raw_summary_requests"] += 1
                before = counts["all_summarize_computes"]
                failed = False
                try:
                    return research_fitness.summarize(
                        series, periods_per_year=periods_per_year)
                except BaseException:
                    failed = True
                    counts["raw_summary_failures"] += 1
                    raise
                finally:
                    computes = counts["all_summarize_computes"] - before
                    counts["raw_summary_computes"] += computes
                    if not computes and not failed:
                        counts["raw_summary_cache_hits"] += 1

        UncachedSummaryCache.__name__ = "UncachedSummaryCache"
        replacement = UncachedSummaryCache
    else:
        class ObservedSummaryCache(original_cache):
            def summarize(self, series, *, periods_per_year=252):
                counts["raw_summary_requests"] += 1
                before = counts["all_summarize_computes"]
                failed = False
                try:
                    return super().summarize(
                        series, periods_per_year=periods_per_year)
                except BaseException:
                    failed = True
                    counts["raw_summary_failures"] += 1
                    raise
                finally:
                    computes = counts["all_summarize_computes"] - before
                    counts["raw_summary_computes"] += computes
                    if not computes and not failed:
                        counts["raw_summary_cache_hits"] += 1

        ObservedSummaryCache.__name__ = "ObservedSummaryCache"
        replacement = ObservedSummaryCache

    research_summary_cache.RawMetricSummaryCache = replacement
    try:
        started = time.perf_counter()
        result = runner(batch, labels, config=config, allow_research=True,
                        lineages=lineages)
        elapsed = time.perf_counter() - started
    finally:
        research_summary_cache.RawMetricSummaryCache = original_cache
        research_fitness.summarize = original_summarize
    return result, elapsed, counts


def _validate_summary_activity(mode, counts):
    names = (
        "raw_summary_requests", "raw_summary_computes",
        "raw_summary_cache_hits", "raw_summary_failures",
        "all_summarize_computes",
    )
    if any(type(counts.get(name)) is not int or counts[name] < 0 for name in names):
        raise ValueError("cache qualification requires nonnegative integer counters")
    requests, computes, hits = (counts[name] for name in names[:3])
    if (requests == 0 or counts["raw_summary_failures"] != 0
            or counts["all_summarize_computes"] < computes):
        raise ValueError("cache qualification requires successful RAW summary activity")
    if mode == "uncached":
        if computes != requests or hits != 0:
            raise ValueError("cache qualification requires an uncached recomputation baseline")
    elif mode == "cached":
        if hits == 0 or computes == 0 or computes + hits != requests:
            raise ValueError("cache qualification requires actual cached hits and misses")
    else:
        raise ValueError("cache qualification received an unknown mode")


def _result_signature(result, batch, split, automatic_audit):
    automatic_audit.validate_result(result, batch, split)
    factors = {}
    for factor_id in batch.factor_ids:
        item = result.factors[factor_id]
        if item.plan is None or item.plan_identity != item.plan.identity:
            raise ValueError("selected plan identity is missing or changed")
        if item.materialization_error is not None or item.status == "materialization_failed":
            raise ValueError("optimizer returned a materialization failure")
        if item.status not in {"raw_retained", "baseline_accepted", "improved"}:
            raise ValueError(
                f"optimizer cache qualification failed for {factor_id}: {item.status}"
            )
        ledger = _jsonable([dict(row) for row in item.candidates])
        joint = _jsonable(dict(getattr(item, "joint_diagnostics", {}) or {}))
        factors[factor_id] = {
            "status": item.status,
            "selected_family": item.selected_family,
            "reason": item.reason,
            "plan_identity": item.plan_identity,
            "train_gain": item.train_gain,
            "validation_lower_bound": item.validation_lower_bound,
            "validation_candidate_identity": item.validation_candidate_identity,
            "validation_coverage": item.validation_coverage,
            "materialization_error": (
                None if item.materialization_error is None
                else str(item.materialization_error)
            ),
            "candidate_count": len(ledger),
            "candidate_ledger_sha256": _digest(_canonical_json(ledger).encode("utf-8")),
            "joint_diagnostics": joint,
            "joint_diagnostics_sha256": _digest(_canonical_json(joint).encode("utf-8")),
            "training_diagnostics": _jsonable(
                getattr(item, "training_diagnostics", None)),
            "baseline_diagnostics": _jsonable(
                getattr(item, "baseline_diagnostics", None)),
        }
    output = result.optimized
    signature = {
        "factor_ids": list(output.factor_ids),
        "values": _array_identity(output.values),
        "validity": _array_identity(output.validity),
        "factors": factors,
    }
    fingerprint = _digest(_canonical_json(signature).encode("utf-8"))
    return signature, fingerprint


def _default_config():
    from factor_optimizer.research_batch import BatchOptimizationConfig
    return BatchOptimizationConfig()


def run_benchmark(*, report, source_loader=None, auto_runner=None,
                  resource_check=None, environment_check=None, config=None):
    from factor_optimizer.research_batch import optimize_factor_batch
    from factor_optimizer.research_summary_cache import MAX_PANEL_KEY_BYTES

    destination = Path(report)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    environment_admission = check_environment if environment_check is None else environment_check
    resource_admission = check_resource_headroom if resource_check is None else resource_check
    source_loader = _default_source_loader if source_loader is None else source_loader
    runner = optimize_factor_batch if auto_runner is None else auto_runner
    config = _default_config() if config is None else config
    environment_admission()
    resource_admission()

    hashes_before = source_hashes()
    runtime_before = runtime_context()
    (batch, labels, provenance, lineages, split, loaded_bytes,
     automatic_audit, scaling_audit, source_audit) = _load_and_validate(source_loader, config)
    source_fingerprint = source_audit.source_fingerprint
    cohort_identity = source_fingerprint(batch, labels, provenance)
    modes = []
    warmup = None
    signatures = []
    for run_index, mode in enumerate(("cached",) + ABBA_MODES):
        is_warmup = run_index == 0
        resource_admission()
        before = source_fingerprint(batch, labels, provenance)
        if source_hashes() != hashes_before:
            raise RuntimeError("benchmark source files changed before optimizer call")
        if runtime_context() != runtime_before:
            raise RuntimeError("benchmark runtime changed before optimizer call")
        if before != cohort_identity:
            raise RuntimeError("bound F16 cohort changed before optimizer call")
        result, elapsed, counts = _run_with_cache_mode(
            mode, batch, labels, config=config, lineages=lineages, runner=runner)
        after = source_fingerprint(batch, labels, provenance)
        if source_hashes() != hashes_before:
            raise RuntimeError("benchmark source files changed during optimizer call")
        if runtime_context() != runtime_before:
            raise RuntimeError("benchmark runtime changed during optimizer call")
        if after != cohort_identity:
            raise RuntimeError("optimizer mutated its bound F16 cohort")
        signature, fingerprint = _result_signature(
            result, batch, split, automatic_audit)
        _validate_summary_activity(mode, counts)
        row = {
            "mode": mode, "elapsed_seconds": elapsed,
            "summarize_counts": counts, "result_fingerprint": fingerprint,
            "optimized_values": signature["values"],
            "optimized_validity": signature["validity"],
            "factor_results": {
                factor_id: {
                    "status": row["status"], "selected_family": row["selected_family"],
                    "plan_identity": row["plan_identity"],
                    "candidate_count": row["candidate_count"],
                    "candidate_ledger_sha256": row["candidate_ledger_sha256"],
                    "joint_diagnostics_sha256": row["joint_diagnostics_sha256"],
                } for factor_id, row in signature["factors"].items()
            },
            "cohort_fingerprint_before": before,
            "cohort_fingerprint_after": after,
        }
        if is_warmup:
            warmup = row
        else:
            modes.append(row)
        signatures.append(fingerprint)
        del result

    hashes_after = source_hashes()
    runtime_after = runtime_context()
    runtime_fingerprint_before = _digest(_canonical_json(runtime_before).encode("utf-8"))
    runtime_fingerprint_after = _digest(_canonical_json(runtime_after).encode("utf-8"))
    if hashes_before != hashes_after:
        raise RuntimeError("benchmark source files changed during ABBA calls")
    if runtime_fingerprint_before != runtime_fingerprint_after:
        raise RuntimeError("benchmark runtime changed during ABBA calls")
    if len(set(signatures)) != 1:
        raise RuntimeError("cached and uncached F16 results differ in output, ledger, or diagnostics")

    pairs = []
    for left, right in ((modes[0], modes[1]), (modes[2], modes[3])):
        uncached = left if left["mode"] == "uncached" else right
        cached = left if left["mode"] == "cached" else right
        ratio = (cached["elapsed_seconds"] / uncached["elapsed_seconds"]
                 if uncached["elapsed_seconds"] > 0 else None)
        pairs.append({
            "order": [left["mode"], right["mode"]],
            "uncached_seconds": uncached["elapsed_seconds"],
            "cached_seconds": cached["elapsed_seconds"],
            "cached_over_uncached_ratio": ratio,
            "relative_time_reduction": None if ratio is None else 1.0 - ratio,
        })
    reductions = [pair["relative_time_reduction"] for pair in pairs
                  if pair["relative_time_reduction"] is not None]
    payload = {
        "schema": "factor_optimizer.raw_summary_cache_abba.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dates": DATE_COUNT, "assets": ASSET_COUNT, "factors": FACTOR_COUNT,
            "factor_ids": list(batch.factor_ids),
            "full_market_or_multiyear_claim": False,
            "test_evaluated": False,
            "semantics": "strict F16 cohort, TRAIN candidate search and frozen-winner VALIDATION only",
        },
        "cohort": {
            "fingerprint": cohort_identity,
            "manifest_uri": provenance["manifest_uri"],
            "loaded_source_bytes": loaded_bytes,
            "batch_shape": list(batch.values.shape),
            "split_identity": split.identity,
            "train_days": len(split.train_indices),
            "validation_days_reserved": len(split.validation_indices),
            "test_days_reserved": len(split.test_indices),
        },
        "cache": {
            "key_panel_byte_cap": MAX_PANEL_KEY_BYTES,
            "report_note": "one-entry exact-content TRAIN RAW summary cache; benchmark does not claim cache changes any research semantics",
        },
        "ordering": list(ABBA_MODES),
        "warmup": warmup,
        "config": asdict(config),
        "runs": modes,
        "paired_timing": {
            "pairs": pairs,
            "median_relative_time_reduction": (
                float(np.median(reductions)) if reductions else None
            ),
            "interpretation": "descriptive paired timings only; no significance claim",
        },
        "equivalence": {
            "all_outputs_and_ledgers_identical": True,
            "runs_compared": len(signatures),
            "result_fingerprint": signatures[0],
        },
        "resources": {
            "admission": "existing strict-cohort RAM/disk guard before load and before each run",
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "rss_scope": "cumulative process ru_maxrss; not an individual per-run peak",
            "input_copy_note": "one DataAccess load reused for all four optimizer calls; no dataset copy/staging tree",
        },
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "runtime_fingerprint_before": runtime_fingerprint_before,
        "runtime_fingerprint_after": runtime_fingerprint_after,
        "report_limit_bytes_strictly_less_than": MAX_REPORT_BYTES,
        "report_overwrite": "refused via exclusive create",
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) >= MAX_REPORT_BYTES:
        raise ValueError("report must be strictly smaller than 1 MiB")
    from audit_real_training_methods_oct03 import write_report_exclusive
    written = write_report_exclusive(destination, payload)
    return {"report_path": str(written), **payload}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="new report path; existing files are refused")
    args = parser.parse_args(argv)
    result = run_benchmark(report=args.report)
    print(json.dumps({
        "report_path": result["report_path"], "scope": result["scope"],
        "equivalence": result["equivalence"],
        "paired_timing": result["paired_timing"],
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
