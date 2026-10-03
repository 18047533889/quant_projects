"""Paired real-cohort benchmark for prepared TRAIN labels; never a selector.

The caller must explicitly run this script. It loads the strict F16 cohort once,
uses its nested F4 prefix, and compares the existing prepared_split path with a
benchmark-only wrapper that removes only that keyword from paired_series.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/sunhaiwei/quant_projects")
SCRIPTS = ROOT / "factor_optimizer/scripts"
FACTOR_COUNT = 16
PREFIX_SIZE = 4
MAX_FACTOR_BYTES = 128 * 1024**2
MAX_BATCH_BYTES = 2 * 1024**3
MIN_MEMORY_BYTES = 32 * 1024**3
MIN_DISK_BYTES = 2 * 1024**3
MAX_REPORT_BYTES = 1 * 1024**2
WARMUPS = 2
PAIRED_BLOCKS = 3
PAIR_SEED = 20261004
_RUNTIME_FILES = (
    "factor_optimizer/factor_optimizer/research_batch.py",
    "factor_optimizer/factor_optimizer/research_fitness.py",
    "factor_optimizer/factor_optimizer/adapters/repair_execution.py",
    "factor_optimizer/factor_optimizer/research_baseline.py",
    "factor_optimizer/factor_optimizer/research_batch_diagnostics.py",
    "factor_optimizer/scripts/audit_real_automatic_oct03.py",
    "factor_optimizer/scripts/audit_real_training_methods_oct03.py",
    "factor_optimizer/scripts/audit_real_scaling_cohort_oct03.py",
    "factor_optimizer/scripts/benchmark_prepared_joint_real_oct04.py",
)


def _canonical(value):
    """Normalize result values without hiding unsupported or nonfinite data."""
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _canonical(item) for key, item in
                sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported result value {type(value).__qualname__}")


def _runtime_fingerprint():
    digest = hashlib.sha256()
    for relative in _RUNTIME_FILES:
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    import platform
    digest.update(platform.python_version().encode())
    digest.update(np.__version__.encode())
    for distribution in ("pandas", "polars", "factor-engine", "quant-evaluator"):
        try:
            package_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_version = "not-installed"
        digest.update(distribution.encode())
        digest.update(package_version.encode())
    return digest.hexdigest()


def _json_digest(value):
    encoded = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value):
    if value is None:
        return None
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    if array.dtype.hasobject:
        # Object buffers contain addresses, not coordinate values. Preserve
        # logical scalar types and fail closed for unsupported object payloads.
        cells = []
        for cell in array.flat:
            if cell is None:
                cells.append(("none", None))
            elif isinstance(cell, (str, np.str_)):
                cells.append(("str", str(cell)))
            elif isinstance(cell, (bool, np.bool_)):
                cells.append(("bool", bool(cell)))
            elif isinstance(cell, (int, np.integer)):
                cells.append(("int", str(int(cell))))
            elif isinstance(cell, (float, np.floating)):
                if isinstance(cell, np.floating) and cell.dtype.itemsize > 8:
                    raise TypeError("extended-precision object coordinates are unsupported")
                cells.append(("float", float(cell).hex()))
            else:
                raise TypeError(f"unsupported object coordinate {type(cell).__qualname__}")
        digest.update(json.dumps(cells, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))
    else:
        digest.update(memoryview(array).cast("B"))
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": digest.hexdigest()}


def _rss_sampler():
    """Sample this process RSS; unlike ru_maxrss this is per-run, not cumulative."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    state = {"peak": 0, "stop": False}

    def sample():
        while not state["stop"]:
            try:
                resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
                state["peak"] = max(state["peak"], resident_pages * page_size)
            except (OSError, ValueError, IndexError):
                pass
            time.sleep(.05)

    thread = threading.Thread(target=sample, name="prepared-ab-rss", daemon=True)
    thread.start()
    return state, thread


def _result_snapshot(result, batch, split, config):
    from audit_real_automatic_oct03 import validate_result

    validate_result(result, batch, split)
    if result.split != split or result.config != config:
        raise ValueError("optimizer result split/config differs from the prescribed run")
    factors = {}
    for factor_id, item in result.factors.items():
        if item.plan is None or item.plan_identity != item.plan.identity:
            raise ValueError("selected plan identity is missing or changed")
        ledger = [dict(row) for row in item.candidates]
        diagnostics = {
            "training": item.training_diagnostics,
            "baseline": item.baseline_diagnostics,
            "joint": item.joint_diagnostics,
        }
        factors[factor_id] = _canonical({
            "status": item.status, "selected_family": item.selected_family,
            "reason": item.reason, "plan_identity": item.plan_identity,
            "train_gain": item.train_gain,
            "validation_lower_bound": item.validation_lower_bound,
            "validation_candidate_identity": item.validation_candidate_identity,
            "validation_coverage": item.validation_coverage,
            "materialization_error": item.materialization_error,
            "candidate_count": len(ledger),
            "candidate_ledger_sha256": _json_digest(ledger),
            "diagnostics_sha256": _json_digest(diagnostics),
            "candidate_evaluated": (item.training_diagnostics or {})
                .get("candidate_budget", {}).get("evaluated", 0),
        })
    output = result.optimized
    return _canonical({
        "factor_ids": output.factor_ids,
        "values": _array_digest(output.values),
        "validity": _array_digest(output.validity),
        "time_axis": _array_digest(output.time_axis.values),
        "asset_axis": _array_digest(output.asset_axis.values),
        "context_refs": output.context_refs, "factors": factors,
        "split": {"identity": split.identity, "train": split.train_indices,
                  "validation": split.validation_indices, "test": split.test_indices},
        "config": vars(config),
        "execution_mode": result.execution_mode,
        "test_evaluated": result.test_evaluated,
    })


def _without_prepared_split(original):
    """Return a benchmark-only paired_series wrapper changing one keyword."""
    def wrapper(*args, **kwargs):
        if "prepared_split" not in kwargs or kwargs["prepared_split"] is None:
            return original(*args, **kwargs)
        kwargs = dict(kwargs)
        kwargs.pop("prepared_split")
        return original(*args, **kwargs)
    return wrapper


def _paired_orders(seed):
    rng = random.Random(seed)
    warmups = ["prepared", "unprepared"]
    rng.shuffle(warmups)
    measured = []
    for _ in range(PAIRED_BLOCKS):
        order = ["prepared", "unprepared"]
        rng.shuffle(order)
        measured.append(order)
    return warmups, measured


def _invoke_pair(*, mode, runner, batch, labels, lineages, config, split,
                 resource_check, source_check, pair_module, snapshotter=None):
    resource_check()
    source_check()
    original = pair_module.paired_series
    rss_state, rss_thread = {"peak": 0, "stop": False}, None
    elapsed = None
    try:
        if mode == "unprepared":
            pair_module.paired_series = _without_prepared_split(original)
        rss_state, rss_thread = _rss_sampler()
        started = time.perf_counter()
        result = runner(batch, labels, config=config, allow_research=True,
                        lineages=lineages)
        elapsed = time.perf_counter() - started
        rss_state["stop"] = True
        if rss_thread is not None:
            rss_thread.join(timeout=1)
        snapshot = (snapshotter(result) if snapshotter is not None else
                    _result_snapshot(result, batch, split, config))
    finally:
        rss_state["stop"] = True
        if rss_thread is not None:
            rss_thread.join(timeout=1)
        pair_module.paired_series = original
    return snapshot, elapsed, rss_state["peak"]


def _assert_same_result(left, right, scope):
    if left != right:
        raise RuntimeError(f"result mismatch in {scope}")


def run_audit(*, source_loader=None, auto_runner=None, resource_check=None,
              config=None, seed=PAIR_SEED):
    """Run two warmups and three randomized paired blocks on strict nested F4."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from audit_real_training_methods_oct03 import (
        MAX_REPORT_BYTES as HELPER_MAX_REPORT_BYTES,
        check_environment, check_resource_headroom, source_fingerprint,
        validate_source,
    )
    from audit_real_automatic_oct03 import validate_result  # noqa: F401
    from factor_optimizer.research_batch import (
        BatchOptimizationConfig, automatic_time_split, optimize_factor_batch,
    )
    from factor_optimizer import research_fitness
    from quant_evaluator.contracts.factor_batch import FactorBatch

    if str(ROOT / "factor_optimizer/examples") not in sys.path:
        sys.path.insert(0, str(ROOT / "factor_optimizer/examples"))
    from cos_batch_audit import load_cos_sample

    loader = load_cos_sample if source_loader is None else source_loader
    runner = optimize_factor_batch if auto_runner is None else auto_runner
    check_environment()
    admit = (resource_check if resource_check is not None else
             lambda: check_resource_headroom(minimum_memory_bytes=MIN_MEMORY_BYTES,
                                             minimum_disk_bytes=MIN_DISK_BYTES))
    admit()
    cohort, labels, provenance, lineages = loader(
        n_factors=FACTOR_COUNT, n_assets=256, include_lineages=True,
        max_factor_bytes=MAX_FACTOR_BYTES, max_batch_factor_bytes=MAX_BATCH_BYTES,
        coverage_policy="strict")
    from audit_real_scaling_cohort_oct03 import _validate_cohort
    loaded_bytes = _validate_cohort(cohort, provenance, lineages)
    if len(cohort.factor_ids) != FACTOR_COUNT:
        raise ValueError("strict F16 cohort must retain exactly 16 factors")
    from audit_real_scaling_cohort_oct03 import _leaf_source
    for factor_id in cohort.factor_ids:
        index = cohort.factor_ids.index(factor_id)
        one = FactorBatch((factor_id,), cohort.time_axis, cohort.asset_axis,
                          cohort.values[:, :, index:index+1],
                          validity=(None if cohort.validity is None else
                                    cohort.validity[:, :, index:index+1]))
        validate_source(one, labels, _leaf_source(provenance, factor_id),
                        max_factor_bytes=MAX_FACTOR_BYTES,
                        max_batch_factor_bytes=MAX_BATCH_BYTES)
        del one
    prefix_ids = cohort.factor_ids[:PREFIX_SIZE]
    batch = FactorBatch(prefix_ids, cohort.time_axis, cohort.asset_axis,
                        cohort.values[:, :, :PREFIX_SIZE],
                        validity=(None if cohort.validity is None else
                                  cohort.validity[:, :, :PREFIX_SIZE]),
                        context_refs=cohort.context_refs)
    prefix_provenance = dict(provenance)
    prefix_provenance["retained_factor_ids"] = list(prefix_ids)
    prefix_provenance["sources"] = [row for row in provenance["sources"]
                                    if row["factor"] in prefix_ids]
    leaf_lineages = {key: lineages[key] for key in prefix_ids}
    config = BatchOptimizationConfig() if config is None else config
    if config != BatchOptimizationConfig():
        raise ValueError("prepared-split A/B protocol requires the unchanged default config")
    split = automatic_time_split(labels, config)
    if split.identity != provenance.get("asset_selection_split"):
        raise ValueError("optimizer split differs from TRAIN asset-selection split")
    fingerprint = source_fingerprint(batch, labels, prefix_provenance)
    cohort_fingerprint = source_fingerprint(cohort, labels, provenance)
    runtime_fingerprint = _runtime_fingerprint()

    def source_check():
        if (source_fingerprint(batch, labels, prefix_provenance) != fingerprint
                or source_fingerprint(cohort, labels, provenance) != cohort_fingerprint):
            raise RuntimeError("source data/provenance drift detected")
        if _runtime_fingerprint() != runtime_fingerprint:
            raise RuntimeError("optimizer source/runtime drift detected")

    warmup_modes, measured_orders = _paired_orders(seed)
    trials = []
    snapshots = {}
    for index, mode in enumerate(warmup_modes):
        snapshot, elapsed, rss = _invoke_pair(
            mode=mode, runner=runner, batch=batch, labels=labels,
            lineages=leaf_lineages, config=config, split=split,
            resource_check=admit, source_check=source_check,
            pair_module=research_fitness)
        snapshots[mode] = snapshot
        trials.append({"phase": "warmup", "index": index, "mode": mode,
                       "elapsed_seconds": elapsed, "sampled_peak_rss_bytes": rss})
    _assert_same_result(snapshots["prepared"], snapshots["unprepared"],
                        "prepared/unprepared warmup")

    blocks = []
    for block, order in enumerate(measured_orders):
        pair = {}
        for mode in order:
            snapshot, elapsed, rss = _invoke_pair(
                mode=mode, runner=runner, batch=batch, labels=labels,
                lineages=leaf_lineages, config=config, split=split,
                resource_check=admit, source_check=source_check,
                pair_module=research_fitness)
            pair[mode] = snapshot
            trials.append({"phase": "measured", "block": block,
                           "mode": mode, "elapsed_seconds": elapsed,
                           "sampled_peak_rss_bytes": rss})
        _assert_same_result(pair["prepared"], pair["unprepared"],
                            f"paired block {block}")
        blocks.append({"block": block, "order": order, "exact_result_match": True})
    source_check()
    attempts = sum(row["candidate_count"] for row in
                   snapshots["prepared"]["factors"].values())
    evaluated = sum(row["candidate_evaluated"] for row in
                    snapshots["prepared"]["factors"].values())
    report = {
        "audit": "prepared-joint-real-ab-oct04-v1",
        "source": prefix_provenance, "source_fingerprint": fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_scope": list(_RUNTIME_FILES) + [
            "python version", "numpy version", "pandas/polars/FactorEngine/QuantEvaluator installed versions"],
        "factor_ids": list(prefix_ids), "batch_shape": list(batch.values.shape),
        "config": _canonical(vars(config)),
        "split": {"identity": split.identity,
                  "train_days": len(split.train_indices),
                  "validation_days_reserved": len(split.validation_indices),
                  "test_days_reserved": len(split.test_indices)},
        "test_evaluated": False, "loaded_f16_source_bytes": loaded_bytes,
        "pair_seed": seed, "warmups": WARMUPS, "paired_blocks": PAIRED_BLOCKS,
        "comparison": "prepared_split versus paired_series wrapper dropping only prepared_split",
        "exact_result_equality": True, "blocks": blocks, "trials": trials,
        "timing_scope": "optimize_factor_batch API call including its internals; excludes result validation and hashing",
        "oom_observed": False,
        "candidate_attempted": attempts, "candidate_evaluated": evaluated,
        "background_load": {
            "loadavg": Path("/proc/loadavg").read_text().strip().split()[:3],
            "mem_available_bytes": next((int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")), None),
        },
        "limits": {"minimum_memory_admission_bytes": MIN_MEMORY_BYTES,
                   "minimum_disk_headroom_bytes": MIN_DISK_BYTES,
                   "report_limit_bytes": min(MAX_REPORT_BYTES, HELPER_MAX_REPORT_BYTES)},
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if len(encoded.encode("utf-8")) > min(MAX_REPORT_BYTES, HELPER_MAX_REPORT_BYTES):
        raise ValueError("JSON report exceeds the 1 MiB output limit")
    return report


def _preflight_report_path(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    parent = path.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise OSError("report parent directory must exist and be writable")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="explicitly run the real-cohort paired workload")
    parser.add_argument("--report", type=Path,
                        help="write full JSON report to this new exclusive path")
    args = parser.parse_args(argv)
    if not args.execute:
        if args.report is not None:
            parser.error("--report requires explicit --execute")
        print(json.dumps({
            "mode": "dry_run", "audit": "prepared-joint-real-ab-oct04-v1",
            "cohort": "strict F16 manifest-bound source; first-four prefix",
            "warmups": WARMUPS, "randomized_paired_blocks": PAIRED_BLOCKS,
            "seed": PAIR_SEED, "test_evaluated": False,
            "minimum_memory_admission_bytes": MIN_MEMORY_BYTES,
            "requires": "--execute --report NEW_PATH",
        }, ensure_ascii=False, separators=(",", ":")))
        return 0
    if args.report is None:
        parser.error("--execute requires --report NEW_PATH")
    report_path = _preflight_report_path(args.report)
    report = run_audit()
    from audit_real_training_methods_oct03 import write_report_exclusive
    write_report_exclusive(report_path, report)
    print(json.dumps({"report_path": str(report_path), "test_evaluated": False,
                      "paired_blocks": report["paired_blocks"],
                      "oom_observed": report["oom_observed"]},
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
