"""Bounded synthetic A/B for TRAIN U-shape shared-rank reuse.

Run each mode in a separate process for independent peak-RSS readings. Keep
input_cells <= MAX_INPUT_CELLS; the panel is synthetic and never reads project
data. Each JSON line records fixture seeds/configuration and source provenance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
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
from unittest.mock import patch

import numpy as np

from factor_optimizer.adapters import repair_execution
from factor_optimizer.research_batch import (
    BatchOptimizationConfig,
    optimize_factor_batch,
)
import factor_optimizer.shape_rank_reuse as shape_rank_reuse
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


MAX_INPUT_CELLS = 3_000_000
MAX_JSON_RECORD_BYTES = 64 * 1024
OPTIMIZER_SEED = 73
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
PROVENANCE_PATHS = (
    "factor_optimizer/scripts/benchmark_u_shape_rank_reuse.py",
    "factor_optimizer/scripts/run_u_shape_rank_reuse_matrix.py",
    "factor_optimizer/factor_optimizer/shape_rank_reuse.py",
    "factor_optimizer/factor_optimizer/adapters/repair_execution.py",
    "factor_optimizer/factor_optimizer/research_batch.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *PROVENANCE_PATHS],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        relevant_worktree_dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        relevant_worktree_dirty = None
    return {
        "git_head": revision,
        "relevant_worktree_dirty": relevant_worktree_dirty,
        "relevant_file_sha256": {
            name: _sha256_file(PROJECT_ROOT / name)
            for name in PROVENANCE_PATHS
            if (PROJECT_ROOT / name).is_file()
        },
    }


def _host_context() -> dict:
    try:
        load_avg_1m = round(os.getloadavg()[0], 3)
    except (AttributeError, OSError):
        load_avg_1m = None
    available_memory_mib = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_memory_mib = round(int(line.split()[1]) / 1024, 1)
                break
    except (OSError, ValueError, IndexError):
        pass
    return {
        "cpu_count": os.cpu_count(),
        "load_avg_1m": load_avg_1m,
        "available_memory_mib": available_memory_mib,
        "load_context_note": "host-wide snapshot; does not attribute other processes",
    }


def _write_compact_record(path: Path, payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    record = encoded + "\n"
    if len(record.encode("utf-8")) > MAX_JSON_RECORD_BYTES:
        raise ValueError(f"JSON run record exceeds {MAX_JSON_RECORD_BYTES} bytes")
    with path.open("x", encoding="utf-8") as output:
        output.write(record)
    return encoded


def _cache_counter_record(caches, expected_class) -> dict:
    names = ("hits", "misses", "rank_calls", "bypasses", "evictions", "retained_bytes")
    per_instance = [
        {name: int(getattr(cache, name)) for name in names}
        for cache in caches
    ]
    totals = {
        name: sum(item[name] for item in per_instance)
        for name in names
    }
    return {
        "instance_count": len(caches),
        "class_identity_preserved": bool(caches) and all(
            type(cache) is expected_class for cache in caches
        ),
        "totals": totals,
        "per_instance": per_instance,
    }


def run(*, time_points: int, assets: int, mode: str, seed: int,
        families: tuple[str, ...] | None = None,
        maximum_candidates: int = 128) -> dict:
    if time_points < 100 or assets < 20:
        raise ValueError("use at least 100 dates and 20 assets")
    if time_points * assets > MAX_INPUT_CELLS:
        raise ValueError(f"panel exceeds bounded {MAX_INPUT_CELLS:,}-cell limit")
    if mode not in {"cached", "uncached"}:
        raise ValueError("mode must be cached or uncached")

    source_provenance = _source_provenance()
    host_context = _host_context()
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((time_points, assets))
    time_axis = AxisRef("time", "int", time_points, np.arange(time_points))
    asset_axis = AxisRef("asset", "str", assets,
                         np.array([f"a{i}" for i in range(assets)]))
    batch = FactorBatch(("synthetic_u",), time_axis, asset_axis, values[:, :, None])
    labels = LabelBundle(
        "synthetic_u_rank_reuse_bench",
        values * values,
        1,
        decision_time=tuple(range(time_points)),
        label_start_time=tuple(range(1, time_points + 1)),
        label_end_time=tuple(range(2, time_points + 2)),
        asset_axis=asset_axis,
    )

    fingerprint = {"calls": 0, "seconds": 0.0}
    fe_rank_calls = {"count": 0}
    fp_rank_shape_calls = {"count": 0}
    observed_plan_identities = set()
    observed_frames = {
        "cached_fingerprint_rows": [],
        "cached_fe_rank_rows": [],
        "uncached_rank_shape_execute_rows": [],
    }
    original_key = shape_rank_reuse._frame_key
    original_fe_rank = repair_execution._execute_fe_cs_rank
    original_execute = repair_execution.ValueRepairPlan.execute
    original_cache_class = shape_rank_reuse.RankFeatureCache
    cache_instances = []

    def capture_cache(*args, **kwargs):
        cache = original_cache_class(*args, **kwargs)
        cache_instances.append(cache)
        return cache

    def timed_key(*args, **kwargs):
        frame = args[0] if args else kwargs.get("frame")
        if frame is not None:
            observed_frames["cached_fingerprint_rows"].append(len(frame))
        started = time.perf_counter()
        try:
            return original_key(*args, **kwargs)
        finally:
            fingerprint["calls"] += 1
            fingerprint["seconds"] += time.perf_counter() - started

    def timed_fe_rank(frame):
        fe_rank_calls["count"] += 1
        observed_frames["cached_fe_rank_rows"].append(len(frame))
        return original_fe_rank(frame)

    def timed_execute(self, *args, **kwargs):
        if self.transform == "rank_shape":
            fp_rank_shape_calls["count"] += 1
            identity = getattr(self, "identity", None)
            if identity is not None:
                observed_plan_identities.add(identity)
            frame = args[0] if args else kwargs.get("values")
            if frame is not None:
                observed_frames["uncached_rank_shape_execute_rows"].append(len(frame))
        return original_execute(self, *args, **kwargs)

    optimization_config = BatchOptimizationConfig(
        selection_objective="rank_ic",
        families=tuple(families or ("U_SHAPE_REPAIR", "INVERTED_U_REPAIR")),
        maximum_candidates=maximum_candidates,
        bootstrap_draws=99,
        seed=OPTIMIZER_SEED,
    )
    started = time.perf_counter()
    with ExitStack() as patches:
        # A/B modes deliberately override production admission so tiny synthetic
        # panels still exercise both kernels; this is not normal gate behavior.
        patches.enter_context(patch.object(
            shape_rank_reuse, "should_admit_u_shape_rank_reuse",
            lambda *args, **kwargs: mode == "cached",
        ))
        if mode == "uncached":
            patches.enter_context(patch.object(
                shape_rank_reuse,
                "apply_u_shape_from_rank",
                lambda *args, **kwargs: None,
            ))
        patches.enter_context(patch.object(
            shape_rank_reuse, "RankFeatureCache", capture_cache
        ))
        patches.enter_context(patch.object(shape_rank_reuse, "_frame_key", timed_key))
        patches.enter_context(patch.object(
            repair_execution, "_execute_fe_cs_rank", timed_fe_rank
        ))
        patches.enter_context(patch.object(
            repair_execution.ValueRepairPlan, "execute", timed_execute
        ))
        original_apply_u_shape = shape_rank_reuse.apply_u_shape_from_rank
        def observe_apply(plan, *args, **kwargs):
            if getattr(plan, "transform", None) == "rank_shape":
                observed_plan_identities.add(plan.identity)
            return original_apply_u_shape(plan, *args, **kwargs)
        patches.enter_context(patch.object(
            shape_rank_reuse, "apply_u_shape_from_rank", observe_apply
        ))
        result = optimize_factor_batch(
            batch,
            labels,
            config=optimization_config,
            allow_research=True,
        )
    elapsed = time.perf_counter() - started

    inferred_train_rows = result.split.validation_start * assets
    if mode == "cached":
        cache_observed = (
            observed_frames["cached_fingerprint_rows"]
            + observed_frames["cached_fe_rank_rows"]
        )
        if not observed_frames["cached_fingerprint_rows"]:
            raise RuntimeError("cached run did not observe a fingerprinted TRAIN frame")
        if not observed_frames["cached_fe_rank_rows"]:
            raise RuntimeError("cached run did not observe an FE-ranked TRAIN frame")
        if len(set(cache_observed)) != 1:
            raise RuntimeError(f"inconsistent cached TRAIN frame lengths: {cache_observed}")
        actual_train_rows = cache_observed[0]
        uncached_train_rows = []
    else:
        execute_rows = observed_frames["uncached_rank_shape_execute_rows"]
        if not execute_rows:
            raise RuntimeError("uncached run did not observe a rank-shape TRAIN execution")
        actual_train_rows = execute_rows[0]
        uncached_train_rows = []
        for frame_rows in execute_rows:
            if frame_rows != actual_train_rows:
                break
            uncached_train_rows.append(frame_rows)
        if not uncached_train_rows:
            raise RuntimeError("uncached run had no consistent initial TRAIN executions")
    if actual_train_rows != inferred_train_rows:
        raise RuntimeError(
            f"observed TRAIN frame has {actual_train_rows} rows, "
            f"split implies {inferred_train_rows}"
        )

    selected = result.factors["synthetic_u"]
    evidence = [
        (
            item["family"],
            item["parameters"],
            item["status"],
            item.get("plan_identity"),
            item.get("train_gain"),
            item.get("coverage"),
        )
        for item in selected.candidates
    ]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str)
    training_rows = result.split.validation_start * assets
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "admission_semantics": {
            "mode": "forced_for_ab",
            "reuse_admitted": mode == "cached",
            "production_gate_bypassed": True,
        },
        "panel_shape": [time_points, assets],
        "input_cells": time_points * assets,
        "training_frame_rows": actual_train_rows,
        "training_frame_observations": {
            "actual_rows": actual_train_rows,
            "split_inferred_rows": inferred_train_rows,
            "consistent": actual_train_rows == inferred_train_rows,
            **observed_frames,
            "uncached_initial_train_execute_rows": uncached_train_rows,
        },
        "rank_cache_bytes_expected": training_rows * np.dtype(np.float64).itemsize
        if mode == "cached" else 0,
        "rank_cache_budget_bytes": shape_rank_reuse.DEFAULT_MAX_BYTES,
        "elapsed_seconds": elapsed,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "fingerprint_calls": fingerprint["calls"],
        "fingerprint_seconds": fingerprint["seconds"],
        "fe_average_rank_calls": fe_rank_calls["count"],
        "observed_distinct_train_rank_plan_identities": len(observed_plan_identities),
        "fp_rank_shape_execute_calls": fp_rank_shape_calls["count"],
        "candidate_count": len(evidence),
        "candidate_budget": dict(selected.training_diagnostics.get("candidate_budget", {})),
        "candidate_evidence_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "selected_family": selected.selected_family,
        "selected_plan_identity": selected.plan_identity,
        "train_gain": selected.train_gain,
        "validation_lower_bound": selected.validation_lower_bound,
        "fixture": {
            "panel_seed": seed,
            "rng": "numpy.random.default_rng/PCG64",
            "panel_values": "standard_normal",
            "label_values": "panel_values ** 2",
            "label_horizon": 1,
            "optimizer_seed": OPTIMIZER_SEED,
            "batch_optimization_config": {
                **asdict(optimization_config),
                "families": list(optimization_config.families),
            },
            "allow_research": True,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "execution_limits": {
            "max_input_cells": MAX_INPUT_CELLS,
            "mode_per_process": True,
        },
        "source": source_provenance,
        "host_context": host_context,
        "rank_cache_metrics": _cache_counter_record(
            cache_instances, original_cache_class
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-points", type=int, required=True)
    parser.add_argument("--assets", type=int, required=True)
    parser.add_argument("--mode", choices=("cached", "uncached"), required=True)
    parser.add_argument("--seed", type=int, default=20261001,
                        help="synthetic panel seed; optimizer seed is fixed separately")
    parser.add_argument("--families", help="comma-separated family names; defaults to U and inverted-U")
    parser.add_argument("--maximum-candidates", type=int, default=128)
    parser.add_argument("--json-out", type=Path,
                        help="exclusive-create path for a compact JSON run record")
    args = parser.parse_args()
    if args.maximum_candidates < 1:
        parser.error("--maximum-candidates must be positive")
    families = tuple(part for part in args.families.split(",") if part) if args.families else None
    payload = run(time_points=args.time_points, assets=args.assets,
                  mode=args.mode, seed=args.seed, families=families,
                  maximum_candidates=args.maximum_candidates)
    if args.json_out is not None:
        _write_compact_record(args.json_out, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
