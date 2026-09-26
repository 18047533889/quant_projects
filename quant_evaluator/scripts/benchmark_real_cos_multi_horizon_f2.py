#!/usr/bin/env python3
"""Bounded full-market F2 two-horizon CUDA tile-reuse A/B.

Run with the same DataAccess COS environment as benchmark_real_cos_metric_batch,
under an external 900-second timeout. Writes only a small JSON summary.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np

from quant_evaluator.api.evaluate_many import evaluate_many
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.scripts.benchmark_real_cos_metric_batch import MANIFEST_SHA256, _plain
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

ROOT = Path("/home/sunhaiwei/quant_projects")
DEFAULT_OUTPUT = ROOT / "quant_evaluator/docs/benchmarks/real_cos_multi_horizon_f2_20260927.json"
METRICS = ("rank_ic_series",)
ORDER = ("sequential", "shared", "shared", "sequential")


def _second_horizon(first):
    """Compound adjacent real AdjVwap periods; unknown last period is invalid."""
    shifted = np.empty_like(first.values)
    shifted[:-1] = first.values[1:]
    shifted[-1] = np.nan
    adjacent = np.array([
        end == start
        for end, start in zip(first.label_end_time[:-1], first.label_start_time[1:])
    ] + [False], dtype=bool)
    with np.errstate(invalid="ignore", over="ignore"):
        values = (1.0 + first.values) * (1.0 + shifted) - 1.0
    validity = (
        first.validity
        & np.concatenate((first.validity[1:], np.zeros_like(first.validity[:1])), axis=0)
        & adjacent[:, None] & np.isfinite(values)
    )
    label = replace(
        first, target_id="adj_vwap_tplus1_to_tplus3_compound_return",
        horizon=2, values=values, validity=validity,
        label_end_time=tuple(first.label_end_time[1:]) + (first.label_end_time[-1],),
        source_ref="data_access:ashare_stock_daily_adj:AdjVwap:adjacent_compound_h2",
    )
    return label, int(adjacent.sum()), int(validity.sum())


def _same_field(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            return bool(np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
        return bool(np.array_equal(a, b))
    return _plain(left) == _plain(right)


def _compare(reference, candidate, labels):
    report = {}
    for label in labels:
        left, right = reference[label.target_id], candidate[label.target_id]
        field_checks = {}
        same_keys = set(left.artifacts) == set(right.artifacts)
        if same_keys:
            for metric in left.artifacts:
                a, b = left.artifacts[metric], right.artifacts[metric]
                field_checks[metric] = (
                    {field.name: _same_field(getattr(a, field.name), getattr(b, field.name))
                     for field in fields(a)}
                    if type(a) is type(b) and is_dataclass(a) else {"artifact_type": False}
                )
        checks = {
            "config_hash": left.config_hash == right.config_hash,
            "artifact_keys": same_keys,
            "artifact_fields": field_checks,
            "grouped_metrics": _same_field(left.grouped_metrics, right.grouped_metrics),
            "factor_artifacts": _same_field(left.factor_artifacts, right.factor_artifacts),
            "semantic_provenance": _same_field(
                left.metadata.get("provenance"), right.metadata.get("provenance")),
        }
        checks["pass"] = (
            checks["config_hash"] and checks["artifact_keys"]
            and all(all(group.values()) for group in field_checks.values())
            and checks["grouped_metrics"] and checks["factor_artifacts"]
            and checks["semantic_provenance"]
        )
        report[label.target_id] = checks
    return report


def main(output: Path) -> None:
    started = time.perf_counter()
    batch, first, source = load_real_batch(
        factors=2, days=0, assets=5500, max_object_mib=64,
        max_total_mib=128, manifest_sha256=MANIFEST_SHA256,
    )
    shape = (batch.num_times, batch.num_assets, batch.num_factors)
    if shape != (2586, 5461, 2):
        raise RuntimeError(f"unexpected bound shape: {shape}")
    load_seconds = time.perf_counter() - started
    second, adjacent_days, valid_cells = _second_horizon(first)
    labels = (first, second)

    uploads = []
    original_stage = DeviceEvaluationSession.stage_factors

    def stage(session, values, factor_ids, layout="T,F,N"):
        uploads.append(len(factor_ids))
        return original_stage(session, values, factor_ids, layout)

    DeviceEvaluationSession.stage_factors = stage
    runs = []
    reference = None
    for route in ORDER:
        uploads.clear()
        gc.collect()
        started = time.perf_counter()
        if route == "sequential":
            result = {
                lb.target_id: evaluate(batch, lb, metrics=METRICS, backend="cuda_strict")
                for lb in labels
            }
        else:
            result = evaluate_many(batch, labels, metrics=METRICS, backend="cuda_strict")
        seconds = time.perf_counter() - started
        parity = None if reference is None else _compare(reference, result, labels)
        if reference is None:
            reference = result
        metadata = [result[lb.target_id].metadata for lb in labels]
        runs.append({
            "route": route,
            "seconds": round(seconds, 6),
            "factor_uploads": len(uploads),
            "factor_upload_widths": list(uploads),
            "h2d_bytes": (
                sum(item["h2d_bytes"] for item in metadata)
                if route == "sequential" else metadata[0]["h2d_bytes"]),
            "d2h_bytes": (
                sum(item["d2h_bytes"] for item in metadata)
                if route == "sequential" else metadata[0]["d2h_bytes"]),
            "peak_vram_bytes": max(item["peak_vram"] for item in metadata),
            "counter_scope": metadata[0].get(
                "device_session_counter_scope", "per_evaluation_session"),
            "config_hashes": {
                lb.target_id: result[lb.target_id].config_hash for lb in labels
            },
            "parity_with_first_sequential": parity,
        })
        if parity is not None and not all(check["pass"] for check in parity.values()):
            raise AssertionError(f"full artifact parity failed in {route}: {parity}")
        if route == "shared":
            del result

    sequential = statistics.median(
        run["seconds"] for run in runs if run["route"] == "sequential")
    shared = statistics.median(
        run["seconds"] for run in runs if run["route"] == "shared")
    summary = {
        "manifest_sha256": MANIFEST_SHA256,
        "source_factors": [
            {key: item[key] for key in ("factor_id", "uri", "sha256", "bytes", "etag",
                                        "manifest_sha256", "source_status")}
            for item in source["sources"]
        ],
        "label_source": source["label"],
        "shape": shape,
        "horizon_ids": [lb.target_id for lb in labels],
        "horizon2_construction": "adjacent real AdjVwap returns compounded; last row invalid",
        "horizon2_adjacent_days": adjacent_days,
        "horizon2_valid_cells": valid_cells,
        "load_seconds": round(load_seconds, 6),
        "run_order": ORDER,
        "runs": runs,
        "median_sequential_seconds": sequential,
        "median_shared_seconds": shared,
        "speedup": sequential / shared,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "all_parity_pass": all(
            all(check["pass"] for check in run["parity_with_first_sequential"].values())
            for run in runs if run["parity_with_first_sequential"] is not None
        ),
    }
    if not summary["all_parity_pass"]:
        raise AssertionError("A/B parity failed")
    output.write_text(json.dumps(_plain(summary), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(output), "all_parity_pass": True,
        "median_sequential_seconds": sequential,
        "median_shared_seconds": shared,
        "speedup": sequential / shared,
    }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.output)
