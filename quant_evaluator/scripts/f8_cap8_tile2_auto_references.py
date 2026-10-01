"""Strict fresh-reference validator for F8 auto cap-8/effective-tile-2."""
from __future__ import annotations

import json
import re
from pathlib import Path


F8_SHAPE = (2586, 5461, 8)
RANK_PAIR = ("rank_ic", "rank_ic_series")
EXPECTED_ORDERS = {("cpu", "cuda_strict"), ("cuda_strict", "cpu")}
HASH_FIELDS = ("source_request_identity_sha256", "source_snapshot_id",
               "source_manifest_sha256", "factor_ids_sha256",
               "source_request_fingerprint")


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_cap8_tile2_references(paths, selected_metrics, manifest_sha256):
    """Validate two opposite-order, exact tile-2 A/Bs for a future cap-8 route.

    Source memory budgets may differ from the cap-8 auto invocation. Request
    fingerprints and source identities may not: tile cap is intentionally not
    part of the semantic request fingerprint.
    """
    if tuple(selected_metrics) != RANK_PAIR or len(paths) != 2:
        raise ValueError("cap-8/tile-2 verification requires the exact F8 rank pair")
    reports = []
    for path in paths:
        path = Path(path)
        if path.stat().st_size > 1024**2:
            raise ValueError("reference report exceeds 1 MiB")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete"
                or report.get("kind") != "real_cos_whole_source_batch_ab.v1"
                or report.get("manifest_sha256") != manifest_sha256
                or tuple(report.get("shape", ())) != F8_SHAPE
                or report.get("factor_dtype") != "float64"
                or report.get("tile_size") != 2
                or tuple(report.get("metric_ids", ())) != RANK_PAIR
                or not report.get("comparison", {}).get("pass")
                or report.get("comparison", {}).get("compared_factor_count") != 8
                or report.get("comparison", {}).get("compared_metric_count") != 20696
                or tuple(report.get("run_order", ())) not in EXPECTED_ORDERS
                or report.get("source_adapter") != "cos"
                or report.get("cos_prefetch") != "auto"
                or report.get("prefetch_mode") != "auto"
                or report.get("prefetch_objects") is not True
                or report.get("prefetch_window") != 2):
            raise ValueError("reference is not a complete exact F8 tile-2 COS A/B")
        run_list = report.get("runs")
        if not isinstance(run_list, list) or len(run_list) != 2:
            raise ValueError("reference must contain exactly two backend runs")
        runs = {run.get("backend_requested"): run for run in run_list}
        if (set(runs) != {"cpu", "cuda_strict"}
                or runs["cpu"].get("backend_used") != "cpu"
                or runs["cuda_strict"].get("backend_used") != "cuda"):
            raise ValueError("reference does not contain the requested CPU/CUDA runs")
        for run in runs.values():
            if (run.get("source_adapter") != "cos"
                    or run.get("cos_prefetch") != "auto"
                    or run.get("prefetch_mode") != "auto"
                    or run.get("prefetch_objects") is not True
                    or run.get("prefetch_window") != 2
                    or run.get("source_manifest_sha256") != manifest_sha256
                    or run.get("factor_tiles_processed") != 4
                    or run.get("tile_ranges") != [[0, 2], [2, 4], [4, 6], [6, 8]]
                    or any(not _is_sha256(run.get(field)) for field in HASH_FIELDS)):
                raise ValueError("reference run lacks exact source or request identity")
        if any(runs["cpu"][field] != runs["cuda_strict"][field] for field in HASH_FIELDS):
            raise ValueError("CPU and CUDA reference requests have different identities")
        for metric, expected_shape, expected_values in (
                ("rank_ic", [8], 8),
                ("rank_ic_series", [2586, 8], 2586 * 8)):
            entry = report["comparison"]["metrics"].get(metric, {})
            if (not entry.get("pass")
                    or entry.get("artifact_kind") != ("scalar" if metric == "rank_ic" else "series")
                    or entry.get("cpu_shape") != expected_shape
                    or entry.get("cuda_shape") != expected_shape
                    or entry.get("compared_value_count") != expected_values
                    or not entry.get("finite_mask_equal")
                    or not entry.get("observation_counts_equal")
                    or any(not _is_sha256(entry.get(field)) for field in (
                        "cpu_values_sha256", "cuda_values_sha256",
                        "cpu_observation_counts_sha256", "cuda_observation_counts_sha256"))
                    or entry.get("cpu_observation_counts_sha256")
                    != entry.get("cuda_observation_counts_sha256")):
                raise ValueError(f"reference metric receipt is incomplete: {metric}")
        reports.append((report, runs))

    first, second = reports
    if first[0]["run_order"] == second[0]["run_order"]:
        raise ValueError("tile-2 references must have opposite run orders")
    for field in HASH_FIELDS:
        if first[1]["cpu"][field] != second[1]["cpu"][field]:
            raise ValueError("opposite-order references use different source requests")
    for metric in RANK_PAIR:
        left = first[0]["comparison"]["metrics"][metric]
        right = second[0]["comparison"]["metrics"][metric]
        for field in ("cpu_values_sha256", "cuda_values_sha256",
                      "cpu_observation_counts_sha256", "cuda_observation_counts_sha256",
                      "cpu_shape", "cuda_shape", "finite_value_count"):
            if left.get(field) != right.get(field):
                raise ValueError(f"opposite-order references disagree for {metric}: {field}")
    result = dict(first[0])
    result["verified_source_request_fingerprint"] = first[1]["cpu"]["source_request_fingerprint"]
    return result
