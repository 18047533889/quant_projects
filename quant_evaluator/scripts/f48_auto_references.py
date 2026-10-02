"""Strict opposite-order source A/B validator for the F48 auto envelope."""
from __future__ import annotations
import json
import math
import numbers
import re
from pathlib import Path


F48_SHAPE = (2586, 5461, 48)
F48_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
EXPECTED_ORDERS = {("cpu", "cuda_strict"), ("cuda_strict", "cpu")}
HASH_FIELDS = ("source_request_identity_sha256", "source_snapshot_id",
               "source_manifest_sha256", "factor_ids_sha256",
               "source_request_fingerprint")
SOURCE_FIELDS = ("source_adapter", "cos_prefetch", "prefetch_objects",
                 "prefetch_mode", "prefetch_window")
EXPECTED_SOURCE = {
    "source_adapter": "cos", "cos_prefetch": "auto",
    "prefetch_objects": True, "prefetch_mode": "auto", "prefetch_window": 2,
}
EXPECTED_RANGES = [[start, start + 2] for start in range(0, 48, 2)]


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same_source_settings(report):
    return (type(report) is dict
            and all(report.get(field) == expected
                    for field, expected in EXPECTED_SOURCE.items()))


EXPECTED_CALLER_SETTINGS = {
    "cos_prefetch_workers": 2,
    "max_prefetch_memory_mib": 512,
    "max_object_mib": 128,
    "max_total_mib": 4096,
}
MIN_AVAILABLE_RAM_BYTES = 32 * 1024**3
MIN_REQUIRED_DISK_BYTES = 5 * 1024**3


def _valid_preflight(receipt):
    if type(receipt) is not dict or receipt.get("pass") is not True:
        return False
    fields = ("available_ram_bytes", "minimum_available_ram_bytes",
              "cos_cache_disk_free_bytes", "required_disk_bytes")
    if any(type(receipt.get(field)) is not int for field in fields):
        return False
    available, minimum, disk_free, disk_required = (receipt[field] for field in fields)
    return (minimum >= MIN_AVAILABLE_RAM_BYTES and available >= minimum
            and disk_required >= MIN_REQUIRED_DISK_BYTES
            and disk_free >= disk_required)


def _valid_caller_settings(report):
    if "caller_settings" not in report:
        return True  # Legacy receipts did not emit these attestations.
    settings = report["caller_settings"]
    return (type(settings) is dict
            and all(type(settings.get(key)) is int and settings[key] == expected
                    for key, expected in EXPECTED_CALLER_SETTINGS.items()))


def validate_f48_auto_references(paths, selected_metrics, manifest_sha256):
    """Validate two stored F48 receipts with opposite order and CUDA faster in both.

    Historical receipts created before caller_settings was emitted cannot attest
    max-object/max-total limits, COS prefetch worker count, or prefetch memory.
    source_request_fingerprint is semantic request identity (axes, labels, factors,
    dtype, snapshot and metrics), not a caller-resource settings hash. Do not
    infer historical resource settings from it or retrofit missing receipt fields.
    Source adapter/mode/window and source-memory fields available in older receipts
    are still checked below; newer receipts additionally record caller_settings.
    """
    if (not isinstance(selected_metrics, (tuple, list))
            or tuple(selected_metrics) != F48_METRICS
            or not isinstance(paths, (tuple, list)) or len(paths) != 2):
        raise ValueError("F48 verification requires the exact mixed-three profile")
    if not _is_sha256(manifest_sha256):
        raise ValueError("F48 verification requires a valid manifest SHA-256")

    reports = []
    for path in paths:
        path = Path(path)
        try:
            with path.open("rb") as stream:
                payload = stream.read(1024**2 + 1)
        except OSError as exc:
            raise ValueError("reference report cannot be read") from exc
        if len(payload) > 1024**2:
            raise ValueError("reference report exceeds 1 MiB")
        try:
            report = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("reference report is not valid UTF-8 JSON") from exc
        if type(report) is not dict:
            raise ValueError("reference report root must be an object")
        comparison = report.get("comparison")
        if type(comparison) is not dict:
            raise ValueError("reference comparison must be an object")
        cpu_preflight = report.get("preflight_before_cpu")
        cuda_preflight = report.get("preflight_before_cuda")
        if type(cpu_preflight) is not dict or type(cuda_preflight) is not dict:
            raise ValueError("reference preflight receipts must be objects")
        shape = report.get("shape")
        metric_ids = report.get("metric_ids")
        run_order = report.get("run_order")
        if (not isinstance(shape, (list, tuple))
                or not isinstance(metric_ids, (list, tuple))
                or not isinstance(run_order, (list, tuple))):
            raise ValueError("reference shape, metrics, and run order must be arrays")
        if (not all(isinstance(name, str) for name in run_order)
                or report.get("status") != "complete"
                or report.get("kind") != "real_cos_whole_source_batch_ab.v1"
                or report.get("manifest_sha256") != manifest_sha256
                or tuple(shape) != F48_SHAPE
                or report.get("factor_dtype") != "float64"
                or report.get("tile_size") != 2
                or tuple(metric_ids) != F48_METRICS
                or comparison.get("pass") is not True
                or comparison.get("compared_factor_count") != 48
                or comparison.get("compared_metric_count") != 144
                or tuple(run_order) not in EXPECTED_ORDERS
                or not _same_source_settings(report)
                or not _valid_caller_settings(report)
                or not _valid_preflight(cpu_preflight)
                or not _valid_preflight(cuda_preflight)):
            raise ValueError("reference is not a complete exact F48 COS tile-2 A/B")

        run_list = report.get("runs")
        if type(run_list) is not list or len(run_list) != 2 or any(type(r) is not dict for r in run_list):
            raise ValueError("reference must contain exactly two backend runs")
        backend_names = [run.get("backend_requested") for run in run_list]
        if (not all(isinstance(name, str) for name in backend_names)
                or len(set(backend_names)) != 2):
            raise ValueError("reference backend names must be unique strings")
        runs = {run["backend_requested"]: run for run in run_list}
        if (set(runs) != {"cpu", "cuda_strict"}
                or runs["cpu"].get("backend_used") != "cpu"
                or runs["cuda_strict"].get("backend_used") != "cuda"):
            raise ValueError("reference does not contain the requested CPU/CUDA runs")
        for run in runs.values():
            seconds = run.get("seconds")
            if (any(run.get(field) != value for field, value in EXPECTED_SOURCE.items())
                    or run.get("source_manifest_sha256") != manifest_sha256
                    or run.get("factor_tiles_processed") != 24
                    or run.get("tile_ranges") != EXPECTED_RANGES
                    or run.get("max_source_memory_bytes") != 4 * 1024**3
                    or any(not _is_sha256(run.get(field)) for field in HASH_FIELDS)
                    or isinstance(seconds, bool) or not isinstance(seconds, numbers.Real)
                    or not math.isfinite(float(seconds)) or seconds <= 0):
                raise ValueError("reference run lacks exact source identity or valid timing")
        if any(runs["cpu"][field] != runs["cuda_strict"][field]
               for field in HASH_FIELDS):
            raise ValueError("CPU and CUDA reference requests have different identities")
        cuda_vram = runs["cuda_strict"].get("vram_budget_bytes")
        if (isinstance(cuda_vram, bool) or not isinstance(cuda_vram, numbers.Real)
                or not math.isfinite(float(cuda_vram)) or cuda_vram < 14 * 1024**3):
            raise ValueError("reference CUDA run lacks the 14 GiB effective VRAM budget")
        if not runs["cuda_strict"]["seconds"] < runs["cpu"]["seconds"]:
            raise ValueError("strict CUDA must be faster than CPU in each reference run")

        metrics = comparison.get("metrics")
        if type(metrics) is not dict:
            raise ValueError("reference metric comparisons must be an object")
        if set(metrics) != set(F48_METRICS):
            raise ValueError("reference metric comparison set is incomplete")
        for metric in F48_METRICS:
            entry = metrics[metric]
            if type(entry) is not dict:
                raise ValueError(f"reference metric receipt must be an object: {metric}")
            error = entry.get("max_abs_error")
            if (entry.get("pass") is not True or entry.get("artifact_kind") != "scalar"
                    or entry.get("cpu_shape") != [48] or entry.get("cuda_shape") != [48]
                    or entry.get("compared_value_count") != 48
                    or entry.get("finite_value_count") != 48
                    or entry.get("finite_mask_equal") is not True
                    or entry.get("observation_counts_equal") is not True
                    or entry.get("observation_counts_shape_valid") is not True
                    or isinstance(error, bool) or not isinstance(error, numbers.Real)
                    or not math.isfinite(float(error)) or error < 0 or error > 1e-10
                    or any(not _is_sha256(entry.get(field)) for field in (
                        "cpu_values_sha256", "cuda_values_sha256",
                        "cpu_observation_counts_sha256",
                        "cuda_observation_counts_sha256"))
                    or entry.get("cpu_observation_counts_sha256")
                    != entry.get("cuda_observation_counts_sha256")):
                raise ValueError(f"reference metric receipt is incomplete: {metric}")
        reports.append((report, runs))

    first, second = reports
    if first[0]["run_order"] == second[0]["run_order"]:
        raise ValueError("F48 references must have opposite run orders")
    for field in ("manifest_sha256", "shape", "factor_dtype", "tile_size", "metric_ids",
                  *SOURCE_FIELDS):
        if first[0].get(field) != second[0].get(field):
            raise ValueError(f"opposite-order references disagree for {field}")
    for field in HASH_FIELDS:
        if first[1]["cpu"][field] != second[1]["cpu"][field]:
            raise ValueError("opposite-order references use different source requests")
    for metric in F48_METRICS:
        left = first[0]["comparison"]["metrics"][metric]
        right = second[0]["comparison"]["metrics"][metric]
        for field in ("cpu_values_sha256", "cuda_values_sha256",
                      "cpu_observation_counts_sha256", "cuda_observation_counts_sha256",
                      "cpu_shape", "cuda_shape", "finite_value_count"):
            if left.get(field) != right.get(field):
                raise ValueError(f"opposite-order references disagree for {metric}: {field}")
    result = dict(first[0])
    result["verified_source_request_fingerprint"] = first[1]["cpu"][
        "source_request_fingerprint"]
    return result
