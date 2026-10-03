"""Current-source verifier for counterbalanced F48 Pearson strict A/B receipts.

This module deliberately does not alter the historical reference validator or
source-auto routing. The caller must bind the pair to independently captured
current source and manifest digests.
"""
from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
import re

from quant_evaluator.scripts.source_tree_provenance import (
    SOURCE_DIRECTORIES,
    SOURCE_FILES,
)

SHAPE = (2586, 5461, 48)
METRICS = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
ORDERS = {("cpu", "cuda_strict"), ("cuda_strict", "cpu")}
EXPECTED_SCOPE = {
    "directories": list(SOURCE_DIRECTORIES),
    "files": list(SOURCE_FILES),
}
HASH_FIELDS = (
    "source_request_identity_sha256",
    "source_snapshot_id",
    "source_manifest_sha256",
    "factor_ids_sha256",
    "source_request_fingerprint",
)
MIN_RAM_BYTES = 32 * 1024**3
MIN_DISK_BYTES = 5 * 1024**3
MIN_VRAM_BYTES = 14 * 1024**3
MAX_REPORT_BYTES = 1024**2
EXPECTED_TILE_RANGES = [[start, start + 4] for start in range(0, 48, 4)]


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_report(path):
    try:
        with Path(path).open("rb") as stream:
            payload = stream.read(MAX_REPORT_BYTES + 1)
        if len(payload) > MAX_REPORT_BYTES:
            raise ValueError("report exceeds 1 MiB")
        report = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("current-source A/B report is unreadable or invalid JSON") from exc
    if type(report) is not dict:
        raise ValueError("current-source A/B report root must be an object")
    return report


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _valid_preflight(item) -> bool:
    fields = ("available_ram_bytes", "minimum_available_ram_bytes",
              "cos_cache_disk_free_bytes", "required_disk_bytes")
    if type(item) is not dict or item.get("pass") is not True:
        return False
    if any(type(item.get(key)) is not int for key in fields):
        return False
    ram, minimum, disk, required = (item[key] for key in fields)
    return (minimum >= MIN_RAM_BYTES and ram >= minimum
            and required >= MIN_DISK_BYTES and disk >= required)


def _finite_positive(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(converted) and converted > 0


def _finite_nonnegative(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(converted) and converted >= 0


def _valid_tile_ranges(value) -> bool:
    if type(value) is not list or len(value) != len(EXPECTED_TILE_RANGES):
        return False
    for actual, expected in zip(value, EXPECTED_TILE_RANGES):
        if (type(actual) is not list or len(actual) != 2
                or any(type(boundary) is not int for boundary in actual)
                or actual != expected):
            return False
    return True


def _validate_provenance(report, expected_source_sha256):
    provenance = report.get("source_provenance")
    verification = report.get("source_provenance_verification")
    if type(provenance) is not dict or type(verification) is not dict:
        raise ValueError("current source provenance receipt is missing")
    if (provenance.get("algorithm") != "sha256_path_length_prefixed_v1"
            or provenance.get("aggregate_sha256") != expected_source_sha256
            or provenance.get("scope") != EXPECTED_SCOPE
            or not _positive_int(provenance.get("file_count"))):
        raise ValueError("source hash or declared harness scope differs from caller-bound scope")
    if (verification.get("pass") is not True
            or verification.get("status") != "unchanged"
            or verification.get("after_aggregate_sha256") != expected_source_sha256):
        raise ValueError("source provenance changed during the A/B run")
    versions = provenance.get("versions")
    if (type(versions) is not dict
            or any(not isinstance(versions.get(key), str) or not versions[key]
                   for key in ("python", "numpy", "pandas"))):
        raise ValueError("source provenance runtime package versions are incomplete")


def _validate_run(run, backend, expected_manifest_sha256):
    is_cuda = backend == "cuda_strict"
    expected_backend_used = "cuda" if is_cuda else "cpu"
    if (type(run) is not dict
            or run.get("backend_requested") != backend
            or run.get("backend_used") != expected_backend_used
            or run.get("source_adapter") != "cos"
            or run.get("effective_max_tile_size") != 4
            or type(run.get("effective_max_tile_size")) is not int
            or run.get("declared_source_tile_size") != 4
            or type(run.get("declared_source_tile_size")) is not int
            or run.get("admitted_source_tile_size") != 4
            or type(run.get("admitted_source_tile_size")) is not int
            or run.get("factor_tiles_processed") != 12
            or type(run.get("factor_tiles_processed")) is not int
            or not _valid_tile_ranges(run.get("tile_ranges"))
            or run.get("source_manifest_sha256") != expected_manifest_sha256
            or run.get("cos_prefetch") != "auto"
            or run.get("max_source_memory_bytes") != 4 * 1024**3
            or type(run.get("max_source_memory_bytes")) is not int
            or run.get("prefetch_objects") is not True
            or run.get("prefetch_mode") != "auto"
            or run.get("prefetch_window") != 2
            or not _finite_positive(run.get("seconds"))
            or not _valid_preflight(run.get("preflight"))):
        raise ValueError("run backend, tile coverage, source settings, timing, or preflight is invalid")
    for key in HASH_FIELDS:
        if not _sha256(run.get(key)):
            raise ValueError("run source identity hash is invalid: " + key)
    if is_cuda:
        if (type(run.get("oom_retries")) is not int or run["oom_retries"] != 0
                or not _finite_positive(run.get("vram_budget_bytes"))
                or float(run["vram_budget_bytes"]) < MIN_VRAM_BYTES
                or not _positive_int(run.get("peak_vram_bytes"))
                or run["peak_vram_bytes"] > run["vram_budget_bytes"]
                or not _positive_int(run.get("pool_reserved_peak_bytes"))
                or not _finite_nonnegative(run.get("h2d_bytes"))
                or not _finite_nonnegative(run.get("d2h_bytes"))):
            raise ValueError("strict CUDA device admission, VRAM, or OOM receipt is invalid")
    elif run.get("oom_retries") is not None:
        raise ValueError("CPU run must not claim CUDA OOM retries")


def _validate_comparison(report):
    comparison = report.get("comparison")
    if (type(comparison) is not dict or comparison.get("pass") is not True
            or comparison.get("compared_factor_count") != 48
            or type(comparison.get("compared_factor_count")) is not int
            or comparison.get("compared_metric_count") != 124272
            or type(comparison.get("compared_metric_count")) is not int
            or type(comparison.get("metrics")) is not dict
            or set(comparison["metrics"]) != set(METRICS)):
        raise ValueError("metric comparison receipt is incomplete")
    for metric in METRICS:
        item = comparison["metrics"][metric]
        series = metric == "pearson_ic_series"
        expected_shape = [2586, 48] if series else [48]
        expected_values = 2586 * 48 if series else 48
        expected_finite = 123529 if series else 48
        if (type(item) is not dict or item.get("pass") is not True
                or item.get("artifact_kind") != ("series" if series else "scalar")
                or item.get("cpu_shape") != expected_shape
                or item.get("cuda_shape") != expected_shape
                or item.get("shape_valid") is not True
                or item.get("factor_count") != 48
                or type(item.get("factor_count")) is not int
                or item.get("compared_value_count") != expected_values
                or type(item.get("compared_value_count")) is not int
                or item.get("finite_value_count") != expected_finite
                or type(item.get("finite_value_count")) is not int
                or item.get("finite_mask_equal") is not True
                or item.get("observation_counts_equal") is not True
                or item.get("observation_counts_shape_valid") is not True):
            raise ValueError("metric shape, finite-mask, or observation-count comparison is invalid: " + metric)
        error = item.get("max_abs_error")
        if not _finite_nonnegative(error) or float(error) > 1e-10:
            raise ValueError("metric error bound is invalid: " + metric)
        for key in ("cpu_values_sha256", "cuda_values_sha256",
                    "cpu_observation_counts_sha256", "cuda_observation_counts_sha256"):
            if not _sha256(item.get(key)):
                raise ValueError("metric result hash is invalid: " + metric + "." + key)
        if item["cpu_observation_counts_sha256"] != item["cuda_observation_counts_sha256"]:
            raise ValueError("CPU/CUDA observation-count hashes differ: " + metric)


def _report_hashes(report):
    return {
        metric: {
            key: report["comparison"]["metrics"][metric][key]
            for key in ("cpu_values_sha256", "cuda_values_sha256",
                        "cpu_observation_counts_sha256", "cuda_observation_counts_sha256")
        }
        for metric in METRICS
    }


def validate_current_f48_pearson_ab(
        paths, *, expected_source_sha256, expected_manifest_sha256):
    """Validate two caller-bound, current-source, opposite-order F48 A/B reports.

    This certifies only the receipts and the declared Python source scope. It
    treats run_order as execution order; the harness serializes the run receipts
    in fixed CPU-then-CUDA order. It does not claim thread configuration,
    runtime-closure identity, or routing suitability. Raises ValueError for
    malformed, stale, or unbalanced inputs.
    """
    if (not isinstance(paths, (list, tuple)) or len(paths) != 2
            or not _sha256(expected_source_sha256)
            or not _sha256(expected_manifest_sha256)):
        raise ValueError("two reports and explicit SHA-256 source/manifest identities are required")
    reports = [_read_report(path) for path in paths]
    runs_by_report = []
    hashes_by_report = []
    identities_by_report = []
    runtime_versions_by_report = []

    for report in reports:
        if (report.get("status") != "complete"
                or report.get("kind") != "real_cos_whole_source_batch_ab.v1"
                or report.get("manifest_sha256") != expected_manifest_sha256
                or report.get("shape") != list(SHAPE)
                or report.get("factor_dtype") != "float64"
                or report.get("tile_size") != 4
                or type(report.get("tile_size")) is not int
                or report.get("source_adapter") != "cos"
                or report.get("cos_prefetch") != "auto"
                or report.get("prefetch_objects") is not True
                or report.get("prefetch_mode") != "auto"
                or report.get("prefetch_window") != 2
                or report.get("metric_ids") != list(METRICS)):
            raise ValueError("report is not the exact F48 float64 Pearson-chain width-4 COS profile")
        settings = report.get("caller_settings")
        if (type(settings) is not dict
                or any(type(settings.get(key)) is not int or settings[key] != value
                       for key, value in (
                           ("cos_prefetch_workers", 2),
                           ("max_prefetch_memory_mib", 512),
                           ("max_object_mib", 128),
                           ("max_total_mib", 4096)))):
            raise ValueError("caller resource settings are outside the qualified source envelope")
        order = report.get("run_order")
        if type(order) is not list or tuple(order) not in ORDERS:
            raise ValueError("report run order is not a recognized CPU/CUDA ordering")
        for key in ("preflight_before_cpu", "preflight_before_cuda"):
            if not _valid_preflight(report.get(key)):
                raise ValueError("report preflight is missing or below the RAM/disk gates")
        _validate_provenance(report, expected_source_sha256)
        runtime_versions_by_report.append(report["source_provenance"]["versions"])
        _validate_comparison(report)
        raw_runs = report.get("runs")
        if type(raw_runs) is not list or len(raw_runs) != 2:
            raise ValueError("report must include exactly two ordered backend runs")
        if any(type(run) is not dict for run in raw_runs):
            raise ValueError("run receipt must be an object")
        # run_order is execution chronology, not serialization order.
        runs = {run["backend_requested"]: run for run in raw_runs}
        if set(runs) != {"cpu", "cuda_strict"}:
            raise ValueError("report must contain CPU and strict CUDA runs")
        for backend in ("cpu", "cuda_strict"):
            _validate_run(runs[backend], backend, expected_manifest_sha256)
        cpu, cuda = runs["cpu"], runs["cuda_strict"]
        for key in HASH_FIELDS:
            if cpu.get(key) != cuda.get(key):
                raise ValueError("CPU/CUDA run source identities differ: " + key)
        if not float(cuda["seconds"]) < float(cpu["seconds"]):
            raise ValueError("strict CUDA was not faster in this run order")
        hashes_by_report.append(_report_hashes(report))
        identities_by_report.append({key: cpu[key] for key in HASH_FIELDS})

    if tuple(reports[0]["run_order"]) == tuple(reports[1]["run_order"]):
        raise ValueError("reports must use opposite CPU/CUDA run orders")
    for key in ("python", "numpy", "pandas"):
        if runtime_versions_by_report[0][key] != runtime_versions_by_report[1][key]:
            raise ValueError("opposite-order reports use different runtime package versions")
    for key in HASH_FIELDS:
        if identities_by_report[0][key] != identities_by_report[1][key]:
            raise ValueError("reports do not describe the same source request: " + key)
    if hashes_by_report[0] != hashes_by_report[1]:
        raise ValueError("opposite-order reports disagree on output or observation-count hashes")

    return {
        "pass": True,
        "status": "current_source_f48_pearson_ab_qualified",
        "source_sha256": expected_source_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "shape": list(SHAPE),
        "metric_ids": list(METRICS),
        "run_orders": [list(report["run_order"]) for report in reports],
        "thread_settings": "not_recorded",
        "limitations": [
            "Reports do not record the CPU thread environment or BLAS thread-pool configuration.",
            "Declared source provenance is not a complete runtime closure and does not attest loaded code or external dependencies.",
            "A/B evidence is bounded to this source request and does not certify PIT or production suitability.",
        ],
    }


__all__ = ("validate_current_f48_pearson_ab",)
