"""Strict validator for the two historical F48 Pearson-chain width-4 receipts."""
from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
import re

SHAPE = (2586, 5461, 48)
METRICS = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
ORDERS = {("cpu", "cuda_strict"), ("cuda_strict", "cpu")}
HASHES = ("source_request_identity_sha256", "source_snapshot_id",
          "source_manifest_sha256", "factor_ids_sha256", "source_request_fingerprint")
ARTIFACTS = (
    "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cpu_first_20261003.json",
    "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cuda_first_20261003.json",
)

EXPECTED_PROVENANCE = "b13bb54505d6da85b29726dd4736e4da71369a4f2074774282f9e0388a97046f"
MIN_RAM = 32 * 1024**3
MIN_DISK = 5 * 1024**3


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_report(path):
    try:
        with Path(path).open("rb") as stream:
            payload = stream.read(1024**2 + 1)
        if len(payload) > 1024**2:
            raise ValueError("reference report exceeds 1 MiB")
        report = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("reference report is unreadable or invalid JSON") from exc
    if type(report) is not dict:
        raise ValueError("reference report root must be an object")
    return report


def _valid_preflight(item):
    fields = ("available_ram_bytes", "minimum_available_ram_bytes",
              "cos_cache_disk_free_bytes", "required_disk_bytes")
    if type(item) is not dict or item.get("pass") is not True:
        return False
    if any(type(item.get(key)) is not int for key in fields):
        return False
    ram, min_ram, disk, min_disk = (item[key] for key in fields)
    return min_ram >= MIN_RAM and ram >= min_ram and min_disk >= MIN_DISK and disk >= min_disk


def _validate_references(paths):
    if not isinstance(paths, (list, tuple)) or len(paths) != 2:
        raise ValueError("exactly two opposite-order references are required")
    reports = []
    for path in paths:
        report = _read_report(path)
        comparison = report.get("comparison")
        if type(comparison) is not dict:
            raise ValueError("reference comparison must be an object")
        if (report.get("status") != "complete"
                or report.get("kind") != "real_cos_whole_source_batch_ab.v1"
                or tuple(report.get("shape", ())) != SHAPE
                or tuple(report.get("metric_ids", ())) != METRICS
                or report.get("factor_dtype") != "float64"
                or report.get("tile_size") != 4
                or report.get("source_adapter") != "cos"
                or report.get("cos_prefetch") != "auto"
                or report.get("prefetch_objects") is not True
                or report.get("prefetch_mode") != "auto"
                or report.get("prefetch_window") != 2
                or tuple(report.get("run_order", ())) not in ORDERS
                or comparison.get("pass") is not True
                or comparison.get("compared_factor_count") != 48
                or comparison.get("compared_metric_count") != 124272
                or not _sha256(report.get("manifest_sha256"))):
            raise ValueError("reference is not a complete exact F48 Pearson width-4 A/B")
        raw_runs = report.get("runs")
        if type(raw_runs) is not list or len(raw_runs) != 2 or any(type(r) is not dict for r in raw_runs):
            raise ValueError("reference must contain exactly two run receipts")
        runs = {r.get("backend_requested"): r for r in raw_runs}
        if set(runs) != {"cpu", "cuda_strict"}:
            raise ValueError("reference must contain CPU and strict CUDA runs")
        for name, run in runs.items():
            if (run.get("backend_used") != ("cpu" if name == "cpu" else "cuda")
                    or run.get("source_adapter") != "cos"
                    or run.get("cos_prefetch") != "auto"
                    or run.get("prefetch_objects") is not True
                    or run.get("prefetch_mode") != "auto"
                    or run.get("prefetch_window") != 2
                    or run.get("max_source_memory_bytes") != 4 * 1024**3
                    or run.get("source_manifest_sha256") != report.get("manifest_sha256")
                    or run.get("factor_tiles_processed") != 12
                    or run.get("tile_ranges") != [[i, i + 4] for i in range(0, 48, 4)]):
                raise ValueError("reference run source settings or tile coverage are invalid")
            seconds = run.get("seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, numbers.Real)
                    or not math.isfinite(float(seconds)) or seconds <= 0):
                raise ValueError("reference run timing is invalid")
            if any(not _sha256(run.get(key)) for key in HASHES):
                raise ValueError("reference source identity hashes are invalid")
        cuda_vram = runs["cuda_strict"].get("vram_budget_bytes")
        if (isinstance(cuda_vram, bool) or not isinstance(cuda_vram, numbers.Real)
                or not math.isfinite(float(cuda_vram)) or cuda_vram < 14 * 1024**3
                or runs["cuda_strict"].get("oom_retries") != 0):
            raise ValueError("strict CUDA VRAM/OOM receipt is invalid")
        for key in HASHES:
            if runs["cpu"].get(key) != runs["cuda_strict"].get(key):
                raise ValueError("CPU/CUDA source identities differ")
        if not runs["cuda_strict"]["seconds"] < runs["cpu"]["seconds"]:
            raise ValueError("strict CUDA was not faster than CPU")
        settings = report.get("caller_settings")
        if (type(settings) is not dict or any(type(settings.get(k)) is not int or settings[k] != v
                for k, v in (("cos_prefetch_workers", 2), ("max_prefetch_memory_mib", 512),
                             ("max_object_mib", 128), ("max_total_mib", 4096)))):
            raise ValueError("reference caller resource settings are invalid")
        for key in ("preflight_before_cpu", "preflight_before_cuda"):
            if not _valid_preflight(report.get(key)):
                raise ValueError("reference preflight receipt is invalid")
        provenance = report.get("source_provenance")
        verification = report.get("source_provenance_verification")
        if (type(provenance) is not dict or type(verification) is not dict
                or verification.get("pass") is not True
                or verification.get("status") != "unchanged"
                or not _sha256(provenance.get("aggregate_sha256"))
                or provenance.get("aggregate_sha256") != EXPECTED_PROVENANCE
                or verification.get("after_aggregate_sha256") != provenance.get("aggregate_sha256")):
            raise ValueError("reference current-source provenance verification is invalid")
        comparison = report.get("comparison")
        if type(comparison.get("metrics")) is not dict:
            raise ValueError("reference metric comparison is invalid")
        if set(comparison["metrics"]) != set(METRICS):
            raise ValueError("reference metric set is incomplete")
        for metric in METRICS:
            item = comparison["metrics"][metric]
            series = metric == "pearson_ic_series"
            expected_shape = [2586, 48] if series else [48]
            expected_compared = 2586 * 48 if series else 48
            expected_finite = 123529 if series else 48
            error = item.get("max_abs_error")
            if (type(item) is not dict or item.get("pass") is not True
                    or item.get("artifact_kind") != ("series" if series else "scalar")
                    or item.get("cpu_shape") != expected_shape or item.get("cuda_shape") != expected_shape
                    or item.get("compared_value_count") != expected_compared
                    or item.get("finite_value_count") != expected_finite
                    or item.get("finite_mask_equal") is not True
                    or item.get("observation_counts_equal") is not True
                    or item.get("observation_counts_shape_valid") is not True
                    or isinstance(error, bool) or not isinstance(error, numbers.Real)
                    or not math.isfinite(float(error)) or error < 0 or error > 1e-10
                    or any(not _sha256(item.get(k)) for k in (
                        "cpu_values_sha256", "cuda_values_sha256",
                        "cpu_observation_counts_sha256", "cuda_observation_counts_sha256"))
                    or item.get("cpu_observation_counts_sha256") != item.get("cuda_observation_counts_sha256")):
                raise ValueError("reference metric receipt is incomplete: " + metric)
        reports.append((report, runs))
    left, right = reports
    if left[0]["run_order"] == right[0]["run_order"]:
        raise ValueError("references must use opposite run orders")
    for key in ("manifest_sha256", "shape", "metric_ids", "factor_dtype", "tile_size"):
        if left[0].get(key) != right[0].get(key):
            raise ValueError("references disagree on " + key)
    for key in HASHES:
        if left[1]["cpu"].get(key) != right[1]["cpu"].get(key):
            raise ValueError("references use different source requests")
    for metric in METRICS:
        a = left[0]["comparison"]["metrics"][metric]
        b = right[0]["comparison"]["metrics"][metric]
        for key in ("cpu_values_sha256", "cuda_values_sha256",
                    "cpu_observation_counts_sha256", "cuda_observation_counts_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError("opposite-order metric receipts disagree: " + metric)
    result = dict(left[0])
    result["verified_source_request_fingerprint"] = left[1]["cpu"]["source_request_fingerprint"]
    result["historical_source_aggregate_sha256"] = left[0]["source_provenance"]["aggregate_sha256"]
    return result


def validate_references(paths):
    """Return a normalized reference or raise ValueError for every invalid input."""
    try:
        return _validate_references(paths)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("reference reports are malformed or incomplete") from exc
