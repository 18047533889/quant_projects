"""Bounded, manifest-bound COS method audit on an automatic TRAIN partition.

This runs the fixed exploratory method catalogue against real factor values.
It does not fit/select an optimizer or score VALIDATION/TEST outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from factor_optimizer.research_batch import BatchOptimizationConfig, automatic_time_split

ROOT = Path("/home/sunhaiwei/quant_projects")
EXAMPLES = ROOT / "factor_optimizer/examples"
CACHE_ROOT = Path("/home/sunhaiwei/.cache/quant-dataaccess/research")
FACTOR_POOL = "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values"
MANIFEST_PREFIX = "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/"
FACTOR_COUNT, ASSET_COUNT, DAY_COUNT = 1, 256, 500
MAX_FACTOR_BYTES = 8 * 1024**2
MAX_BATCH_BYTES = 128 * 1024**2
DECLARED_ARRAY_BUDGET_BYTES = 512 * 1024**2
MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
MIN_DISK_HEADROOM_BYTES = 256 * 1024**2
MAX_REPORT_BYTES = 1 * 1024**2
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def check_environment(environ=None):
    """Require configured server-c DataAccess paths before any remote read."""
    env = os.environ if environ is None else environ
    expected = {
        "ASHARE_PARQUET_ROOT": "/home/sunhaiwei/cos_data",
        "DATA_ACCESS_COS_CLI": "admin-cos",
        "DATA_ACCESS_COS_CACHE_ROOT": str(CACHE_ROOT),
    }
    wrong = [key for key, value in expected.items() if env.get(key) != value]
    if wrong:
        raise RuntimeError("server-c DataAccess settings missing or mismatched: " + ", ".join(wrong))


def available_memory_bytes():
    """Return MemAvailable, constrained by any visible cgroup memory limit."""
    try:
        available = None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
        if available is None:
            return None
        for path in (Path("/sys/fs/cgroup/memory.max"),
                     Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
            try:
                limit_text = path.read_text().strip()
            except OSError:
                continue
            if limit_text.isdigit():
                limit = int(limit_text)
                if limit < (1 << 60):
                    current = 0
                    current_path = (Path("/sys/fs/cgroup/memory.current")
                                    if path.name == "memory.max"
                                    else Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
                    try:
                        current = int(current_path.read_text().strip())
                    except (OSError, ValueError):
                        return None
                    available = min(available, max(0, limit - current))
        return available
    except (OSError, ValueError, IndexError):
        return None


def check_resource_headroom(*, memory_bytes=None, cache_root=CACHE_ROOT,
                            minimum_memory_bytes=MIN_AVAILABLE_MEMORY_BYTES,
                            minimum_disk_bytes=MIN_DISK_HEADROOM_BYTES):
    """Admission check before COS reads; leave room for the bounded conversion."""
    if (type(minimum_memory_bytes) is not int
            or minimum_memory_bytes < MIN_AVAILABLE_MEMORY_BYTES
            or type(minimum_disk_bytes) is not int
            or minimum_disk_bytes < MIN_DISK_HEADROOM_BYTES):
        raise ValueError("resource thresholds must be integer limits at least the default floors")
    available = available_memory_bytes() if memory_bytes is None else memory_bytes
    if available is None:
        raise RuntimeError("available RAM could not be determined before COS reads")
    if type(available) is not int or available < 0:
        raise ValueError("available memory must be a nonnegative built-in integer byte count")
    if available < minimum_memory_bytes:
        raise RuntimeError(f"at least {minimum_memory_bytes // 1024**3} GiB available RAM is required before COS reads")
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise RuntimeError("configured DataAccess COS cache root is not an existing directory")
    if shutil.disk_usage(cache_root).free < minimum_disk_bytes:
        raise RuntimeError(f"less than {minimum_disk_bytes // 1024**2} MiB free in the DataAccess COS cache filesystem")


def load_real_source():
    """Load exact manifest-bound factors through the existing DataAccess loader."""
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    from cos_batch_audit import load_cos_sample
    return load_cos_sample(n_factors=FACTOR_COUNT, n_assets=ASSET_COUNT,
        include_lineages=True, max_factor_bytes=MAX_FACTOR_BYTES,
        coverage_policy="isolate")


def _is_int(value):
    return type(value) is int


def validate_source(batch, labels, provenance, *, max_factor_bytes=MAX_FACTOR_BYTES,
                    max_batch_factor_bytes=MAX_BATCH_BYTES):
    """Reject source metadata, dimensions, or payloads beyond the admitted envelope."""
    if (type(max_factor_bytes) is not int or not 0 < max_factor_bytes <= 128*1024**2
            or type(max_batch_factor_bytes) is not int
            or not 0 < max_batch_factor_bytes <= 2*1024**3):
        raise ValueError("source budgets must be positive integer limits within DataAccess caps")
    if not isinstance(provenance, dict):
        raise ValueError("source provenance must be a mapping")
    manifest_uri = provenance.get("manifest_uri")
    if (not isinstance(manifest_uri, str)
            or not re.fullmatch(re.escape(MANIFEST_PREFIX) + r"[0-9a-f]{64}/landing_manifest\.json", manifest_uri)):
        raise ValueError("source manifest URI is outside the authorized content-addressed metadata pool")
    if not isinstance(batch.factor_ids, tuple) or len(batch.factor_ids) != FACTOR_COUNT:
        raise ValueError("loaded factor count differs from the one-factor audit cap")
    if tuple(provenance.get("retained_factor_ids", ())) != tuple(batch.factor_ids):
        raise ValueError("retained factor IDs differ from loaded factor IDs")
    sources = provenance.get("sources")
    if not isinstance(sources, list) or len(sources) != FACTOR_COUNT:
        raise ValueError("expected exactly one manifest-bound factor source")
    if sources[0].get("factor") != batch.factor_ids[0]:
        raise ValueError("manifest source factor ID differs from the loaded batch")
    source = sources[0]
    uri, sha, size = source.get("uri"), source.get("sha256"), source.get("downloaded_bytes")
    if (not isinstance(sha, str) or _HEX64.fullmatch(sha) is None
            or uri != f"{FACTOR_POOL}/{sha}/{batch.factor_ids[0]}.parquet"):
        raise ValueError("factor source URI and content SHA256 do not bind to the authorized pool")
    if not _is_int(size) or not 0 < size <= max_factor_bytes:
        raise ValueError(f"downloaded factor size exceeds the {max_factor_bytes // 1024**2} MiB object cap")
    if size > max_batch_factor_bytes:
        raise ValueError("downloaded factor size exceeds the batch-object cap")
    if source.get("bytes", size) != size:
        raise ValueError("downloaded factor byte count differs from the manifest")
    if source.get("manifest_sha256") is None or _HEX64.fullmatch(str(source["manifest_sha256"])) is None:
        raise ValueError("source manifest SHA256 is missing or invalid")
    if source.get("source_status") not in {"materialized_not_evaluated", "evaluated_optimization_pending"}:
        raise ValueError("source landing status is blocked or unsupported")
    if not isinstance(source.get("expression"), str) or not source["expression"]:
        raise ValueError("source expression is missing")
    if not isinstance(source.get("etag"), str) or not source["etag"]:
        raise ValueError("source ETag is missing")
    shape = tuple(getattr(batch.values, "shape", ()))
    if shape != (DAY_COUNT, ASSET_COUNT, FACTOR_COUNT):
        raise ValueError(f"factor batch shape {shape!r} exceeds or differs from (500, 256, 1)")
    if getattr(batch, "time_axis", None) is None:
        raise ValueError("factor batch time axis is required")
    if not np.array_equal(np.asarray(batch.time_axis.values), np.asarray(labels.decision_time)):
        raise ValueError("factor batch time axis does not exactly match label decision_time")
    if getattr(batch, "asset_axis", None) is None or getattr(labels, "asset_axis", None) is None:
        raise ValueError("factor and label asset axes are required")
    if not np.array_equal(np.asarray(batch.asset_axis.values), np.asarray(labels.asset_axis.values)):
        raise ValueError("factor batch asset axis does not exactly match label asset_axis")
    if getattr(batch.values, "dtype", None) is None or batch.values.dtype.kind not in "fi":
        raise ValueError("factor values must be a numeric array")
    if not batch.values.flags.c_contiguous:
        raise ValueError("factor values must be contiguous for bounded fingerprinting")
    if batch.values.nbytes > DECLARED_ARRAY_BUDGET_BYTES:
        raise ValueError("factor batch array exceeds the 512 MiB audit budget")
    if tuple(getattr(labels.values, "shape", ())) != (DAY_COUNT, ASSET_COUNT):
        raise ValueError("label matrix must align to the bounded 500 by 256 factor panel")
    if labels.values.dtype.kind not in "fi" or not labels.values.flags.c_contiguous:
        raise ValueError("label values must be contiguous numeric data")
    for name, mask, expected in (
        ("factor validity", getattr(batch, "validity", None), shape),
        ("label validity", getattr(labels, "validity", None), (DAY_COUNT, ASSET_COUNT)),
    ):
        if mask is not None and (tuple(getattr(mask, "shape", ())) != expected
                or np.asarray(mask).dtype.kind != "b" or not np.asarray(mask).flags.c_contiguous):
            raise ValueError(f"{name} does not match its admitted panel")
    split = automatic_time_split(labels)
    if provenance.get("asset_selection_split") != split.identity:
        raise ValueError("asset selection split identity differs from the automatic purged TRAIN split")
    if provenance.get("asset_selection_train_days") != len(split.train_indices):
        raise ValueError("asset selection TRAIN-day count differs from the automatic purged TRAIN split")
    if provenance.get("max_factor_bytes") != max_factor_bytes:
        raise ValueError("loader factor-object cap differs from the requested bound")
    if provenance.get("max_batch_factor_bytes") != max_batch_factor_bytes:
        raise ValueError("loader batch-object cap differs from the requested bound")


def _hash_array(hasher, name, value):
    if value is None:
        hasher.update((name + ":none;").encode())
        return
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiub" or not arr.flags.c_contiguous:
        raise ValueError(f"{name} cannot be fingerprinted within the admitted representation")
    hasher.update(name.encode())
    hasher.update(str(arr.dtype).encode())
    hasher.update(json.dumps(list(arr.shape), separators=(",", ":")).encode())
    hasher.update(memoryview(arr).cast("B"))


def source_fingerprint(batch, labels, provenance):
    """Hash bounded in-memory source arrays and their already-bound provenance."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(list(batch.factor_ids), separators=(",", ":")).encode())
    for axis_name, axis in (("time", batch.time_axis.values), ("asset", batch.asset_axis.values)):
        hasher.update(axis_name.encode())
        hasher.update(json.dumps([str(value) for value in axis], separators=(",", ":")).encode())
    _hash_array(hasher, "factor_values", batch.values)
    _hash_array(hasher, "factor_validity", getattr(batch, "validity", None))
    _hash_array(hasher, "label_values", labels.values)
    _hash_array(hasher, "label_validity", getattr(labels, "validity", None))
    if getattr(labels, "asset_axis", None) is None:
        hasher.update(b"label_asset_axis:none;")
    else:
        hasher.update(json.dumps([str(value) for value in labels.asset_axis.values],
                                 separators=(",", ":")).encode())
    for field in ("decision_time", "observation_time", "signal_available_time",
                  "execution_time", "label_start_time", "label_end_time",
                  "calendar_ref", "source_ref", "target_id", "horizon",
                  "execution_delay"):
        value = getattr(labels, field, None)
        normalized = [str(item) for item in value] if isinstance(value, (tuple, list)) else value
        hasher.update(field.encode())
        hasher.update(json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode())
    hasher.update(json.dumps(provenance, sort_keys=True, separators=(",", ":"), default=str).encode())
    return hasher.hexdigest()


def build_report(batch, labels, provenance, methods, *, config=None, fingerprints=None):
    config = config or BatchOptimizationConfig()
    split = automatic_time_split(labels, config)
    report = {
        "audit": "real-factor-method-training-audit-v1",
        "source": provenance,
        "source_fingerprints": fingerprints or {},
        "factor_ids": list(batch.factor_ids),
        "batch_shape": list(batch.values.shape),
        "method_audit_partition": "TRAIN only",
        "method_semantics": "fixed prespecified exploratory catalogue; no fitted optimizer, winner selection, or holdout metric",
        "finite_fraction_scope": "per-method transform over the pre-validation prefix including warmup and purged rows; IC and costed scores use purged TRAIN indices only",
        "split": {"identity": split.identity, "train_days": len(split.train_indices),
                  "validation_days_reserved": len(split.validation_indices),
                  "test_days_reserved": len(split.test_indices)},
        "test_evaluated": False,
        "method_status_counts": dict(sorted(Counter(
            row.get("status", "missing_status") for row in methods).items())),
        "methods": methods,
        "limits": {"factor_count": FACTOR_COUNT, "asset_count": ASSET_COUNT,
                   "date_count": DAY_COUNT, "max_factor_object_bytes": MAX_FACTOR_BYTES,
                   "max_batch_object_bytes": MAX_BATCH_BYTES,
                   "declared_panel_array_budget_bytes": DECLARED_ARRAY_BUDGET_BYTES,
                   "minimum_available_memory_admission_bytes": MIN_AVAILABLE_MEMORY_BYTES,
                   "process_peak_rss_hard_cap": False,
                   "label_read_max_result_bytes": 32 * 1024**2,
                   "selection": "deterministic eligible manifest order; assets selected on automatic purged TRAIN coverage",
                   "claim_boundary": "TRAIN execution diagnostics only; no TEST score, upstream PIT certification, or production admission"},
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("JSON report exceeds the 1 MiB output limit")
    return report


def run_audit(*, source_loader=load_real_source, method_runner=None, config=None,
              resource_check=check_resource_headroom):
    check_environment()
    resource_check()
    batch, labels, provenance, _lineages = source_loader()
    validate_source(batch, labels, provenance)
    config = config or BatchOptimizationConfig()
    before = source_fingerprint(batch, labels, provenance)
    if method_runner is None:
        if str(EXAMPLES) not in sys.path:
            sys.path.insert(0, str(EXAMPLES))
        from method_audit import audit_methods
        method_runner = audit_methods
    methods = method_runner(batch, labels, config=config)
    after = source_fingerprint(batch, labels, provenance)
    if before != after:
        raise RuntimeError("source fingerprint changed during fixed-method audit")
    return build_report(batch, labels, provenance, methods, config=config,
                        fingerprints={"before": before, "after": after})


def write_report_exclusive(path, report):
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    payload = encoded.encode("utf-8")
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError("JSON report exceeds the 1 MiB output limit")
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None,
                        help="write full JSON report to this new exclusive path (maximum 1 MiB)")
    args = parser.parse_args(argv)
    report = run_audit()
    report_path = write_report_exclusive(args.report, report) if args.report else None
    summary = {
        "status": "failed" if report["method_status_counts"].get("failed", 0) else "complete",
        "factors": report["factor_ids"],
        "method_status_counts": report["method_status_counts"],
        "train_days": report["split"]["train_days"],
        "validation_days_reserved": report["split"]["validation_days_reserved"],
        "test_days_reserved": report["split"]["test_days_reserved"],
        "test_evaluated": False,
        "report_path": str(report_path) if report_path else None,
        "source_fingerprint": report["source_fingerprints"]["after"],
    }
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 1 if report["method_status_counts"].get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
