#!/usr/bin/env python3
"""Manifest-only admission check for materialized F48 evaluation.

Reads one manifest through authorized DataAccess. It never reads factor objects
or invokes the evaluator.
"""
from __future__ import annotations

import json
import subprocess

from data_access.core.engine import DuckDBEngine
from data_access.cos.research import read_declared_cos_object
from data_access.registry.loader import DatasetRegistry
from data_access.store import DataAccessStore
from quant_evaluator.scripts.load_real_cos_factor_batch import BASE, _ds

MANIFEST_SHA256 = "b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864"
FACTOR_COUNT = 48
MAX_OBJECT_MIB = 128
MAX_TOTAL_MIB = 2048
EXPECTED_SHAPE = (2586, 5461, 48)
MIN_HEADROOM_BYTES = 24 * 1024**3


def _mem_available_bytes():
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode:
            return {"status": "unavailable", "detail": result.stderr.strip()[:300]}
        devices = []
        for row in result.stdout.splitlines():
            name, total, free = (part.strip() for part in row.split(",", 2))
            devices.append({"name": name, "total_mib": int(total), "free_mib": int(free)})
        return {"status": "ok", "devices": devices}
    except Exception as exc:
        return {"status": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}


def select_records(factors):
    """Select the smallest exact 48 verified records under fixed byte bounds."""
    selected = []
    for factor_id, record in sorted(factors.items()):
        if not isinstance(record, dict) or record.get("verified") is not True:
            continue
        if record.get("status") not in {
            "materialized_not_evaluated", "evaluated_optimization_pending"
        }:
            continue
        size, digest = record.get("bytes"), record.get("sha256")
        if type(size) is not int or not 0 < size <= MAX_OBJECT_MIB * 1024**2:
            continue
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            continue
        uri = f"{BASE}/factor_values/{digest}/{factor_id}.parquet"
        if record.get("uri") != uri:
            continue
        selected.append((factor_id, record))
    selected = sorted(selected, key=lambda item: (item[1]["bytes"], item[0]))[:FACTOR_COUNT]
    if len(selected) != FACTOR_COUNT:
        raise ValueError("manifest has fewer than 48 verified bounded factors")
    if len({record["sha256"] for _, record in selected}) != FACTOR_COUNT:
        raise ValueError("selected factor hashes are not distinct")
    total_bytes = sum(record["bytes"] for _, record in selected)
    if total_bytes > MAX_TOTAL_MIB * 1024**2:
        raise ValueError("selected 48 object bytes exceed 2048 MiB")
    return selected, total_bytes


def preflight():
    uri = f"{BASE}/metadata/{MANIFEST_SHA256}/landing_manifest.json"
    dataset = _ds("source_manifest", uri.rsplit("/", 1)[0],
                  "landing_manifest.json", "json")
    engine = DuckDBEngine(threads=1)
    try:
        store = DataAccessStore(DatasetRegistry({dataset.name: dataset}), engine)
        store.authorize_dataset(dataset.name)
        store._authorize_factor_params(dataset.name, None)
        declared = read_declared_cos_object(store, dataset.name, allow_research=True)
        if declared.content_sha256 != MANIFEST_SHA256:
            raise ValueError("declared manifest hash differs from pinned identity")
        rows = declared.table.to_pylist()
        if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
            raise ValueError("invalid landing manifest structure")
        selected, total_bytes = select_records(rows[0]["factors"])
    finally:
        engine.close()

    t, n, f = EXPECTED_SHAPE
    cube_bytes = t * n * f * 8
    mask_bytes = t * n * f
    # Two overlapping cubes at FactorBatch ownership, four value-sized CPU
    # metric temporaries, the validity mask, and a fixed 2 GiB allowance.
    estimated_peak_bytes = 6 * cube_bytes + mask_bytes + 2 * 1024**3
    available = _mem_available_bytes()
    memory_pass = (available is not None and
                   available >= estimated_peak_bytes + MIN_HEADROOM_BYTES)
    return {
        "status": "ready_to_materialize" if memory_pass else "insufficient_host_memory",
        "pass": memory_pass,
        "manifest_sha256": MANIFEST_SHA256,
        "selected_factor_count": len(selected),
        "selected_factor_ids": [factor_id for factor_id, _ in selected],
        "selected_distinct_hash_count": len({r["sha256"] for _, r in selected}),
        "selected_object_bytes": total_bytes,
        "max_object_bytes": MAX_OBJECT_MIB * 1024**2,
        "max_total_bytes": MAX_TOTAL_MIB * 1024**2,
        "expected_shape_from_source_receipt": list(EXPECTED_SHAPE),
        "materialized_axes_verified": False,
        "materialized_axes_note": "compare after bounded object load; no factor objects read here",
        "estimated_peak_bytes": estimated_peak_bytes,
        "estimated_peak_model": "6 float64 cubes + bool mask + 2 GiB fixed allowance",
        "required_available_headroom_bytes": MIN_HEADROOM_BYTES,
        "mem_available_bytes": available,
        "gpu_memory": _gpu_snapshot(),
    }


def main():
    try:
        report = preflight()
    except Exception as exc:
        safe_messages = {
            "exact object metadata is missing or mismatched",
            "declared manifest hash differs from pinned identity",
            "invalid landing manifest structure",
            "manifest has fewer than 48 verified bounded factors",
            "selected factor hashes are not distinct",
            "selected 48 object bytes exceed 2048 MiB",
        }
        message = str(exc)
        report = {
            "status": "unavailable", "pass": False,
            "failure_type": type(exc).__name__,
            "failure_reason": message if message in safe_messages else "manifest preflight failed",
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
