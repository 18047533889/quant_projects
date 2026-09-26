#!/usr/bin/env python3
"""Bounded full-batch CPU/CUDA/auto A/B on an F8 DataAccess COS panel."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import resource
import statistics
import time
import traceback
from collections.abc import Mapping

import numpy as np

from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
MANIFEST_SHA256 = "b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864"
_BATCH = None
_LABELS = None


def _plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return str(value)
    return value


def _metric_value(value):
    if value is None:
        return None
    return {name: _plain(getattr(value, name)) for name in (
        "metric_id", "value", "valid", "observation_count", "metric_version",
        "warnings", "sample_unit")}


def _worker(connection, backend, repeats):
    try:
        from quant_evaluator.runtime.evaluator import evaluate

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            bundle = evaluate(_BATCH, _LABELS, metrics=METRICS, backend=backend)
            timings.append(time.perf_counter() - start)

        artifacts = {}
        for metric in METRICS:
            artifact = bundle.artifacts[metric]
            values = np.asarray(artifact.values)
            finite_mask = np.isfinite(values)
            explicit_mask = getattr(artifact, "valid_mask", None)
            explicit_counts = getattr(artifact, "counts", None)
            observations = tuple(
                _metric_value(bundle.grouped_metrics[factor_id].get(metric))
                for factor_id in bundle.factor_ids
            )
            descriptor = {
                "type": type(artifact).__name__,
                "artifact_kind": _plain(getattr(artifact, "artifact_kind", None)),
            }
            if is_dataclass(artifact):
                descriptor.update({
                    field.name: _plain(getattr(artifact, field.name))
                    for field in fields(artifact)
                    if field.name not in {"values", "counts", "valid_mask", "provenance"}
                })
            else:
                raise TypeError(f"unsupported non-dataclass artifact: {type(artifact).__name__}")
            artifacts[metric] = {
                "descriptor": descriptor,
                "values": values.tolist(),
                "finite_mask": finite_mask.tolist(),
                "valid_mask": (finite_mask if explicit_mask is None else
                               np.asarray(explicit_mask)).tolist(),
                "counts": (None if explicit_counts is None else
                           np.asarray(explicit_counts).tolist()),
                "provenance_observation_counts": _plain(
                    artifact.provenance.get("observation_counts")),
                "provenance": _plain(artifact.provenance),
                "metric_values": observations,
            }
        meta = bundle.metadata
        connection.send({
            "status": "ok", "backend_requested": backend,
            "seconds": timings, "cold_s": timings[0],
            "warm_median_s": statistics.median(timings[1:]) if len(timings) > 1 else None,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "backend_used": meta.get("backend_used"),
            "auto_backend_reason": meta.get("auto_backend_reason"),
            "auto_backend_profile": meta.get("auto_backend_profile"),
            "metric_backends": _plain(meta.get("metric_backends")),
            "peak_vram": meta.get("peak_vram"),
            "config_hash": bundle.config_hash,
            "execution_receipt": _plain(meta.get("execution_receipt")),
            "artifacts": artifacts,
        })
    except BaseException as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=12)})
    finally:
        connection.close()


def _run_one(context, backend, repeats, timeout_s):
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, backend, repeats))
    process.start()
    sender.close()
    if receiver.poll(timeout_s):
        try:
            result = receiver.recv()
        except EOFError:
            result = {"status": "crash", "exitcode": process.exitcode}
    else:
        result = {"status": "timeout", "timeout_s": timeout_s}
        process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    receiver.close()
    if result.get("status") != "ok":
        raise RuntimeError(f"{backend}: {result}")
    return result


def _same(a, b):
    return json.dumps(_plain(a), sort_keys=True, separators=(",", ":"),
                      allow_nan=True) == json.dumps(_plain(b), sort_keys=True,
                                                    separators=(",", ":"), allow_nan=True)


def _close_values(a, b):
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return left.shape == right.shape and bool(np.allclose(
        left, right, rtol=1e-8, atol=1e-10, equal_nan=True))


def _metric_values_same(a, b):
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if left is None or right is None:
            if left is not right:
                return False
            continue
        if not _same({k: v for k, v in left.items() if k != "value"},
                     {k: v for k, v in right.items() if k != "value"}):
            return False
        if left["value"] is None or right["value"] is None:
            if left["value"] is not right["value"]:
                return False
        elif not _close_values([left["value"]], [right["value"]]):
            return False
    return True


def _compare(reference, candidate):
    result = {"config_hash": reference["config_hash"] == candidate["config_hash"],
              "metrics": {}}
    for metric in METRICS:
        a, b = reference["artifacts"][metric], candidate["artifacts"][metric]
        result["metrics"][metric] = {
            "descriptor": _same(a["descriptor"], b["descriptor"]),
            "values_rtol_1e-8_atol_1e-10": _close_values(a["values"], b["values"]),
            "finite_mask": _same(a["finite_mask"], b["finite_mask"]),
            "valid_mask": _same(a["valid_mask"], b["valid_mask"]),
            "counts": _same(a["counts"], b["counts"]),
            "provenance_observation_counts": _same(
                a["provenance_observation_counts"], b["provenance_observation_counts"]),
            "provenance": _same(a["provenance"], b["provenance"]),
            "metric_values": _metric_values_same(a["metric_values"], b["metric_values"]),
        }
        result["metrics"][metric]["pass"] = all(
            result["metrics"][metric][field] for field in (
                "descriptor", "values_rtol_1e-8_atol_1e-10", "finite_mask",
                "valid_mask", "counts", "provenance_observation_counts",
                "provenance", "metric_values"))
    result["pass"] = result["config_hash"] and all(
        metric["pass"] for metric in result["metrics"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("timeout-s must be positive")

    global _BATCH, _LABELS
    load_start = time.perf_counter()
    _BATCH, _LABELS, source = load_real_batch(
        factors=8, days=0, assets=5500, max_object_mib=64,
        manifest_sha256=MANIFEST_SHA256)
    load_s = time.perf_counter() - load_start
    shape = (_BATCH.num_times, _BATCH.num_assets, _BATCH.num_factors)
    if shape != (2586, 5461, 8):
        raise RuntimeError(f"unexpected bound panel shape: {shape}")

    # Symmetric interleaving reduces order and cache bias across backends.
    order = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
    context = mp.get_context("fork")
    runs = []
    for index, backend in enumerate(order):
        run = _run_one(context, backend, repeats=2, timeout_s=args.timeout_s)
        run["round"] = index + 1
        runs.append(run)
        print(json.dumps({"round": index + 1, "backend": backend,
                          "cold_s": run["cold_s"], "warm_median_s": run["warm_median_s"],
                          "backend_used": run["backend_used"],
                          "auto_backend_reason": run["auto_backend_reason"],
                          "peak_vram": run["peak_vram"],
                          "peak_rss_kib": run["peak_rss_kib"]}), flush=True)

    by_backend = {backend: [run for run in runs if run["backend_requested"] == backend]
                  for backend in ("cpu", "cuda_strict", "auto")}
    reference = by_backend["cpu"][0]
    comparisons = {
        backend: [_compare(reference, run) for run in selected]
        for backend, selected in by_backend.items()
    }
    config_hashes = sorted({run["config_hash"] for run in runs})
    report = {
        "created_utc": datetime.now().astimezone().isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request": {"metrics": METRICS, "backend_order": order,
                    "repeats_per_child": 2, "timeout_s": args.timeout_s,
                    "shape": shape, "dtype": str(_BATCH.values.dtype),
                    "manifest_sha256": MANIFEST_SHA256},
        "source": _plain(source), "load_s": load_s,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "config_hashes": config_hashes,
        "runs": runs, "comparisons_to_first_cpu": comparisons,
        "parity_pass": len(config_hashes) == 1 and all(
            comparison["pass"] for repeated in comparisons.values()
            for comparison in repeated),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                          allow_nan=False) + "\n")
    print(json.dumps({"parity_pass": report["parity_pass"],
                      "config_hashes": config_hashes,
                      "auto_routes": [run["backend_used"] for run in by_backend["auto"]],
                      "output": str(args.output) if args.output else None}), flush=True)
    if not report["parity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
