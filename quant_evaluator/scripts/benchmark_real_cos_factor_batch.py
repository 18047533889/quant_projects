#!/usr/bin/env python3
"""CPU/CUDA/auto A/B on COS-bound factor tiles over A-share sessions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

from quant_evaluator.scripts import benchmark_public_backend_routes as routes
from quant_evaluator.scripts import benchmark_real_cos_metric_batch as full
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

DEFAULT_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
EXTENDED_METRICS = ("rank_ic_series", "ic_ir", "quantile_returns_daily", "quantile_returns_full")
METRICS = DEFAULT_METRICS + EXTENDED_METRICS

def run_extended(ctx, requested_metrics, factors, labels, provenance, args, load_s):
    """Interleaved full-artifact A/B for the exact verified F8 panel."""
    if tuple(requested_metrics) != tuple(m for m in EXTENDED_METRICS if m in requested_metrics):
        raise ValueError("extended metrics must follow EXTENDED_METRICS order")
    if (args.factors, args.days, args.assets) != (8, 0, 5500):
        raise ValueError("extended trial requires the exact F8 full-history request")
    if (args.manifest_sha256 != full.MANIFEST_SHA256 or
            args.max_object_mib != 64 or args.max_total_mib != 256):
        raise ValueError("extended trial requires the verified manifest and 64/256 MiB bounds")
    full._BATCH, full._LABELS = factors, labels
    order = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
    results = {}
    for metric in requested_metrics:
        full.METRICS = (metric,)
        runs = []
        for index, backend in enumerate(order):
            run = full._run_one(ctx, backend, repeats=args.repeats + 1, timeout_s=args.timeout_s)
            runs.append(run)
            print(f"{metric} round={index+1} {backend} cold={run['cold_s']:.3f}s "
                  f"warm={run['warm_median_s']:.3f}s route={run['backend_used']} "
                  f"rss_kib={run['peak_rss_kib']} vram={run['peak_vram']}", flush=True)
        comparisons = [full._compare(runs[0], run) for run in runs]
        results[metric] = {
            "runs": [{k: v for k, v in run.items() if k != "artifacts"} |
                     {"artifact_sha256": hashlib.sha256(json.dumps(
                         full._plain(run["artifacts"]), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
                     for run in runs],
            "comparisons_to_first_cpu": comparisons,
            "pass": all(item["pass"] for item in comparisons),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "DataAccess bound COS landing manifest and registered A-share calendar/AdjVwap",
        "request": {"metrics": requested_metrics, "backend_order": order,
                    "warm_repeats_per_child": args.repeats, "timeout_s": args.timeout_s,
                    "shape": list(factors.values.shape),
                    "manifest_sha256": args.manifest_sha256},
        "load_s": load_s,
        "peak_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "provenance": provenance, "metrics": results,
        "pass": all(result["pass"] for result in results.values()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    if not report["pass"]:
        raise AssertionError("full public artifact parity failed")
    print(json.dumps({"shape":report["request"]["shape"],"load_s":load_s,
                      "parity":{m:x["pass"] for m,x in results.items()}},ensure_ascii=False),flush=True)


def _mem_available_bytes():
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


GPU_VRAM_FRACTION = 0.4
MIN_EFFECTIVE_FREE_VRAM_MIB = 14 * 1024


def _gpu_memory_pass(gpu_memory):
    return (gpu_memory.get("status") == "ok" and any(
        device.get("free_mib", 0) * GPU_VRAM_FRACTION >= MIN_EFFECTIVE_FREE_VRAM_MIB
        for device in gpu_memory.get("devices", [])
    ))


def _gpu_memory_snapshot():
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode != 0:
            return {"status": "unavailable", "detail": result.stderr.strip()[:300]}
        devices = []
        for row in result.stdout.splitlines():
            total, free = (int(part.strip()) for part in row.split(",", 1))
            effective_free = int(free * GPU_VRAM_FRACTION)
            devices.append({
                "total_mib": total, "free_mib": free,
                "effective_free_mib": effective_free,
                "passes_minimum": effective_free >= MIN_EFFECTIVE_FREE_VRAM_MIB,
            })
        return {"status": "ok", "devices": devices}
    except Exception as exc:
        return {"status": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}


def _estimate_f32_peak_bytes(manifest_sha256=None, *, metric_ids=None):
    """Bound the exact verified F32 manifest empirically; keep others conservative."""
    from quant_evaluator.scripts.coverage_memory_profile import measured_coverage_peak_bytes
    coverage_peak = measured_coverage_peak_bytes(manifest_sha256, metric_ids)
    if coverage_peak is not None:
        return coverage_peak
    array_upper_bytes = 3000 * 5500 * 32 * 8
    if manifest_sha256 == full.MANIFEST_SHA256:
        # Same immutable 32-object sample: largest observed parent RSS
        # 13,872,336 KiB; largest F32 worker RSS 29,282,916 KiB across
        # rank, quantile and default mixed double-call batches. Add 20% to their
        # conservative sum, round up to GiB, then require 8 GiB headroom.
        observed_combined = (13_872_336 + 29_282_916) * 1024
        empirical = (observed_combined * 6 + 4) // 5
        gib = 1024**3
        empirical = ((empirical + gib - 1) // gib) * gib
        return max(array_upper_bytes * 8, empirical)
    observed_f24_peak_bytes = 46 * 1024**3
    scaled_f24_peak_bytes = (observed_f24_peak_bytes * 32 + 23) // 24
    return max(array_upper_bytes * 16, scaled_f24_peak_bytes)


def preflight_factor_count_profile(args):
    """Inspect only the bound landing manifest and host memory before data reads."""
    from data_access.core.engine import DuckDBEngine
    from data_access.registry.loader import DatasetRegistry
    from data_access.store import DataAccessStore
    from data_access.cos.research import read_declared_cos_object
    from data_access.core.exceptions import AuthorizationError
    from quant_evaluator.scripts.load_real_cos_factor_batch import BASE, SHA, _ds

    requested_sha = args.manifest_sha256 or full.MANIFEST_SHA256 or SHA
    uri = f"{BASE}/metadata/{requested_sha}/landing_manifest.json"
    dataset = _ds("source_manifest", uri.rsplit("/", 1)[0],
                  "landing_manifest.json", "json")
    engine = DuckDBEngine(threads=1)
    try:
        store = DataAccessStore(DatasetRegistry({dataset.name: dataset}), engine)
        # Preserve DataAccess authorization ordering. The declared-object read
        # is the authority for whether the configured COS path is usable: it
        # may use the DataAccess CLI gateway without S3 credentials.
        store.authorize_dataset(dataset.name)
        store._authorize_factor_params(dataset.name, None)
        declared = read_declared_cos_object(store, dataset.name, allow_research=True)
        if declared.content_sha256 != requested_sha:
            return {"status": "unavailable", "pass": False,
                    "reason": "manifest identity mismatch",
                    "manifest_sha256": requested_sha,
                    "gpu_memory": _gpu_memory_snapshot()}
        rows = declared.table.to_pylist()
        if len(rows) != 1 or not isinstance(rows[0].get("factors"), dict):
            return {"status": "unavailable", "pass": False,
                    "reason": "invalid landing manifest",
                    "gpu_memory": _gpu_memory_snapshot()}
        factors = rows[0]["factors"]
        eligible = []
        for factor_id, record in factors.items():
            if not isinstance(record, dict) or record.get("verified") is not True:
                continue
            if record.get("status") not in {
                "materialized_not_evaluated", "evaluated_optimization_pending"
            }:
                continue
            size, digest = record.get("bytes"), record.get("sha256")
            if type(size) is not int or not 0 < size <= args.profile_max_object_mib * 1024**2:
                continue
            if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                continue
            if record.get("uri") != f"{BASE}/factor_values/{digest}/{factor_id}.parquet":
                continue
            eligible.append((factor_id, size))
        selected = sorted(eligible, key=lambda item: (item[1], item[0]))[:32]
        selected_bytes = sum(size for _, size in selected)
        if len(selected) < 32:
            return {"status": "unavailable", "pass": False,
                    "reason": "manifest has fewer than 32 verified bounded factors",
                    "manifest_sha256": requested_sha, "verified_factor_count": len(eligible),
                    "selected_object_bytes": sum(size for _, size in selected),
                    "mem_available_bytes": _mem_available_bytes(),
                    "gpu_memory": _gpu_memory_snapshot()}
        if len({factor_id for factor_id, _ in selected}) != 32:
            return {"status": "unavailable", "pass": False,
                    "reason": "factor identifiers are not unique",
                    "gpu_memory": _gpu_memory_snapshot()}
        if selected_bytes > args.profile_max_total_mib * 1024**2:
            return {"status": "unavailable", "pass": False,
                    "reason": "32-factor manifest sample exceeds transfer budget",
                    "selected_object_bytes": selected_bytes,
                    "max_total_bytes": args.profile_max_total_mib * 1024**2,
                    "mem_available_bytes": _mem_available_bytes(),
                    "gpu_memory": _gpu_memory_snapshot()}
        # The exact immutable manifest uses observed F32 peaks; unseen
        # manifests retain the conservative F24 extrapolation.
        array_upper_bytes = 3000 * 5500 * 32 * 8
        observed_f24_peak_bytes = 46 * 1024**3
        metric_ids = getattr(args, "metric_ids", None)
        if metric_ids == ("coverage",):
            estimated_peak_bytes = _estimate_f32_peak_bytes(
                requested_sha, metric_ids=metric_ids)
        else:
            estimated_peak_bytes = _estimate_f32_peak_bytes(requested_sha)
        mem_available = _mem_available_bytes()
        gpu_memory = _gpu_memory_snapshot()
        budget_bytes = args.max_working_gib * 1024**3
        memory_pass = (estimated_peak_bytes <= budget_bytes and
                       mem_available is not None and
                       mem_available >= estimated_peak_bytes + 8*1024**3)
        gpu_pass = _gpu_memory_pass(gpu_memory)
        preflight = {
            "status": "ready" if memory_pass and gpu_pass else "insufficient_resources",
            "pass": memory_pass and gpu_pass,
            "gpu_min_effective_free_mib": MIN_EFFECTIVE_FREE_VRAM_MIB,
            "gpu_policy_fraction": GPU_VRAM_FRACTION,
            "gpu_gate_note": "0.4 conservative routes policy; whole-batch evaluator defaults to 0.75",
            "gpu_memory_pass": gpu_pass,
            "manifest_sha256": requested_sha,
            "verified_factor_count": len(eligible),
            "selected_factor_ids": [factor_id for factor_id, _ in selected],
            "selected_object_bytes": selected_bytes,
            "max_object_bytes": args.profile_max_object_mib * 1024**2,
            "max_total_bytes": args.profile_max_total_mib * 1024**2,
            "estimated_peak_bytes": estimated_peak_bytes,
            "memory_profile_metrics": list(metric_ids or ()),
            "estimated_peak_model": (
                "exact F32 coverage: max(8x array bound, 1.2x coverage parent+worker RSS rounded to GiB); "
                "evidence real_cos_f32_coverage_raw_identity_cache_ab_20261001.json"
                if metric_ids == ("coverage",) and requested_sha == full.MANIFEST_SHA256 else
                "exact manifest: max(8x array bound, 1.2x observed F32 parent+worker RSS rounded to GiB)"
                if requested_sha == full.MANIFEST_SHA256 else
                "unseen manifest: max(16x array bound, old F24 CPU peak 46 GiB * 32/24)"),
            "array_upper_bound_bytes": array_upper_bytes,
            "observed_f24_peak_bytes": observed_f24_peak_bytes,
            "max_working_bytes": budget_bytes,
            "mem_available_bytes": mem_available,
            "gpu_memory": gpu_memory,
            "shape_upper_bound": [3000, 5500, 32],
        }
        if preflight["status"] != "ready":
            failures = []
            if not memory_pass:
                failures.append("estimated peak exceeds profile budget or available-memory headroom")
            if not gpu_pass:
                failures.append("effective free VRAM is below the 14 GiB profile minimum")
            preflight["reason"] = "; ".join(failures)
        return preflight
    except AuthorizationError:
        raise
    except Exception as exc:
        message = str(exc)
        # Failed COS CLI metadata requests are collapsed to empty results.
        # Without gateway diagnostics the cause cannot be distinguished.
        if "exact object metadata is missing or mismatched" in message:
            category = "object_metadata_unavailable"
            reason = "COS gateway returned no usable exact-object metadata; cause is undetermined"
        else:
            category = "preflight_error"
            reason = f"manifest preflight failed: {type(exc).__name__}: {message}"
        return {"status": "unavailable", "pass": False,
                "category": category, "reason": reason,
                "manifest_sha256": requested_sha,
                "mem_available_bytes": _mem_available_bytes(),
                "gpu_memory": _gpu_memory_snapshot()}
    finally:
        engine.close()



def run_factor_count_batch_profile(ctx, batch, labels, provenance, args, load_s):
    """Run all default metrics together through the complete public batch evaluator."""
    full._BATCH, full._LABELS = batch, labels
    full.METRICS = DEFAULT_METRICS
    order = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
    runs = []
    for index, backend in enumerate(order):
        run = full._run_one(ctx, backend, repeats=args.repeats + 1,
                            timeout_s=args.timeout_s)
        run["round"] = index + 1
        runs.append(run)
        print(f"F32 mode=whole_batch round={index+1} {backend} "
              f"route={run['backend_used']} cold={run['cold_s']:.3f}s "
              f"warm={run['warm_median_s']:.3f}s rss_kib={run['peak_rss_kib']} "
              f"vram={run['peak_vram']}", flush=True)

    reference = next(run for run in runs if run["backend_requested"] == "cpu")
    comparisons = []
    for run in runs:
        comparison = full._compare(reference, run)
        requested = run["backend_requested"]
        if requested == "cpu":
            route_ok = run.get("backend_used") == "cpu"
        elif requested == "cuda_strict":
            route_ok = run.get("backend_used") == "cuda" and bool(run.get("peak_vram"))
        else:
            route_ok = run.get("backend_used") in {"cpu", "cuda"}
        comparison["requested_backend_used"] = route_ok
        comparison["pass"] = comparison["pass"] and route_ok
        comparisons.append(comparison)
    config_hashes = sorted({run["config_hash"] for run in runs})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "factor_count_profile": "F32", "profile_mode": "whole_batch",
            "metrics": DEFAULT_METRICS, "backend_order": order,
            "repeats_per_child": args.repeats + 1, "timeout_s": args.timeout_s,
            "shape": list(batch.values.shape), "dtype": str(batch.values.dtype),
            "manifest_sha256": args.manifest_sha256 or provenance["sources"][0]["manifest_sha256"],
            "profile_max_object_mib": args.profile_max_object_mib,
            "profile_max_total_mib": args.profile_max_total_mib,
            "max_working_gib": args.max_working_gib,
            "preflight_vram_fraction": GPU_VRAM_FRACTION,
            "cuda_strict_policy": "benchmark_real_cos_metric_batch uses evaluator default GPUExecutionPolicy fraction 0.75; preflight 0.4 is the stricter public routes policy",
        },
        "load_s": load_s, "peak_parent_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
        "provenance": provenance, "config_hashes": config_hashes,
        "runs": [{key: value for key, value in run.items() if key != "artifacts"} | {
            "artifact_sha256": hashlib.sha256(json.dumps(
                full._plain(run["artifacts"]), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
                 for run in runs],
        "comparisons_to_first_cpu": comparisons,
        "pass": len(config_hashes) == 1 and all(item["pass"] for item in comparisons),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                          allow_nan=False) + "\n")
    print(json.dumps({
        "profile_mode": "whole_batch",
        "shape": report["request"]["shape"],
        "actual_routes": [run["backend_used"] for run in runs],
        "parity_pass": report["pass"],
        "output": str(args.output) if args.output else None,
    }, ensure_ascii=False), flush=True)
    if not report["pass"]:
        raise AssertionError("F32 whole-batch public artifact parity failed")

def run_factor_count_profile(ctx, batch, labels, provenance, args, load_s):
    """Interleaved real full-history F32 CPU/CUDA/auto profile."""
    if args.factor_profile_mode == "whole_batch":
        return run_factor_count_batch_profile(ctx, batch, labels, provenance, args, load_s)
    order = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
    metrics = {}
    for metric in DEFAULT_METRICS:
        runs = []
        for index, backend in enumerate(order):
            result = routes.run_one(ctx, metric, backend, args.repeats, args.timeout_s)
            result["round"] = index + 1
            result["backend_requested"] = backend
            runs.append(result)
            print(f"F32 metric={metric} round={index+1} {backend} "
                  f"status={result.get('status')} route={result.get('backend_used')} "
                  f"cold={result.get('cold_s')} warm={result.get('warm_median_s')} "
                  f"rss_kib={result.get('peak_rss_kib')} "
                  f"vram={result.get('gpu_peak_vram')}", flush=True)
        baseline = next((run for run in runs if run["backend_requested"] == "cpu"
                         and run.get("status") == "ok"), None)
        comparisons = []
        for run in runs:
            if baseline is None or run.get("status") != "ok":
                comparisons.append({"pass": False, "reason": "backend run failed"})
                continue
            comparison = routes.parity(baseline, run)
            if run["backend_requested"] == "auto":
                output_fields = ("same_shape", "same_finite", "numeric_1e-8_1e-10",
                                 "same_counts", "same_mask", "same_evidence",
                                 "same_provenance_counts", "same_input_hash")
                comparison["pass"] = all(comparison[field] for field in output_fields)
                comparison["auto_output_pass"] = comparison["pass"]
            if run["backend_requested"] == "cpu":
                comparison["requested_backend_used"] = run.get("backend_used") == "cpu"
                comparison["pass"] &= comparison["requested_backend_used"]
            comparisons.append(comparison)
        metrics[metric] = {
            "runs": [{key: value for key, value in run.items()
                      if key not in {"values", "counts", "valid_mask", "evidence"}}
                     for run in runs],
            "comparisons_to_first_cpu": comparisons,
            "pass": all(run.get("status") == "ok" for run in runs) and
                    all(item.get("pass", False) for item in comparisons),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": {"factor_count_profile": "F32", "metrics": DEFAULT_METRICS,
                    "backend_order": order, "repeats": args.repeats,
                    "timeout_s": args.timeout_s, "shape": list(batch.values.shape),
                    "dtype": str(batch.values.dtype),
                    "manifest_sha256": args.manifest_sha256 or provenance["sources"][0]["manifest_sha256"],
                    "profile_max_object_mib": args.profile_max_object_mib,
                    "profile_max_total_mib": args.profile_max_total_mib,
                    "max_working_gib": args.max_working_gib},
        "load_s": load_s, "peak_parent_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
        "provenance": provenance, "metrics": metrics,
        "pass": all(item["pass"] for item in metrics.values()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"profile_mode": "single", "shape": report["request"]["shape"],
                      "actual_routes": {metric: [run.get("backend_used")
                                                  for run in item["runs"]]
                                        for metric, item in metrics.items()},
                      "parity": {metric: item["pass"] for metric, item in metrics.items()},
                      "output": str(args.output) if args.output else None},
                     ensure_ascii=False), flush=True)
    if not report["pass"]:
        raise AssertionError("F32 public route parity failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=0, help="0 means every complete session")
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--factors", type=int, default=2, help="bound COS factors, 1..16")
    parser.add_argument("--manifest-sha256", type=str, default=None)
    parser.add_argument("--max-object-mib", type=int, default=8)
    parser.add_argument("--max-total-mib", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--factor-count-profile", choices=("32", "64"), default=None,
                        help="run bounded full-history factor-count profile; F64 is reported unavailable")
    parser.add_argument("--factor-profile-mode", choices=("whole_batch", "single"),
                        default="whole_batch",
                        help="for F32, compare all default metrics in one public call or run them singly")
    parser.add_argument("--profile-max-object-mib", type=int, default=128)
    parser.add_argument("--profile-max-total-mib", type=int, default=2048)
    parser.add_argument("--max-working-gib", type=int, default=50)
    parser.add_argument("--preflight-only", action="store_true",
                        help="inspect manifest/resources and exit before factor-object reads")
    parser.add_argument("--run", action="store_true",
                        help="run the full F32 benchmark after a ready preflight")
    parser.add_argument("--timeout-s", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    if args.timeout_s <= 0:
        parser.error("timeout-s must be positive")
    if args.preflight_only and args.run:
        parser.error("--preflight-only and --run are mutually exclusive")
    if args.factor_count_profile and (
        not 1 <= args.profile_max_object_mib <= 128 or
        not 1 <= args.profile_max_total_mib <= 2048 or
        not 1 <= args.max_working_gib <= 128
    ):
        parser.error("factor-count profile limits: object 1..128 MiB, total 1..2048 MiB, working 1..128 GiB")
    requested_metrics = tuple(args.metrics.split(","))
    if not requested_metrics or len(set(requested_metrics)) != len(requested_metrics) or any(
        metric not in METRICS for metric in requested_metrics
    ):
        parser.error("metrics must be a unique subset of " + ",".join(METRICS))
    if any(metric in EXTENDED_METRICS for metric in requested_metrics):
        if requested_metrics != tuple(metric for metric in EXTENDED_METRICS
                                      if metric in requested_metrics):
            parser.error("extended trial metrics must follow EXTENDED_METRICS order")
        if ((args.factors, args.days, args.assets) != (8, 0, 5500) or
                args.manifest_sha256 != full.MANIFEST_SHA256 or
                args.max_object_mib != 64 or args.max_total_mib != 256):
            parser.error("extended trial requires exact F8 verified manifest and 64/256 MiB bounds")
    if args.factor_count_profile == "64":
        print(json.dumps({
            "status": "unavailable", "factor_count": 64,
            "loader_max_factor_count": 32,
            "reason": "the bound DataAccess loader supports at most 32 factors",
        }, ensure_ascii=False), flush=True)
        return
    if args.factor_count_profile == "32":
        if requested_metrics != DEFAULT_METRICS:
            parser.error("factor-count profile uses the default public metrics")
        if args.manifest_sha256 is None:
            args.manifest_sha256 = full.MANIFEST_SHA256
        preflight = preflight_factor_count_profile(args)
        print(json.dumps({"preflight": preflight}, ensure_ascii=False), flush=True)
        if args.preflight_only or preflight["status"] != "ready":
            return
        if not args.run:
            print(json.dumps({"status": "preflight_only",
                              "note": "pass --run to start the full-history F32 benchmark"},
                             ensure_ascii=False), flush=True)
            return
        args.factors, args.days, args.assets = 32, 0, 5500
        args.max_object_mib = args.profile_max_object_mib
        args.max_total_mib = args.profile_max_total_mib
    start = time.perf_counter()
    factors, labels, provenance = load_real_batch(
        factors=args.factors, days=args.days, assets=args.assets,
        max_object_mib=args.max_object_mib, max_total_mib=args.max_total_mib,
        **({"manifest_sha256": args.manifest_sha256} if args.manifest_sha256 else {}),
    )
    load_s = time.perf_counter() - start
    routes._SOURCES = {"base": (factors, labels, {})}
    ctx = mp.get_context("fork")
    if args.factor_count_profile == "32":
        run_factor_count_profile(ctx, factors, labels, provenance, args, load_s)
        return
    if any(metric in EXTENDED_METRICS for metric in requested_metrics):
        run_extended(ctx, requested_metrics, factors, labels, provenance, args, load_s)
        return
    measurements = {}
    for metric in requested_metrics:
        runs = {}
        for backend in ("cpu", "cuda_strict", "auto"):
            result = routes.run_one(ctx, metric, backend, args.repeats, 180)
            if result.get("status") != "ok":
                raise RuntimeError(f"{metric}/{backend}: {result}")
            runs[backend] = result
            print(f"{metric} {backend} cold={result['cold_s']:.3f}s "
                  f"warm={result['warm_median_s']:.3f}s "
                  f"route={result['backend_used']}", flush=True)
        cuda_parity = routes.parity(runs["cpu"], runs["cuda_strict"])
        auto_parity = routes.parity(runs["cpu"], runs["auto"])
        # The shared parity helper also asserts GPU routing. CPU auto is a
        # legitimate policy decision for some metrics; output checks still hold.
        auto_output_pass = all(v for k, v in auto_parity.items()
                               if k in {"same_shape","same_finite","numeric_1e-8_1e-10",
                                        "same_counts","same_mask","same_evidence",
                                        "same_provenance_counts","same_input_hash"})
        measurements[metric] = {
            "runs": {name: {k:v for k,v in result.items()
                             if k not in {"values","counts","valid_mask","evidence"}}
                     for name,result in runs.items()},
            "cuda_parity": cuda_parity,
            "auto_parity": auto_parity,
            "auto_output_pass": auto_output_pass,
        }
        if not cuda_parity["pass"] or not auto_output_pass:
            raise AssertionError(f"full public artifact parity failed for {metric}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "DataAccess bound COS landing manifest and registered A-share calendar/AdjVwap",
        "load_s": load_s,
        "peak_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "provenance": provenance,
        "metrics": measurements,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"days":provenance["days"],"assets":provenance["assets"],
        "load_s":load_s,"peak_parent_rss_kib":report["peak_parent_rss_kib"],
        "parity":{m:{"cuda":x["cuda_parity"]["pass"],"auto":x["auto_output_pass"]}
                  for m,x in measurements.items()}},ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
