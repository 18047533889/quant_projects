"""Bounded F32 COS public-facade measured-auto verification; no value copies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from quant_evaluator import AutoCalibrationOptions, evaluate
from quant_evaluator.runtime.backend_calibration import CalibrationPolicy, _parity_mismatch
from quant_evaluator.scripts.benchmark_real_cos_factor_batch import preflight_factor_count_profile
from quant_evaluator.scripts.benchmark_real_cos_metric_batch import MANIFEST_SHA256
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

METRICS = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")


def measured_route_matches(receipt, metrics):
    """Require adoption evidence, not just a numerically matching result."""
    return (
        receipt.get("auto_backend_reason") == "measured_auto_exact_candidate"
        and receipt.get("auto_backend_policy") == "measured_auto_registry_exact_v1"
        and receipt.get("auto_backend_profile") == "exact_content_process_local"
        and receipt.get("backend_used") == "cuda"
        and all(receipt.get("metric_backends", {}).get(metric) == "cuda"
                for metric in metrics)
    )



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--days", type=int, choices=(0, 500, 1000), default=0)
    parser.add_argument("--repetitions", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--require-measured-route", action="store_true",
                        help="require ordinary auto to adopt measured CUDA evidence")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; refusing to overwrite evidence")
    preflight = preflight_factor_count_profile(SimpleNamespace(
        manifest_sha256=MANIFEST_SHA256, profile_max_object_mib=128,
        profile_max_total_mib=2048, max_working_gib=50,
    ))
    print(json.dumps({"preflight": preflight}, ensure_ascii=False), flush=True)
    if preflight["status"] != "ready":
        raise SystemExit(2)
    if not args.run:
        return
    tick = time.perf_counter()
    batch, label, source = load_real_batch(
        factors=32, days=args.days, assets=5500, max_object_mib=128,
        max_total_mib=2048, manifest_sha256=MANIFEST_SHA256,
    )
    load_seconds = time.perf_counter() - tick
    if (batch.num_assets, batch.num_factors) != (5461, 32):
        raise RuntimeError("unexpected bound F32 asset/factor shape")
    if args.days == 0 and batch.num_times != 2586:
        raise RuntimeError("unexpected bound full-history date count")
    print(json.dumps({"loaded_shape": list(batch.values.shape),
                      "load_seconds": load_seconds}), flush=True)
    options = AutoCalibrationOptions(policy=CalibrationPolicy(
        repetitions=args.repetitions, warmups=1, max_wall_time_seconds=360,
    ))
    tick = time.perf_counter()
    first = evaluate(batch, label, metrics=METRICS, backend="auto", auto_calibration=options)
    calibration_seconds = time.perf_counter() - tick
    print(json.dumps({"calibration": first.metadata["auto_calibration"]}), flush=True)
    tick = time.perf_counter()
    repeat = evaluate(batch, label, metrics=METRICS, backend="auto", auto_calibration=options)
    cache_hit_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    ordinary = evaluate(batch, label, metrics=METRICS, backend="auto")
    ordinary_auto_seconds = time.perf_counter() - tick
    comparisons = {
        "cached": _parity_mismatch(first, repeat, options.policy),
        "ordinary_auto": _parity_mismatch(first, ordinary, options.policy),
    }
    measured_route_pass = measured_route_matches(
        ordinary.metadata["execution_receipt"], METRICS)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape": list(batch.values.shape), "dtype": str(batch.values.dtype),
        "metrics": METRICS, "source": source, "manifest_sha256": MANIFEST_SHA256,
        "load_seconds": load_seconds, "calibration_seconds": calibration_seconds,
        "cache_hit_seconds": cache_hit_seconds, "ordinary_auto_seconds": ordinary_auto_seconds,
        "calibration": first.metadata["auto_calibration"],
        "cache_hit": repeat.metadata["auto_calibration"],
        "ordinary_route": ordinary.metadata["execution_receipt"],
        "comparison_mismatches": comparisons,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pass": all(value is None for value in comparisons.values())
                and repeat.metadata["auto_calibration"]["status"] == "cache_hit"
                and first.metadata["auto_calibration"].get("calibration_record", {}).get("parity") == "pass"
                and (not args.require_measured_route or measured_route_pass),
        "require_measured_route": args.require_measured_route,
        "measured_route_pass": measured_route_pass,
        "limitations": "Alternating paired calibration, not randomized repeated source IO. "
                        "Steady-state winner is exact-content evidence, not universal fastest certification.",
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if len(serialized.encode()) > 256 * 1024:
        raise RuntimeError("bounded evidence report exceeded 256 KiB")
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(json.dumps({"pass": report["pass"], "shape": report["shape"],
                      "cache_hit_seconds": cache_hit_seconds,
                      "ordinary_auto_seconds": ordinary_auto_seconds}), flush=True)
    if not report["pass"]:
        raise AssertionError("measured auto parity/cache validation failed")


if __name__ == "__main__":
    main()
