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
EXTENDED_METRICS = ("rank_ic_series", "ic_ir")
METRICS = DEFAULT_METRICS + EXTENDED_METRICS

def run_extended(ctx, requested_metrics, factors, labels, provenance, args, load_s):
    """Interleaved full-artifact A/B for the exact verified F8 panel."""
    if tuple(requested_metrics) != tuple(m for m in EXTENDED_METRICS if m in requested_metrics):
        raise ValueError("extended metrics must be rank_ic_series,ic_ir in that order")
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
            run = full._run_one(ctx, backend, repeats=args.repeats + 1, timeout_s=180)
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
                    "warm_repeats_per_child": args.repeats,
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
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    requested_metrics = tuple(args.metrics.split(","))
    if not requested_metrics or len(set(requested_metrics)) != len(requested_metrics) or any(
        metric not in METRICS for metric in requested_metrics
    ):
        parser.error("metrics must be a unique subset of " + ",".join(METRICS))
    if any(metric in EXTENDED_METRICS for metric in requested_metrics):
        if requested_metrics != tuple(metric for metric in EXTENDED_METRICS
                                      if metric in requested_metrics):
            parser.error("extended trial accepts rank_ic_series,ic_ir in that order")
        if ((args.factors, args.days, args.assets) != (8, 0, 5500) or
                args.manifest_sha256 != full.MANIFEST_SHA256 or
                args.max_object_mib != 64 or args.max_total_mib != 256):
            parser.error("extended trial requires exact F8 verified manifest and 64/256 MiB bounds")
    start = time.perf_counter()
    factors, labels, provenance = load_real_batch(
        factors=args.factors, days=args.days, assets=args.assets,
        max_object_mib=args.max_object_mib, max_total_mib=args.max_total_mib,
        **({"manifest_sha256": args.manifest_sha256} if args.manifest_sha256 else {}),
    )
    load_s = time.perf_counter() - start
    routes._SOURCES = {"base": (factors, labels, {})}
    ctx = mp.get_context("fork")
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
