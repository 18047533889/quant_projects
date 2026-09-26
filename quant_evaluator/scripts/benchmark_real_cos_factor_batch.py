#!/usr/bin/env python3
"""CPU/CUDA/auto A/B on two COS bound factors over all available A-share sessions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

from quant_evaluator.scripts import benchmark_public_backend_routes as routes
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=0, help="0 means every complete session")
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    start = time.perf_counter()
    factors, labels, provenance = load_real_batch(days=args.days, assets=args.assets)
    load_s = time.perf_counter() - start
    routes._SOURCES = {"base": (factors, labels, {})}
    ctx = mp.get_context("fork")
    measurements = {}
    for metric in METRICS:
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
