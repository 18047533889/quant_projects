#!/usr/bin/env python3
"""Adjacent-shape CPU/CUDA public evaluate tournament on bound real COS factors."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time
import numpy as np
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.scripts import benchmark_public_backend_routes as routes
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

METRICS = ("rank_ic", "quantile_spread")
DEFAULT_SHAPES = (
    (1000, 5000, 1), (1000, 5461, 2),
    (1800, 5000, 2), (2400, 5461, 1),
    (2400, 5000, 2), (2586, 5461, 2),
)

def _slice(batch, labels, t, n, f):
    if t > batch.num_times or n > batch.num_assets or f > batch.num_factors:
        raise ValueError("requested shape exceeds COS input")
    ta = AxisRef("time", batch.time_axis.dtype, t, batch.time_axis.values[-t:])
    aa = AxisRef("asset", batch.asset_axis.dtype, n, batch.asset_axis.values[:n])
    factor = FactorBatch(batch.factor_ids[:f], ta, aa,
        np.ascontiguousarray(batch.values[-t:,:n,:f]),
        validity=np.ascontiguousarray(batch.validity[-t:,:n,:f]))
    updates = dict(values=np.ascontiguousarray(labels.values[-t:,:n]),
        asset_axis=aa, validity=np.ascontiguousarray(labels.validity[-t:,:n]))
    for field in ("decision_time","observation_time","signal_available_time",
                  "execution_time","label_start_time","label_end_time"):
        updates[field] = tuple(getattr(labels, field)[-t:])
    return factor, replace(labels, **updates)

def _write(path, report):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--shape", action="append", default=[],
                        help="repeatable T,N,F; defaults to six representative shapes")
    parser.add_argument("--limit", type=int, default=6,
                        help="maximum shapes to run (1..16)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 16:
        parser.error("--limit must be 1..16")
    if args.shape:
        try:
            shapes = [tuple(map(int, item.split(","))) for item in args.shape]
        except ValueError as exc:
            parser.error(f"invalid --shape: {exc}")
        if any(len(item) != 3 for item in shapes):
            parser.error("each --shape must be T,N,F")
    else:
        shapes = DEFAULT_SHAPES
    shapes = shapes[:args.limit]
    began = time.perf_counter()
    batch, labels, provenance = load_real_batch(days=0, assets=5500)
    report = {
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_provenance":provenance,
        "input_load_s":time.perf_counter()-began,
        "host_parent_peak_rss_kib":None,
        "shapes":[],
        "note":"one cold and one warm evaluation per backend; process order alternated",
    }
    ctx = mp.get_context("fork")
    for ordinal, (t,n,f) in enumerate(shapes):
        sub_batch, sub_labels = _slice(batch, labels, t, n, f)
        routes._SOURCES = {"base":(sub_batch, sub_labels, {})}
        metrics = {}
        for metric in METRICS:
            order = ("cpu","cuda_strict") if ordinal % 2 else ("cuda_strict","cpu")
            runs = {}
            for backend in order:
                result = routes.run_one(ctx,metric,backend,1,args.timeout)
                if result.get("status") != "ok":
                    raise RuntimeError(f"shape={(t,n,f)} metric={metric} backend={backend}: {result}")
                runs[backend] = result
            parity = routes.parity(runs["cpu"],runs["cuda_strict"])
            if not parity["pass"]:
                raise AssertionError(f"CPU/CUDA public output parity failed: {(t,n,f)} {metric} {parity}")
            metrics[metric] = {
                "cpu":{"cold_s":runs["cpu"]["cold_s"],"warm_s":runs["cpu"]["warm_median_s"],
                       "peak_rss_kib":runs["cpu"]["peak_rss_kib"]},
                "cuda":{"cold_s":runs["cuda_strict"]["cold_s"],
                        "warm_s":runs["cuda_strict"]["warm_median_s"],
                        "peak_rss_kib":runs["cuda_strict"]["peak_rss_kib"],
                        "peak_vram":runs["cuda_strict"]["gpu_peak_vram"]},
                "parity":parity,
            }
            print(f"T={t} N={n} F={f} {metric} "
                  f"cpu={metrics[metric]['cpu']['warm_s']:.3f}s "
                  f"cuda={metrics[metric]['cuda']['warm_s']:.3f}s "
                  f"vram={metrics[metric]['cuda']['peak_vram']} "
                  f"parity={parity['pass']}",flush=True)
        report["shapes"].append({"t":t,"n":n,"f":f,"metrics":metrics})
        report["host_parent_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write(args.output,report)
    print(json.dumps({"shapes":len(report["shapes"]),
        "all_parity":all(m["parity"]["pass"] for s in report["shapes"]
                         for m in s["metrics"].values()),
        "input_load_s":report["input_load_s"]}),flush=True)

if __name__ == "__main__":
    main()
