#!/usr/bin/env python3
"""Bounded public auto-route verification on the bound real COS factor panel."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.scripts.benchmark_real_cos_route_region import _slice
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch

OUTPUT = Path("/home/sunhaiwei/quant_projects/quant_evaluator/docs/benchmarks/public_auto_real_cos_verification_20260927.json")
METRICS = ("rank_ic", "quantile_spread")


def _run(batch, labels, metric, backend):
    started = time.perf_counter()
    if backend == "default":
        bundle = evaluate(batch, labels, metrics=(metric,))
    else:
        bundle = evaluate(batch, labels, metrics=(metric,), backend=backend)
    receipt = bundle.metadata["execution_receipt"]
    return bundle, {
        "elapsed_s": time.perf_counter() - started,
        "backend_requested": bundle.metadata["backend_requested"],
        "backend_strategy": bundle.metadata["backend_strategy"],
        "backend_used": bundle.metadata["backend_used"],
        "auto_backend_policy": bundle.metadata["auto_backend_policy"],
        "auto_backend_profile": bundle.metadata["auto_backend_profile"],
        "auto_backend_reason": bundle.metadata["auto_backend_reason"],
        "config_hash": bundle.config_hash,
        "receipt_hash": receipt["receipt_hash"],
        "receipt_config_matches": receipt["config_hash"] == bundle.config_hash,
        "peak_vram": bundle.metadata.get("peak_vram"),
    }


def _compare(left, right, metric, factor_ids):
    la, ra = left.artifacts[metric], right.artifacts[metric]
    same_shape = la.values.shape == ra.values.shape
    same_finite = same_shape and bool(np.array_equal(np.isfinite(la.values), np.isfinite(ra.values)))
    numeric = same_shape and bool(np.allclose(
        la.values, ra.values, rtol=1e-8, atol=1e-10, equal_nan=True))
    counts = (getattr(la, "counts", None), getattr(ra, "counts", None))
    same_counts = all(v is None for v in counts) or (
        all(v is not None for v in counts) and bool(np.array_equal(*counts)))
    masks = (getattr(la, "valid_mask", None), getattr(ra, "valid_mask", None))
    same_mask = all(v is None for v in masks) or (
        all(v is not None for v in masks) and bool(np.array_equal(*masks)))
    same_values = all(
        left.get_metric(metric, fid).valid == right.get_metric(metric, fid).valid
        and left.get_metric(metric, fid).observation_count == right.get_metric(metric, fid).observation_count
        and np.isclose(left.get_metric(metric, fid).value,
                       right.get_metric(metric, fid).value,
                       rtol=1e-8, atol=1e-10, equal_nan=True)
        for fid in factor_ids
    )
    same_config = left.config_hash == right.config_hash
    return {
        "pass": bool(all((same_shape, same_finite, numeric, same_counts,
                           same_mask, same_values, same_config))),
        "same_shape": bool(same_shape),
        "same_finite": bool(same_finite),
        "numeric_1e-8_1e-10": bool(numeric),
        "same_counts": bool(same_counts),
        "same_mask": bool(same_mask),
        "same_metric_values": bool(same_values),
        "same_config_hash": bool(same_config),
    }


def main():
    started = time.perf_counter()
    batch, labels, provenance = load_real_batch(days=0, assets=5500)
    assert (batch.num_times, batch.num_assets, batch.num_factors) == (2586, 5461, 2)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "DataAccess bound COS landing manifest and registered AdjVwap",
        "input_shape": [batch.num_times, batch.num_assets, batch.num_factors],
        "input_load_s": time.perf_counter() - started,
        "source_manifest_sha256": [s["manifest_sha256"] for s in provenance["sources"]],
        "full_panel": {},
        "cold_start_edge": {},
    }
    for metric in METRICS:
        bundles, runs = {}, {}
        for backend in ("cpu", "cuda_strict", "auto", "default"):
            bundles[backend], runs[backend] = _run(batch, labels, metric, backend)
            print(f"full {metric} {backend}: {runs[backend]['elapsed_s']:.3f}s "
                  f"{runs[backend]['backend_used']}", flush=True)
        parity = {other: _compare(bundles["cpu"], bundles[other], metric, batch.factor_ids)
                  for other in ("cuda_strict", "auto", "default")}
        assert all(item["pass"] for item in parity.values()), (metric, parity)
        assert all(runs[b]["backend_used"] == "cuda" for b in ("cuda_strict", "auto", "default"))
        assert all(runs[b]["auto_backend_profile"] == "real_cos_region_20260927"
                   for b in ("auto", "default"))
        assert all(item["receipt_config_matches"] for item in runs.values())
        report["full_panel"][metric] = {"runs": runs, "parity_vs_cpu": parity}
    edge_batch, edge_labels = _slice(batch, labels, 1000, 5000, 1)
    bundles, runs = {}, {}
    for backend in ("cpu", "auto", "default"):
        bundles[backend], runs[backend] = _run(edge_batch, edge_labels, "rank_ic", backend)
        print(f"edge rank_ic {backend}: {runs[backend]['elapsed_s']:.3f}s "
              f"{runs[backend]['backend_used']}", flush=True)
    parity = {other: _compare(bundles["cpu"], bundles[other], "rank_ic", edge_batch.factor_ids)
              for other in ("auto", "default")}
    assert all(item["pass"] for item in parity.values()), parity
    assert all(runs[b]["backend_used"] == "cpu" for b in ("auto", "default"))
    assert all(runs[b]["auto_backend_reason"] == "shape_outside_certified_range"
               for b in ("auto", "default"))
    report["cold_start_edge"] = {"shape": [1000, 5000, 1], "metric": "rank_ic",
                                 "runs": runs, "parity_vs_cpu": parity}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
