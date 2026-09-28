"""Bounded public A/B for the final five factors of the verified F61 COS sample."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_metric_batch as full

AXIS_INDEX = Path(__file__).resolve().parents[1] / "docs/benchmarks/real_cos_f61_axis_index_20260929.json"
OUTPUT = Path(__file__).resolve().parents[1] / "docs/benchmarks/real_cos_f61_tail5_mixed_three_20260929.json"
F61_LABEL_HASH = "42cdc998916cc284d739e0e1423441bee65f5e3a393c5ff6013efbd26363cf2b"
ORDER = ("cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu")
METRICS = full.DEFAULT_METRICS


def select_tail(mapping, index_path=AXIS_INDEX):
    """Bind the last five selected factors to the checked F61 common-axis index."""
    records = tiles.select_records(mapping, 61, 128, 6144)
    dates, assets, sources = tiles.read_axis_index(index_path, full.MANIFEST_SHA256, records)
    if len(records) != 61 or len(sources) != 61:
        raise ValueError("verified F61 selection is incomplete")
    return records[-5:], sources[-5:], dates, assets


def receipt_check(run, expected_backend):
    receipt = run.get("execution_receipt")
    if not isinstance(receipt, dict):
        return False
    actual = run.get("backend_used")
    if (receipt.get("backend_requested") != expected_backend or
            receipt.get("backend_used") != actual or
            receipt.get("config_hash") != run.get("config_hash") or
            not isinstance(receipt.get("receipt_hash"), str) or
            len(receipt["receipt_hash"]) != 64 or
            receipt.get("metric_backends") != run.get("metric_backends")):
        return False
    if expected_backend == "cpu" and actual != "cpu":
        return False
    if expected_backend == "cuda_strict" and actual != "cuda":
        return False
    return isinstance(receipt.get("metric_backends"), dict) and all(
        receipt["metric_backends"].get(metric) == actual for metric in METRICS)


def run_ab(batch, labels, timeout_s):
    full._BATCH, full._LABELS, full.METRICS = batch, labels, METRICS
    context = mp.get_context("fork")
    runs = []
    for round_number, backend in enumerate(ORDER, 1):
        if not tiles.memory_preflight(5, 128, 50)["pass"]:
            raise RuntimeError("host or COS cache resources fell below the F61 tail gate")
        run = full._run_one(context, backend, repeats=2, timeout_s=timeout_s)
        run["round"] = round_number
        runs.append(run)
        print(json.dumps({"round": round_number, "backend_requested": backend,
                          "backend_used": run["backend_used"],
                          "cold_s": run["cold_s"], "warm_median_s": run["warm_median_s"],
                          "peak_vram": run["peak_vram"]}), flush=True)
    reference = runs[0]
    comparisons = [full._compare(reference, run) for run in runs]
    receipts = [receipt_check(run, backend) for run, backend in zip(runs, ORDER)]
    config_hashes = {run["config_hash"] for run in runs}
    return {"runs": runs, "comparisons_to_first_cpu": comparisons,
            "receipt_checks": receipts,
            "parity_pass": (len(config_hashes) == 1 and all(receipts) and
                            all(item["pass"] for item in comparisons))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 30 <= args.timeout_s <= 900:
        parser.error("timeout-s must be 30..900 seconds")
    if args.output.resolve() == AXIS_INDEX.resolve():
        parser.error("output cannot replace the F61 axis index")
    started = time.perf_counter()
    try:
        records, sources, common_dates, common_assets = select_tail(
            tiles.read_manifest(full.MANIFEST_SHA256))
        preflight = tiles.memory_preflight(5, 128, 50)
    except Exception as exc:
        print(json.dumps({"status": "unavailable", "reason":
                          f"{type(exc).__name__}: {exc}"}), flush=True)
        if args.run:
            raise SystemExit(2) from exc
        return
    print(json.dumps({"status": "ready" if preflight["pass"] else "insufficient_resources",
                      "preflight": preflight, "manifest_sha256": full.MANIFEST_SHA256,
                      "axis_index": str(AXIS_INDEX),
                      "factor_ids": [row[0] for row in records],
                      "source_bytes": sum(row[3] for row in records)}), flush=True)
    if not args.run:
        return
    if not preflight["pass"]:
        raise SystemExit(2)
    dates, assets, labels = tiles.load_labels(common_dates, common_assets, 0, 5500)
    if (len(dates), len(assets)) != (2586, 5461) or labels.content_hash != F61_LABEL_HASH:
        raise ValueError("F61 decision/asset axes or label content changed")
    if not tiles.memory_preflight(5, 128, 50)["pass"]:
        raise RuntimeError("host or COS cache resources changed before factor reads")
    batch = tiles.make_tile(
        tiles.iter_frames(records, full.MANIFEST_SHA256, 128), sources,
        dates, assets, labels, required_dates=common_dates,
        required_assets=common_assets)
    if (batch.num_times, batch.num_assets, batch.num_factors) != (2586, 5461, 5):
        raise ValueError("F61 tail FactorBatch shape changed")
    if tuple(batch.factor_ids) != tuple(row[0] for row in records):
        raise ValueError("F61 tail factor order changed")
    ab = run_ab(batch, labels, args.timeout_s)
    report = {"status": "complete", "kind": "research_f61_tail5_mixed_three_ab.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "manifest_sha256": full.MANIFEST_SHA256,
              "axis_index_body_sha256": json.loads(AXIS_INDEX.read_text())["body_sha256"],
              "label_content_hash": labels.content_hash,
              "factor_ids": list(batch.factor_ids), "sources": sources,
              "shape": [2586, 5461, 5], "metric_ids": METRICS,
              "backend_order": ORDER, "preflight": preflight,
              "elapsed_s": time.perf_counter() - started,
              "peak_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              **ab}
    tiles.write_collection_atomic(args.output, full._plain(report))
    print(json.dumps({"parity_pass": ab["parity_pass"], "output": str(args.output),
                      "auto_routes": [run["backend_used"] for run in ab["runs"]
                                      if run["backend_requested"] == "auto"]}), flush=True)
    if not ab["parity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
