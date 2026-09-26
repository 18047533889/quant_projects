#!/usr/bin/env python3
"""Read-only full-report A/B on manifest-bound COS factors.

Writes only bounded timing/parity metadata when --output is provided.  Raw
factor values, labels, and ReportEvaluation arrays stay in process memory.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import re
import resource
import time

import numpy as np

from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_batch
from quant_evaluator.scripts.load_real_cos_factor_batch import load_real_batch


def _manifest_sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError(
            "must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _stop_worker(process: mp.Process) -> None:
    """Stop a stuck worker with bounded TERM/KILL waits."""
    if not process.is_alive():
        return
    process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    if process.is_alive():
        raise RuntimeError(f"worker pid {process.pid} did not stop after terminate/kill")


def _worker(conn, values, labels, options, backend):
    try:
        runs = []
        for repeat in ("cold", "warm"):
            started = time.perf_counter()
            result = evaluate_report_batch(values, labels, backend=backend, **options)
            runs.append({
                "repeat": repeat,
                "seconds": round(time.perf_counter() - started, 3),
                "backend_used": result.backend_used,
                "ic_numerics": result.ic_numerics,
                "selection_reason": result.backend_selection_reason,
                "fallback_reason": result.backend_fallback_reason,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            })
        conn.send(("ok", runs, result))
    except Exception as error:
        conn.send(("error", type(error).__name__, str(error)))
    finally:
        conn.close()


def _parity(reference, actual, factor_ids):
    failed = []
    max_ic_diff = 0.0
    max_quantile_diff = 0.0
    for factor_id in factor_ids:
        left = reference.factors[factor_id]
        right = actual.factors[factor_id]
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                same = np.allclose(a, b, equal_nan=True, rtol=1e-8, atol=1e-10)
                difference = a - b
                if np.isfinite(difference).any():
                    delta = float(np.nanmax(np.abs(difference)))
                    if field.name == "rank_ic_series":
                        max_ic_diff = max(max_ic_diff, delta)
                    elif field.name == "quantile_returns":
                        max_quantile_diff = max(max_quantile_diff, delta)
            elif isinstance(a, float):
                same = bool(np.isclose(a, b, equal_nan=True, rtol=1e-8, atol=1e-10))
            else:
                same = a == b
            if not same:
                failed.append([factor_id, field.name])
    return {
        "fields_per_factor": len(fields(reference.factors[factor_ids[0]])),
        "checked_fields": [field.name for field in fields(reference.factors[factor_ids[0]])],
        "field_comparisons": len(factor_ids) * len(fields(reference.factors[factor_ids[0]])),
        "factor_count": len(factor_ids),
        "rtol": 1e-8,
        "atol": 1e-10,
        "failed": failed,
        "max_rank_ic_absolute_difference": max_ic_diff,
        "max_quantile_return_absolute_difference": max_quantile_diff,
        "directions": {name: actual.factors[name].direction for name in factor_ids},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=2, help="manifest-bound factors, 1..16")
    parser.add_argument("--days", type=int, default=0, help="0 selects every complete session")
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--manifest-sha256", type=_manifest_sha256, required=True)
    parser.add_argument("--max-object-mib", type=int, default=64)
    parser.add_argument("--max-total-mib", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    factors, labels, provenance = load_real_batch(
        factors=args.factors, days=args.days, assets=args.assets,
        max_object_mib=args.max_object_mib, max_total_mib=args.max_total_mib,
        manifest_sha256=args.manifest_sha256,
    )
    load_seconds = round(time.perf_counter() - started, 3)
    factor_ids = tuple(factors.factor_ids)
    training_periods = int((np.asarray(factors.time_axis.values, dtype="datetime64[ns]")
                            <= np.datetime64("2018-06-30")).sum())
    options = dict(
        factor_ids=factor_ids, n_quantiles=10, min_assets=20,
        min_ic_periods=20, direction_training_periods=training_periods,
        commission_rate=0.0001,
    )
    context = mp.get_context("fork")
    results, runs = {}, {}
    for backend in ("numba", "cuda_strict", "cpu", "auto"):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker, args=(child, factors.values, labels.values, options, backend))
        try:
            process.start()
            child.close()
            if not parent.poll(300):
                raise TimeoutError(f"{backend} exceeded 300 seconds")
            payload = parent.recv()
            process.join(10)
            if process.is_alive():
                _stop_worker(process)
                raise TimeoutError(f"{backend} did not exit after returning a result")
            if payload[0] != "ok" or process.exitcode != 0:
                raise RuntimeError(f"{backend}: {payload}; exit={process.exitcode}")
        finally:
            child.close()
            parent.close()
            if process.pid is not None:
                if process.is_alive():
                    _stop_worker(process)
                process.close()
        _, runs[backend], results[backend] = payload
        print(f"{backend}: {runs[backend]}", flush=True)
    parity = {backend: _parity(results["cpu"], results[backend], factor_ids)
              for backend in ("numba", "cuda_strict", "auto")}
    if any(result["failed"] for result in parity.values()):
        raise AssertionError(f"report parity failed: {parity}")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": "DataAccess manifest-bound COS factor values and registered AdjVwap",
            "manifest_sha256": args.manifest_sha256,
            "max_object_mib": args.max_object_mib,
            "max_total_mib": args.max_total_mib,
            "factors": [{key: source[key] for key in ("factor_id", "sha256", "bytes")}
                        for source in provenance["sources"]],
            "shape": list(factors.values.shape),
            "date_span": provenance["date_span"],
            "label": provenance["label"],
            "factor_finite_ratio": provenance["factor_finite_ratio"],
            "label_finite_ratio": provenance["label_finite_ratio"],
            "missing_adj_vwap_partitions": provenance["missing_adj_vwap_partitions"],
        },
        "report_parameters": {
            "training_end": "2018-06-30", "training_periods": training_periods,
            "n_quantiles": 10, "min_assets": 20, "min_ic_periods": 20,
            "commission_rate": 0.0001,
        },
        "load_seconds": load_seconds,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runs": runs,
        "parity_vs_exact_cpu": parity,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({"shape": summary["source"]["shape"],
                      "parity": {key: not value["failed"] for key, value in parity.items()}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
