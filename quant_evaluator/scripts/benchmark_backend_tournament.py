"""Bounded IC backend benchmark. Run as a module from the repository root.

This measures the public compute_daily_ic kernel API, not evaluate() facade.
Panels are reproducible synthetic stock-shaped data, never claimed real COS.
Each backend gets an isolated process, a timeout, and identical panel content.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from datetime import date, timedelta
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback

import numpy as np


BACKENDS = ("exact", "numba", "polars", "gpu")
_REAL_REGISTERED_PANEL = None


def panel(t, n, f, seed):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    rng = np.random.default_rng(seed)
    values = rng.standard_normal((t, n, f), dtype=np.float32)
    labels = rng.standard_normal((t, n), dtype=np.float32)
    values += labels[:, :, None] * np.float32(0.03)
    np.multiply(np.round(values * 20), 0.05, out=values)
    values[rng.random(values.shape) < 0.04] = np.nan
    labels[rng.random(labels.shape) < 0.03] = np.nan
    fb = FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(f)),
        time_axis=AxisRef("t", "int", t),
        asset_axis=AxisRef("a", "str", n), values=values,
    )
    lb = LabelBundle(
        target_id="r", values=labels, horizon=1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
    )
    return fb, lb


def real_cos_panel():
    """Use only the existing bounded, authorized DataAccess sample loader."""
    examples = Path(__file__).resolve().parents[2] / "factor_optimizer" / "examples"
    sys.path.insert(0, str(examples))
    from cos_batch_audit import load_cos_sample
    fb, lb, provenance = load_cos_sample(n_factors=2, n_assets=256)
    return fb, lb, {"days": fb.num_times, "assets": fb.num_assets,
                    "factors": len(fb.factor_ids),
                    "manifest_uri": provenance["manifest_uri"],
                    "date_span": provenance["date_span"]}


def real_registered_panel(start=date(2023, 9, 1), end=date(2026, 8, 31)):
    """Read bounded chunks of the configured adjusted daily stock dataset."""
    import pandas as pd
    from data_access import get_store
    from data_access.read.query_budget import QueryBudget
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    mirror_root = os.environ.get("ASHARE_PARQUET_ROOT")
    if not mirror_root:
        raise ValueError("ASHARE_PARQUET_ROOT must name the configured local mirror")
    local_dir = Path(mirror_root) / "StockDailyBarAdj"
    file_days = sorted(
        date.fromisoformat(p.stem)
        for p in local_dir.glob("*.parquet")
        if len(p.stem) == 10 and start.isoformat() <= p.stem <= end.isoformat()
    )
    if not 40 <= len(file_days) <= 800:
        raise ValueError(f"expected 40..800 existing local daily files, got {len(file_days)}")
    # Only request dates represented by existing local partitions. Split at
    # every absent calendar day so DataAccess never has to treat a holiday as
    # a required missing object. The registered read still validates scope.
    groups = []
    for day in file_days:
        if not groups or (day - groups[-1][-1]).days != 1 or len(groups[-1]) >= 7:
            groups.append([day])
        else:
            groups[-1].append(day)
    pieces = []
    source_rows = 0
    store = get_store()
    for group in groups:
        table = store.read(
            "ashare_stock_daily_adj",
            columns=["TradeDate", "Symbol", "AdjVwap"],
            time_range=(group[0].isoformat(), group[-1].isoformat()),
            result="arrow",
            query_budget=QueryBudget(
                max_scan_files=7, max_rows=40_000,
                max_result_bytes=4 * 1024**2, max_elapsed_ms=60_000,
                require_columns=True, require_time_range=True,
            ),
        ).to_arrow()
        source_rows += table.num_rows
        if table.num_rows:
            chunk = table.to_pandas().pivot(
                index="TradeDate", columns="Symbol", values="AdjVwap",
            )
            pieces.append(chunk)
    if not pieces:
        raise ValueError("registered adjusted daily dataset returned no rows")
    panel = pd.concat(pieces).sort_index()
    if panel.index.has_duplicates or panel.columns.has_duplicates:
        raise ValueError("registered adjusted daily dataset has duplicate keys")
    prices = panel.to_numpy(dtype=np.float64)
    prices[~np.isfinite(prices) | (prices <= 0)] = np.nan
    if len(prices) < 40 or prices.shape[1] < 1000:
        raise ValueError(f"insufficient real panel: {prices.shape}")
    idx = np.arange(21, len(prices) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Both signals stop at t-1; labels use t to t+1.
        factor5 = prices[idx - 1] / prices[idx - 6] - 1
        factor20 = prices[idx - 1] / prices[idx - 21] - 1
        labels = prices[idx + 1] / prices[idx] - 1
    values = np.stack((factor5, factor20), axis=-1)
    values[~np.isfinite(values)] = np.nan
    labels[~np.isfinite(labels)] = np.nan
    days = pd.to_datetime(panel.index).to_numpy(dtype="datetime64[ns]")
    selected_days = days[idx]
    next_days = days[idx + 1]
    symbols = np.asarray(panel.columns, dtype=str)
    fb = FactorBatch(
        factor_ids=("lagged_momentum_5d", "lagged_momentum_20d"),
        time_axis=AxisRef("time", "datetime64[ns]", len(idx), selected_days),
        asset_axis=AxisRef("asset", "str", len(symbols), symbols),
        values=np.ascontiguousarray(values),
    )
    lb = LabelBundle(
        target_id="next_day_adjusted_vwap_return",
        values=np.ascontiguousarray(labels), horizon=1,
        decision_time=tuple(selected_days),
        label_start_time=tuple(selected_days),
        label_end_time=tuple(next_days),
    )
    return fb, lb, {
        "dataset": "ashare_stock_daily_adj", "source_rows": source_rows,
        "local_partition_days": len(file_days), "read_groups": len(groups),
        "gaps_over_three_calendar_days": sum(
            (b - a).days > 3 for a, b in zip(file_days, file_days[1:])
        ),
        "calendar_days": len(prices), "days": len(idx),
        "assets": len(symbols), "factors": 2,
        "date_span": [str(panel.index.min()), str(panel.index.max())],
        "signal": "AdjVwap[t-1]/AdjVwap[t-6 or t-21]-1",
        "label": "AdjVwap[t+1]/AdjVwap[t]-1",
    }


def worker(conn, case, method, backend, seed, repeats):
    try:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        start = time.perf_counter()
        if case == "real_registered":
            fb, lb, source = _REAL_REGISTERED_PANEL
        elif case == "real_cos":
            fb, lb, source = real_cos_panel()
        else:
            t, n, f = case
            fb, lb = panel(t, n, f, seed)
            source = {"days": t, "assets": n, "factors": f}
        constructed = time.perf_counter()
        from quant_evaluator.metrics.ic import compute_daily_ic
        times = []
        for _ in range(repeats + 1):
            tick = time.perf_counter()
            ic, counts = compute_daily_ic(
                fb, lb, method=method, min_assets=20, backend=backend,
            )
            times.append(time.perf_counter() - tick)
        conn.send({
            "status": "ok", "construct_s": constructed - start,
            "source": source,
            "cold_s": times[0], "warm_s": times[1:],
            "warm_median_s": statistics.median(times[1:]),
            "end_to_end_cold_s": constructed - start + times[0],
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "ic": np.asarray(ic, dtype=np.float64),
            "counts": np.asarray(counts, dtype=np.int32),
        })
    except BaseException as exc:
        conn.send({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=5)})
    finally:
        conn.close()


def run_one(ctx, case, method, backend, seed, repeats, timeout_s):
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=worker, args=(sender, case, method, backend, seed, repeats),
    )
    process.start()
    sender.close()
    if receiver.poll(timeout_s):
        try:
            answer = receiver.recv()
        except EOFError:
            answer = {"status": "crash", "exitcode": process.exitcode}
    else:
        answer = {"status": "timeout", "timeout_s": timeout_s}
        process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    receiver.close()
    return answer


def parity(ref, candidate):
    x, y = ref["ic"], candidate["ic"]
    same_finite = bool(np.array_equal(np.isfinite(x), np.isfinite(y)))
    same_counts = bool(np.array_equal(ref["counts"], candidate["counts"]))
    finite = np.isfinite(x) & np.isfinite(y)
    close = bool(np.allclose(x, y, rtol=1e-8, atol=1e-10, equal_nan=True))
    return {
        "pass": same_finite and same_counts and close,
        "same_finite": same_finite, "same_counts": same_counts,
        "allclose_1e-8_1e-10": close,
        "max_abs_ic": float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="120x300x1,1250x5000x1")
    ap.add_argument("--methods", default="pearson,spearman")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=180)
    ap.add_argument("--max-input-gib", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=260926)
    ap.add_argument("--real-cos", action="store_true",
                    help="also read the bounded existing DataAccess COS sample")
    ap.add_argument("--real-registered", action="store_true",
                    help="read three years of registered adjusted daily stocks in bounded chunks")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cases = ([tuple(map(int, spec.split("x"))) for spec in args.cases.split(",")]
             if args.cases else [])
    if args.real_cos:
        cases.append("real_cos")
    if args.real_registered:
        cases.append("real_registered")
    methods, backends = args.methods.split(","), args.backends.split(",")
    if any(len(c) != 3 or min(c) < 1 for c in cases if not isinstance(c, str)):
        ap.error("cases must be positive TxNxF triples")
    if any(m not in ("pearson", "spearman") for m in methods):
        ap.error("unsupported method")
    if any(b not in BACKENDS for b in backends) or "exact" not in backends:
        ap.error("backends must include exact")
    if args.repeats < 1 or args.timeout_s <= 0 or args.max_input_gib <= 0:
        ap.error("repeats, timeout-s and max-input-gib must be positive")
    for t, n, f in (c for c in cases if not isinstance(c, str)):
        input_gib = t * n * (f + 1) * 4 / 2**30
        if input_gib > args.max_input_gib:
            ap.error(f"{t}x{n}x{f} input {input_gib:.2f} GiB exceeds cap")
    ctx = mp.get_context("fork" if args.real_registered else "spawn")
    source_load_s = None
    if args.real_registered:
        global _REAL_REGISTERED_PANEL
        tick = time.perf_counter()
        _REAL_REGISTERED_PANEL = real_registered_panel()
        source_load_s = time.perf_counter() - tick
    script_path = Path(__file__).resolve()
    polars_path = script_path.parents[1] / "backends" / "polars_backend.py"
    report = {"kind": "registered_real_and_or_synthetic_ic_kernel_tournament"
              if args.real_registered else
              ("mixed_real_cos_and_synthetic_ic_kernel_tournament" if args.real_cos else
               "synthetic_ic_kernel_tournament"), "seed": args.seed,
              "registered_source_load_s": source_load_s,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {
                  "benchmark_script": hashlib.sha256(script_path.read_bytes()).hexdigest(),
                  "polars_backend": hashlib.sha256(polars_path.read_bytes()).hexdigest(),
              },
              "repeats": args.repeats, "timeout_s": args.timeout_s,
              "parity": {"rtol": 1e-8, "atol": 1e-10,
                         "counts_and_finite_mask_exact": True}, "cases": {}}
    for case in cases:
        for method in methods:
            key = (case if isinstance(case, str) else
                   f"{case[0]}x{case[1]}x{case[2]}") + f"_{method}"
            report["cases"][key] = {}
            raw = {}
            for backend in backends:
                print(f"{key} {backend}: running", flush=True)
                answer = run_one(ctx, case, method, backend, args.seed,
                                 args.repeats, args.timeout_s)
                raw[backend] = answer
                report["cases"][key][backend] = {
                    k: v for k, v in answer.items()
                    if k not in ("ic", "counts", "traceback")
                }
                print(f"{key} {backend}: {report['cases'][key][backend]}", flush=True)
            if raw["exact"]["status"] == "ok":
                for backend in backends:
                    if raw[backend]["status"] == "ok":
                        report["cases"][key][backend]["parity_vs_exact"] = parity(
                            raw["exact"], raw[backend],
                        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
