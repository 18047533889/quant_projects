# -*- coding: utf-8 -*-
"""Batch 3 follow-up: numerical verification of the three intraday kernels on
REAL minute data.

Why a separate harness: the generic daily harness drives a one-row-per-day panel,
for which these kernels are vacuously empty (they aggregate intraday moments per
session).  Here we build a wide MINUTE panel from ``~/cos_data/StockMinuteBarAdj``
(N symbols x 5 sessions x 240 bars) and compare the registered polars kernel with
the certified ``pandas_numpy`` reference.

Both sides return a DAILY wide panel, so the comparison aligns on
(normalised date index x symbol column names) -- never on raw array position, and
never on integer position, because a pivoting kernel is free to order its output
columns differently from its input.

Checks:
  1. NaN MASK identity
  2. max absolute deviation over co-finite cells
  3. IDEMPOTENCY (same input twice -> bitwise-identical output)
  4. runtime marshal probe (pl.DataFrame.to_pandas calls, must be 0)
  5. classification flip (canonical_polars_kind production_mode=True)
"""
from __future__ import annotations

import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
REPO = "/home/sunhaiwei/quant_projects"
MINUTE = os.path.expanduser("~/cos_data/StockMinuteBarAdj")
DAYS = int(os.environ.get("INTRADAY_DAYS", "5"))
NSYM = int(os.environ.get("INTRADAY_NSYM", "20"))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import canonical_polars_kind  # noqa: E402

OPS = ["intra_realized_variance", "intra_realized_skewness", "intra_realized_kurtosis"]
IDX_COLS = ("date", "timestamp", "time", "QuoteTime", "TradeDate")

load_all()

PROBE = {"n": 0}
_orig_tp = pl.DataFrame.to_pandas


def _counting(self, *a, **k):
    PROBE["n"] += 1
    return _orig_tp(self, *a, **k)


def load_minute_panel():
    files = sorted(glob.glob(os.path.join(MINUTE, "*.parquet")))[:DAYS]
    if not files:
        raise SystemExit(f"no minute parquet under {MINUTE}")
    syms = (pl.read_parquet(files[-1]).group_by("Symbol").len()
            .sort("len", descending=True).head(NSYM)["Symbol"].to_list())
    long = (pl.scan_parquet(files).select(["QuoteTime", "Symbol", "AdjClose"])
            .filter(pl.col("Symbol").is_in(syms)).collect())
    long = long.with_columns(
        pl.col("QuoteTime").dt.convert_time_zone("Asia/Shanghai")
        .dt.replace_time_zone(None).alias("date")
    ).select(["date", "Symbol", "AdjClose"]).drop_nulls().sort("date")
    wide = (long.pivot(values="AdjClose", index="date", on="Symbol")
            .sort("date").select(["date", *syms]))
    pdf = wide.to_pandas().set_index("date")
    pdf.index = pd.Index(pd.to_datetime(pdf.index))
    pdf.index.name = "date"
    meta = {"days": DAYS, "symbols": len(syms), "minute_bars": int(wide.height),
            "sessions_from": str(pdf.index[0]), "sessions_to": str(pdf.index[-1]),
            "source": MINUTE}
    return pdf, wide, meta, list(syms)


def as_frame(obj):
    """Normalise a daily wide result to a pandas frame indexed by normalised date."""
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, pl.DataFrame):
        idx = [c for c in obj.columns if c in IDX_COLS]
        if idx:
            labels = pd.to_datetime(obj[idx[0]].to_pandas())
            df = obj.drop(idx[0]).to_pandas()
            df.index = pd.Index(labels)
        else:
            df = obj.to_pandas()
    else:
        raise TypeError(type(obj))
    df.columns = [str(c) for c in df.columns]
    df.index = pd.Index(pd.to_datetime(df.index).normalize())
    return df.sort_index()


def main():
    pdf, wide, meta, syms = load_minute_panel()
    print("MINUTE PANEL", json.dumps(meta), flush=True)
    results = []
    pl.DataFrame.to_pandas = _counting
    try:
        for canon in OPS:
            rec = {"operator": canon}
            try:
                pop = OperatorRegistry.get(canon, "pandas_numpy")
                plop = OperatorRegistry.get(canon, "polars", mode="any")
                kind = canonical_polars_kind(canon, production_mode=True)
                rec["polars_kind"] = getattr(kind, "value", str(kind))

                ref = pop.calculate(pdf)
                PROBE["n"] = 0
                t0 = time.perf_counter()
                got1 = plop._calculate_series(wide)
                rec["_polars_s"] = time.perf_counter() - t0
                rec["to_pandas_calls"] = PROBE["n"]

                t1 = time.perf_counter()
                got2 = plop._calculate_series(wide)
                rec["_polars_2nd_s"] = time.perf_counter() - t1

                t2 = time.perf_counter()
                pop.calculate(pdf)
                rec["_pandas_s"] = time.perf_counter() - t2

                rf, nf1, nf2 = as_frame(ref), as_frame(got1), as_frame(got2)
                rec["shape_ref_raw"] = list(rf.shape)
                rec["shape_new_raw"] = list(nf1.shape)
                rec["cols_ref_sorted"] = sorted(map(str, rf.columns)) == sorted(map(str, nf1.columns))
                # align on the (date x symbol) intersection of BOTH sides
                idx = sorted(set(rf.index) & set(nf1.index))
                order = sorted(set(rf.columns) & set(nf1.columns))
                rec["n_rows_aligned"] = len(idx)
                rec["n_cols_aligned"] = len(order)
                if not idx or not order:
                    raise AssertionError(
                        f"no overlap: ref index {list(rf.index)[:2]} vs new {list(nf1.index)[:2]}")
                a = rf.reindex(index=idx, columns=order).to_numpy(dtype=float)
                b = nf1.reindex(index=idx, columns=order).to_numpy(dtype=float)
                c = nf2.reindex(index=idx, columns=order).to_numpy(dtype=float)

                rec["nan_mask_identical"] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
                fin = np.isfinite(a) & np.isfinite(b)
                rec["n_finite"] = int(fin.sum())
                rec["finite_frac_ref"] = round(float(np.isfinite(a).mean()), 6)
                rec["max_abs_dev"] = (float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None)
                rec["idempotent"] = bool(
                    np.array_equal(np.isnan(c), np.isnan(b)) and np.array_equal(c[fin], b[fin]))
                rec["ok"] = True
            except Exception as e:  # pragma: no cover
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            results.append(rec)
            print(json.dumps(rec, default=str), flush=True)
    finally:
        pl.DataFrame.to_pandas = _orig_tp

    ok = [r for r in results if r.get("ok")]
    summary = {
        "panel": meta,
        "n_operators": len(results),
        "n_ok": len(ok),
        "all_nan_mask_identical": all(r["nan_mask_identical"] for r in ok) if ok else None,
        "all_idempotent": all(r["idempotent"] for r in ok) if ok else None,
        "all_zero_marshal": all(r["to_pandas_calls"] == 0 for r in ok) if ok else None,
        "all_flipped_to_polars_native": all(r.get("polars_kind") == "polars_native" for r in ok) if ok else None,
        "total_finite_cells_compared": sum(r["n_finite"] for r in ok),
        "max_abs_dev_over_all": max([r["max_abs_dev"] or 0.0 for r in ok], default=None),
        "failures": [r["operator"] for r in results if not r.get("ok")],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=1, default=str))
    out = os.environ.get("INTRADAY_OUT") or f"{REPO}/evidence/_verify_batch3_intraday.json"
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1, default=str)
    print("WROTE", out)


if __name__ == "__main__":
    main()
