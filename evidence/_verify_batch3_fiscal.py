# -*- coding: utf-8 -*-
"""Batch 3 follow-up: numerical verification of ``fin_quarter_from_cumulative``.

Why a separate harness: the generic daily harness passes float price panels for
every operand, but this operator's contract is
``(x: cumulative_flow, period_id: fiscal label, fiscal_quarter: 1..4)``.  Feeding
it prices makes BOTH sides degenerate and the comparison vacuous (the first run
recorded ``nan_mask_identical=False`` on an all-NaN reference, i.e. an
unverifiable case), so it must be driven with a structurally valid fiscal input.

Input construction (all on REAL axes):
  * real symbols and the real daily trading calendar (25 x 972, 2019-2022) taken
    from ~/cos_data/StockDailyBarAdj, so the period walk runs over real dates;
  * the per-quarter flow driver is the real quarterly SUM of AdjAmount for each
    symbol (a genuine positive cash-flow-like series);
  * ``x`` = YTD cumulative of that quarterly flow (resets each fiscal year),
    ``period_id`` = the quarter label of each date, ``fiscal_quarter`` = 1..4.

Caveat stated plainly: the VALUE driver is real-but-not-a-financial-statement
(quarterly traded amount, not reported revenue).  This verifies the fiscal-ordinal
period walk and the equivalence of the two implementations on real axes; it is
NOT a restatement-parity test against reported filings.
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
DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")
START = os.environ.get("VER_START", "2019-01-01")
END = os.environ.get("VER_END", "2022-12-30")
NSYM = int(os.environ.get("VER_NSYM", "25"))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import canonical_polars_kind  # noqa: E402

CANON = "fin_quarter_from_cumulative"
load_all()

PROBE = {"n": 0}
_orig_tp = pl.DataFrame.to_pandas


def _counting(self, *a, **k):
    PROBE["n"] += 1
    return _orig_tp(self, *a, **k)


def load_panels():
    files = [f for f in sorted(glob.glob(os.path.join(DATA, "*.parquet")))
             if START <= os.path.basename(f)[:10] <= END]
    syms = (pl.read_parquet(files[-1]).filter(pl.col("AdjAmount").is_not_null())
            .sort("AdjAmount", descending=True).head(NSYM)["Symbol"].to_list())
    cnt = (pl.scan_parquet(files).filter(pl.col("Symbol").is_in(syms))
           .group_by("Symbol").len().collect())
    present = {r["Symbol"]: r["len"] for r in cnt.iter_rows(named=True)}
    syms = [s for s in syms if present.get(s, 0) >= 0.95 * len(files)]
    long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", "AdjAmount"])
            .filter(pl.col("Symbol").is_in(syms)).collect())
    wide = (long.pivot(values="AdjAmount", index="TradeDate", on="Symbol")
            .sort("TradeDate").select(["TradeDate", *syms]))
    pdf = wide.to_pandas().set_index("TradeDate")
    pdf.index.name = "date"
    return pdf, syms, len(files)


def build_fiscal(pdf, syms):
    idx = pd.DatetimeIndex(pdf.index)
    qper = idx.to_period("Q")
    # real quarterly traded amount per symbol, then YTD cumulative within the year
    q = pdf.copy()
    q["__q"] = qper.astype(str)
    qsum = q.groupby("__q", sort=True)[list(syms)].sum()
    yearly = {}
    for label, row in qsum.iterrows():
        year = label[:4]
        prev = yearly.get(year)
        cum = row if prev is None else prev + row
        yearly[year] = cum
        cum.to_frame().T.to_numpy()
        yearly.setdefault("__series", []).append((label, cum.copy()))
    ytd = pd.DataFrame({lbl: cum for lbl, cum in yearly["__series"]}).T
    ytd = ytd.reindex(sorted(ytd.index))
    ytd.index.name = "__q"

    period_id = pd.DataFrame(
        {s: [str(p) for p in qper] for s in syms}, index=idx)
    fiscal_quarter = pd.DataFrame(
        {s: [int(str(p)[-1]) for p in qper] for s in syms}, index=idx).astype("float64")
    # map each date to its own quarter's YTD cumulative
    x = pd.DataFrame(
        {s: ytd.loc[[str(p) for p in qper], s].to_numpy() for s in syms}, index=idx)
    return x, period_id, fiscal_quarter


def main():
    pdf, syms, ndays = load_panels()
    x, period_id, fiscal_quarter = build_fiscal(pdf, syms)
    meta = {"days": int(len(x)), "symbols": len(syms), "files": ndays,
            "start": str(x.index[0].date()), "end": str(x.index[-1].date()),
            "value_driver": "real quarterly SUM(AdjAmount), YTD cumulative",
            "source": DATA}
    print("PANEL", json.dumps(meta), flush=True)

    pop = OperatorRegistry.get(CANON, "pandas_numpy")
    plop = OperatorRegistry.get(CANON, "polars", mode="any")

    x_pl = pl.from_pandas(x.reset_index())
    pid_pl = pl.from_pandas(period_id.reset_index())
    fq_pl = pl.from_pandas(fiscal_quarter.reset_index())

    kind = canonical_polars_kind(CANON, production_mode=True)
    rec = {"operator": CANON, "polars_kind": getattr(kind, "value", str(kind))}

    pl.DataFrame.to_pandas = _counting
    try:
        ref = pop.calculate(x, period_id, fiscal_quarter)
        PROBE["n"] = 0
        t0 = time.perf_counter()
        got1 = plop._calculate_series(x_pl, pid_pl, fq_pl)
        rec["_polars_s"] = time.perf_counter() - t0
        rec["to_pandas_calls"] = PROBE["n"]
        t1 = time.perf_counter()
        got2 = plop._calculate_series(x_pl, pid_pl, fq_pl)
        rec["_polars_2nd_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        pop.calculate(x, period_id, fiscal_quarter)
        rec["_pandas_s"] = time.perf_counter() - t2
    finally:
        pl.DataFrame.to_pandas = _orig_tp

    def frame(obj):
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        else:
            drop = [c for c in obj.columns if c in ("date", "TradeDate")]
            df = (obj.drop(drop[0]) if drop else obj).to_pandas()
            if drop:
                df.index = pd.Index(pd.to_datetime(obj[drop[0]].to_pandas()))
        df.columns = [str(c) for c in df.columns]
        return df.sort_index()

    rf, nf1, nf2 = frame(ref), frame(got1), frame(got2)
    order = sorted(set(rf.columns) & set(nf1.columns))
    rows = sorted(set(rf.index) & set(nf1.index))
    rec["shape_ref_raw"] = list(rf.shape)
    rec["shape_new_raw"] = list(nf1.shape)
    rec["n_rows_aligned"] = len(rows)
    rec["n_cols_aligned"] = len(order)
    rec["finite_frac_ref"] = round(float(np.isfinite(
        rf.reindex(index=rows, columns=order).to_numpy(dtype=float)).mean()), 6)
    a = rf.reindex(index=rows, columns=order).to_numpy(dtype=float)
    b = nf1.reindex(index=rows, columns=order).to_numpy(dtype=float)
    c = nf2.reindex(index=rows, columns=order).to_numpy(dtype=float)
    rec["nan_mask_identical"] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    fin = np.isfinite(a) & np.isfinite(b)
    rec["n_finite"] = int(fin.sum())
    rec["max_abs_dev"] = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None
    rec["idempotent"] = bool(np.array_equal(np.isnan(c), np.isnan(b))
                             and np.array_equal(c[fin], b[fin]))
    rec["ok"] = True
    print(json.dumps(rec, default=str), flush=True)

    out = os.environ.get("FISCAL_OUT") or f"{REPO}/evidence/_verify_batch3_fiscal.json"
    json.dump({"summary": meta, "results": [rec]}, open(out, "w"), indent=1, default=str)
    print("WROTE", out)


if __name__ == "__main__":
    main()
