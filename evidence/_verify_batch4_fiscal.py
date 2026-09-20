# -*- coding: utf-8 -*-
"""Batch 4 fiscal verification: 3 period-aware operators on REAL axes.

fiscal_standardized_surprise / fin_average_balance / ttm_from_cumulative need a
structurally valid fiscal input (x: per-period flow, period_id: fiscal label,
fiscal_quarter: 1..4).  Feeding price panels makes both sides degenerate, so we
construct, on the REAL daily calendar and REAL symbols:
  * quarterly flow  = real quarterly SUM of AdjAmount per symbol
  * x[date]         = that fiscal year's YTD cumulative of the quarterly flow
                      (mapped back onto every trading date of the quarter)
  * period_id[date] = quarter label, fiscal_quarter[date] = 1..4
Value driver is real-but-not-a-financial-statement; this verifies the fiscal
period walk and pandas/polars equivalence, not restatement parity.
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

OPS = ["fiscal_standardized_surprise", "fin_average_balance", "ttm_from_cumulative"]

load_all()

PROBE = {"n": 0}
_orig_tp = pl.DataFrame.to_pandas


def _counting(self, *a, **k):
    PROBE["n"] += 1
    return _orig_tp(self, *a, **k)


def load_daily():
    files = [f for f in sorted(glob.glob(os.path.join(DATA, "*.parquet")))
             if START <= os.path.basename(f)[:10] <= END]
    syms = (pl.read_parquet(files[-1]).filter(pl.col("AdjAmount").is_not_null())
            .sort("AdjAmount", descending=True).head(NSYM)["Symbol"].to_list())
    long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", "AdjAmount"])
            .filter(pl.col("Symbol").is_in(syms)).collect())
    wide = (long.pivot(values="AdjAmount", index="TradeDate", on="Symbol")
            .sort("TradeDate"))
    pdf = wide.to_pandas().set_index("TradeDate")
    pdf.index = pd.to_datetime(pdf.index)
    pdf.index.name = "date"
    return pdf, list(syms), {"days": len(pdf), "symbols": len(syms),
                             "start": str(pdf.index[0].date()), "end": str(pdf.index[-1].date())}


def build_fiscal(pdf, syms):
    idx = pdf.index
    qper = idx.to_period("Q")
    qlabels = pd.Index([str(p) for p in qper])
    qsum = pdf.groupby(qlabels)[syms].sum()  # quarter x symbol flow
    years = sorted({lbl[:4] for lbl in qsum.index})
    ytd_rows = {}
    for year in years:
        cum = None
        for q in (1, 2, 3, 4):
            lbl = f"{year}Q{q}"
            if lbl not in qsum.index:
                continue
            cum = qsum.loc[lbl] if cum is None else cum + qsum.loc[lbl]
            ytd_rows[lbl] = cum
    ytd = pd.DataFrame(ytd_rows).T.sort_index()  # quarter x symbol YTD cumulative
    x = pd.DataFrame({s: ytd.loc[qlabels, s].to_numpy() for s in syms}, index=idx)
    x.index.name = "date"
    period_id = pd.DataFrame({s: qlabels.to_numpy() for s in syms}, index=idx)
    period_id.index.name = "date"
    fq = pd.DataFrame({s: [int(l[-1]) for l in qlabels] for s in syms}, index=idx).astype("float64")
    fq.index.name = "date"
    return x, period_id, fq


def fill_kwargs(ps):
    kw = {}
    for name, spec in (ps or {}).items():
        d = getattr(spec, "default", None)
        if d is not None and "missing" not in type(d).__name__.lower():
            kw[name] = d
            continue
        dt, lo = getattr(spec, "dtype", None), (getattr(spec, "min", None) or 1)
        ch = getattr(spec, "choices", None)
        if ch:
            kw[name] = ch[0]
        elif dt is bool:
            kw[name] = True
        elif dt is int:
            kw[name] = max(int(lo), 2)
        elif dt is float:
            kw[name] = max(float(lo), 0.5)
    return kw


def bind(canonical, md, pools):
    """pools: dict of operand-name -> panel; returns (pd_args, pl_args, kw)."""
    pn = list(getattr(md, "param_names", []) or [])
    ps = getattr(md, "param_specs", None) or {}
    frames, scalars = [], []
    for nm in pn:
        if nm in ps:
            scalars.append(nm)
        else:
            frames.append(nm)
    kw = fill_kwargs({k: ps[k] for k in scalars})
    pd_args = [pools["pd"][nm] for nm in frames]
    pl_args = [pools["pl"][nm] for nm in frames]
    return pd_args, pl_args, kw


def as_frame(obj):
    if isinstance(obj, pl.DataFrame):
        obj = obj.to_pandas()
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(type(obj))
    df = obj.copy()
    df.columns = [str(c) for c in df.columns]
    idxc = [c for c in df.columns if c.lower() in ("date", "tradedate")]
    if idxc:
        df = df.set_index(idxc[0])
    try:
        df.index = pd.Index(pd.to_datetime(df.index).normalize())
    except Exception:
        pass
    return df.sort_index()


def main():
    pdf, syms, meta = load_daily()
    x, period_id, fq = build_fiscal(pdf, syms)
    pools = {
        "pd": {"x": x, "period_id": period_id, "fiscal_quarter": fq},
        "pl": {k: pl.from_pandas(v.reset_index()) for k, v in
               {"x": x, "period_id": period_id, "fiscal_quarter": fq}.items()},
    }
    print("FISCAL PANEL", json.dumps(meta), flush=True)
    results = []
    pl.DataFrame.to_pandas = _counting
    try:
        for canon in OPS:
            rec = {"operator": canon}
            try:
                pop = OperatorRegistry.get(canon, "pandas_numpy")
                plop = OperatorRegistry.get(canon, "polars", mode="any")
                md = pop.metadata
                pd_args, pl_args, kw = bind(canon, md, pools)
                rec["frames"] = [type(a).__name__ for a in pd_args]
                rec["kwargs"] = kw
                kind = canonical_polars_kind(canon, production_mode=True)
                rec["polars_kind_after"] = getattr(kind, "value", str(kind))

                ref = pop.calculate(*pd_args, **kw)
                PROBE["n"] = 0
                t0 = time.perf_counter()
                _call = getattr(plop, "_calculate_series", None) or plop.calculate
                got1 = _call(*pl_args, **kw)
                rec["_polars_s"] = time.perf_counter() - t0
                rec["to_pandas_calls"] = PROBE["n"]
                got2 = _call(*pl_args, **kw)

                rf, nf1, nf2 = as_frame(ref), as_frame(got1), as_frame(got2)
                rec["ref_shape"] = list(rf.shape); rec["ref_cols"] = [str(c) for c in rf.columns[:4]]
                rec["ref_idx"] = [str(i) for i in rf.index[:3]]
                rec["new_shape"] = list(nf1.shape); rec["new_cols"] = [str(c) for c in nf1.columns[:4]]
                rec["new_idx"] = [str(i) for i in nf1.index[:3]]
                # fiscal kernels may key output by period label, not date; align
                # on whatever index both sides share, else on positional rows.
                # fiscal outputs may be keyed by period labels on one side and
                # dates on the other; when the label spaces do not intersect,
                # fall back to POSITIONAL comparison (same input axes on both
                # sides, columns sorted by name to survive pivot ordering).
                rf_c, nf1_c, nf2_c = rf.sort_index(), nf1.sort_index(), nf2.sort_index()
                if not (set(rf_c.index) & set(nf1_c.index)) or not (set(rf_c.columns) & set(nf1_c.columns)):
                    rf_c = rf_c.sort_index(axis=1)
                    nf1_c = nf1_c.sort_index(axis=1)
                    nf2_c = nf2_c.sort_index(axis=1)
                    if rf_c.shape != nf1_c.shape:
                        raise AssertionError(
                            f"shape mismatch ref={rf_c.shape} new={nf1_c.shape} "
                            f"ref_idx={list(map(str, rf_c.index[:3]))} new_idx={list(map(str, nf1_c.index[:3]))}")
                    a = rf_c.to_numpy(dtype=float)
                    b = nf1_c.to_numpy(dtype=float)
                    c = nf2_c.to_numpy(dtype=float)
                    rec["align_mode"] = "positional"
                else:
                    idx = sorted(set(rf_c.index) & set(nf1_c.index), key=str)
                    order = sorted(set(rf_c.columns) & set(nf1_c.columns), key=str)
                    rec["n_rows_aligned"], rec["n_cols_aligned"] = len(idx), len(order)
                    if not idx or not order:
                        raise AssertionError("no overlap between reference and polars output")
                    a = rf_c.reindex(index=idx, columns=order).to_numpy(dtype=float)
                    b = nf1_c.reindex(index=idx, columns=order).to_numpy(dtype=float)
                    c = nf2_c.reindex(index=idx, columns=order).to_numpy(dtype=float)
                    rec["align_mode"] = "label"
                rec["nan_mask_identical"] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
                fin = np.isfinite(a) & np.isfinite(b)
                rec["n_finite"] = int(fin.sum())
                rec["finite_frac_ref"] = round(float(np.isfinite(a).mean()), 6)
                rec["max_abs_dev"] = (float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None)
                rec["idempotent"] = bool(
                    np.array_equal(np.isnan(c), np.isnan(b)) and np.array_equal(c[fin], b[fin]))
                rec["ok"] = True
            except Exception as e:
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
        "all_flipped_to_polars_native": all(r.get("polars_kind_after") == "polars_native" for r in ok) if ok else None,
        "total_finite_cells_compared": sum(r["n_finite"] for r in ok),
        "max_abs_dev_over_all": max([r["max_abs_dev"] or 0.0 for r in ok], default=None),
        "failures": [r["operator"] for r in results if not r.get("ok")],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=1, default=str))
    out = os.environ.get("VER_OUT") or f"{REPO}/evidence/_verify_batch4_fiscal.json"
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1, default=str)
    print("WROTE", out)


if __name__ == "__main__":
    main()
