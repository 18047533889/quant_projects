# -*- coding: utf-8 -*-
"""Batch 4 intraday verification: 12 intra_* operators on REAL minute data.

Minute wide panels built from ~/cos_data/StockMinuteBarAdj (20 symbols x 5
sessions x 240 bars).  Operand panels: close (AdjClose), amount (AdjAmount),
volume (Volume), returns (per-symbol minute-over-minute pct change), flow
(amount * sign(returns)), value (AdjAmount mirror).  Scalar params filled from
param_specs (same fill rules as the daily harness).

Checks (per operator): NaN mask identity vs the pandas_numpy reference, max abs
deviation on co-finite cells, idempotency, runtime marshal probe
(pl.DataFrame.to_pandas == 0), classification flip.
"""
from __future__ import annotations

import glob
import inspect
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

OPS = [
    "intra_jump_ratio", "intra_signed_jump_ratio", "intra_jump_variation",
    "intra_vwap_reversion_speed", "intra_amihud", "intra_volume_profile_jsd",
    "intra_path_efficiency", "intra_same_slot_momentum", "intra_entropy",
    "intra_segment_return", "intraday_impact_asymmetry", "intraday_impact_beta",
]
IDX_COLS = ("date", "timestamp", "time", "QuoteTime", "TradeDate")

load_all()

PROBE = {"n": 0}
_orig_tp = pl.DataFrame.to_pandas


def _counting(self, *a, **k):
    PROBE["n"] += 1
    return _orig_tp(self, *a, **k)


NAME_MAP = {"close": "close_wide", "amount": "amount_wide", "volume": "volume_wide",
            "returns": "returns_wide", "flow": "flow_wide", "value": "value_wide"}


def load_minute_panel():
    files = sorted(glob.glob(os.path.join(MINUTE, "*.parquet")))[:DAYS]
    if not files:
        raise SystemExit(f"no minute parquet under {MINUTE}")
    syms = (pl.read_parquet(files[-1]).group_by("Symbol").len()
            .sort("len", descending=True).head(NSYM)["Symbol"].to_list())
    long = (pl.scan_parquet(files)
            .select(["QuoteTime", "Symbol", "AdjClose", "Volume", "AdjAmount"])
            .filter(pl.col("Symbol").is_in(syms)).collect())
    long = long.with_columns(
        pl.col("QuoteTime").dt.convert_time_zone("Asia/Shanghai")
        .dt.replace_time_zone(None).alias("date")
    ).drop_nulls().sort("date")
    pools_pd, pools_pl = {}, {}
    for col in ("AdjClose", "Volume", "AdjAmount"):
        w = (long.select(["date", "Symbol", col])
             .pivot(values=col, index="date", on="Symbol").sort("date"))
        pdf = w.to_pandas().set_index("date")
        pdf.index.name = "date"
        pools_pd[col] = pdf
        pools_pl[col] = pl.from_pandas(pdf.reset_index())
    close_pd = pools_pd["AdjClose"]
    ret_pd = close_pd / close_pd.shift(1) - 1.0
    amt_pd = pools_pd["AdjAmount"]
    flow_pd = amt_pd * np.sign(ret_pd).fillna(0.0)
    pools_pd["returns"] = ret_pd
    pools_pd["flow"] = flow_pd
    pools_pd["value"] = amt_pd
    for k in ("returns", "flow", "value"):
        pools_pl[k] = pl.from_pandas(pools_pd[k].reset_index())
    # rename to the operand names the bind step expects
    _REN = {"AdjClose": "close_wide", "AdjAmount": "amount_wide",
            "Volume": "volume_wide", "returns": "returns_wide",
            "flow": "flow_wide", "value": "value_wide"}
    pools_pd = {_REN.get(k, k): v for k, v in pools_pd.items()}
    pools_pl = {_REN.get(k, k): v for k, v in pools_pl.items()}
    meta = {"days": DAYS, "symbols": len(syms), "minute_bars": int(pools_pd["close_wide"].shape[0]),
            "sessions_from": str(pools_pd["close_wide"].index[0]),
            "sessions_to": str(pools_pd["close_wide"].index[-1]), "source": MINUTE}
    return pools_pd, pools_pl, meta


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


def bind(canonical, md, pool):
    pn = list(getattr(md, "param_names", []) or [])
    ps = getattr(md, "param_specs", None) or {}
    frames, scalars = [], []
    for nm in pn:
        if nm in ps:
            scalars.append(nm)
            continue
        if nm in NAME_MAP:
            frames.append(NAME_MAP[nm])
        else:
            frames.append("close_wide")
    args = [pool[c] for c in frames]
    kw = {k: v for k, v in fill_kwargs({k: ps[k] for k in scalars}).items()}
    return args, kw


def as_frame(obj):
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
    try:
        df.index = pd.Index(pd.to_datetime(df.index).normalize())
    except Exception:
        pass
    return df.sort_index()


def main():
    pd_pool, pl_pool, meta = load_minute_panel()
    print("MINUTE PANEL", json.dumps(meta), flush=True)
    results = []
    pl.DataFrame.to_pandas = _counting
    try:
        for canon in OPS:
            rec = {"operator": canon}
            try:
                pop = OperatorRegistry.get(canon, "pandas_numpy")
                plop = OperatorRegistry.get(canon, "polars", mode="any")
                md = pop.metadata
                pd_args, kw = bind(canon, md, pd_pool)
                pl_args, _ = bind(canon, md, pl_pool)
                rec["inputs"] = pd_args and [type(a).__name__ for a in pd_args]
                rec["kwargs"] = kw
                kind = canonical_polars_kind(canon, production_mode=True)
                rec["polars_kind_after"] = getattr(kind, "value", str(kind))

                ref = pop.calculate(*pd_args, **kw)
                PROBE["n"] = 0
                t0 = time.perf_counter()
                got1 = plop._calculate_series(*pl_args, **kw)
                rec["_polars_s"] = time.perf_counter() - t0
                rec["to_pandas_calls"] = PROBE["n"]

                rf, nf1 = as_frame(ref), as_frame(got1)
                got2 = plop._calculate_series(*pl_args, **kw)
                nf2 = as_frame(got2)

                idx = sorted(set(rf.index) & set(nf1.index))
                order = sorted(set(rf.columns) & set(nf1.columns))
                rec["n_rows_aligned"] = len(idx)
                rec["n_cols_aligned"] = len(order)
                if not idx or not order:
                    raise AssertionError(
                        f"no overlap: ref {list(rf.index)[:2]}x{list(rf.columns)[:2]} "
                        f"vs new {list(nf1.index)[:2]}x{list(nf1.columns)[:2]}")
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
    out = os.environ.get("VER_OUT") or f"{REPO}/evidence/_verify_batch4_intraday.json"
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1, default=str)
    print("WROTE", out)


if __name__ == "__main__":
    main()
