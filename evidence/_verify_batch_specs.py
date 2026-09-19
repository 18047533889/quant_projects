# -*- coding: utf-8 -*-
"""Independent numerical verification for declared polars specs (step 3 batches).

For each operator, on REAL data (default: 25 symbols x ~4 years):
  1. NaN MASK IDENTITY between the pandas_numpy reference and the polars kernel
  2. max absolute deviation on co-finite cells
  3. IDEMPOTENCY: same input twice -> bitwise-identical output
  4. runtime marshal probe: number of pl.DataFrame.to_pandas calls (must be 0 to
     justify POLARS_NATIVE_EXPR / POLARS_NUMPY_KERNEL rather than a delegate)
  5. classification flip: canonical_polars_kind(production_mode=True)

Usage:  python3 evidence/_verify_batch_specs.py op1 op2 ...   (or --from-patch)
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
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
load_all()

POOL = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "Volume", "AdjAmount", "Return", "AdjVwap"]
NAME_MAP = {"x": "AdjClose", "y": "AdjClose", "z": "AdjClose", "price": "AdjClose",
            "close": "AdjClose", "feature": "AdjClose", "series": "AdjClose",
            "signal": "AdjClose", "value": "AdjClose", "ret": "Return",
            "open": "AdjOpen", "high": "AdjHigh", "low": "AdjLow",
            "activity": "Volume", "volume": "Volume", "vol": "Volume",
            "amount": "AdjAmount", "vwap": "AdjVwap"}
SECOND = {"cs_quantile_resid", "cs_spline_resid", "ts_beta", "cs_isotonic_residual"}

PROBE = {"n": 0}
_orig_tp = pl.DataFrame.to_pandas


def _counting(self, *a, **k):
    PROBE["n"] += 1
    return _orig_tp(self, *a, **k)


def load_panels():
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    files = [f for f in files if START <= os.path.basename(f)[:10] <= END]
    syms = (pl.read_parquet(files[-1]).filter(pl.col("AdjAmount").is_not_null())
              .sort("AdjAmount", descending=True).head(NSYM)["Symbol"].to_list())
    cnt = (pl.scan_parquet(files).filter(pl.col("Symbol").is_in(syms))
             .group_by("Symbol").len().collect())
    present = {r["Symbol"]: r["len"] for r in cnt.iter_rows(named=True)}
    syms = [s for s in syms if present.get(s, 0) >= 0.95 * len(files)]
    long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", *POOL])
              .filter(pl.col("Symbol").is_in(syms)).collect())
    pd_p, pl_p = {}, {}
    for c in POOL:
        w = (long.select(["TradeDate", "Symbol", c]).pivot(values=c, index="TradeDate", on="Symbol").sort("TradeDate"))
        pdf = w.to_pandas().set_index("TradeDate"); pdf.index.name = "date"
        pd_p[c] = pdf; pl_p[c] = pl.from_pandas(pdf.reset_index())
    return pd_p, pl_p, {"days": len(pd_p["AdjClose"]), "symbols": len(syms),
                        "start": str(pd_p["AdjClose"].index[0].date()),
                        "end": str(pd_p["AdjClose"].index[-1].date())}


def bind(canonical, md, pool):
    pn = list(getattr(md, "param_names", []) or [])
    ps = getattr(md, "param_specs", None) or {}
    frames = []
    for nm in pn:
        if nm in ps:
            break
        frames.append(nm)
    if not frames:
        # logical ops register an empty param_names list; fall back to the concrete
        # calculate() signature and take the leading DataFrame-annotated params.
        import inspect
        try:
            sig = inspect.signature(OperatorRegistry.get(canonical, "pandas_numpy").calculate)
        except Exception:
            return None, []
        for nm, p in sig.parameters.items():
            if nm in ("self", "kwargs", "args") or p.kind in (
                    inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            ann = str(p.annotation).lower()
            if "dataframe" in ann:
                frames.append(nm)
            elif p.default is inspect.Parameter.empty and not frames:
                continue
            else:
                break
    if not frames:
        return None, []
    if canonical in SECOND and len(frames) >= 2:
        return [pool["AdjClose"], pool["Volume"]], ["AdjClose", "Volume"]
    cols = []
    for i, nm in enumerate(frames):
        if nm in NAME_MAP:
            cols.append(NAME_MAP[nm])
        elif nm.startswith("x") and nm[1:].isdigit():
            cols.append(POOL[(int(nm[1:]) - 1) % len(POOL)])
        else:
            cols.append(POOL[i % len(POOL)])
    return [pool[c] for c in cols], cols


def fill_kwargs(ps):
    ov = {"order": 3, "delay": 1, "normalize": True, "n": 1}
    kw = {}
    for name, spec in (ps or {}).items():
        if name in ov:
            kw[name] = ov[name]; continue
        d = getattr(spec, "default", None)
        if d is not None and "missing" not in type(d).__name__.lower():
            kw[name] = d; continue
        dt, lo = getattr(spec, "dtype", None), (getattr(spec, "min", None) or 1)
        ch = getattr(spec, "choices", None)
        if ch:
            kw[name] = ch[0]
        elif dt is bool:
            kw[name] = True
        elif dt is int:
            kw[name] = max(int(lo), 20)
        elif dt is float:
            kw[name] = max(float(lo), 0.5)
    return kw


def as_array(obj):
    """Column-name aligned view of a wide result.

    A pivoting kernel orders its output columns by pivot discovery order, which
    need not equal the input panel's column order; comparing the raw numpy blocks
    positionally would then compare different symbols against each other.  Sort
    both sides by column name so the comparison is symbol-for-symbol.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.reindex(columns=sorted(map(str, obj.columns))).to_numpy(dtype=float)
    if isinstance(obj, pl.DataFrame):
        cols = sorted(c for c in obj.columns if c != "date")
        return obj.select(cols).to_numpy().astype(float)
    raise TypeError(type(obj))


def main():
    ops = [a for a in sys.argv[1:] if not a.startswith("--")]
    pd_p, pl_p, meta = load_panels()
    print("PANEL", json.dumps(meta), flush=True)
    results = []
    pl.DataFrame.to_pandas = _counting
    try:
        for canon in ops:
            rec = {"operator": canon}
            try:
                pop = OperatorRegistry.get(canon, "pandas_numpy")
                plop = OperatorRegistry.get(canon, "polars", mode="any")
                md = pop.metadata
                pd_args, cols = bind(canon, md, pd_p)
                pl_args, _ = bind(canon, md, pl_p)
                kw = fill_kwargs(getattr(md, "param_specs", None) or {})
                rec["inputs"] = cols
                rec["kwargs"] = {k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v))
                                 for k, v in kw.items()}
                kind = canonical_polars_kind(canon, production_mode=True)
                rec["polars_kind_after"] = getattr(kind, "value", str(kind))

                ref = pop.calculate(*pd_args, **kw)
                PROBE["n"] = 0
                t0 = time.perf_counter()
                got1 = plop._calculate_series(*pl_args, **kw)
                rec["_delegate_s"] = time.perf_counter() - t0
                rec["to_pandas_calls"] = PROBE["n"]

                a = as_array(ref); b = as_array(got1)
                rec["shape_ref"] = list(a.shape); rec["shape_new"] = list(b.shape)
                rec["nan_mask_identical"] = bool(a.shape == b.shape and
                                                 np.array_equal(np.isnan(a), np.isnan(b)))
                fin = np.isfinite(a) & np.isfinite(b)
                rec["n_finite"] = int(fin.sum())
                rec["finite_frac_ref"] = round(float(np.isfinite(a).mean()), 6)
                rec["max_abs_dev"] = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None

                t1 = time.perf_counter()
                got2 = plop._calculate_series(*pl_args, **kw)
                rec["_delegate_2nd_s"] = time.perf_counter() - t1
                c = as_array(got2)
                rec["idempotent"] = bool(c.shape == b.shape and
                                         np.array_equal(np.isnan(c), np.isnan(b)) and
                                         np.array_equal(c[fin], b[fin]))
                # pandas reference timing for the record
                t2 = time.perf_counter()
                pop.calculate(*pd_args, **kw)
                rec["_pandas_s"] = time.perf_counter() - t2
                rec["ok"] = True
            except Exception as e:
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
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
        "max_abs_dev_over_all": max([r["max_abs_dev"] or 0.0 for r in ok], default=None),
        "all_idempotent": all(r["idempotent"] for r in ok) if ok else None,
        "all_zero_marshal": all(r["to_pandas_calls"] == 0 for r in ok) if ok else None,
        "all_flipped_to_polars_native": all(r.get("polars_kind_after") == "polars_native" for r in ok) if ok else None,
        "failures": [r["operator"] for r in results if not r.get("ok")],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=1, default=str))
    out = os.environ.get("VER_OUT") or f"{REPO}/evidence/_verify_batch_specs.json"
    prev = []
    if os.path.exists(out):
        try:
            prev = json.load(open(out))
        except Exception:
            prev = []
    json.dump(prev + [{"summary": summary, "results": results}], open(out, "w"), indent=1, default=str)
    print("WROTE", out)


if __name__ == "__main__":
    main()
