# -*- coding: utf-8 -*-
"""Step 3 prep: for the highest-frequency operators that lack a PhysicalImplementationSpec,
decide HONESTLY whether the registered polars kernel is
  - a polars expression        -> POLARS_NATIVE_EXPR
  - a per-column numpy kernel  -> POLARS_NUMPY_KERNEL
  - a pandas round-trip        -> not eligible for this batch

Evidence used (no guessing):
  * the kernel BODY (inspect.getsource) for pl.* / np.* / pandas tokens
  * a RUNTIME marshal probe (patch pl.DataFrame.to_pandas, execute on real data)
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
REPO = "/home/sunhaiwei/quant_projects"
DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import (  # noqa: E402
    get_physical_spec, _kernel_is_delegate, _module_is_delegate, _source_is_delegate)
load_all()

TOP_N = int(os.environ.get("TOP_N", "45"))

PROBE = {"n": 0}
_orig = pl.DataFrame.to_pandas


def counting(self, *a, **k):
    PROBE["n"] += 1
    return _orig(self, *a, **k)


POOL = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "Volume", "AdjAmount", "Return", "AdjVwap"]
NAME_MAP = {"x": "AdjClose", "y": "AdjClose", "z": "AdjClose", "price": "AdjClose",
            "close": "AdjClose", "feature": "AdjClose", "series": "AdjClose",
            "signal": "AdjClose", "value": "AdjClose", "ret": "Return",
            "open": "AdjOpen", "high": "AdjHigh", "low": "AdjLow",
            "activity": "Volume", "volume": "Volume", "vol": "Volume",
            "amount": "AdjAmount", "vwap": "AdjVwap"}
SECOND = {"cs_quantile_resid", "cs_spline_resid", "ts_beta", "cs_isotonic_residual"}


def load_panels(lo="2021-01-01", hi="2022-06-30", nsym=8):
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    files = [f for f in files if lo <= os.path.basename(f)[:10] <= hi]
    syms = (pl.read_parquet(files[-1]).sort("AdjAmount", descending=True).head(nsym)["Symbol"].to_list())
    long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", *POOL])
              .filter(pl.col("Symbol").is_in(syms)).collect())
    out = {}
    for c in POOL:
        w = (long.select(["TradeDate", "Symbol", c]).pivot(values=c, index="TradeDate", on="Symbol").sort("TradeDate"))
        pdf = w.to_pandas().set_index("TradeDate"); pdf.index.name = "date"
        out[c] = (pdf, pl.from_pandas(pdf.reset_index()))
    return {k: v[0] for k, v in out.items()}, {k: v[1] for k, v in out.items()}


def bind(canonical, md, panels):
    pn = list(getattr(md, "param_names", []) or [])
    ps = getattr(md, "param_specs", None) or {}
    frames = []
    for nm in pn:
        if nm in ps:
            break
        frames.append(nm)
    if not frames:
        return None
    if canon == "composition_normalized_entropy":
        return [panels[c] for c in POOL[:8]]
    if canonical in SECOND and len(frames) >= 2:
        return [panels["AdjClose"], panels["Volume"]]
    out = []
    for i, nm in enumerate(frames):
        if nm in NAME_MAP:
            col = NAME_MAP[nm]
        elif nm in ("period_end", "period_id", "report_date"):
            col = "AdjClose"
        elif nm.startswith("x") and nm[1:].isdigit():
            col = POOL[(int(nm[1:]) - 1) % len(POOL)]
        else:
            col = POOL[i % len(POOL)]
        out.append(panels[col])
    return out


def fill_kwargs(ps):
    ov = {"order": 3, "delay": 1, "normalize": True, "require_consecutive": True, "min_patterns": 20}
    kw = {}
    for name, spec in (ps or {}).items():
        if name in ov:
            kw[name] = ov[name]; continue
        d = getattr(spec, "default", None)
        if d is not None and not type(d).__name__ == "_MissingType":
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


def kernel_tokens(op):
    toks = {"pl_expr": 0, "np_call": 0, "pd_call": 0, "to_pandas": 0, "list_api": 0, "rolling": 0, "group_by": 0}
    for mn in ("_calculate_series", "calc"):
        k = getattr(op, mn, None)
        if k is None:
            continue
        try:
            body = inspect.getsource(k)
        except Exception:
            body = ""
        toks["pl_expr"] += len(re.findall(r"\bpl\.(col|when|lit|concat_list|element|Series|DataFrame|select|first|last|max_horizontal|min_horizontal|coalesce|any_horizontal|all_horizontal)\b", body))
        toks["np_call"] += len(re.findall(r"\bnp\.\w+", body))
        toks["pd_call"] += len(re.findall(r"\bpandas\.\w+|\bpd\.\w+", body))
        toks["to_pandas"] += body.count("to_pandas")
        toks["list_api"] += body.count(".list.")
        toks["rolling"] += body.count(".rolling(")
        toks["group_by"] += body.count("group_by") + body.count(".over(")
    return toks


def classify_kernel(op, toks):
    if _kernel_is_delegate(op) or toks["to_pandas"] > 0:
        return "pandas_delegate"
    if toks["pl_expr"] > 0 and toks["np_call"] <= 2:
        return "polars_native_expr"
    if toks["pl_expr"] > 0 and toks["np_call"] > 2:
        return "polars_expr_plus_numpy"
    if toks["np_call"] > 0:
        return "polars_numpy_kernel"
    return "unknown"


freq = {d["operator"]: d["rows"] for d in json.load(
    open(f"{REPO}/evidence/factor_catalog_20260916/r57c_operator_frequency.json"))}
cands = []
for canon, entry in OperatorRegistry._catalog.items():
    p = OperatorRegistry.get(canon, "polars", mode="any")
    if p is None or get_physical_spec(p) is not None:
        continue
    r = freq.get(canon, 0)
    if r > 0:
        cands.append((r, canon, p))
cands.sort(reverse=True)
cands = cands[:TOP_N]

panels, panels_pl = load_panels()
pl.DataFrame.to_pandas = counting
rows = []
try:
    for r, canon, op in cands:
        md = getattr(op, "metadata", None)
        toks = kernel_tokens(op)
        kind = classify_kernel(op, toks)
        rec = {"operator": canon, "rows": r, "class": type(op).__name__,
               "module": type(op).__module__, "kernel_kind": kind, "tokens": toks}
        try:
            src = inspect.getsource(type(op))
            rec["has_phys_spec_attr"] = "_physical_spec" in src
        except Exception:
            pass
        frames = bind(canon, md, panels)
        if frames:
            kw = fill_kwargs(getattr(md, "param_specs", None) or {})
            plargs = []
            for f in frames:
                for c, pp in panels.items():
                    if pp is f:
                        plargs.append(panels_pl[c]); break
                else:
                    plargs.append(pl.from_pandas(f.reset_index()))
            try:
                PROBE["n"] = 0
                t0 = time.perf_counter()
                out = op._calculate_series(*plargs, **kw)
                rec["probe"] = {"to_pandas_calls": PROBE["n"], "seconds": time.perf_counter() - t0,
                                "n_frames": len(frames), "ok": True}
                if isinstance(out, pl.DataFrame):
                    num = out.select([c for c in out.columns if c != "date"]).to_numpy()
                    rec["probe"]["finite_frac"] = round(float(np.isfinite(num.astype(float)).mean()), 4)
                    rec["probe"]["out_cols"] = len(out.columns) - (1 if "date" in out.columns else 0)
            except Exception as e:
                rec["probe"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}
        else:
            rec["probe"] = {"ok": False, "error": "unbindable"}
        rows.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "tokens"}), flush=True)
finally:
    pl.DataFrame.to_pandas = _orig

json.dump(rows, open(f"{REPO}/evidence/_step3_kernel_kinds.json", "w"), indent=1)
print("\n=== SELECTABLE (no marshal, kernel not delegate) ===")
for rec in rows:
    if rec.get("probe", {}).get("to_pandas_calls") == 0 and rec["kernel_kind"] != "pandas_delegate":
        print(f"  {rec['operator']:38s} rows={rec['rows']:6d} {rec['kernel_kind']:26s} "
              f"finite={rec['probe'].get('finite_frac')}  "
              f"{rec['module'].split('.')[-1]}.{rec['class']}")
