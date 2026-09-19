# -*- coding: utf-8 -*-
"""Step 1 (REAL DATA): is the polars "delegate" slot worth keeping?

For the highest-frequency POLARS_PANDAS_DELEGATE operators in the R57c catalog,
time on the SAME real A-share panel:
  (a) the pandas_numpy reference implementation  -> "pandas"
  (b) the registered polars delegate slot        -> "delegate"  (pl -> pd -> compute -> pl)

Also isolates the irreducible marshaling penalty and measures the cost of
breaking a native polars lazy chain by inserting a delegate.

Data: ~/cos_data/StockDailyBarAdj (one parquet per trading day, real adjusted bars).
No synthetic data. No writes to the data directory.
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
START = os.environ.get("BENCH_START", "2019-01-01")
END = os.environ.get("BENCH_END", "2022-12-31")
NSYM = int(os.environ.get("BENCH_NSYM", "30"))
TARGET_SECONDS = float(os.environ.get("BENCH_TARGET_S", "0.6"))
MAX_ITERS = int(os.environ.get("BENCH_MAX_ITERS", "30"))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.cleaned_operators.common._polars_bridge import (  # noqa: E402
    to_pandas_panel,
    from_pandas_panel,
)

load_all()

MISSING = object()
try:
    from factor_engine.backend.contracts import MISSING as _M  # type: ignore
    MISSING = _M
except Exception:
    for mod in ("factor_engine.cleaned_operators.base_pandas",
                "factor_engine.cleaned_operators.base",
                "factor_engine.backend.operator_capability"):
        try:
            m = __import__(mod, fromlist=["MISSING"])
            if hasattr(m, "MISSING"):
                MISSING = m.MISSING
                break
        except Exception:
            pass


# --------------------------------------------------------------------------
# 1. Load REAL data
# --------------------------------------------------------------------------
def load_real_panels():
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    files = [f for f in files if START <= os.path.basename(f)[:10] <= END]
    if not files:
        raise SystemExit("no parquet files in range")
    # pick liquid symbols from the most recent file in range, require they are
    # present on >=95% of the trading days in range
    probe = pl.read_parquet(files[-1]).filter(pl.col("AdjAmount").is_not_null())
    syms = (probe.sort("AdjAmount", descending=True)
                 .head(NSYM)["Symbol"].to_list())
    n_expect = len(files)
    cnt = (pl.scan_parquet(files)
             .filter(pl.col("Symbol").is_in(syms))
             .group_by("Symbol").len()
             .collect())
    present = {r["Symbol"]: r["len"] for r in cnt.iter_rows(named=True)}
    syms = [s for s in syms if present.get(s, 0) >= 0.95 * n_expect]
    cols = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "Volume", "AdjAmount",
            "Return", "AdjVwap"]
    long = (pl.scan_parquet(files)
              .select(["TradeDate", "Symbol", *cols])
              .filter(pl.col("Symbol").is_in(syms))
              .collect())
    panels_pd, panels_pl = {}, {}
    for col in cols:
        wide = (long.select(["TradeDate", "Symbol", col])
                    .pivot(values=col, index="TradeDate", on="Symbol")
                    .sort("TradeDate"))
        pdf = wide.to_pandas().set_index("TradeDate")
        pdf.index.name = "date"
        panels_pd[col] = pdf
        panels_pl[col] = pl.from_pandas(pdf.reset_index())
    meta = {
        "n_trading_days": len(panels_pd["AdjClose"]),
        "n_symbols": len(panels_pd["AdjClose"].columns),
        "date_start": str(panels_pd["AdjClose"].index[0].date()),
        "date_end": str(panels_pd["AdjClose"].index[-1].date()),
        "cells": int(panels_pd["AdjClose"].size),
    }
    return panels_pd, panels_pl, meta


# --------------------------------------------------------------------------
# 2. Bind real panels to an operator's positional contract
# --------------------------------------------------------------------------
POOL = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "Volume", "AdjAmount",
        "Return", "AdjVwap"]

NAME_MAP = {
    "x": "AdjClose", "y": "AdjClose", "z": "AdjClose", "price": "AdjClose",
    "close": "AdjClose", "feature": "AdjClose", "series": "AdjClose",
    "signal": "AdjClose", "value": "AdjClose", "ret": "Return",
    "open": "AdjOpen", "high": "AdjHigh", "low": "AdjLow",
    "activity": "Volume", "volume": "Volume", "vol": "Volume",
    "amount": "AdjAmount", "vwap": "AdjVwap",
}

# ops whose 2nd frame is an explanatory variable -> use a different real column
SECOND_IS_EXPLANATORY = {"cs_quantile_resid", "cs_spline_resid", "ts_beta",
                         "cs_isotonic_residual"}


def bind_panels(canonical, param_names, param_specs, panels):
    """Leading param_names not declared in param_specs are the panel inputs."""
    frames = []
    for nm in param_names:
        if nm in param_specs:
            break
        frames.append(nm)
    if not frames:
        return None
    out = []
    if canonical == "composition_normalized_entropy":
        out = [panels[c] for c in POOL[:8]]
    elif canonical in SECOND_IS_EXPLANATORY and len(frames) >= 2:
        out = [panels["AdjClose"], panels["Volume"]]
    else:
        for i, nm in enumerate(frames):
            if nm in NAME_MAP:
                col = NAME_MAP[nm]
            elif nm in ("period_end", "period_id", "report_date"):
                col = "AdjClose"          # date-like frame: not representable
            elif nm.startswith("x") and nm[1:].isdigit():
                col = POOL[(int(nm[1:]) - 1) % len(POOL)]
            else:
                col = POOL[i % len(POOL)]
            out.append(panels[col])
    return out


def fill_kwargs(param_specs):
    overrides = {"order": 3, "delay": 1, "normalize": True,
                 "require_consecutive": True, "min_patterns": 20}
    kw = {}
    for name, spec in (param_specs or {}).items():
        d = getattr(spec, "default", MISSING)
        if name in overrides:
            kw[name] = overrides[name]
            continue
        if d is not MISSING and d is not None:
            kw[name] = d
            continue
        dt = getattr(spec, "dtype", None)
        lo = getattr(spec, "min", None) or 1
        choices = getattr(spec, "choices", None)
        if choices:
            kw[name] = choices[0]
        elif dt is bool:
            kw[name] = True
        elif dt is int:
            kw[name] = max(int(lo), 20)
        elif dt is float:
            kw[name] = max(float(lo), 0.5)
        # otherwise omit -> leave to the reference's own default
    return kw


# --------------------------------------------------------------------------
# 3. Timing
# --------------------------------------------------------------------------
def timeit(fn, target=TARGET_SECONDS, max_iters=MAX_ITERS, min_iters=3):
    fn()
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    n = int(max(min_iters, min(max_iters, target / one if one > 0 else max_iters)))
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {"n_iters": n, "median": statistics.median(samples),
            "min": min(samples), "max": max(samples)}


def main():
    print("loading real data ...", flush=True)
    panels_pd, panels_pl, meta = load_real_panels()
    print("PANEL", json.dumps(meta), flush=True)

    cls = json.load(open(f"{REPO}/evidence/_operator_classification.json"))
    freq = {d["operator"]: d["rows"] for d in json.load(
        open(f"{REPO}/evidence/factor_catalog_20260916/r57c_operator_frequency.json"))}
    cands = sorted([(freq.get(o, 0), o) for o, c in cls.items()
                    if c == "polars_pandas_delegate" and freq.get(o, 0) > 0],
                   reverse=True)
    top = [o for _, o in cands[:16]]

    # --- irreducible marshaling floor at THIS panel size ---
    plc = panels_pl["AdjClose"]
    pdc = panels_pd["AdjClose"]
    m = timeit(lambda: to_pandas_panel(plc))
    to_pd = m["median"]
    m2 = timeit(lambda: from_pandas_panel(plc, pdc.copy()))
    from_pd = m2["median"]
    m3 = timeit(lambda: to_pandas_panel(plc).copy(deep=True))
    to_pd_deep = m3["median"]
    print(json.dumps({"MARSHAL_FLOOR_SEC": {
        "to_pandas_panel": to_pd,
        "from_pandas_panel": from_pd,
        "to_pandas_panel_plus_deepcopy": to_pd_deep,
        "roundtrip_total": to_pd_deep + from_pd,
    }}), flush=True)

    rows = []
    for canonical in top:
        entry = {"operator": canonical, "rows": freq.get(canonical, 0)}
        try:
            pop = OperatorRegistry.get(canonical, "pandas_numpy")
            plop = OperatorRegistry.get(canonical, "polars", mode="any")
            if pop is None or plop is None:
                entry["status"] = "no_ref_or_slot"
                rows.append(entry)
                print(entry, flush=True)
                continue
            md = pop.metadata
            pn = list(getattr(md, "param_names", []) or [])
            ps = getattr(md, "param_specs", None) or {}
            frames = bind_panels(canonical, pn, ps, panels_pd)
            if not frames:
                entry["status"] = "unbindable_no_panel_param"
                rows.append(entry)
                print(entry, flush=True)
                continue
            kw = fill_kwargs(ps)
            pd_args = tuple(frames)
            pl_args = tuple(bind_panels(canonical, pn, ps, panels_pl))
            entry["n_frames"] = len(frames)
            entry["kwargs"] = {k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v))
                               for k, v in kw.items()}

            t_pd = timeit(lambda: pop.calculate(*pd_args, **kw))
            t_del = timeit(lambda: plop._calculate_series(*pl_args, **kw))
            entry.update({
                "status": "OK",
                "pandas_median_s": t_pd["median"], "pandas_min_s": t_pd["min"],
                "delegate_median_s": t_del["median"], "delegate_min_s": t_del["min"],
                "multiplier_median": t_del["median"] / t_pd["median"],
                "multiplier_min": t_del["min"] / t_pd["min"],
                "delegate_minus_pandas_s_median": t_del["median"] - t_pd["median"],
                "n_iters": t_del["n_iters"],
            })
            # equivalence spot-check (same values on both paths?)
            try:
                r_pd = pop.calculate(*pd_args, **kw)
                r_pl = plop._calculate_series(*pl_args, **kw)
                a = np.asarray(r_pd.to_numpy(dtype=float), dtype=float)
                b = np.asarray(r_pl.select(
                    [c for c in r_pl.columns if c != "date"]).to_numpy(), dtype=float)
                entry["nan_mask_identical"] = bool(a.shape == b.shape and
                                                   np.array_equal(np.isnan(a), np.isnan(b)))
                fin = np.isfinite(a) & np.isfinite(b)
                entry["max_abs_dev"] = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else None
            except Exception as e:
                entry["equiv_error"] = f"{type(e).__name__}: {e}"
            rows.append(entry)
            print(json.dumps(entry), flush=True)
        except Exception as e:
            entry["status"] = f"ERR {type(e).__name__}: {e}"
            rows.append(entry)
            print(json.dumps(entry), flush=True)

    # --- cost of breaking a native polars lazy chain with a delegate ---
    native_probe = []
    # faithful: 6-element native expression chain on one real column
    c0 = [c for c in panels_pl["AdjClose"].columns if c != "date"][0]
    chain = panels_pl["AdjClose"].lazy()
    e = pl.col(c0)
    for _ in range(6):
        e = (e + 1.0).log().abs()
    t_nat = timeit(lambda: chain.with_columns(e.alias("__n")).collect())
    # delegate break: collect + full marshal to pandas and back, then continue
    def break_chain():
        _df = chain.with_columns(e.alias("__n")).collect()
        _pdf = to_pandas_panel(_df)
        _ = from_pandas_panel(pl.from_pandas(_pdf), _pdf)
    t_brk = timeit(break_chain)
    # a real delegate op inside the pipeline
    real_delta = None
    try:
        op = OperatorRegistry.get("ts_cusum_pressure", "polars", mode="any")
        pdp = panels_pd["AdjClose"]
        plp = panels_pl["AdjClose"]
        def native_pipe():
            return chain.with_columns(e.alias("__n")).collect()
        def delegate_pipe():
            _base = chain.with_columns(e.alias("__n")).collect()
            _ = op._calculate_series(_base, reference_window=20, drift=0.5)
            return _base
        tn = timeit(native_pipe)
        td = timeit(delegate_pipe)
        real_delta = {"native_collect_s": tn["median"],
                      "collect_plus_one_delegate_s": td["median"],
                      "delta_s": td["median"] - tn["median"]}
    except Exception as e:
        real_delta = {"error": f"{type(e).__name__}: {e}"}
    native_probe = {
        "native_lazy_chain_6expr_collect_s": t_nat["median"],
        "collect_plus_marshal_roundtrip_s": t_brk["median"],
        "marshal_break_overhead_s": t_brk["median"] - t_nat["median"],
        "marshal_break_overhead_x": t_brk["median"] / t_nat["median"],
        "with_real_delegate_op": real_delta,
    }
    print(json.dumps({"PIPELINE_BREAK": native_probe}, indent=1), flush=True)

    out = {"panel": meta, "operators": rows, "pipeline_break": native_probe,
           "marshaling": {"to_pandas_panel_s": to_pd,
                          "from_pandas_panel_s": from_pd,
                          "roundtrip_total_s": to_pd_deep + from_pd}}
    with open(f"{REPO}/evidence/_step1_delegate_bench_real.json", "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", f"{REPO}/evidence/_step1_delegate_bench_real.json", flush=True)


if __name__ == "__main__":
    main()
