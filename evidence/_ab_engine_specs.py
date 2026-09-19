# -*- coding: utf-8 -*-
"""DECISIVE TEST: does declaring POLARS_NATIVE_EXPR for these ops actually speed up the
engine, or does it route work to a SLOWER polars kernel?

Runs the same factor expressions on real data through the real FactorEngine with
backend=pandas and backend=polars_long, twice:
  SPECS=on   -> current working tree (batch-2 specs declared)
  SPECS=off  -> the same process with the batch-2 specs removed from the classes
                (i.e. the pre-batch-2 state)

Reports wall time per (factor, backend, specs) plus the polars_long routing flags.
"""
from __future__ import annotations

import glob
import json
import os
import time
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
REPO = "/home/sunhaiwei/quant_projects"
SYMS = os.environ.get("AB_SYMS", "000001.SZ,600000.SH,600519.SH,000002.SZ,600036.SH,"
                                  "601318.SH,000858.SZ,600030.SH,002415.SZ,300750.SZ,"
                                  "601166.SH,000333.SZ,600276.SH,002594.SZ,601888.SH,"
                                  "600887.SH,000651.SZ,601012.SH,002304.SZ,600585.SH")
START = os.environ.get("AB_START", "2019-01-02")
END = os.environ.get("AB_END", "2022-12-30")
SPECS = os.environ.get("SPECS", "on") == "on"
REPS = int(os.environ.get("AB_REPS", "3"))
UNIVERSE = os.environ.get("AB_UNIVERSE", "fixed")   # fixed | topN


def resolve_universe():
    if UNIVERSE == "fixed":
        return SYMS.split(",")
    n = int(UNIVERSE.replace("top", ""))
    import glob as _g
    files = sorted(_g.glob(os.path.expanduser("~/cos_data/StockDailyBarAdj/*.parquet")))
    files = [f for f in files if "2019-01-01" <= os.path.basename(f)[:10] <= END]
    return (pl.read_parquet(files[-1]).filter(pl.col("AdjAmount").is_not_null())
            .sort("AdjAmount", descending=True).head(n)["Symbol"].to_list())


UNIV = resolve_universe()

BATCH2 = {
    "factor_engine.cleaned_operators.common.cross_sectional": ["RankPolars", "CsPctRankPolars", "CrossSectionalMeanPolars"],
    "factor_engine.cleaned_operators.common.polars_auto": ["AndPolarsAuto"],
    "factor_engine.cleaned_operators.common.time_series": ["TSSumPolars", "TSZScorePolars", "TSDeltaPolars",
                                                           "TSDelayPolars", "TSMaxPolars", "TSMinPolars"],
    "factor_engine.cleaned_operators.common.polars_extended": ["TanhPolars"],
    "factor_engine.cleaned_operators.common.polars_daily_native": ["TSSharpeNative", "CSMadZscoreNative"],
}

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import canonical_polars_kind, get_physical_spec  # noqa: E402
from factor_engine.storage.factory import build_data_source  # noqa: E402
from factor_engine.backend.factory import build_backend  # noqa: E402
from factor_engine.runtime.engine import FactorEngine  # noqa: E402
from factor_engine.api.factor import Factor  # noqa: E402
from factor_engine.api.columns import col  # noqa: E402
from factor_engine.api.cleaned_ops import make_cleaned_call_factory  # noqa: E402

load_all()

# also cover the factory-generated compare classes (gt / lt) via the registry
FACTORY_OPS = ["gt", "lt"]

REMOVED = []


def strip_specs():
    """Remove _physical_spec from the batch-2 classes (recreate the pre-batch-2 state)."""
    import importlib
    targets = []
    for mod, classes in BATCH2.items():
        m = importlib.import_module(mod)
        for cls in classes:
            targets.append(getattr(m, cls, None))
    # dynamic compare classes
    for op in FACTORY_OPS:
        p = OperatorRegistry.get(op, "polars", mode="any")
        if p is not None:
            targets.append(type(p))
    for cls in targets:
        if cls is None:
            continue
        if "_physical_spec" in cls.__dict__:
            REMOVED.append(cls.__name__)
            delattr(cls, "_physical_spec")


_o = {n: make_cleaned_call_factory(n) for n in
      ["ts_mean", "ts_std", "multiply", "add", "rank", "cs_pct_rank", "ts_sum", "abs", "gt", "lt", "tanh"]}
C = col("AdjClose")
V = col("Volume")
FACTORS = {
    "ts_mean_20": _o["ts_mean"](C, 20),
    "rank": _o["rank"](C),
    "cs_pct_rank": _o["cs_pct_rank"](C),
    "ts_sum_20": _o["ts_sum"](C, 20),
    "gt_const": _o["gt"](C, 10.0),
    "tanh": _o["tanh"](C),
    "mixed_rank_std": _o["multiply"](_o["rank"](C), _o["ts_std"](C, 20)),
}


def run_once(backend, expr):
    ds = build_data_source({
        "type": "data_access", "dataset": "ashare_stock_daily_adj",
        "start_date": START, "end_date": END, "instrument_filter": UNIV,
    })
    eng = FactorEngine(build_backend(backend), ds, run_mode="research")
    t0 = time.perf_counter()
    r = eng.run(Factor(name="ab", expr=expr), market="ashare")
    dt = time.perf_counter() - t0
    return dt, r


def main():
    if not SPECS:
        strip_specs()
        print("SPECS=off  -> removed _physical_spec from:", REMOVED, flush=True)
    else:
        print("SPECS=on", flush=True)

    kinds = {}
    for op in ["rank", "cs_pct_rank", "ts_sum", "ts_mean", "gt", "tanh", "cs_mean",
               "ts_zscore", "ts_delta", "ts_delay", "ts_max", "ts_min", "ts_sharpe",
               "cs_mad_zscore", "and_", "lt"]:
        try:
            k = canonical_polars_kind(op, production_mode=True)
            kinds[op] = getattr(k, "value", str(k))
        except Exception as e:
            kinds[op] = f"ERR {type(e).__name__}"

    out = {"specs": "on" if SPECS else "off", "kinds": kinds, "factors": {}, "panel": {
        "symbols": len(UNIV), "start": START, "end": END, "reps": REPS}}
    for fname, expr in FACTORS.items():
        rec = {}
        for backend in ("pandas", "polars_long"):
            times = []
            flags = {}
            err = None
            for _ in range(REPS):
                try:
                    dt, r = run_once(backend, expr)
                    times.append(dt)
                    flags = {k: r.get(k) for k in
                             ("used_polars_long_path", "used_polars_long_native",
                              "polars_expr", "polars_expr_fallback") if k in r}
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    break
            rec[backend] = {"median_s": min(times) if times else None,
                            "reps": len(times), "flags": flags, "error": err}
            print(f"{fname:16s} {backend:12s} "
                  f"{'%.3f s' % min(times) if times else 'ERR ' + str(err)[:60]} {flags}", flush=True)
        p, q = rec["pandas"]["median_s"], rec["polars_long"]["median_s"]
        rec["polars_vs_pandas_x"] = (p / q) if (p and q) else None
        out["factors"][fname] = rec
    fn_out = f"{REPO}/evidence/_ab_engine_specs_{UNIVERSE}_{out['specs']}.json"
    json.dump(out, open(fn_out, "w"), indent=1, default=str)
    print("\nSPECS =", out["specs"])
    for f, r in out["factors"].items():
        print(f"  {f:16s} pandas={r['pandas']['median_s']} polars_long={r['polars_long']['median_s']} "
              f"speedup={r['polars_vs_pandas_x']}")
    print("WROTE", fn_out)


if __name__ == "__main__":
    main()
