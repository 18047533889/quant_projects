# -*- coding: utf-8 -*-
"""Capture the exact SQL the auto/SQL route emits, then profile it.

Why: the E-mode benchmark shows ``duckdb_sql_full`` taking ~80s per factor
(275s for a single recorded query) while polars_long does the same four
factors in 49s.  The route is now correct but the query is pathological, so
the next step is to read the emitted SQL and let DuckDB say where the time
goes.

Captures, for each executed SQL statement:
  * the full query text
  * the store kwargs (view_columns / read_datasets)
  * ``EXPLAIN``            -> the chosen plan
  * ``EXPLAIN ANALYZE``    -> per-operator timing (only when --analyze)

Usage:
    python3 _wb_probe_sql_profile.py --n 1 [--analyze]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.expanduser("~/cos_data"))
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")

PROJECT = os.path.expanduser("~/quant_projects")
for _p in (PROJECT, PROJECT + "/quant_evaluator", PROJECT + "/factor_preprocess",
           PROJECT + "/vectorbt_qs", PROJECT + "/data_access", PROJECT + "/jobs"):
    sys.path.insert(0, _p)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--analyze", action="store_true")
ap.add_argument("--out", default="/tmp/sql_profile.json")
args = ap.parse_args()

import factor_engine.cleaned_operators  # noqa: F401,E402
from factor_engine.api.factor import Factor  # noqa: E402
from factor_engine.api.cleaned_ops import make_cleaned_call_factory  # noqa: E402
from factor_engine.api.columns import col  # noqa: E402
from factor_engine.backend.factory import build_backend  # noqa: E402
from factor_engine.storage.factory import build_data_source  # noqa: E402
from factor_engine.runtime.engine import FactorEngine  # noqa: E402
from data_access.runtime.mode_identity import (  # noqa: E402
    set_runtime_mode_identity,
    reset_runtime_mode_identity,
)


def build_formulas(n):
    o = {}
    for nm in ["rank", "zscore", "ts_mean", "ts_std", "ts_corr", "ts_rank",
               "ts_delta", "ts_sum", "ts_min", "ts_max", "log", "abs", "delay"]:
        o[nm] = make_cleaned_call_factory(nm)
    C, V, A = col("AdjClose"), col("Volume"), col("AdjAmount")
    F = {
        "b01_ts_mean_c10": o["ts_mean"](C, 10),
        "b02_ts_std_c20": o["ts_std"](C, 20),
        "b03_rank_c": o["rank"](C),
        "b04_zscore_v": o["zscore"](V),
    }
    return dict(list(F.items())[:n])


DS_SPEC = {
    "type": "data_access",
    "dataset": "ashare_stock_daily_adj",
    "start_date": "2019-01-02",
    "end_date": "2026-08-24",
}

# ---- hook store.sql so we can profile exactly what the engine runs ---------
import data_access  # noqa: E402

CAPTURED: list[dict] = []
_orig_get_store = data_access.get_store
_STORE = {"obj": None, "sql": None}


def _wrap_store(store):
    if _STORE["obj"] is store:
        return store
    _STORE["obj"] = store
    _STORE["sql"] = store.sql

    def sql(query, **kwargs):
        t0 = time.perf_counter()
        out = _STORE["sql"](query, **kwargs)
        el = time.perf_counter() - t0
        rec = {"query": query, "elapsed_s": round(el, 3),
               "kwargs_keys": sorted(kwargs),
               "view_columns": {k: list(v) for k, v in (kwargs.get("view_columns") or {}).items()},
               "read_datasets": list(kwargs.get("read_datasets") or [])}
        CAPTURED.append(rec)
        print(f"[sql] {el:8.2f}s len={len(query)} datasets={rec['read_datasets']}", flush=True)
        if args.analyze:
            conn = getattr(store, "_conn", None)
            if conn is not None:
                for label, prefix in (("plan", "EXPLAIN "), ("profile", "EXPLAIN ANALYZE ")):
                    try:
                        rows = conn.execute(prefix + query).fetchall()
                        rec[label] = "\n".join(str(r[0]) for r in rows)
                    except Exception as exc:  # noqa: BLE001
                        rec[label] = f"<{type(exc).__name__}: {str(exc)[:300]}>"
        return out

    store.sql = sql
    return store


def get_store_patched(*a, **k):
    return _wrap_store(_orig_get_store(*a, **k))


data_access.get_store = get_store_patched
try:
    import data_access as _da_mod
    _da_mod.get_store = get_store_patched
except Exception:  # noqa: BLE001
    pass
# the executor does ``from data_access import get_store`` at call time
import factor_engine.backend.sql_pushdown.executor as _ex  # noqa: E402
_ex.get_store = get_store_patched

forms = build_formulas(args.n)
tok = set_runtime_mode_identity("interactive_research", source="probe_sql_profile")
report = {"n": args.n, "forms": list(forms)}
try:
    ds = build_data_source(DS_SPEC)
    eng = FactorEngine(build_backend("auto"), ds, run_mode="research")
    fl = [Factor(name=k, expr=v) for k, v in forms.items()]
    t0 = time.perf_counter()
    out = eng.run_many(fl, market="ashare", result_policy="return")
    report["batch_run_s"] = round(time.perf_counter() - t0, 3)
    report["routes"] = {str(k): str(v)[:200]
                        for k, v in (out.get("backend_paths") or {}).items()}
    ds.close()
finally:
    reset_runtime_mode_identity(tok)

report["sql_calls"] = CAPTURED
json.dump(report, open(args.out, "w"), indent=1, default=str)
print("\n=== route summary ===", flush=True)
print(json.dumps(report["routes"], ensure_ascii=False, indent=1)[:2000], flush=True)
print(f"\nbatch_run_s={report['batch_run_s']} sql_calls={len(CAPTURED)}", flush=True)
print(f"written to {args.out}", flush=True)
if CAPTURED and not args.analyze:
    print("\n=== FIRST QUERY (full) ===", flush=True)
    print(CAPTURED[0]["query"], flush=True)
