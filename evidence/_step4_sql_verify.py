# -*- coding: utf-8 -*-
"""Step 4: verify DuckDB / SQL pushdown wiring (VERIFY ONLY - no source changes).

Checks:
  1. what ``_data_source_kind`` returns for a real ``DataAccessSource`` in BOTH
     implementations (plan_cost_router / cleaned_bridge)
  2. whether ``DataAccessSource`` exposes ``capabilities``
  3. how many canonical operators advertise an SQL slot; how many carry an
     explicit SQL / duckdb execution kind
  4. an end-to-end factor run under pandas / polars_long / duckdb_sql / auto,
     capturing the SQL runtime fields (used_sql_pushdown, sql_fully_pushed,
     sql_query_count, sql_full_execution_failed)
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

REPO = "/home/sunhaiwei/quant_projects"
SYMS = os.environ.get("S4_SYMS", "000001.SZ,600000.SH,600519.SH,000002.SZ,600036.SH").split(",")
START = os.environ.get("S4_START", "2022-01-03")
END = os.environ.get("S4_END", "2022-06-30")

import factor_engine.backend.sql_backend as sb  # noqa: E402
from factor_engine.backend.runtime_events import merge_runtime as _mr  # noqa: E402

CAPTURED: list[dict] = []
_orig_merge = sb._merge_runtime


def _patched(ctx, **fields):
    if any(k.startswith("sql_") or k in ("used_sql_pushdown", "compile_fully_sql",
                                         "execution_fully_sql") for k in fields):
        CAPTURED.append(dict(fields))
    return _orig_merge(ctx, **fields)


sb._merge_runtime = _patched

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import get_physical_spec  # noqa: E402
from factor_engine.backend.contracts import ExecutionKind  # noqa: E402
from factor_engine.storage.factory import build_data_source  # noqa: E402
from factor_engine.backend.factory import build_backend  # noqa: E402
from factor_engine.runtime.engine import FactorEngine  # noqa: E402
from factor_engine.api.factor import Factor  # noqa: E402
from factor_engine.api.columns import col  # noqa: E402
from factor_engine.api.cleaned_ops import make_cleaned_call_factory  # noqa: E402

load_all()

out: dict = {}

# ---------- 1/2. data source kind ----------
ds = build_data_source({
    "type": "data_access",
    "dataset": "ashare_stock_daily_adj",
    "start_date": START,
    "end_date": END,
    "instrument_filter": SYMS,
})
out["data_source"] = {
    "class": type(ds).__name__,
    "mro": [c.__name__ for c in type(ds).__mro__[:5]],
    "has_capabilities_attr": hasattr(ds, "capabilities"),
    "capabilities_repr": (repr(getattr(ds, "capabilities", None))[:200]
                          if hasattr(ds, "capabilities") else None),
    "inner_class": type(getattr(ds, "inner", None) or getattr(ds, "_inner", None)).__name__,
}


class _Ctx:
    def __init__(self, d):
        self.data_source = d


ctx_probe = _Ctx(ds)
res_kinds = {}
for mod, path in (("plan_cost_router", "factor_engine.backend.plan_cost_router"),
                  ("cleaned_bridge", "factor_engine.backend.cleaned_bridge")):
    try:
        m = __import__(path, fromlist=["_data_source_kind"])
        res_kinds[mod] = m._data_source_kind(ctx_probe)
    except Exception as e:
        res_kinds[mod] = f"ERR {type(e).__name__}: {e}"
out["_data_source_kind"] = res_kinds
out["expected"] = "duckdb (DataAccessSource is a parquet/duckdb-capable source)"

# ---------- 3. SQL slot census ----------
cat = OperatorRegistry._catalog
sql_slot, sql_real, sql_real_ops, sql_slot_no_impl = 0, 0, [], []
for canon in sorted(cat):
    bks = tuple(map(str, cat[canon].get("backends") or ()))
    if not any("sql" in b.lower() for b in bks):
        continue
    sql_slot += 1
    found = False
    for b in bks:
        if "sql" not in b.lower():
            continue
        op = OperatorRegistry.get(canon, b, mode="any")
        if op is None:
            continue
        spec = get_physical_spec(op)
        if spec is not None and getattr(spec.execution_kind, "value", None) in (
                ExecutionKind.DUCKDB_NATIVE_SQL.value, ExecutionKind.SQL_PYTHON_UDF.value):
            found = True
            break
    if found:
        sql_real += 1
        sql_real_ops.append(canon)
    else:
        sql_slot_no_impl.append(canon)
out["sql_slot_census"] = {
    "operators_with_sql_slot": sql_slot,
    "with_explicit_sql_execution_kind": sql_real,
    "sql_slot_without_sql_impl": len(sql_slot_no_impl),
    "examples_without_impl": sql_slot_no_impl[:25],
}

# ---------- 4. end-to-end routing ----------
_o = {n: make_cleaned_call_factory(n) for n in
      ["ts_mean", "ts_std", "multiply", "add", "rank", "ts_sum", "abs", "lt", "where"]}
C = col("AdjClose")
FACTORS = {
    "ts_mean_20": _o["ts_mean"](C, 20),
    "arith": _o["add"](_o["ts_mean"](C, 5), _o["multiply"](_o["ts_std"](C, 20), 2)),
    "rank_tsmean": _o["rank"](_o["ts_mean"](C, 10)),
    "where_lt": _o["where"](_o["lt"](C, 10.0), _o["ts_sum"](C, 20), _o["abs"](C)),
}
run_log = {}
for bname in ("pandas", "polars_long", "duckdb_sql", "auto"):
    for fname, expr in FACTORS.items():
        key = f"{bname}|{fname}"
        CAPTURED.clear()
        rec = {}
        try:
            ds2 = build_data_source({
                "type": "data_access", "dataset": "ashare_stock_daily_adj",
                "start_date": START, "end_date": END, "instrument_filter": SYMS,
            })
            eng = FactorEngine(build_backend(bname), ds2, run_mode="research")
            r = eng.run(Factor(name=fname, expr=expr), market="ashare")
            s = r["result"]
            rec["ok"] = True
            rec["rows"] = int(len(s))
            rec["nonnull"] = int(s.notna().sum())
            rec["flags"] = {k: r.get(k) for k in
                            ("used_polars_long_path", "used_polars_long_native",
                             "polars_expr", "polars_expr_fallback")
                            if k in r}
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        merged = {}
        for d in CAPTURED:
            merged.update(d)
        rec["sql_runtime"] = {k: merged.get(k) for k in
                              ("used_sql_pushdown", "sql_fully_pushed", "sql_partial_pushed",
                               "sql_full_execution_failed", "sql_query_count",
                               "sql_dialect", "compile_fully_sql", "execution_fully_sql",
                               "sql_subtree_count", "sql_fallback_subtree_count")
                              if k in merged}
        run_log[key] = rec
        print(key, json.dumps(rec, default=str)[:400], flush=True)
out["runs"] = run_log

json.dump(out, open(f"{REPO}/evidence/_step4_sql_verify.json", "w"), indent=1, default=str)
print("\n=== SUMMARY ===")
print(json.dumps({k: out[k] for k in ("data_source", "_data_source_kind", "expected",
                                      "sql_slot_census")}, indent=1, default=str)[:3000])
print("WROTE _step4_sql_verify.json")
