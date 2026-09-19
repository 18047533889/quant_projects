# -*- coding: utf-8 -*-
"""A/B acceptance test for the average-rank SQL rewrite.

Compiles every SQL-capable op that routes through the average-rank helper,
executes the compiled SQL against a DuckDB view built from the real
``ashare_stock_daily_adj`` parquet over a bounded window, and digests the
result.  Run it once per emitter revision (old correlated form vs new window
form) and diff the two JSON reports.

The view intentionally injects ``+Inf`` / ``-Inf`` cells and a synthetic group
column so the ``exclude_nan=True`` branch (``cs_rank`` / ``rank``) and the
``exclude_nan=False`` / descending branch (``cs_pct_rank`` / ``group_rank`` /
``group_percentile``) are both exercised.

Usage:
    python3 _wb_verify_rank_ab.py --tag new --out /tmp/rank_ab_new.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT = os.path.expanduser("~/quant_projects")
for _p in (PROJECT, PROJECT + "/quant_evaluator", PROJECT + "/factor_preprocess",
           PROJECT + "/vectorbt_qs", PROJECT + "/data_access", PROJECT + "/jobs"):
    sys.path.insert(0, _p)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", default="2024-01-02")
ap.add_argument("--end", default="2024-06-28")
args = ap.parse_args()

import duckdb  # noqa: E402

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends  # noqa: E402
from factor_engine.backend.sql_pushdown.emitter import (  # noqa: E402
    SqlDialect,
    compile_plan_to_sql,
    plan_is_sql_capable,
)
from factor_engine.planner.logical_plan import PlanNode  # noqa: E402

load_all()
register_sql_backends()


def col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def lit(v) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": v})


PLANS = {
    # 0-1 average rank (ascending, exclude_nan=True)  -> normalised=False
    "rank": PlanNode(op="rank", inputs=[col("AdjClose")]),
    "cs_rank": PlanNode(op="cs_rank", inputs=[col("AdjClose")]),
    # pct average rank (exclude_nan from RankSpec.inf_policy)
    "rank_pct": PlanNode(op="rank_pct", inputs=[col("AdjClose")]),
    "cs_pct_rank": PlanNode(op="cs_pct_rank", inputs=[col("AdjClose")]),
    # grouped: 0-1 and pct, ascending + descending
    "group_rank": PlanNode(op="group_rank", inputs=[col("AdjClose"), col("industry")]),
    "group_percentile_top": PlanNode(
        op="group_percentile", inputs=[col("AdjClose"), col("industry"), lit(0.5)],
        attrs={"side": "top"},
    ),
    "group_percentile_bottom": PlanNode(
        op="group_percentile", inputs=[col("AdjClose"), col("industry"), lit(0.5)],
        attrs={"side": "bottom"},
    ),
    # third, independently written copy of the same correlated subquery
    "cs_rank_normalize": PlanNode(op="cs_rank_normalize", inputs=[col("AdjClose")]),
    # control: must not change (ROW_NUMBER based)
    "ts_rank": PlanNode(op="ts_rank", inputs=[col("AdjClose")], attrs={"d": 5}),
}

DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")
files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
window = [
    f for f in files
    if args.start <= os.path.basename(f)[:10] <= args.end
]
if not window:
    print("no parquet files in window", flush=True)
    sys.exit(2)

con = duckdb.connect()
con.execute("SET threads TO 4")
file_list = ", ".join("'" + f + "'" for f in window)
# A deterministic group key (7 groups) and two deliberate non-finite cells so the
# exclude_nan=True and =False branches are both reached.
con.execute(
    f"""
    CREATE OR REPLACE VIEW d AS
    SELECT
      "TradeDate" AS "TradeDate",
      "Symbol" AS "Symbol",
      CASE
        WHEN "Symbol" = (SELECT min("Symbol") FROM read_parquet([{file_list}]))
             AND "TradeDate" = DATE '{args.start}' THEN 'Infinity'::DOUBLE
        WHEN "Symbol" = (SELECT max("Symbol") FROM read_parquet([{file_list}]))
             AND "TradeDate" = DATE '{args.start}' THEN '-Infinity'::DOUBLE
        ELSE "AdjClose"
      END AS "AdjClose",
      (CAST(unicode(substr("Symbol", 1, 1)) AS INTEGER) % 7) + 1 AS "industry"
    FROM read_parquet([{file_list}])
    WHERE "TradeDate" >= DATE '{args.start}' AND "TradeDate" <= DATE '{args.end}'
    """
)

report = {"tag": args.tag, "window": [args.start, args.end],
          "n_files": len(window), "cases": {}}


def flush() -> None:
    """Persist after every case: the correlated form can be OOM-killed."""
    json.dump(report, open(args.out, "w"), indent=1, default=str)


def digest(rows) -> dict:
    """Order-independent + order-sensitive digest of the (ts, inst, value) rows."""
    vals = [r[2] for r in rows]
    nn = [v for v in vals if v is not None]
    import math

    finite = [v for v in nn if math.isfinite(v)]
    arr = sorted((str(r[0]), str(r[1]), ("" if r[2] is None else repr(float(r[2]))))
                 for r in rows)
    h = hashlib.sha256("\n".join(f"{a}|{b}|{c}" for a, b, c in arr).encode()).hexdigest()
    return {
        "rows": len(rows),
        "non_null": len(nn),
        "n_nan": sum(1 for v in nn if math.isnan(v)),
        "n_inf": sum(1 for v in nn if math.isinf(v)),
        "sum": float(sum(finite)),
        "min": float(min(finite)) if finite else None,
        "max": float(max(finite)) if finite else None,
        "sha256_sorted": h,
    }


for name, plan in PLANS.items():
    entry = {"sql_capable": bool(plan_is_sql_capable(plan))}
    if not entry["sql_capable"]:
        report["cases"][name] = entry
        print(f"  {name:<24} NOT SQL-CAPABLE (skipped)", flush=True)
        flush()
        continue
    try:
        compiled = compile_plan_to_sql(
            plan, dataset="d", time_column="TradeDate",
            instrument_column="Symbol", dialect=SqlDialect.DUCKDB,
        )
    except Exception as exc:  # noqa: BLE001
        entry["compile_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        report["cases"][name] = entry
        print(f"  {name:<24} COMPILE FAILED {entry['compile_error']}", flush=True)
        flush()
        continue
    if compiled is None:
        entry["compiled"] = None
        report["cases"][name] = entry
        print(f"  {name:<24} compiled=None", flush=True)
        flush()
        continue

    import re

    sql = re.sub(r"\{\{\s*d\s*\}\}", '"d"', compiled.query)
    entry["sql"] = sql
    try:
        rows = con.execute(sql).fetchall()
    except Exception as exc:  # noqa: BLE001
        entry["exec_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"][name] = entry
        print(f"  {name:<24} EXEC FAILED {entry['exec_error']}", flush=True)
        flush()
        continue
    try:
        d = digest(rows)
    except Exception as exc:  # noqa: BLE001
        entry["digest_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"][name] = entry
        print(f"  {name:<24} DIGEST FAILED {entry['digest_error']}", flush=True)
        flush()
        continue
    entry.update(d)
    entry["correlated_sql"] = ("cnt_le" in compiled.query) or ("countIf" in compiled.query)
    entry["window_sql"] = "RANK() OVER" in compiled.query
    report["cases"][name] = entry
    print(f"  {name:<24} rows={d['rows']:<8} nonnull={d['non_null']:<8} "
          f"nan={d['n_nan']:<5} inf={d['n_inf']:<3} sum={d['sum']:.10g} "
          f"corr={entry['correlated_sql']} win={entry['window_sql']}", flush=True)
    flush()

json.dump(report, open(args.out, "w"), indent=1, default=str)
print(f"\nwritten {args.out}", flush=True)
