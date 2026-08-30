#!/usr/bin/env python
"""Build StockMinuteBarAdj from StockMinuteBar * StockDailyBarAdj.Factor.

Idempotent: existing target days are skipped. Progress appended to
/tmp/minute_adj_build.log every 20 days.
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = os.path.expanduser("~/cos_data")
MIN_DIR = os.path.join(BASE, "StockMinuteBar")
DAILY_DIR = os.path.join(BASE, "StockDailyBarAdj")
OUT_DIR = os.path.join(BASE, "StockMinuteBarAdj")
LOG = "/tmp/minute_adj_build.log"

SCHEMA = pa.schema([
    pa.field("TradeDate", pa.date32()),
    pa.field("QuoteTime", pa.timestamp("ms", tz="UTC")),
    pa.field("Symbol", pa.string()),
    pa.field("AdjOpen", pa.float64()),
    pa.field("AdjHigh", pa.float64()),
    pa.field("AdjLow", pa.float64()),
    pa.field("AdjClose", pa.float64()),
    pa.field("AdjPreClose", pa.float64()),
    pa.field("Volume", pa.uint64()),
    pa.field("AdjAmount", pa.float64()),
    pa.field("AdjVwap", pa.float64()),
    pa.field("UpdateTime", pa.timestamp("ms", tz="UTC")),
])


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build_day(day):
    mpath = os.path.join(MIN_DIR, day)
    dpath = os.path.join(DAILY_DIR, day)

    m = pd.read_parquet(mpath)

    if not os.path.exists(dpath):
        return {"day": day, "status": "SKIP_NO_DAILY", "rows": len(m)}

    d = pd.read_parquet(dpath, columns=["Symbol", "Factor"])
    fmap = d.set_index("Symbol")["Factor"]
    fac = m["Symbol"].map(fmap)

    n_total = len(m)
    keep = fac.notna().values
    n_keep = int(keep.sum())
    n_drop = n_total - n_keep
    drop_rate = n_drop / n_total if n_total else 0.0

    if n_keep == 0:
        return {"day": day, "status": "SKIP_ALL_DROPPED", "rows": n_total}

    m = m[keep].copy()
    f = fac[keep].to_numpy(dtype=np.float64)

    m["AdjOpen"] = m["Open"].to_numpy() * f
    m["AdjHigh"] = m["High"].to_numpy() * f
    m["AdjLow"] = m["Low"].to_numpy() * f
    m["AdjClose"] = m["Close"].to_numpy() * f
    m["AdjPreClose"] = np.full(len(m), np.nan, dtype=np.float64)
    m["AdjAmount"] = m["Amount"].to_numpy() * f
    m["AdjVwap"] = m["Vwap"].to_numpy() * f

    m["TradeDate"] = pd.to_datetime(m["TradeDate"]).dt.date

    out = m[
        ["TradeDate", "QuoteTime", "Symbol",
         "AdjOpen", "AdjHigh", "AdjLow", "AdjClose", "AdjPreClose",
         "Volume", "AdjAmount", "AdjVwap", "UpdateTime"]
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, day)
    tmp_path = out_path + ".tmp"
    table = pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, tmp_path, version="2.6")
    os.replace(tmp_path, out_path)

    return {"day": day, "status": "OK", "rows": n_total, "kept": n_keep,
            "drop": n_drop, "drop_rate": drop_rate}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    days = sorted(os.listdir(MIN_DIR))
    if not days:
        log("ERROR: no minute files found")
        return 1

    existing = set(os.listdir(OUT_DIR))
    todo = [d for d in days if d not in existing]
    log(f"total days: {len(days)} | already built: {len(days) - len(todo)} | "
        f"to build: {len(todo)} | range: {days[0]} .. {days[-1]}")

    t0 = time.time()
    stats = {"OK": 0, "SKIP_NO_DAILY": 0, "SKIP_ALL_DROPPED": 0}
    total_kept = 0
    total_drop = 0
    total_rows = 0

    for i, day in enumerate(todo, 1):
        try:
            r = build_day(day)
        except Exception as e:
            log(f"FAIL {day}: {e}\n{traceback.format_exc()}")
            stats["FAIL"] = stats.get("FAIL", 0) + 1
            continue

        stats[r["status"]] = stats.get(r["status"], 0) + 1
        if r["status"] == "OK":
            total_rows += r["rows"]
            total_kept += r["kept"]
            total_drop += r["drop"]

        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / el if el > 0 else 0
            log(f"progress {i}/{len(todo)} | OK={stats['OK']} "
                f"no_daily={stats.get('SKIP_NO_DAILY', 0)} "
                f"all_dropped={stats.get('SKIP_ALL_DROPPED', 0)} "
                f"fail={stats.get('FAIL', 0)} | {rate:.2f} days/s | "
                f"elapsed {el/60:.1f} min")

    el = time.time() - t0
    drop_rate_all = total_drop / total_rows if total_rows else 0.0
    log(f"DONE in {el/60:.1f} min | built={stats['OK']} | "
        f"skipped_no_daily={stats.get('SKIP_NO_DAILY', 0)} | "
        f"skipped_all_dropped={stats.get('SKIP_ALL_DROPPED', 0)} | "
        f"fail={stats.get('FAIL', 0)} | rows_kept={total_kept} | "
        f"rows_dropped={total_drop} | overall_drop_rate={drop_rate_all:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
