#!/usr/bin/env python3
"""Check whether daily source panels are invariant over overlapping query windows."""
import json
import os

import numpy as np
import pandas as pd

os.environ["ASHARE_PARQUET_ROOT"] = "/home/sunhaiwei/cos_data"
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
os.environ["DATA_ACCESS_RUN_MODE"] = "interactive_research"

from factor_engine.storage.sources.data_access_source import DataAccessSource

SYMBOLS = ["000001.SZ", "000002.SZ", "000063.SZ", "000333.SZ", "000651.SZ", "000858.SZ", "600000.SH", "600519.SH"]
STARTS = ["2025-01-01", "2025-04-03", "2025-10-21", "2025-11-07"]


def load(start, field, physical):
    source = DataAccessSource(
        dataset="ashare_stock_daily_adj",
        fields={field: physical},
        start_date=start,
        end_date="2026-04-30",
        instrument_filter=SYMBOLS,
        run_mode="interactive_research",
        production=False,
        read_auto=False,
    )
    try:
        value = source.load_column(field)
        if isinstance(value, pd.DataFrame):
            value = value.stack(dropna=False)
        return value.sort_index()
    finally:
        source.close()


baseline = {field: load(STARTS[0], field, physical) for field, physical in (("close", "AdjClose"), ("ret", "Return"))}
for start in STARTS[1:]:
    for field, physical in (("close", "AdjClose"), ("ret", "Return")):
        short = load(start, field, physical)
        long = baseline[field].reindex(short.index)
        a = long.to_numpy(dtype=float)
        b = short.to_numpy(dtype=float)
        finite = np.isfinite(a) & np.isfinite(b)
        rolling = None
        if field == "close" and isinstance(short.index, pd.MultiIndex):
            date_level = "TradeDate" if "TradeDate" in short.index.names else short.index.names[0]
            symbol_level = "Symbol" if "Symbol" in short.index.names else short.index.names[1]
            long_roll = baseline[field].groupby(level=symbol_level, group_keys=False).rolling(252).mean()
            short_roll = short.groupby(level=symbol_level, group_keys=False).rolling(252).mean()
            # groupby.rolling may prepend the group key; compare only common tail values.
            long_tail = np.asarray(long_roll.groupby(level=0).tail(4), dtype=float)
            short_tail = np.asarray(short_roll.groupby(level=0).tail(4), dtype=float)
            rolling = {
                "date_level": date_level,
                "symbol_level": symbol_level,
                "manual_rolling_tail_max_abs_error": float(np.nanmax(np.abs(long_tail - short_tail))),
            }
        print(json.dumps({
            "start": start,
            "field": field,
            "long_rows": int(len(baseline[field])),
            "short_rows": int(len(short)),
            "overlap_index_equal": bool(short.index.equals(long.index)),
            "finite_mask_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
            "max_abs_error": float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0,
            "max_rel_error": float(np.max(np.abs(a[finite] - b[finite]) / np.maximum(np.abs(a[finite]), np.finfo(float).tiny))) if finite.any() else 0.0,
            "index_names": list(short.index.names),
            "manual_rolling_252": rolling,
        }))
