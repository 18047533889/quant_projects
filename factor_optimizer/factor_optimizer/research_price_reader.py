"""Session-preserving bounded reads of registered adjusted VWAP prices.

The query cap is not a total RSS guarantee. The caller admits the final panel
together with all other resident cohort buffers before calling this reader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from data_access.read.query_budget import QueryBudget

QUERY_RESULT_BYTES = 32 * 1024**2
ESTIMATED_LONG_ROW_BYTES = 128


def read_session_vwap(store, sessions, assets):
    """Return float64 prices on the exact supplied session and asset axes.

Missing prices remain NaN, never forward-filled or removed from the calendar.
Duplicate daily observations and off-axis responses fail closed.
"""
    calendar = pd.DatetimeIndex(sessions)
    if (calendar.empty or calendar.hasnans or calendar.has_duplicates
            or not calendar.is_monotonic_increasing or calendar.tz is not None
            or not calendar.equals(calendar.normalize())):
        raise ValueError("sessions must be unique increasing naive midnight dates")
    if len(calendar) > 10000:
        raise ValueError("session read is limited to 10000 calendar rows")
    asset_ids = tuple(assets)
    if (not 1 <= len(asset_ids) <= 5000
            or any(not isinstance(a, str) or not a for a in asset_ids)
            or len(set(asset_ids)) != len(asset_ids)):
        raise ValueError("assets must contain 1..5000 unique nonempty strings")
    chunk_days = min(64, QUERY_RESULT_BYTES //
                     (ESTIMATED_LONG_ROW_BYTES * len(asset_ids)))
    values = np.full((len(calendar), len(asset_ids)), np.nan, dtype=np.float64)
    receipts = []
    for offset in range(0, len(calendar), chunk_days):
        window = calendar[offset:offset + chunk_days]
        handle = store.read(
            "ashare_stock_daily_adj", columns=["TradeDate", "Symbol", "AdjVwap"],
            time_range=(window[0], window[-1]), instrument_filter=list(asset_ids),
            result="arrow", query_budget=QueryBudget(
                max_scan_files=len(window), max_rows=len(window) * len(asset_ids),
                max_result_bytes=QUERY_RESULT_BYTES,
            ),
        )
        try:
            rows = handle.to_pandas()
            identity = getattr(handle, "read_identity", None)
            receipts.append({
                "dataset": "ashare_stock_daily_adj",
                "first_session": window[0].isoformat(),
                "last_session": window[-1].isoformat(),
                "rows": len(rows),
                "max_result_bytes": QUERY_RESULT_BYTES,
                "read_identity_digest": getattr(identity, "digest", None),
                "source_snapshot": getattr(identity, "source_snapshot", None),
                "provenance_status": getattr(identity, "provenance_status", "unavailable"),
            })
        finally:
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        required = {"TradeDate", "Symbol", "AdjVwap"}
        if not required.issubset(rows.columns):
            raise ValueError("VWAP response missing required date/asset/price columns")
        if rows.empty:
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(rows["TradeDate"]))
        if (dates.hasnans or dates.tz is not None
                or not dates.equals(dates.normalize())
                or not dates.isin(window).all()
                or not rows["Symbol"].isin(asset_ids).all()):
            raise ValueError("VWAP response contains invalid or off-axis observations")
        rows = rows.assign(TradeDate=dates)
        if rows.duplicated(["TradeDate", "Symbol"]).any():
            raise ValueError("VWAP response contains duplicate daily asset observations")
        if rows["AdjVwap"].dtype.kind not in "fiu":
            raise ValueError("VWAP response prices must be real numeric values")
        wide = rows.pivot(index="TradeDate", columns="Symbol", values="AdjVwap")
        values[offset:offset + len(window)] = wide.reindex(
            index=window, columns=asset_ids).to_numpy(dtype=np.float64)
        del rows, wide
    panel = pd.DataFrame(values, index=calendar, columns=asset_ids)
    panel.attrs["data_access_price_reads"] = receipts
    return panel
