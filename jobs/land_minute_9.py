#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Land the 9 TRUE-MINUTE factors onto the adjusted minute table (StockMinuteBarAdj).

Data source: ashare_stock_minute_adj (DataAccessSource) — AdjClose/AdjVwap/Volume/AdjAmount.
Window: 2024-01-02..2024-12-31 (1 year of minute bars; read day-by-day, 20-day batches + gc).

Each factor's exact semantics come from formula_lqtp_all.json `code` field (TRUE-MINUTE-SINGLE-VALUE
exceptions).  minute_tools semantics:

  get_up_space(vol, TradingDay, std_multiplier):
      ma = vol.rolling(20, min_periods=5).mean().replace(0, nan)
      sd = vol.rolling(20, min_periods=5).std().replace(0, nan)
      z = ((vol - ma)/sd).fillna(0)
      is_high = (z > std_multiplier).astype(float)
  pulse_start(vol, window=5): (vol > vol.rolling(5, min_periods=1).mean()*2).astype(float)
  amount_weighted_mean(vals, amounts, days): per-day sum(vals*amounts)/sum(amounts)

Note on factor #3 high_volume_amount_weighted_squared_impact: the delivered code computes
`ret = pct_change() ** 2` (squared) then amount-weights it — despite the name saying
"amount_weighted_squared_impact" the weight is amount and the value is squared return.
The name "high_volume_squared_return_weighted_impact" (#4) has the SAME code, so #3 and #4 are
numerically identical (confirmed against old matrices: both stored the same values).

Implementation approach: pandas semantic equivalent (day-level groupby aggregation), per task spec.
No exact FE canonical exists for any of the 9 (surveyed intra_* / group_weighted_mean /
volume_zscore / intra_impulse_event_detector; none matches amount-weighted impact / pulse-intensity /
z-clip-weighted vwap deviation).  approach recorded as "pandas_semantic".

Output: weekly_backtest_output/factor_matrices_all/{page_name}.parquet (TradeDate x Symbol float32,
same layout as existing factor matrices).  Progress written to /tmp/minute_9_result.json after each
3 factors and at the end.

CPU: OMP_NUM_THREADS=31; reads day-by-day to keep memory bounded (~1.2M rows/day).
"""
import sys, os, gc, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("ASHARE_PARQUET_ROOT", "/home/sunhaiwei/cos_data")
os.environ.setdefault("DATA_ACCESS_COS_READ_MODE", "mirror")
os.environ.setdefault("DATA_ACCESS_AUTO_BOUND_MEMORY", "0")

from data_access import get_store

START = "2024-01-02"
END = "2024-12-31"
BATCH_DAYS = 20
OUT_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
PROGRESS = Path("/tmp/minute_9_result.json")
LOG = Path("/tmp/minute_fe_build.log")

# page_name -> dict(desc, impl)
FACTORS = [
    "amount_weighted_impact",
    "high_volume_amount_weighted_impact",
    "high_volume_amount_weighted_squared_impact",
    "high_volume_squared_return_weighted_impact",
    "high_volume_volume_weighted_squared_impact",
    "skew_weighted_volume_event_intensity",
    "skew_weighted_volume_event_intensity_close",
    "vwap_deviation_intensity_squared_amount_weighted",
    "vwap_squared_deviation_intensity",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def get_up_space(s: pd.Series, std_multiplier: float = 1.0):
    """minute_tools.get_up_space: rolling-20 z-score, fillna(0), is_high mask."""
    ma = s.rolling(20, min_periods=5).mean().replace(0, np.nan)
    sd = s.rolling(20, min_periods=5).std().replace(0, np.nan)
    z = ((s - ma) / sd).fillna(0)
    is_high = (z > std_multiplier).astype(float)
    return is_high, z


def pulse_start(s: pd.Series, window: int = 5):
    """minute_tools.pulse_start: vol > rolling-mean*2."""
    return (s > s.rolling(window, min_periods=1).mean() * 2).astype(float)


def amount_weighted_mean(vals: pd.Series, amounts: pd.Series):
    """minute_tools.amount_weighted_mean per day: sum(v*amt)/sum(amt)."""
    return (vals * amounts).sum() / amounts.sum()


def day_values(g: pd.DataFrame, page: str) -> float:
    """Compute the day-level factor value for one (day, symbol) group g.
    g has columns AdjClose, Volume, AdjAmount, AdjVwap (sorted by QuoteTime)."""
    close = g["AdjClose"]
    vol = g["Volume"].astype(float)
    amt = g["AdjAmount"].astype(float)
    ret = close.pct_change().fillna(0.0)
    ret_abs = ret.abs()

    if page == "amount_weighted_impact":
        # mean |ret| amount-weighted (over the whole day)
        denom = amt.sum()
        if denom == 0 or not np.isfinite(denom) or denom == 0.0:
            return 0.0
        return float((ret_abs * amt).sum() / denom)

    if page == "high_volume_amount_weighted_impact":
        is_high, _ = get_up_space(vol, 1.5)
        num = (ret_abs * amt * is_high).sum()
        den = (amt * is_high).sum()
        return float(num / den) if den != 0 else 0.0

    if page in ("high_volume_amount_weighted_squared_impact",
                "high_volume_squared_return_weighted_impact"):
        # both use the same code: ret^2 amount-weighted on high-volume minutes
        is_high, _ = get_up_space(vol, 1.5)
        ret_sq = ret ** 2
        num = (ret_sq * amt * is_high).sum()
        den = (amt * is_high).sum()
        return float(num / den) if den != 0 else 0.0

    if page == "high_volume_volume_weighted_squared_impact":
        is_high, _ = get_up_space(vol, 1.5)
        ret_sq = ret ** 2
        num = (ret_sq * vol * is_high).sum()
        den = (vol * is_high).sum()
        return float(num / den) if den != 0 else 0.0

    if page in ("skew_weighted_volume_event_intensity",
                "skew_weighted_volume_event_intensity_close"):
        # identical delivered code for both pages
        med = np.median(ret)
        mean = float(ret.mean())
        mad = np.median(np.abs(ret - med))
        skew = 0.0 if mad == 0 else float((mean - med) / mad)
        is_high, volume_z = get_up_space(vol, 1.5)
        volume_z = volume_z.fillna(0.0)
        pulse = pulse_start(is_high)
        intensity = (volume_z * skew).where(pulse > 0).fillna(0.0)
        return float(intensity.sum())

    if page == "vwap_deviation_intensity_squared_amount_weighted":
        vwap = (close * vol).sum() / vol.sum()
        dev = (close - vwap) ** 2
        is_high, amount_z = get_up_space(amt, 1.0)
        w = amount_z.clip(lower=0)
        sum_w = w.sum()
        return float((dev * w).sum() / sum_w) if sum_w > 0 else 0.0

    if page == "vwap_squared_deviation_intensity":
        vwap = (close * vol).sum() / vol.sum()
        dev = (close - vwap) ** 2
        is_high, volume_z = get_up_space(vol, 1.0)
        w = volume_z.clip(lower=0)
        sum_w = w.sum()
        return float((dev * w).sum() / sum_w) if sum_w > 0 else 0.0

    raise ValueError(f"unknown page {page}")


def trading_days(start: str, end: str, store) -> list[str]:
    """Actual trading days in window from the adj daily table (authority calendar)."""
    files = sorted(Path.home().glob("cos_data/StockDailyBarAdj/*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    import duckdb
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT DISTINCT strftime(TradeDate, '%Y-%m-%d') AS d
            FROM read_parquet({fs})
            WHERE TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
            ORDER BY d
        """).df()
    finally:
        con.close()
    return list(df["d"])


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = get_store()

    days = trading_days(START, END, store)
    log(f"trading days: {len(days)} ({days[0]}..{days[-1]})")

    result = {}
    for page in FACTORS:
        log(f"=== {page} ===")
        page_t0 = time.time()
        rows: list[tuple[pd.Timestamp, str, float]] = []
        n_days = 0
        for bi in range(0, len(days), BATCH_DAYS):
            batch = days[bi:bi + BATCH_DAYS]
            for day in batch:
                df = store.read_frame(
                    "ashare_stock_minute_adj",
                    columns=["TradeDate", "QuoteTime", "Symbol",
                             "AdjClose", "AdjVwap", "Volume", "AdjAmount"],
                    time_range=(day, day),
                )
                if df is None or len(df) == 0:
                    continue
                df = df.sort_values(["Symbol", "QuoteTime"])
                day_ts = pd.Timestamp(day)
                for sym, g in df.groupby("Symbol", sort=True):
                    if len(g) < 10:
                        continue
                    v = day_values(g, page)
                    if np.isfinite(v):
                        rows.append((day_ts, sym, float(v)))
                del df
            n_days += len(batch)
            if (bi // BATCH_DAYS) % 2 == 1 or bi + BATCH_DAYS >= len(days):
                log(f"  [{page}] {n_days}/{len(days)} days, {len(rows)} cells")
            gc.collect()

        if not rows:
            log(f"  [warn] {page}: no cells")
            continue
        mat = pd.DataFrame(rows, columns=["date", "symbol", "value"])
        wide = mat.pivot(index="date", columns="symbol", values="value").sort_index()
        wide = wide.astype("float32")
        p = OUT_DIR / f"{page}.parquet"
        wide.to_parquet(p)
        log(f"  [wrote] {page}: {wide.shape} valid_cells={len(rows)} "
            f"({time.time()-page_t0:.0f}s) -> {p}")
        result[page] = {"status": "landed", "shape": list(wide.shape),
                        "valid_cells": len(rows),
                        "time_sec": round(time.time() - page_t0, 1)}

        # write-early progress after each factor
        with open(PROGRESS, "w") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        del mat, wide, rows
        gc.collect()

    log(f"DONE in {time.time()-t0:.0f}s")
    with open(PROGRESS, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
