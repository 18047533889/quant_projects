#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extend the 9 TRUE-MINUTE factor matrices to 2026-01-02..2026-08-27.

The original landing window (land_minute_9.py) was 2024-01-02..2024-12-31.
This script recomputes the same 9 factors on the 2026 window (minute bars
StockMinuteBarAdj), producing
  weekly_backtest_output/factor_matrices_all_2026minute/{page}.parquet
(rows 2026-01-02..2026-08-27, cols = symbols present that day).
Implementation is a vectorized-per-day equivalent of land_minute_9.day_values
(validated 0-error against the reference on 2024-01-02). Output columns are
the same symbols as the 2026 daily-wide Vwap_adj panel (5461 syms incl. new
listings; those with no minute coverage stay NaN).

CPU: 31 workers, OMP_NUM_THREADS=31.  OMP threads also constrain pandas
groupby transforms.  Write-early per 5 days.
"""
import os, sys, gc, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
OUT_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all_2026minute"
MIN_ROOT = Path.home() / "cos_data" / "StockMinuteBarAdj"
VWAP_F = PROJECT / "lightgbm_qs/data/build/ohlcv_adj_wide/Vwap_adj.parquet"
LOG = Path("/tmp/rob26_minute_extend.log")
PROGRESS = Path("/tmp/rob26_minute_extend.json")

PAGES = [
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

START = "2026-01-02"
END = "2026-08-27"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def compute_day(df: pd.DataFrame, syms: np.ndarray, code2idx: dict, bounds: np.ndarray):
    """All 9 factors for one day. df sorted by Symbol/QuoteTime. Returns (out, codes)."""
    vol = df["Volume"].astype(float)
    amt = df["AdjAmount"].astype(float)
    close = df["AdjClose"].astype(float)
    g = df.groupby("Symbol", sort=True)
    vol_ma = g["Volume"].transform(lambda s: s.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    vol_sd = g["Volume"].transform(lambda s: s.rolling(20, min_periods=5).std()).replace(0, np.nan)
    z_vol = ((vol - vol_ma) / vol_sd).fillna(0.0)
    amt_ma = g["AdjAmount"].transform(lambda s: s.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    amt_sd = g["AdjAmount"].transform(lambda s: s.rolling(20, min_periods=5).std()).replace(0, np.nan)
    z_amt = ((amt - amt_ma) / amt_sd).fillna(0.0)
    ret = g["AdjClose"].pct_change().fillna(0.0)
    ret_abs = ret.abs()
    ret_sq = ret ** 2

    def rs(x):
        return np.add.reduceat(x.to_numpy(), bounds[:-1])

    is_high_vol = (z_vol > 1.5).astype(float)
    out = {}
    den = rs(amt)
    num = rs(ret_abs * amt)
    out["amount_weighted_impact"] = num / np.where(den == 0, np.nan, den)

    den = rs(amt * is_high_vol)
    num = rs(ret_abs * amt * is_high_vol)
    out["high_volume_amount_weighted_impact"] = num / np.where(den == 0, np.nan, den)

    den = rs(amt * is_high_vol)
    num = rs(ret_sq * amt * is_high_vol)
    v = num / np.where(den == 0, np.nan, den)
    out["high_volume_amount_weighted_squared_impact"] = v
    out["high_volume_squared_return_weighted_impact"] = v

    den = rs(vol * is_high_vol)
    num = rs(ret_sq * vol * is_high_vol)
    out["high_volume_volume_weighted_squared_impact"] = num / np.where(den == 0, np.nan, den)

    # skew factor: per-symbol median-based skew; pulse = is_high > rolling5mean*2
    def grp_skew(s):
        med = s.median()
        mean = s.mean()
        mad = np.median(np.abs(s - med))
        return 0.0 if mad == 0 else (mean - med) / mad

    skew_map = ret.groupby(df["Symbol"]).apply(grp_skew, include_groups=False)
    skew = df["Symbol"].map(skew_map).fillna(0.0)
    pulse_mean = is_high_vol.groupby(df["Symbol"]).transform(
        lambda s: s.rolling(5, min_periods=1).mean() * 2)
    pulse_bin = (is_high_vol > pulse_mean).astype(float)
    intensity = (z_vol * skew).where(pulse_bin > 0).fillna(0.0)
    out["skew_weighted_volume_event_intensity"] = rs(intensity)
    out["skew_weighted_volume_event_intensity_close"] = rs(intensity)

    vwap = rs(close * vol) / np.where(rs(vol) == 0, np.nan, rs(vol))
    vwap_b = np.array([vwap[code2idx[s]] for s in syms])
    dev = (close - vwap_b) ** 2
    w = z_amt.clip(lower=0)
    den = rs(w)
    num = rs(dev * w)
    out["vwap_deviation_intensity_squared_amount_weighted"] = num / np.where(den == 0, np.nan, den)
    w = z_vol.clip(lower=0)
    den = rs(w)
    num = rs(dev * w)
    out["vwap_squared_deviation_intensity"] = num / np.where(den == 0, np.nan, den)
    return out


def main():
    os.environ["OMP_NUM_THREADS"] = "31"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vw = pd.read_parquet(VWAP_F)
    # extend the vwap panel to 08-27 by appending StockDailyBarAdj rows
    from jobs.adj_vwap_common import load_adj_vwap
    tail = load_adj_vwap("2026-08-25", "2026-08-27")
    vw_full = pd.concat([vw, tail]).sort_index()
    # drop the NaN return row: last row (08-27) has no t+2 return => robustness
    # script already handles that; keep all dates.
    univ_cols = vw_full.columns.tolist()
    univ_pos = {s: i for i, s in enumerate(univ_cols)}
    day_files = sorted(MIN_ROOT.glob("*.parquet"))
    days = [d for d in day_files if START <= d.stem <= END]
    log(f"days {len(days)} ({days[0].stem}..{days[-1].stem}) univ_cols {len(univ_cols)}")

    # init accumulators: index = all days; columns = univ_cols (5461)
    days_str = [d.stem for d in days]
    acc = {p: pd.DataFrame(np.nan, index=days_str, columns=univ_cols, dtype="float32") for p in PAGES}
    t0 = time.time()
    done = 0
    result = {}
    for i, f in enumerate(days):
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        df = df.sort_values(["Symbol", "QuoteTime"])
        syms = df["Symbol"].values
        codes = pd.unique(syms)
        code2idx = {s: i for i, s in enumerate(codes)}
        idx_arr = np.array([code2idx[s] for s in syms])
        bounds = np.flatnonzero(np.diff(idx_arr) != 0) + 1
        bounds = np.concatenate([[0], bounds, [len(df)]])
        out = compute_day(df, syms, code2idx, bounds)
        for p in PAGES:
            row_vals = out[p]
            for j, s in enumerate(codes):
                c = univ_pos.get(s)
                if c is not None:
                    acc[p].iat[i, c] = np.float32(row_vals[j])
        done += 1
        if done % 5 == 0 or i == len(days) - 1:
            el = time.time() - t0
            log(f"  {done}/{len(days)} days {el:.0f}s ({el/max(done,1)*done:.0f}s est total) "
                f"rate {el/max(done,1):.1f}s/day")
            # write-early
            for p in PAGES:
                (OUT_DIR / f"{p}.parquet").write_bytes(_df_to_parquet(acc[p]))
            PROGRESS.write_text(json.dumps({"done": done, "total": len(days)}))
        del df, out
        gc.collect()

    for p in PAGES:
        acc[p].to_parquet(OUT_DIR / f"{p}.parquet")
        log(f"[wrote] {p} {acc[p].shape}")
    log(f"DONE {done} days {time.time()-t0:.0f}s")


def _df_to_parquet(df):
    import io
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


if __name__ == "__main__":
    main()
