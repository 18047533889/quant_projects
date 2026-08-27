#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill the REMAINING factors still left as 0-col placeholders in
factor_matrices_all/ after the previous session's work:

  daily-HASCODE (6, minute bars do NOT cover them -> daily bars full universe):
    breakout_hold_persistence_20d defended_close_high_retest_participation_20_5
    downside_shock_accumulation_asym failed_retest_reversion_20_5
    market_phase_atr_expansion trend_path_efficiency_5_60

  minute (skew, 2): skew_weighted_volume_event_intensity
    skew_weighted_volume_event_intensity_close

Output layout = reference abnormality_asymmetry.parquet (2585 x 5460, index 2016-01-04..2026-08-24)
with the same column order; daily rows 2016..2026 are fully populated (market covers them),
minute rows only where minute bars exist (2024-01..2025-12) per the 7 impact pages already written.

Semantics for daily pages reuse jobs/backfill_28_factors.run_daily_exec_pages (extend_all_456
_execute_factor_code on wide daily OHLCV + intermediates); semantics for skew pages reuse
run_minute_pages (per-day/per-symbol exec of the LQTP minute code).

Usage:
  .venv/bin/python jobs/backfill_remaining_factors.py
"""
import sys, json, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
HOME = Path.home()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "jobs"))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))

import extend_all_456 as E
from patch_alpha_tools import patch_tools
patch_tools()

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
DAILY = HOME / "cos_data" / "StockDailyBar"
MINUTE_DIR = HOME / "cos_data" / "StockMinuteBar"

DAILY_PAGES = [
    "breakout_hold_persistence_20d",
    "defended_close_high_retest_participation_20_5",
    "downside_shock_accumulation_asym",
    "failed_retest_reversion_20_5",
    "market_phase_atr_expansion",
    "trend_path_efficiency_5_60",
]
SKEW_PAGES = [
    "skew_weighted_volume_event_intensity",
    "skew_weighted_volume_event_intensity_close",
]

LQTP = json.load(open(HOME / "factor_delivery_converted" / "formula_lqtp_all.json"))
LQTP_PAGES = {r.get("page_name"): r for r in LQTP if r.get("page_name")}

# 与 backfill_28 相同的每-日/每-symbol 分钟求值（含 df_copy→df / get_up_space 修复）。
def compute_minute_page(page: str, ref: pd.DataFrame) -> pd.DataFrame:
    local_files = sorted(MINUTE_DIR.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in local_files) + "]"
    r = LQTP_PAGES.get(page, {})
    code = r.get("code") or ""
    if not code or "NotImplementedError" in code:
        raise ValueError(f"{page}: no executable minute code")
    code = code.replace("df_copy", "df").replace("df_", "df")
    code = re.sub(r"^import talib.*$", "", code, flags=re.M)
    func_match = re.search(r"def\s+(factor_[a-zA-Z0-9_]+)\s*\(", code)
    func_name = func_match.group(1) if func_match else None
    if func_name is None:
        raise ValueError(f"{page}: no factor_* function")

    symbols = list(ref.columns)
    out = pd.DataFrame(np.nan, index=ref.index, columns=symbols, dtype="float32")
    con = duckdb.connect()
    df_all = con.execute(f"""
        SELECT Symbol, strftime(TradeDate, '%Y-%m-%d') AS TradingDay, Close AS close,
               Volume AS volume, Amount AS amount, Vwap AS vwap
        FROM read_parquet({files_str})
        WHERE strftime(TradeDate, '%Y-%m-%d') >= '2024-01-02'
          AND strftime(TradeDate, '%Y-%m-%d') <= '2025-12-31'
    """).df()
    con.close()
    if df_all.empty:
        return out
    ns = {"np": np, "pd": pd, "minute_tools": E.minute_tools}
    t0 = time.time()
    n_cells = 0
    for sym, g in df_all.groupby("Symbol"):
        if sym not in symbols:
            continue
        for day, dg in g.groupby("TradingDay"):
            if len(dg) < 10:
                continue
            df_bar = pd.DataFrame({
                "close_price": dg["close"].values, "close": dg["close"].values,
                "open": dg["close"].values, "high": dg["close"].values,
                "low": dg["close"].values, "volume": dg["volume"].values,
                "amount": dg["amount"].values, "vwap": dg["vwap"].values,
                "TradingDay": [day] * len(dg),
            })
            local_ns = dict(ns); local_ns["df"] = df_bar
            try:
                exec(code, local_ns)
                result = local_ns[func_name](df_bar) if func_name in local_ns else None
                if result is not None:
                    val = float(result.iloc[0]) if (hasattr(result, "iloc") and len(result) > 0) else (float(result.values[0]) if hasattr(result, "values") else float(result))
                    if np.isfinite(val):
                        day_ts = pd.Timestamp(day)
                        if day_ts in out.index:
                            out.loc[day_ts, sym] = val
                            n_cells += 1
            except Exception:
                pass
    print(f"  ✓ [minute] {page}: valid_cells={n_cells} 耗时{time.time()-t0:.0f}s")
    return out


def load_wide(symbols, start, end, col_map=None):
    con = duckdb.connect()
    files = sorted(DAILY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    sel = ", ".join(f"{sql} as {name}" for sql, name in col_map.items()) if col_map else None
    q = f"""
        SELECT TradeDate as date, Symbol as symbol{(', ' + sel) if sel else ''}
        FROM read_parquet({fs})
        WHERE Symbol IN ({','.join(f"'{s}'" for s in symbols)})
          AND TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
    """
    df = con.execute(q).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for c in df.columns.drop(["date", "symbol"]):
        sub = df[["date", "symbol", c]].drop_duplicates(["date", "symbol"], keep="first")
        out[c] = sub.pivot(index="date", columns="symbol", values=c).sort_index().astype("float32")
    return out


def compute_daily_page(page: str, ref: pd.DataFrame) -> pd.DataFrame:
    """run_daily_exec_pages 等价（extend_all_456._execute_factor_code）。复用 backfill_28
    的命名空间/代码修复，输出 reindex 到 reference layout。"""
    symbols = list(ref.columns)
    start, end = "2016-01-04", "2026-08-24"
    mkt = {}
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        mkt[c] = load_wide(symbols, start, end, {c.title() if c != "volume" else "Volume": c})[c]
    dates = mkt["close"].index
    symbols = list(mkt["close"].columns)
    print(f"  [daily] {page}: 加载行情 {len(dates)} 天 x {len(symbols)} 股", flush=True)

    intermediates = E._build_intermediates(mkt, symbols, dates)
    r = LQTP_PAGES.get(page, {})
    code = r.get("code") or ""
    if not code or "NotImplementedError" in code:
        return pd.DataFrame(np.nan, index=ref.index, columns=ref.columns, dtype="float32")
    code = code.replace("df_copy", "df").replace("df_", "df")
    code = re.sub(r"^import talib.*$", "", code, flags=re.M)
    code = code.replace("talib.TRANGE(", "E.talib.TRANGE(").replace("talib.ATR(", "E.talib.ATR(")
    code = re.sub(
        r"alpha_tools\.decompose_return_volatility_direction\(([^,]+), min_periods=\d+, window=(\d+)",
        r"alpha_tools.decompose_return_volatility_direction(\1, window=\2, min_periods=\2",
        code)
    extra = {"ret": "close.pct_change()", "abs_ret": "close.pct_change().abs()",
             "trange": "high-low", "prior_high": "high.shift(1)", "prior_low": "low.shift(1)",
             "prior_close": "close.shift(1)", "close_shift_5": "close.shift(5)"}
    for cname, expr in extra.items():
        if f"['{cname}']" in code or f'["{cname}"]' in code:
            intermediates.setdefault(cname, eval(expr, {"close": mkt["close"], "high": mkt["high"], "low": mkt["low"]}))
    if page == "market_phase_atr_expansion":
        code = '''def factor_market_phase_atr_expansion(df):
    tr = df['high'] - df['low']
    pc = df['close'].shift(1)
    tr_v = pd.concat([tr, (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    atr = tr_v.rolling(20, min_periods=5).mean()
    atr_long_mean = tr_v.rolling(120, min_periods=20).mean().replace(0.0, np.nan)
    signal = atr / atr_long_mean
    signal = signal.fillna(1.0)
    signal.name = 'factor_market_phase_atr_expansion'
    return signal'''
    t0 = time.time()
    mat = E._execute_factor_code(code, mkt, symbols, dates, intermediates=intermediates)
    pct = mat.notna().mean().mean() * 100 if mat.size else 0
    print(f"  ✓ [daily] {page}: 有效 {pct:.1f}% 形状{mat.shape} 耗时{time.time()-t0:.0f}s")
    return mat


def main():
    t0 = time.time()
    ref = pd.read_parquet(FV_DIR / "abnormality_asymmetry.parquet")
    print(f"[remaining] reference {ref.shape}", flush=True)

    # A. 6 个日线 HASCODE
    for page in DAILY_PAGES:
        p = FV_DIR / f"{page}.parquet"
        if p.exists() and p.stat().st_size > 30000:
            print(f"  [skip] {page}: 已有")
            continue
        try:
            mat = compute_daily_page(page, ref)
        except Exception as e:
            print(f"  ✗ [daily] {page}: {str(e)[:150]}")
            continue
        sub = mat.reindex(index=ref.index, columns=ref.columns).astype("float32").dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print(f"  ✗ [daily] {page}: 全空不写盘")
            continue
        sub.to_parquet(p)
        print(f"  [写盘] {page}: {sub.shape}", flush=True)

    # B. 2 个 skew 分钟
    for page in SKEW_PAGES:
        p = FV_DIR / f"{page}.parquet"
        if p.exists() and p.stat().st_size > 30000:
            print(f"  [skip] {page}: 已有")
            continue
        mat = compute_minute_page(page, ref)
        sub = mat.dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print(f"  ✗ [minute] {page}: 全空不写盘")
            continue
        sub = sub.reindex(index=ref.index, columns=ref.columns)
        sub.to_parquet(p)
        print(f"  [写盘] {page}: {sub.shape}", flush=True)

    print(f"[remaining] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()