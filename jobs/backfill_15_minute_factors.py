#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill the '15 minute factors' that remain empty (0-col placeholders) in
weekly_backtest_output/factor_matrices_all/.

The last session had already populated:
  DSL-only (13): amount_volume_corr kurt_volume_std minute_cond_vol_corr_std
    minute_corr_cond_vol_median minute_mutation_skew_mean
    minute_price_deviation_zscore_mean minute_range_skew skew_intraday_range
    std_ret vol_ret_corr volume_delta_corr vwap_deviation vwap_deviation_end_of_day
  daily HASCODE (6): breakout_hold_persistence_20d defended_close_high_retest_participation_20_5
    downside_shock_accumulation_asym failed_retest_reversion_20_5 market_phase_atr_expansion
    trend_path_efficiency_5_60

Still 0-col placeholders -> 7 'high_volume_*'/'impact'/'vwap_deviation_intensity' pages which
are ALL minute-tools factors with real LQTP python code. They execute the per-day / per-symbol
minute-bar code in backfill_28_factors.run_minute_pages, but write over the FULL universe index
(2585 days x 5460 symbols) to match the reference layout of the other factor pages.

Output column layout = reference (abnormality_asymmetry.parquet: 2585 x 5460) with the SAME
column order — the half-computed sample-universe columns are dropped and replaced by the full
per-symbol columns, so every page is (2585, 5460).

Usage:
  .venv/bin/python jobs/backfill_15_minute_factors.py
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
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))

import extend_all_456 as E
from compute_minute_factors import _build_minute_tools
from patch_alpha_tools import patch_tools
patch_tools()  # top-level alpha_tools/minute_tools helpers

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
MINUTE_DIR = HOME / "cos_data" / "StockMinuteBar"

PAGES = [
    "amount_weighted_impact",
    "high_volume_amount_weighted_impact",
    "high_volume_amount_weighted_squared_impact",
    "high_volume_squared_return_weighted_impact",
    "high_volume_volume_weighted_squared_impact",
    "vwap_squared_deviation_intensity",
    "vwap_deviation_intensity_squared_amount_weighted",
    "skew_weighted_volume_event_intensity",
    "skew_weighted_volume_event_intensity_close",
]

# 日线 HASCODE 页：minute-bar 数据里没有这些（它们是纯日线 code）。这些文件是
# 0-col 占位；run_daily_exec_pages 已有结果（1854 days × 5447 syms）但只在
# 日线脚本里出现。backfill_15 不管它们。
DAILY_HASCODE_PAGES = {
    "breakout_hold_persistence_20d", "defended_close_high_retest_participation_20_5",
    "downside_shock_accumulation_asym", "failed_retest_reversion_20_5",
    "market_phase_atr_expansion", "trend_path_efficiency_5_60",
}

LQTP = json.load(open(HOME / "factor_delivery_converted" / "formula_lqtp_all.json"))
LQTP_PAGES = {r.get("page_name"): r for r in LQTP if r.get("page_name")}


def load_reference(page: str) -> pd.DataFrame:
    """Live full-universe reference matrix (index + column order, no values)."""
    ref = pd.read_parquet(FV_DIR / "abnormality_asymmetry.parquet")
    return ref


def compute_minute_page(page: str, ref: pd.DataFrame, smoke: bool = False) -> pd.DataFrame:
    """Evaluate a minute factor's LQTP code for every symbol in ref.columns across all days.

    Reuses the exact per-day/per-symbol evaluation of backfill_28_factors.run_minute_pages
    but loops over the reference universe and cumulates into the reference index. Only the
    rows that exist in the minute-bar files (2024-01..2025-12) receive real values.
    """
    local_files = sorted(MINUTE_DIR.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in local_files) + "]"

    r = LQTP_PAGES.get(page, {})
    code = r.get("code") or ""
    if not code or "NotImplementedError" in code:
        raise ValueError(f"{page}: no executable code")

    # 命名空间修复（与 backfill_28 相同：df_copy→df、talib 移除、get_up_space 3参）
    code = code.replace("df_copy", "df").replace("df_", "df")
    code = re.sub(r"^import talib.*$", "", code, flags=re.M)
    func_match = re.search(r"def\s+(factor_[a-zA-Z0-9_]+)\s*\(", code)
    func_name = func_match.group(1) if func_match else None
    if func_name is None:
        raise ValueError(f"{page}: no factor_* function")

    symbols = list(ref.columns)
    out = pd.DataFrame(np.nan, index=ref.index, columns=symbols, dtype="float32")

    con = duckdb.connect()
    # 分钟文件只有 2024-01 .. 2025-12 两年；只对范围内日期求值。
    start_d, end_d = "2024-01-02", "2025-12-31"
    df_all = con.execute(f"""
        SELECT Symbol, strftime(TradeDate, '%Y-%m-%d') AS TradingDay, Close AS close,
               Volume AS volume, Amount AS amount, Vwap AS vwap
        FROM read_parquet({files_str})
        WHERE strftime(TradeDate, '%Y-%m-%d') >= '{start_d}'
          AND strftime(TradeDate, '%Y-%m-%d') <= '{end_d}'
    """).df()
    con.close()
    if df_all.empty:
        print(f"  [分钟] {page}: 无分钟数据（{start_d}~{end_d}），返回空矩阵")
        return out

    ns = {"np": np, "pd": pd, "minute_tools": E.minute_tools}
    t0 = time.time()
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
                result = None
                if func_name in local_ns:
                    result = local_ns[func_name](df_bar)
                if result is not None:
                    val = float(result.iloc[0]) if (hasattr(result, "iloc") and len(result) > 0) else (float(result.values[0]) if hasattr(result, "values") else float(result))
                    if np.isfinite(val):
                        day_ts = pd.Timestamp(day)
                        if day_ts in out.index:
                            out.loc[day_ts, sym] = val
            except Exception:
                pass
    pct = out.notna().sum().sum() / max(out.size, 1) * 100
    print(f"  ✓ [分钟] {page}: 有效 {pct:.1f}% 形状{out.shape} 耗时{time.time()-t0:.0f}s")
    return out


def main():
    t0 = time.time()
    ref = load_reference("abnormality_asymmetry")
    print(f"[15min] reference {ref.shape} (index {ref.index[0].date()}..{ref.index[-1].date()})", flush=True)

    for page in PAGES:
        p = FV_DIR / f"{page}.parquet"
        if p.exists() and p.stat().st_size > 30000:
            print(f"  [skip] {page}: 已有回填")
            continue
        t1 = time.time()
        try:
            mat = compute_minute_page(page, ref)
        except Exception as e:
            print(f"  ✗ [分钟] {page}: {str(e)[:150]}")
            continue
        sub = mat.dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print(f"  ✗ [分钟] {page}: 全空，不写盘")
            continue
        sub = sub.reindex(index=ref.index, columns=ref.columns)
        sub.to_parquet(p)
        print(f"  [写盘] {page}: {sub.shape} 列{int(sub.notna().sum().sum())} 耗时{time.time()-t1:.0f}s", flush=True)

    print(f"[15min] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()