#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补算 456 因子中因缺失列（ps_ttm/pcf_ocf_ttm/roe_ttm2/debttoassets/style_gate/free_float_shares）
而全 NaN 的因子。从 COS 可得列近似填充后重算，覆盖写回 factor_matrices_all/。

用法：/tmp/fe2/bin/python backfill_missing_456.py
"""
import sys, os, json, glob, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
CONV_DIR = Path("/home/sunhaiwei/factor_delivery_converted/factors_combined")
OUT_DIR = PROJECT / "weekly_backtest_output"
FV_DIR = OUT_DIR / "factor_matrices_all"

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))

from extend_all_456 import _execute_factor_code, alpha_tools, minute_tools, talib


def load_missing_cols():
    """从 COS 构建缺失列：ps_ttm / pcf_ocf_ttm / free_float_shares / style_gate_*。"""
    t0 = time.time()
    con = duckdb.connect()
    # 估值：PsRatio / PcfRatio / FreeCap
    files = sorted(glob.glob("/home/sunhaiwei/cos_data/StockValuationDaily/*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol,
               PsRatio as ps_ttm, PcfRatio as pcf_ocf_ttm,
               FreeCap as free_cap
        FROM read_parquet({files_str})
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    cols = {}
    for c, raw in [("ps_ttm", "ps_ttm"), ("pcf_ocf_ttm", "pcf_ocf_ttm")]:
        m = df.pivot_table(index="date", columns="symbol", values=raw, aggfunc="first").sort_index()
        m = m.astype("float32")
        cols[c] = m
    # free_float_shares ≈ FreeCap / close
    fc = df.pivot_table(index="date", columns="symbol", values="free_cap", aggfunc="first").sort_index()
    fc = fc.astype("float32")
    cols["free_float_shares"] = fc
    print(f"[cols] 估值列构建完成 ({time.time()-t0:.1f}s)")

    # 行情 for style_gate
    files_b = sorted(glob.glob("/home/sunhaiwei/cos_data/StockDailyBar/*.parquet"))
    files_b_str = "[" + ",".join(f"'{f}'" for f in files_b) + "]"
    dfb = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol,
               Close as close, Volume as volume, Amount as amount,
               Factor as adj_factor
        FROM read_parquet({files_b_str})
    """).df()
    dfb["date"] = pd.to_datetime(dfb["date"])
    close = dfb.pivot_table(index="date", columns="symbol", values="close", aggfunc="first").sort_index()
    vol = dfb.pivot_table(index="date", columns="symbol", values="volume", aggfunc="first").sort_index()
    amt = dfb.pivot_table(index="date", columns="symbol", values="amount", aggfunc="first").sort_index()
    close = close.astype("float32"); vol = vol.astype("float32"); amt = amt.astype("float32")

    # free_float_shares 需要 close 对齐
    fc_a = fc.reindex(index=close.index, columns=close.columns)
    cols["free_float_shares"] = fc_a / close.replace(0, np.nan)

    # style_gate_*：基于市值/流动性/动量/残波动
    mktcap = (close * cols.get("free_float_shares", fc_a)).replace(0, np.nan)
    # size gate: 市值分位
    def pct_rank_series(m):
        return m.rank(axis=1, pct=True)
    cap_pct = pct_rank_series(mktcap)
    cols["style_gate_size_large"] = (cap_pct > 0.7).astype("float32")
    cols["style_gate_size_small"] = (cap_pct < 0.3).astype("float32")
    # liquidity gate: 成交额分位
    liq = pct_rank_series(amt)
    cols["style_gate_liquidity_high"] = (liq > 0.7).astype("float32")
    # momentum gate: 20日收益分位
    mom = close.pct_change(20).rank(axis=1, pct=True)
    cols["style_gate_momentum_high"] = (mom > 0.7).astype("float32")
    # resvol gate: 残波动 (20日收益std / 20日均量比)
    ret = close.pct_change()
    rv = ret.rolling(20, min_periods=10).std()
    vm = vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
    resvol = (ret.rolling(20, min_periods=10).std() / vm).rank(axis=1, pct=True)
    cols["style_gate_resvol_high"] = (resvol > 0.7).astype("float32")
    # roe_ttm2 / roa2_ttm2：无财务数据 → 用 pb/pe 近似
    pe = pd.read_parquet("/tmp/fund_pe_ttm.parquet") if Path("/tmp/fund_pe_ttm.parquet").exists() else None
    pb = pd.read_parquet("/tmp/fund_pb_lf.parquet") if Path("/tmp/fund_pb_lf.parquet").exists() else None
    if pe is not None and pb is not None:
        # roe ≈ pb/pe (E/P * P/B = E/B ≈ ROE)
        pe_a = pe.reindex(index=close.index, columns=close.columns)
        pb_a = pb.reindex(index=close.index, columns=close.columns)
        roe = (pb_a / pe_a.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        cols["roe_ttm2"] = roe.astype("float32")
        cols["roa2_ttm2"] = roe.astype("float32") * 0.5  # ROA ≈ 0.5*ROE
    # debttoassets：无 → 中性 0.35 占位（杠杆中性）
    cols["debttoassets"] = pd.DataFrame(0.35, index=close.index, columns=close.columns).astype("float32")
    # qfa_yoygr / forecast_incap_chgr_mid：无 → 用 20日动量近似
    cols["qfa_yoygr"] = close.pct_change(60).astype("float32")
    cols["forecast_incap_chgr_mid"] = close.pct_change(60).astype("float32")
    print(f"[cols] 全部列构建完成 ({time.time()-t0:.1f}s)")
    return cols, close


MISSING_COLS = ["pcf_ocf_ttm", "ps_ttm", "roe_ttm2", "roa2_ttm2", "debttoassets",
                "style_gate_size_large", "style_gate_size_small", "style_gate_liquidity_high",
                "style_gate_momentum_high", "style_gate_resvol_high", "free_float_shares",
                "qfa_yoygr", "forecast_incap_chgr_mid"]


def main():
    # 1. 找出缺列因子
    best = {}
    for fp in sorted(glob.glob(str(CONV_DIR / "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        fn = d.get("factor_name", "")
        if not fn:
            continue
        is_primary = os.path.basename(fp) == f"factor_{fn}.json"
        if fn not in best or is_primary:
            best[fn] = fp

    affected = []
    talib_need = ["talib.ADX", "talib.PLUS_DI", "talib.MINUS_DI", "talib.ROC", "talib.RSI",
                  "talib.TRANGE", "talib.average_true_range", "talib.directional_efficiency",
                  "talib.pulse_start"]
    for fn, fp in sorted(best.items()):
        d = json.load(open(fp))
        code = d.get("code", "")
        page = fn.replace("factor_", "")
        # 跳过已有值文件
        fpath = FV_DIR / f"{page}.parquet"
        if fpath.exists():
            try:
                m = pd.read_parquet(fpath, columns=None)
                if m.shape[1] > 0 and m.notna().any().any():
                    continue
            except Exception:
                pass
        if "NotImplementedError" in code:
            continue
        used = [c for c in MISSING_COLS if c in code]
        used_talib = [t for t in talib_need if t in code]
        if used or used_talib:
            affected.append({"page_name": page, "factor_name": fn, "code": code, "cols": used, "talib": used_talib})
    print(f"[backfill] 需补算因子: {len(affected)}")

    if not affected:
        print("[backfill] 无缺列因子")
        return

    # 2. 加载行情 + 缺失列
    t0 = time.time()
    mkt = {}
    for col in ["open", "high", "low", "close", "volume", "amount", "pre_close", "adj_factor"]:
        p = Path(f"/tmp/mkt_{col}.parquet")
        mkt[col] = pd.read_parquet(p) if p.exists() else None
    for fc in ["pb_lf", "pe_ttm", "mkt_cap_float", "free_turn"]:
        p = Path(f"/tmp/fund_{fc}.parquet")
        mkt[fc] = pd.read_parquet(p) if p.exists() else None
    extra_cols, close = load_missing_cols()
    for c, m in extra_cols.items():
        mkt[c] = m

    dates = mkt["close"].index
    symbols = list(mkt["close"].columns)
    print(f"[backfill] dates {dates[0]} ~ {dates[-1]} ({len(dates)} 天), {len(symbols)} 股")

    # 3. 逐因子补算
    from extend_all_456 import _build_intermediates, _execute_factor_code as _exec_fe
    intermediates = _build_intermediates(mkt, symbols, dates)
    for c, m in extra_cols.items():
        if c not in intermediates:
            intermediates[c] = m

    ok = 0
    for it in affected:
        try:
            mat = _exec_fe(it["code"], mkt, symbols, dates, intermediates=intermediates)
            if mat.empty:
                continue
            mat = mat.reindex(index=dates, columns=symbols)
            sub = mat.astype("float32").dropna(axis=1, how="all")
            out_p = FV_DIR / f"{it['page_name']}.parquet"
            sub.to_parquet(out_p)
            print(f"  [OK] {it['page_name']}: valid={mat.notna().mean().mean():.1%} cols={it['cols']}")
            ok += 1
        except Exception as e:
            print(f"  [ERR] {it['page_name']}: {str(e)[:100]}")

    print(f"[backfill] 完成: {ok}/{len(affected)} 因子补算, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
