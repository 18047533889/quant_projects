#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建 /tmp/mkt_*.parquet + /tmp/fund_*.parquet 缓存 —— 后复权口径（喂 StockDailyBarAdj）。
所有行情列使用后复权 Adj 值（AdjX = X×Factor）：
  open←AdjOpen, high←AdjHigh, low←AdjLow, close←AdjClose, pre_close←AdjPreClose,
  amount←AdjAmount, vwap←AdjVwap, volume←Volume(不变), adj_factor←Factor
估值/市值仍用 StockValuationDaily（市值不参与复权，市值本身是实时值）。
财务指标（debttoassets/roe_ttm2/roa2_ttm2/ps_ttm/pcf_ocf_ttm/qfa_yoygr 等）来自
StockBalance/StockIndicator/StockValuationDaily，本身不复权，原样用（2026-08-28 扩展）。
"""
import duckdb, glob, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

t0 = time.time()

# ---- 行情 StockDailyBarAdj（后复权） ----
print("[1] 行情(后复权 Adj) ...")
files = sorted(glob.glob("/home/sunhaiwei/cos_data/StockDailyBarAdj/*.parquet"))
files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
con = duckdb.connect()
df = con.execute(f"""
    SELECT TradeDate as date, Symbol as symbol,
           AdjOpen as open, AdjHigh as high, AdjLow as low, AdjClose as close,
           AdjPreClose as pre_close, Volume as volume, AdjAmount as amount,
           AdjVwap as vwap, Factor as adj_factor
    FROM read_parquet({files_str})
""").df()
print(f"  行数 {len(df)} ({time.time()-t0:.1f}s)")
df["date"] = pd.to_datetime(df["date"])
for col in ["open", "high", "low", "close", "pre_close", "volume", "amount", "vwap", "adj_factor"]:
    m = df.pivot_table(index="date", columns="symbol", values=col, aggfunc="first").sort_index()
    m = m.astype("float32")
    m.to_parquet(f"/tmp/mkt_{col}.parquet")
    print(f"  mkt_{col}: {m.shape}")

# ---- 估值 StockValuationDaily ----
print("[2] 估值 ...")
files_v = sorted(glob.glob("/home/sunhaiwei/cos_data/StockValuationDaily/*.parquet"))
files_v_str = "[" + ",".join(f"'{f}'" for f in files_v) + "]"
dfv = con.execute(f"""
    SELECT TradeDate as date, Symbol as symbol,
           PbRatio as pb_lf, PeRatio as pe_ttm,
           CirculatingMarketCap as mkt_cap_float, TurnoverRatio as free_turn,
           PsRatio as ps_ttm, PcfRatio as pcf_ocf_ttm, FreeCap as free_cap
    FROM read_parquet({files_v_str})
""").df()
print(f"  行数 {len(dfv)} ({time.time()-t0:.1f}s)")
dfv["date"] = pd.to_datetime(dfv["date"])
for col in ["pb_lf", "pe_ttm", "mkt_cap_float", "free_turn", "ps_ttm", "pcf_ocf_ttm", "free_cap"]:
    m = dfv.pivot_table(index="date", columns="symbol", values=col, aggfunc="first").sort_index()
    m = m.astype("float32")
    m.to_parquet(f"/tmp/fund_{col}.parquet")
    print(f"  fund_{col}: {m.shape}")

# free_float_shares = FreeCap / 未复权 Close（股本，不复权）
print("[3] free_float_shares (FreeCap/未复权Close) ...")
files_b = sorted(glob.glob("/home/sunhaiwei/cos_data/StockDailyBar/*.parquet"))
files_b_str = "[" + ",".join(f"'{f}'" for f in files_b) + "]"
dfb = con.execute(f"SELECT TradeDate as date, Symbol as symbol, Close as close_raw FROM read_parquet({files_b_str})").df()
dfb["date"] = pd.to_datetime(dfb["date"])
close_raw = dfv[["date", "symbol"]].merge(dfb, on=["date", "symbol"], how="left") if False else dfb
close_raw_m = dfb.pivot_table(index="date", columns="symbol", values="close_raw", aggfunc="first").sort_index().astype("float32")
close_raw_m.to_parquet("/tmp/fund_close_raw.parquet")
fc_m = dfv.pivot_table(index="date", columns="symbol", values="free_cap", aggfunc="first").sort_index().astype("float32")
ffs = (fc_m / close_raw_m.replace(0, np.nan)).astype("float32")
ffs.to_parquet("/tmp/fund_free_float_shares.parquet")
print(f"  fund_free_float_shares: {ffs.shape}")
del dfb

# ---- 财务事件表：StockBalance → debttoassets；StockIndicator → roe/roa TTM + yoy ----
# 事件快照表（每文件只含当日新披露公司），按 PubDate 排序后 ffill 到交易日（无前视）。
print("[4] StockBalance → debttoassets ...")
files_bal = sorted(glob.glob("/home/sunhaiwei/cos_data/StockBalance/*.parquet"))
fs_bal = "[" + ",".join(f"'{f}'" for f in files_bal) + "]"
dfbal = con.execute(f"""
    SELECT PubDate as pub, Symbol as symbol,
           TotalLiability*100.0/NULLIF(TotalAssets,0) as debttoassets
    FROM read_parquet({fs_bal}) WHERE TotalAssets > 0
""").df()
dfbal["pub"] = pd.to_datetime(dfbal["pub"])


def _events_to_daily(df, val, name):
    """事件轴(pub) → 日频面板：searchsorted 对齐 + ffill（只用 pub<=date 的信息，无前视）。"""
    d = df[["pub", "symbol", val]].dropna(subset=[val]).sort_values(["symbol", "pub"])
    ev = d.pivot_table(index="pub", columns="symbol", values=val, aggfunc="last").sort_index()
    pos = ev.index.searchsorted(dfv_dates, side="right") - 1
    valid = pos >= 0
    out = pd.DataFrame(np.nan, index=dfv_dates, columns=ev.columns, dtype="float64")
    out.iloc[valid, :] = ev.values[pos[valid]]
    out = out.ffill().reindex(index=dfv_dates, columns=symbols_all).astype("float32")
    out.to_parquet(f"/tmp/fund_{name}.parquet")
    print(f"  fund_{name}: {out.shape} nonnull={np.isfinite(out.values).sum()/out.size*100:.1f}%")
    return out


_ref = pd.read_parquet("/tmp/mkt_close.parquet")
dfv_dates = _ref.index
symbols_all = _ref.columns
del _ref
_events_to_daily(dfbal, "debttoassets", "debttoassets")
del dfbal

print("[5] StockIndicator → roe_ttm2/roa2_ttm2/qfa_yoygr ...")
files_i = sorted(glob.glob("/home/sunhaiwei/cos_data/StockIndicator/*.parquet"))
fs_i = "[" + ",".join(f"'{f}'" for f in files_i) + "]"
dfi = con.execute(f"""
    SELECT PubDate as pub, Symbol as symbol, ReportPeriodEndDate as rpt,
           Roe as roe, Roa as roa, IncRevenueYearOnYear as rev_yoy
    FROM read_parquet({fs_i})
""").df()
dfi["pub"] = pd.to_datetime(dfi["pub"])
dfi["rpt"] = pd.to_datetime(dfi["rpt"])
# 单季 Roe/Roa(%) → TTM：按报告期滚动 4 季求和（要求报告期连续 85~100 天间隔）
ttm = dfi[["pub", "symbol", "rpt", "roe", "roa"]].sort_values(["symbol", "rpt"])
g = ttm.groupby("symbol", group_keys=False)
ttm["roe_ttm"] = g["roe"].transform(lambda s: s.rolling(4, min_periods=4).sum())
ttm["roa_ttm"] = g["roa"].transform(lambda s: s.rolling(4, min_periods=4).sum())
ttm["gap"] = g["rpt"].diff().dt.days
ttm["ok"] = ttm["gap"].between(85, 100)
ttm["ok4"] = ttm.groupby("symbol")["ok"].transform(lambda s: s.rolling(3, min_periods=3).min().fillna(0).astype(bool))
_events_to_daily(ttm[ttm["ok4"]][["pub", "symbol", "roe_ttm"]].rename(columns={"roe_ttm": "v"}).sort_values(["symbol", "pub"]), "v", "roe_ttm2")
_events_to_daily(ttm[ttm["ok4"]][["pub", "symbol", "roa_ttm"]].rename(columns={"roa_ttm": "v"}).sort_values(["symbol", "pub"]), "v", "roa2_ttm2")
_events_to_daily(dfi, "rev_yoy", "qfa_yoygr")
_events_to_daily(dfi, "rev_yoy", "forecast_incap_chgr_mid")
del dfi, ttm

# ---- style_gate_*（截面分位门控，backfill_missing_456 口径） ----
print("[6] style_gate_* ...")
cap_pct = fc_m.rank(axis=1, pct=True)
(cap_pct > 0.7).astype("float32").to_parquet("/tmp/fund_style_gate_size_large.parquet")
(cap_pct < 0.3).astype("float32").to_parquet("/tmp/fund_style_gate_size_small.parquet")
amt = pd.read_parquet("/tmp/mkt_amount.parquet").reindex(index=dfv_dates, columns=symbols_all)
(amt.rank(axis=1, pct=True) > 0.7).astype("float32").to_parquet("/tmp/fund_style_gate_liquidity_high.parquet")
close_adj = pd.read_parquet("/tmp/mkt_close.parquet").reindex(index=dfv_dates, columns=symbols_all)
(close_adj.pct_change(20).rank(axis=1, pct=True) > 0.7).astype("float32").to_parquet("/tmp/fund_style_gate_momentum_high.parquet")
print(f"  style_gate_size_large/small + liquidity_high + momentum_high: {cap_pct.shape}")

print(f"[done] 后复权缓存重建完成 ({time.time()-t0:.1f}s)")
