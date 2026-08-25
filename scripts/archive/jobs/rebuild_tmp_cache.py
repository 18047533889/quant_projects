#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建 /tmp/mkt_*.parquet + /tmp/fund_*.parquet 缓存（从 ~/cos_data 用 duckdb）。"""
import duckdb, glob, time, warnings
import pandas as pd
warnings.filterwarnings("ignore")

t0 = time.time()

# ---- 行情 StockDailyBar ----
print("[1] 行情 ...")
files = sorted(glob.glob("/home/sunhaiwei/cos_data/StockDailyBar/*.parquet"))
files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
con = duckdb.connect()
df = con.execute(f"""
    SELECT TradeDate as date, Symbol as symbol,
           Open as open, High as high, Low as low, Close as close,
           PreClose as pre_close, Volume as volume, Amount as amount,
           Vwap as vwap, Factor as adj_factor
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
           CirculatingMarketCap as mkt_cap_float, TurnoverRatio as free_turn
    FROM read_parquet({files_v_str})
""").df()
print(f"  行数 {len(dfv)} ({time.time()-t0:.1f}s)")
dfv["date"] = pd.to_datetime(dfv["date"])
for col in ["pb_lf", "pe_ttm", "mkt_cap_float", "free_turn"]:
    m = dfv.pivot_table(index="date", columns="symbol", values=col, aggfunc="first").sort_index()
    m = m.astype("float32")
    m.to_parquet(f"/tmp/fund_{col}.parquet")
    print(f"  fund_{col}: {m.shape}")

print(f"[done] 缓存重建完成 ({time.time()-t0:.1f}s)")
