# -*- coding: utf-8 -*-
"""直接用 COS 表内现成的复权因子 Factor 列重建后复权 OHLCV 面板。

真源: /home/sunhaiwei/cos_data/StockDailyBar/<date>.parquet
  (COS: clean_data/ashare/lqtp_data/StockDailyBar/, 本地镜像已全量落盘)

表内每交易日每股票一行, 列:
  Open High Low Close PreClose Volume Amount Return Factor Vwap IsSuspend
其中:
  - Close/Open/High/Low/Vwap : 未复权原始价
  - Factor                    : 后复权累积因子 (表内现成, 直接读, 不自己推导)
  - Volume                    : 成交量(股数), 不复权
  - Amount                    : 成交额, 后复权 = 原始 × Factor
后复权价 = 原始价 × Factor  (用表内 Factor 列, 分段常数, 分红/送转日跳升)

输出: data/build/ohlcv_adj.parquet (long: date/asset/field/value)
  + data/build/ohlcv_adj_wide/ 每字段 wide 面板
"""
import os, time, glob
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BAR_DIR = os.environ.get("BAR_DIR", "/home/sunhaiwei/cos_data/StockDailyBar")
OUT = os.path.join(ROOT, "data/build/ohlcv_adj.parquet")
OUT_WIDE = os.path.join(ROOT, "data/build/ohlcv_adj_wide")
LOG = "/tmp/build_ohlcv_adj_table.log"

PRICE_FIELDS = ["Open", "High", "Low", "Close", "Vwap"]
# Volume 不复权; Amount 后复权(×Factor)
VOL_FIELDS = ["Volume"]
AMT_FIELDS = ["Amount"]

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    t0 = time.time()
    os.makedirs(OUT_WIDE, exist_ok=True)
    files = sorted(glob.glob(os.path.join(BAR_DIR, "*.parquet")))
    plog(f"[{time.strftime('%H:%M:%S')}] 交易日文件: {len(files)} from {BAR_DIR}")

    # 逐日读取 -> 后复权价 = 原始 × Factor (用表内 Factor 列)
    frames = []
    wide_cache = {f: [] for f in PRICE_FIELDS + VOL_FIELDS + AMT_FIELDS}
    date_list = []
    for fi, f in enumerate(files):
        d = pd.read_parquet(f)
        d = d[["TradeDate", "Symbol", "Factor"] + PRICE_FIELDS + VOL_FIELDS + AMT_FIELDS]
        d = d.rename(columns={"TradeDate": "date", "Symbol": "asset"})
        d["date"] = pd.to_datetime(d["date"])
        F = d["Factor"].to_numpy()
        # 价格字段: 后复权 = 原始 × Factor
        for col in PRICE_FIELDS:
            d[col] = d[col].to_numpy() * F
        # Amount: 后复权 = 原始 × Factor (成交量额=价×量)
        for col in AMT_FIELDS:
            d[col] = d[col].to_numpy() * F
        # Volume: 不复权, 保持原值
        # 收集 long
        m = d.melt(id_vars=["date", "asset"], value_vars=PRICE_FIELDS + VOL_FIELDS + AMT_FIELDS,
                   var_name="field", value_name="value")
        frames.append(m)
        # 收集 wide (per-field date x asset)
        for col in PRICE_FIELDS + VOL_FIELDS + AMT_FIELDS:
            wide_cache[col].append(d[["date", "asset", col]].pivot_table(index="date", columns="asset", values=col))
        if (fi + 1) % 300 == 0:
            plog(f"  {fi+1}/{len(files)} {time.time()-t0:.0f}s")
    allf = pd.concat(frames, ignore_index=True)
    allf = allf[["date", "asset", "field", "value"]]
    allf.to_parquet(OUT, index=False)
    plog(f"WROTE {OUT} {allf.shape}")
    # wide
    for col in PRICE_FIELDS + VOL_FIELDS + AMT_FIELDS:
        w = pd.concat(wide_cache[col]).sort_index()
        w.to_parquet(os.path.join(OUT_WIDE, f"{col}_adj.parquet"))
    plog(f"[{time.strftime('%H:%M:%S')}] DONE total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
