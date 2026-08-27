# -*- coding: utf-8 -*-
"""建立后复权 OHLCV 面板 (data/build/ohlcv_adj.parquet, long: date/asset/field/value)。

复权口径(用户强调 + COS 字典 2026-08-08 实盘验证):
  后复权价 = 原始价 × Factor   (Factor = 后复权累积因子, 分段常数, 分红/送转日跳升)
  - Open/High/Low/Close/Vwap : × Factor  (后复权)
  - Volume(成交量, 股数)      : 不复权 (股数不随分红变化)
  - Amount(成交额, 元)        : × Factor (成交额=价×量, 后复权额=后复权价×量)
  - 财务/基本面               : 不复权

Factor 来源: vwap_trad_adj / vwap_trad (已验证 = 后复权Vwap/未复权Vwap, 分段常数)。
输出: data/build/ohlcv_adj.parquet (date, asset, field, value)  + 每字段一个 wide 面板
"""
import os, time
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
RAW = os.path.join(ROOT, "data/panel/lqtp_raw")
VWAP_RAW = os.path.join(ROOT, "data/panel/vwap_trad.parquet")
VWAP_ADJ = os.path.join(ROOT, "data/panel/vwap_trad_adj.parquet")
OUT = os.path.join(ROOT, "data/build/ohlcv_adj.parquet")
OUT_WIDE = os.path.join(ROOT, "data/build/ohlcv_adj_wide")
LOG = "/tmp/build_ohlcv_adj.log"

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def load_raw(field):
    df = pd.read_parquet(os.path.join(RAW, f"{field}.parquet"))
    df["date"] = pd.to_datetime(df["date"].astype(str))
    return df.pivot_table(index="date", columns="asset", values="value")

def main():
    t0 = time.time()
    os.makedirs(OUT_WIDE, exist_ok=True)
    plog(f"[{time.strftime('%H:%M:%S')}] 计算 Factor = vwap_trad_adj / vwap_trad ...")
    vraw = pd.read_parquet(VWAP_RAW); vraw.index = pd.to_datetime(vraw.index)
    vadj = pd.read_parquet(VWAP_ADJ); vadj.index = pd.to_datetime(vadj.index)
    F = vadj / vraw
    F = F.replace([np.inf, -np.inf], np.nan)
    plog(f"  Factor panel {F.shape} 值域 {float(F.min().min()):.2f}..{float(F.max().max()):.2f}")

    # 价格字段: 后复权 = 原始 × Factor
    price_fields = ["Open", "High", "Low", "Close"]
    # 量/额: Volume 不复权, Amount 后复权(×Factor)
    frames = []
    for field in price_fields:
        raw = load_raw(field)
        adj = raw * F
        adj.to_parquet(os.path.join(OUT_WIDE, f"{field}_adj.parquet"))
        s = adj.stack().rename("value").reset_index()
        s["field"] = field
        frames.append(s)
        plog(f"  {field}: 后复权 {adj.shape} 值域 {float(adj.min().min()):.2f}..{float(adj.max().max()):.2f}")

    # Vwap 后复权 (直接用 vwap_trad_adj)
    vadj_s = vadj.stack().rename("value").reset_index()
    vadj_s["field"] = "Vwap"
    frames.append(vadj_s)
    vadj.to_parquet(os.path.join(OUT_WIDE, "Vwap_adj.parquet"))
    plog(f"  Vwap: 后复权 {vadj.shape}")

    # Volume 不复权
    vol = load_raw("Volume")
    vol.to_parquet(os.path.join(OUT_WIDE, "Volume_raw.parquet"))
    s = vol.stack().rename("value").reset_index(); s["field"] = "Volume"
    frames.append(s)
    plog(f"  Volume: 不复权 {vol.shape}")

    # Amount 后复权 (×Factor)
    amt = load_raw("Amount")
    amt_adj = amt * F
    amt_adj.to_parquet(os.path.join(OUT_WIDE, "Amount_adj.parquet"))
    s = amt_adj.stack().rename("value").reset_index(); s["field"] = "Amount"
    frames.append(s)
    plog(f"  Amount: 后复权 {amt_adj.shape}")

    allf = pd.concat(frames, ignore_index=True)
    allf.columns = ["date", "asset", "value", "field"]
    allf = allf[["date", "asset", "field", "value"]]
    allf.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] WROTE {OUT} {allf.shape} total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
