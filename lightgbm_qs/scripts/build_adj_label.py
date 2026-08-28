# -*- coding: utf-8 -*-
"""重建后复权口径 10 日前向静替标签（vwap-to-vwap，全局硬性口径）。

发现：现有 data/panel/fwd_ret10.parquet 基于 【未复权】vwap_trad.parquet，
除权/送转日 vwap 跳空被当成收益（如 000001.SZ 2016-06-16 送转：
  未复权 10 日收益 = -16.9%（错误，跳空算真收益）
  后复权 10 日收益 = +1.17%（正确）
per-asset 受影响标签日 20..224 天，297 个资产全部受影响，合计约 3.2 万行。
预测端已改用 data/panel/vwap_trad_adj.parquet（后复权）。本脚本重建正确标签。

口径：adj10[t] = vwap_trad_adj[t+10]/vwap_trad_adj[t] - 1（vwap-to-vwap）
用法：/home/sunhaiwei/quant_projects/.venv/bin/python scripts/build_adj_label.py
"""
import os, time
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
ADJ = os.path.join(ROOT, "data/panel/vwap_trad_adj.parquet")
OUT = os.path.join(ROOT, "data/panel/fwd_ret10_adj.parquet")
LOG = "/tmp/build_adj_label.log"

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] load {ADJ} ...")
    adj = pd.read_parquet(ADJ)
    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index()
    plog(f"  adj shape={adj.shape}")

    adj10 = (adj.shift(-10) / adj - 1.0).astype(np.float64)
    adj10 = adj10.where(np.isfinite(adj10))

    long = adj10.stack().rename("fwd").reset_index()
    long.columns = ["date", "asset", "fwd"]
    long["date"] = long["date"].astype(str)
    long = long.sort_values(["date", "asset"]).reset_index(drop=True)
    long.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] WROTE {OUT} {long.shape} total {time.time()-t0:.0f}s")
    return long

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
