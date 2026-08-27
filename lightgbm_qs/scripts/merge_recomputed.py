# -*- coding: utf-8 -*-
"""用后复权重算的 159 个纯价格因子替换特征矩阵中对应列, 生成最终后复权特征矩阵。

输入:
  - data/build/features_full2.parquet (2166 因子, 原始)
  - data/build/factors_recomputed/*.parquet (159 个后复权重算纯价格因子, wide date x asset)
输出:
  - data/build/features_full2_adj.parquet (2166 因子, 其中 159 个已替换为后复权版本)

替换规则: 对每个重算因子名, 在特征矩阵中找到含该名的列(带池前缀), 用后复权值覆盖。
其余因子(财务/水平型/LQTP 等)保留原值。
"""
import os, time, glob
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
IN = os.path.join(ROOT, "data/build/features_full2.parquet")
REC_DIR = os.path.join(ROOT, "data/build/factors_recomputed")
OUT = os.path.join(ROOT, "data/build/features_full2_adj.parquet")
LOG = "/tmp/merge_recomputed.log"

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] 读取 {IN} ...")
    df = pd.read_parquet(IN)
    df["date"] = pd.to_datetime(df["date"])
    feats = [c for c in df.columns if c not in ("date", "asset")]
    plog(f"  特征矩阵 {df.shape} 因子={len(feats)}")

    # 重算因子 wide 面板
    rec_files = sorted(glob.glob(os.path.join(REC_DIR, "*.parquet")))
    plog(f"  重算因子文件: {len(rec_files)}")

    # 建立 date->asset 网格索引
    df["_key"] = df["date"].astype(str) + "|" + df["asset"]
    replaced = 0
    for f in rec_files:
        name = os.path.basename(f)[:-8]
        # 找到特征矩阵中含该名的列
        match = [c for c in feats if name in c]
        if not match:
            continue
        rec = pd.read_parquet(f)
        rec.index = pd.to_datetime(rec.index)
        # 转 long
        rec_long = rec.stack().rename("v").reset_index()
        rec_long.columns = ["date", "asset", "v"]
        rec_long["date"] = rec_long["date"].astype(str)
        rec_long["_key"] = rec_long["date"] + "|" + rec_long["asset"]
        rec_map = rec_long.set_index("_key")["v"]
        for col in match:
            df[col] = df["_key"].map(rec_map).to_numpy()
            replaced += 1
        del rec, rec_long, rec_map
    plog(f"  替换因子列: {replaced}")

    df = df.drop(columns=["_key"])
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] WROTE {OUT} {df.shape} total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
