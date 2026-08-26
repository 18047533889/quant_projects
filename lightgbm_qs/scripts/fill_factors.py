# -*- coding: utf-8 -*-
"""因子值补全：把 features_all 里的稀疏因子补成满时间线。
规则：按 asset 分组做 ffill(前值延续) + bfill(起始段用后值) + 残余 NaN 用当日截面中位数兜底。
全空列(某因子完全无值)才丢弃。输出 features_filled.parquet。
"""
import pandas as pd, numpy as np
ROOT="/home/sunhaiwei/quant_projects/lightgbm_qs"
df=pd.read_parquet(f"{ROOT}/data/build/features_all.parquet")
df["date"]=pd.to_datetime(df["date"])
cols=[c for c in df.columns if c not in ("date","asset")]
print("before:", df.shape, "因子", len(cols))

# 每资产时间序列内 ffill+bfill
df=df.sort_values(["asset","date"]).reset_index(drop=True)
df[cols]=df[cols].groupby(df["asset"]).transform(lambda s: s.ffill().bfill())

# 残余 NaN: 每日期截面中位数兜底
def fill_cs(g):
    med=g[cols].median(skipna=True)
    return g[cols].fillna(med)
df[cols]=df.groupby("date")[cols].transform(lambda g: g.fillna(g.median(skipna=True)))

# 全空列丢弃
kept=[c for c in cols if df[c].notna().mean()>0]
df=df[["date","asset"]+kept]
df=df.sort_values(["date","asset"]).reset_index(drop=True)
df.to_parquet(f"{ROOT}/data/build/features_filled.parquet")
print("after:", df.shape, "因子", len(kept))
print("NaN residual:", round(df[kept].isna().mean().mean(),5))
