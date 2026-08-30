# -*- coding: utf-8 -*-
"""后复权中性化标签 fwd_adj_neu: vwap_trad_adj 的 10 日前向收益(后复权Vwap, 分红送转还原)
按日期做截面中性化(demean + zscore, 以 tradable 297 资产为截面)。

背景(用户 2026-08-27 强调"非常重要"):
  - 模型预测目标必须是中性化后的收益, 不能是绝对收益(才能传给组合优化端)
  - 收益基准必须是复权数据: 组合端已用 vwap_trad_adj(后复权Vwap)
  - 原 fwd_ret10.parquet 用的是 未复权Vwap(除权跳空被当成收益, 如 000001.SZ 2016-06 送转 -17.8%)

输出: data/build/fwd_adj_neu.parquet (date,asset,fwd_neu)
  fwd_neu = (adj10_t - 截面mean) / 截面std  (逐日截面中性化, 无前视)
用法: python3.12 build_label_adj.py
"""
import os, time
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
ADJ = os.path.join(ROOT, "data/panel/vwap_trad_adj.parquet")
OUT = os.path.join(ROOT, "data/build/fwd_adj_neu.parquet")
LOG = "/tmp/build_label_adj.log"

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

    # 10日持有期月收益 (后复权Vwap, COS TargetVwapReturnH10 官方口径):
    #   = AdjVwap[t+11] / AdjVwap[t+1] - 1  (T+1 建仓, T+11 平仓, 即跨 10 个交易日)
    # 已与 COS StockDailyBarAdj.TargetVwapReturnH10 逐点对拍(12 个样本点 diff=0)。
    # 旧口径 shift(-10)/shift(-1) 少算 1 天(9 交易日)会系统性偏低, 已修正。
    adj10 = (adj.shift(-11) / adj.shift(-1) - 1.0).astype(np.float64)
    adj10 = adj10.where(np.isfinite(adj10))

    # 逐日截面中性化: 每行减去横截面 mean, 除以横截面 std (仅用该日有效资产)
    # 向量化: 对整个矩阵一次计算, 不逐行 apply (避免 object-dtype 问题)
    def cs_neutralize_matrix(M):
        ok = np.isfinite(M)
        n_ok = ok.sum(axis=1)
        mu = np.where(n_ok >= 5, np.nanmean(M, axis=1), np.nan)
        sd = np.where(n_ok >= 5, np.nanstd(M, axis=1, ddof=1), np.nan)
        out = np.full_like(M, np.nan, dtype=np.float64)
        denom = np.where((sd > 0) & np.isfinite(mu), sd, np.nan)
        safe = np.where(np.isfinite(denom)[:, None] & np.isfinite(mu)[:, None] & ok,
                        (M - mu[:, None]) / denom[:, None], np.nan)
        out[ok] = safe[ok]
        return out

    plog("  cross-sectional neutralization per date ...")
    neu_arr = cs_neutralize_matrix(adj10.to_numpy(dtype=np.float64, copy=True))
    neu = pd.DataFrame(neu_arr, index=adj10.index, columns=adj10.columns)

    # 转 long
    plog("  to long form ...")
    long = neu.stack().rename("fwd_neu").reset_index()
    long.columns = ["date", "asset", "fwd_neu"]
    long["date"] = long["date"].astype(str)
    long = long.sort_values(["date", "asset"]).reset_index(drop=True)
    long.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] WROTE {OUT} {long.shape} total {time.time()-t0:.0f}s")
    return long

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
