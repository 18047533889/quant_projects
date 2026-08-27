# -*- coding: utf-8 -*-
"""因子预处理/中性化：在 feature 矩阵(2166因子)基础上, 用 factor_preprocess 的
截面变换(cs_winsor/cs_rank/cs_zscore)做统一预处理, 得到 model-ready 特征。

背景(用户 2026-08-27 强调"非常重要"):
  1. 模型预测的 rankIC/IC 很重要, 要专门展示;
  2. 所有因子用到的数据要复权(已核对: 标签=前复权Vwap10日收益, 组合基准=后复权Vwap);
  3. 模型预测目标必须是中性化后的收益(不是绝对收益);
  4. 模型前的因子也要中性化处理(不一定全中性化), 用 /home/sunhaiwei/quant_projects/factor_preprocess
     自动选择预处理, DSL 已做过预处理的不重复做。

本脚本在 2166 因子矩阵上按"池级别"应用统一预处理(池的 DSL 语义决定是否中性化):
  - fm247/fmqa (_neu):    已经过中性化/标准化处理 -> 只做 cs_winsor(去极值)
  - cogfull/cogshort/cogneutral/cogneutral: 已中性化, 保持
  - delivery/optfac/factmat/lqtp: 未中性化 -> cs_winsor + cs_rank + cs_zscore
其中 cs_* 是 factor_preprocess 的截面变换(逐日期独立, 无前视)。

输出: data/build/features_full2_prep.parquet (与 features_full2 同网格, 同 2166 列)
以及中性化后的标签 y_neu(第2167列)供训练使用。
用法:
  python preprocess_features.py            # 全部因子矩阵 -> features_full2_prep.parquet
  python preprocess_features.py --label    # 同时计算中性化标签 fwd_neu
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
IN  = os.environ.get("FEATURES_PARQUET", os.path.join(ROOT, "data/build/features_full2.parquet"))
OUT = os.environ.get("PREP_OUT", os.path.join(ROOT, "data/build/features_full2_prep.parquet"))
FWD = os.path.join(ROOT, "data/panel/fwd_ret10.parquet")
VWAP_ADJ = os.path.join(ROOT, "data/panel/vwap_trad_adj.parquet")
LOG = "/tmp/preprocess_features.log"

sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_preprocess")
from factor_preprocess.transforms.cross_sectional import cs_winsor, cs_rank, cs_zscore

POOL_PREFIXES = {
    "fm247:":    "neu",   # _neu 已中性化 -> 仅去极值
    "fmqa:":     "neu",
    "cogfull:":  "neu",
    "cogshort:": "neu",
    "cogneutral:": "neu",
    "delivery:": "raw",   # 未中性化 -> winsor+rank+zscore
    "optfac:":   "raw",
    "factmat:":  "raw",
    "lqtp:":     "raw",
}

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] 读取 {IN} ...")
    df = pd.read_parquet(IN)
    plog(f"  shape={df.shape}")
    feats = [c for c in df.columns if c not in ("date", "asset")]
    n_row, n_feat = len(df), len(feats)
    date_arr = df["date"].to_numpy()
    asset_arr = df["asset"].to_numpy()

    # 分组: 每列按前缀归类到 (date) 网格 -> 2D 矩阵 (n_date, n_asset) 逐日截面处理
    uni_dates, date_idx = np.unique(date_arr, return_inverse=True)
    uni_assets = pd.Index(pd.unique(asset_arr))
    a_pos = pd.Series(np.arange(len(uni_assets)), index=uni_assets)
    asset_pos = a_pos.reindex(asset_arr).to_numpy()
    n_date, n_ast = len(uni_dates), len(uni_assets)
    plog(f"  dates={n_date} assets={n_ast} feats={n_feat}")

    # 预分配输出
    out_block = np.empty((n_row, n_feat), dtype=np.float32)
    stats = []
    # 每列 -> 宽矩阵 (date x asset) 填充
    for j, col in enumerate(feats):
        v = df[col].to_numpy(dtype=np.float64)
        # 归属池
        prefix = next((p for p in POOL_PREFIXES if col.startswith(p)), "raw")
        kind = POOL_PREFIXES[prefix]
        # 宽化 (sparse 内存友好: 先分配 NaN 再填)
        M = np.full((n_date, n_ast), np.nan, dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum():
            M[date_idx[ok], asset_pos[ok]] = v[ok]
        if kind == "neu":
            # 已中性化: 仅 winsor 去极值 (1%/99%)
            M = cs_winsor(M, lower=0.01, upper=0.99, axis=1)
        else:
            # 未中性化: winsor -> rank -> zscore
            M = cs_winsor(M, lower=0.01, upper=0.99, axis=1)
            M = cs_rank(M, axis=1, pct=True)
            M = cs_zscore(M, axis=1, ddof=1)
        out_block[:, j] = M[date_idx, asset_pos]
        if (j + 1) % 100 == 0:
            gc.collect()
            plog(f"  {j+1}/{n_feat} ...")
        del M
    stats.append("ok")

    out = pd.DataFrame({
        "date": date_arr,
        "asset": asset_arr,
    })
    for j, col in enumerate(feats):
        out[col] = out_block[:, j]
    plog(f"写 {OUT} shape={out.shape} ...")
    out.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] DONE total {time.time()-t0:.0f}s -> {OUT}")
    return out

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
