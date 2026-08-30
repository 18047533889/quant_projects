# -*- coding: utf-8 -*-
"""#22 step3: 34 条新列 prep — 完全复用 preprocess_features_flip 的 per-pool 变换。

新因子来源为 factor_matrices_all(新挖) + 分钟因子矩阵, 都是 raw(未中性化),
与 optfac/factmat/delivery/lqtp 同策略: cs_winsor + cs_rank(pct) + cs_zscore(ddof=1)。
复用 factor_preprocess 库 (禁止手写变换)。
输出 data/build/features_inc35_prep.parquet (date, asset + 34 列, 排除冗余 alphasage_rank_amount)。
"""
import os, sys, time, json
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PROG = "/tmp/fac818_progress.log"
IN = os.path.join(BUILD, "features_inc35.parquet")
OUT = os.path.join(BUILD, "features_inc35_prep.parquet")
DROP = {"alphasage_rank_amount"}   # 冗余 >0.9 (lqtp:wzr_price_volume_039)

sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_preprocess")
from factor_preprocess.transforms.cross_sectional import cs_winsor, cs_rank, cs_zscore


def plog(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    plog("===== step3: preprocess 34 new features =====")
    df = pd.read_parquet(IN)
    feats = [c for c in df.columns if c not in ("date", "asset") and c not in DROP]
    plog(f"input {df.shape} -> prepping {len(feats)} cols")
    date_arr = df["date"].to_numpy()
    asset_arr = df["asset"].to_numpy()
    uni_dates, date_idx = np.unique(date_arr, return_inverse=True)
    uni_assets = pd.Index(pd.unique(asset_arr))
    a_pos = pd.Series(np.arange(len(uni_assets)), index=uni_assets)
    asset_pos = a_pos.reindex(asset_arr).to_numpy()
    n_date, n_ast = len(uni_dates), len(uni_assets)
    out_block = np.empty((len(df), len(feats)), dtype=np.float32)
    for j, col in enumerate(feats):
        v = df[col].to_numpy(dtype=np.float64)
        M = np.full((n_date, n_ast), np.nan, dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum():
            M[date_idx[ok], asset_pos[ok]] = v[ok]
        M = cs_winsor(M, lower=0.01, upper=0.99, axis=1)
        M = cs_rank(M, axis=1, pct=True)
        M = cs_zscore(M, axis=1, ddof=1)
        out_block[:, j] = M[date_idx, asset_pos]
        if (j + 1) % 10 == 0:
            plog(f"  {j+1}/{len(feats)} ({time.time()-t0:.0f}s)")
        del M
    out = pd.DataFrame({"date": date_arr, "asset": asset_arr})
    for j, col in enumerate(feats):
        out[col] = out_block[:, j]
    out.to_parquet(OUT, index=False)
    plog(f"DONE -> {OUT} shape={out.shape} total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
