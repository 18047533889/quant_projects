# -*- coding: utf-8 -*-
"""后复权一致性审计: 判断每个因子是否 scale-invariant(比率型, 已与后复权一致)。

原理: 后复权 = 原始价 × 后复权累积因子 F(每资产分段常数, 分红日跳升)。
  - 比率型因子(close/high, 收益, 收益zscore, 距离开销等): 分子分母同乘 F, F 抵消 → 值不变
    → 已与后复权一致, 无需重算
  - 水平型因子(裸价格 level, 绝对量): 乘 F 后值改变 → 需要后复权重算

判定: 对每因子, 计算其值与 F 的截面相关 |corr|。|corr|>0.3 → 水平型(需重算);
      |corr|<=0.3 → 比率型(已一致)。同时输出因子值量纲(是否 ~0.01 收益量级)。

用法: python3.12 audit_adj_consistency.py <pool_dir> <out_csv>
输出: 每因子一行 {factor, corr_with_F, scale_invariant, value_scale, n}
"""
import os, sys, glob, time
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
RAW = os.path.join(ROOT, "data/panel/vwap_trad.parquet")
ADJ = os.path.join(ROOT, "data/panel/vwap_trad_adj.parquet")
CORR_THRESH = 0.3

def main():
    pool_dir, out_csv = sys.argv[1], sys.argv[2]
    raw = pd.read_parquet(RAW)
    adj = pd.read_parquet(ADJ)
    # 用 3 个代表性资产算 F, 取中位(避免单资产异常)
    assets = raw.columns[:3].tolist()
    Fs = []
    for c in assets:
        Fs.append((adj[c] / raw[c]).rename(c))
    F = pd.concat(Fs, axis=1).median(axis=1).rename("F")

    files = sorted(glob.glob(os.path.join(pool_dir, "*.parquet")))
    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        name = os.path.basename(f)[:-8]
        try:
            d = pd.read_parquet(f)
            # 取第一个资产列
            col = d.columns[0]
            v = d[col].rename("factor")
            m = pd.concat([v, F], axis=1).dropna()
            if len(m) < 100:
                rows.append((name, np.nan, "insufficient", np.nan, len(m)))
                continue
            u = np.unique(m.factor)
            if len(u) < 3:
                rows.append((name, 0.0, "constant", float(m.factor.mean()), len(m)))
                continue
            c = np.corrcoef(m.factor, m.F)[0, 1]
            if not np.isfinite(c):
                rows.append((name, np.nan, "nan", float(m.factor.mean()), len(m)))
                continue
            scale_inv = "scale_invariant" if abs(c) <= CORR_THRESH else "LEVEL_BASED"
            rows.append((name, float(c), scale_inv, float(m.factor.mean()), len(m)))
        except Exception as e:
            rows.append((name, np.nan, "error:" + type(e).__name__, np.nan, 0))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows, columns=["factor", "corr_with_F", "kind", "value_mean", "n"])
    out.to_csv(out_csv, index=False)
    print(f"WROTE {out_csv}  total={len(out)}  "
          f"scale_invariant={(out.kind=='scale_invariant').sum()}  "
          f"LEVEL_BASED={(out.kind=='LEVEL_BASED').sum()}  "
          f"constant={(out.kind=='constant').sum()}  "
          f"other={(~out.kind.isin(['scale_invariant','LEVEL_BASED','constant'])).sum()}")

if __name__ == "__main__":
    main()
