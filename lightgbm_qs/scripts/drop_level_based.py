# -*- coding: utf-8 -*-
"""剔除 20 个水平型(level-based)因子 —— 它们依赖绝对价格水平, 未复权输入下与后复权不一致,
且公式在仓库内不可恢复(外部 OAP/Tak/JQ delivery), 无法用后复权数据重算。

审计依据(scripts/audit_adj_consistency.py): 这 20 个因子值与后复权累积因子 F 的截面相关 |corr|>0.3,
其余 1372 个 LQTP 因子为比率型(scale-invariant, 分子分母同乘 F 抵消, 已与后复权一致)。

输出: data/build/features_full2_prep_adj.parquet (2166-20=2146 列, 剔除水平型因子)
用法: python3.12 drop_level_based.py
"""
import os, time
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
IN = os.path.join(ROOT, "data/build/features_full2_prep.parquet")
OUT = os.path.join(ROOT, "data/build/features_full2_prep_adj.parquet")
LOG = "/tmp/drop_level_based.log"

# 20 个水平型因子(审计确认, 需剔除)
LEVEL_BASED = [
    "JQ_ALPHA_081", "JQ_ALPHA_144",
    "OAP_AUDIT2_MomOffSeason", "OAP_IdioVolAHT", "OAP_Illiquidity",
    "OAP_LRreversal", "OAP_RESCAN4_VolSD", "OAP_STD_FIX_MomOffSeason",
    "OAP_VERIFIED_UNIFIED_Illiquidity", "OAP_VERIFIED_UNIFIED_LRreversal",
    "OAP_VERIFIED_UNIFIED_MomOffSeason", "OAP_VERIFIED_UNIFIED_VolSD",
    "OAP_recover_IdioVolAHT", "OAP_recover_Illiquidity", "OAP_recover_VolSD",
    "TDX_NEW_LON_V02_LOW_STATE_AMT", "Tak_Sent_063", "Tak_Vol_066",
    "alpha_20260703_d485a3b3", "ext_c26a3d0b",
]

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
    drop = [c for c in feats if any(lb in c for lb in LEVEL_BASED)]
    keep = [c for c in feats if c not in drop]
    plog(f"  剔除水平型因子 {len(drop)} 个: {drop}")
    plog(f"  保留 {len(keep)} 个因子")
    out = df[["date", "asset"] + keep]
    out.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] WROTE {OUT} {out.shape} total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
