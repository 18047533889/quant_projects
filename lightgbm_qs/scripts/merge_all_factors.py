# -*- coding: utf-8 -*-
"""合并全部本地因子池 → 统一筛 bug 清零 + rank_ic>0.015 → 全特征矩阵。

口径（用户 2026-08-26 定死）：
  本地 /home/sunhaiwei/quant_projects/lightgbm_qs/data/factor_pools/ 下全部因子，
  即 fm247(247) + fmqa(48) + cogfull(108) + cogshort(110) + cogneutral(26)
  + delivery(246 个 factor_values_test.parquet, wide: date x asset)
  ≈ 785 个因子候选。全部跑 rank_ic>0.015 筛选，选中的进特征矩阵。

三种格式统一：
  - long : datetime/asset/factor_value
  - wide : index=date, cols=asset
  - delivery: index=TradeDate(date), cols=asset（其实与 wide 同构）

约束：
  - 只保留 tradable 资产（vwap_trad.parquet 的列）
  - rank_ic 用 10 日前瞻收益（Vwap_t+10/Vwap_t-1）逐日截面秩相关
  - 特征矩阵：行=(date,asset)，列=选中因子，保留 NaN(模型原生处理)
记忆友好：逐因子处理、不进大 DataFrame 峰值；进度写 /tmp/build_merge.log
"""
import os, glob, sys, time, numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
os.makedirs(f"{ROOT}/data/build", exist_ok=True)
LOG = "/tmp/build_merge.log"
def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

plog(f"[{time.strftime('%H:%M:%S')}] ===== 合并全部本地因子池 开始 =====")
trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad.parquet").columns)
fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")   # date x asset
fwd.index = pd.to_datetime(fwd.index)
fwd_long = fwd.stack().rename("fwd").reset_index()
fwd_long.columns = ["date", "asset", "fwd"]
fwd_long["date"] = fwd_long["date"].dt.date
trad_set = set(trad)
plog(f"tradable assets: {len(trad)}   forward window: {fwd_long.date.min()}..{fwd_long.date.max()}")

def load_factor(full_path, fmt, trad_set):
    """统一拉成长格式 DataFrame[date,asset,fv]，只含 tradable 资产。fmt: long|wide"""
    try:
        if fmt == "long":
            df = pq.read_table(full_path, columns=["datetime", "asset", "factor_value"]).to_pandas()
            df = df[df.asset.isin(trad_set)]
            df = df.copy()
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
            return df[["date", "asset", "factor_value"]].rename(columns={"factor_value": "fv"})
        else:  # wide (含 delivery: TradeDate index, asset cols)
            df = pq.read_table(full_path).to_pandas()
            df.index = pd.to_datetime(df.index).date
            cols = [c for c in df.columns if c in trad_set]
            if not cols:
                return None
            w = df[cols].stack().rename("fv").reset_index()
            w.columns = ["date", "asset", "fv"]
            w["fv"] = pd.to_numeric(w["fv"], errors="coerce")
            return w
    except Exception as e:
        plog(f"    加载失败 {os.path.basename(full_path)}: {e}")
        return None

# ---------- 枚举全部因子 ----------
pools = [
    ("fm247",      "long",  "data/factor_pools/fm247"),
    ("fmqa",       "long",  "data/factor_pools/fmqa"),
    ("cogfull",    "wide",  "data/factor_pools/cogfull"),
    ("cogshort",   "wide",  "data/factor_pools/cogshort"),
    ("cogneutral", "wide",  "data/factor_pools/cogneutral"),
    ("delivery",   "wide",  "data/factor_pools/delivery"),   # 目录嵌套: <hash>/factor_values_test.parquet
    ("optfac",     "wide",  "data/factor_pools/optimized_factors"),   # 本地优化后因子(428)
    ("factmat",    "wide",  "data/factor_pools/factor_matrices"),     # 本地因子矩阵(61)
    ("lqtp",       "wide",  "data/factor_pools/lqtp"),                # LQTP 因子(本地+gRPC, 1272)
]
# delivery 是目录嵌套，单独展开（name 取父目录 hash，保证 246 个各自唯一）
candidates = []   # (pool, name, path, fmt)
for pool, fmt, d in pools:
    base = f"{ROOT}/{d}"
    if pool == "delivery":
        files = sorted(glob.glob(f"{base}/*/factor_values_test.parquet"))
    else:
        files = sorted(glob.glob(f"{base}/*.parquet"))
    for fp in files:
        if pool == "delivery":
            name = os.path.basename(os.path.dirname(fp))   # 父目录 hash = 因子 id
        else:
            name = os.path.basename(fp)
            if name.endswith("_neu.parquet"):
                name = name[:-len("_neu.parquet")]
            elif name.endswith(".parquet"):
                name = name[:-len(".parquet")]
        candidates.append((pool, name, fp, fmt))
plog(f"候选因子总数(全部池): {len(candidates)}")

# ---------- 两遍跑：先算 rank_ic（可并行，但按池串行省内存）----------
ic_rows = []
feat_map = {}   # (pool:name) -> long df
for pool, name, fp, fmt in candidates:
    key = f"{pool}:{name}"
    df = load_factor(fp, fmt, trad_set)
    if df is None or len(df) < 500:
        ic_rows.append((key, pool, np.nan, 0, True))
        continue
    m = df.merge(fwd_long, on=["date", "asset"], how="inner")
    if len(m) < 1500:
        ic_rows.append((key, pool, np.nan, 0, True))
        continue
    try:
        ic = m.groupby("date").apply(
            lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False)
        ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    except Exception as e:
        plog(f"  rank_ic 失败 {key}: {e}")
        ic_rows.append((key, pool, np.nan, 0, True))
        continue
    if len(ic) < 50:
        ic_rows.append((key, pool, np.nan, 0, True))
        continue
    rank_ic = float(ic.mean())
    ic_rows.append((key, pool, rank_ic, len(ic), False))
    del m, ic
    plog(f"  {key}: rank_ic={rank_ic:.4f} n={len(df)}")

icdf = pd.DataFrame(ic_rows, columns=["name", "pool", "rank_ic", "n_dates", "drop"])
icdf.to_csv(f"{ROOT}/data/build/rankic_all_factors.csv", index=False)
plog(f"== 完成 rank_ic 计算，共 {len(icdf)} 因子 ==")

# ---------- 筛选 rank_ic>0.015 ----------
sel = icdf[(icdf["rank_ic"] > 0.015) & (~icdf["drop"])].sort_values("rank_ic", ascending=False)
plog(f"== 选中因子 (rank_ic>0.015): {len(sel)} ==")
sel.to_csv(f"{ROOT}/data/build/selected_factors_full.csv", index=False)
sel_names = sel["name"].tolist()

# ---------- 完全重复去重：按因子值高度相关(>0.98)聚类，每簇只留 rank_ic 最高者 ----------
plog("开始按因子值相关性去重（完全重复 = corr>0.98 簇内只留 rank_ic 最高）…")
CORR_THRESH = 0.98
# 逐对抽样计算相关性（每因子取重叠区间的稀疏样本，控制内存）
samples = {}
for key in sel_names:
    pool, _, fp, fmt = [c for c in candidates if f"{c[0]}:{c[1]}" == key][0]
    df = load_factor(fp, fmt, trad_set)
    if df is None:
        continue
    # 抽样：最多 40k 行，保证 (date,asset) 对齐可比
    smp = df.sample(n=min(40000, len(df)), random_state=42)
    samples[key] = smp.set_index(["date", "asset"])["fv"]
    del df

skeys = list(samples)
to_drop = set()
for i in range(len(skeys)):
    for j in range(i + 1, len(skeys)):
        a, b = skeys[i], skeys[j]
        s = pd.concat([samples[a], samples[b]], axis=1, join="inner").dropna()
        if len(s) < 500:
            continue
        corr = s.iloc[:, 0].corr(s.iloc[:, 1])
        if corr >= CORR_THRESH:
            # 保留 rank_ic 高者，丢低者
            ic_a = sel.loc[sel["name"] == a, "rank_ic"].iloc[0]
            ic_b = sel.loc[sel["name"] == b, "rank_ic"].iloc[0]
            drop_this = a if ic_a <= ic_b else b
            to_drop.add(drop_this)
            plog(f"  去重: {drop_this} (corr={corr:.3f}, 与 {'/' .join(sorted({a,b}-{drop_this}))})")
plog(f"== 相似去重后去除 {len(to_drop)} 个完全重复因子，剩 {len(sel_names)-len(to_drop)} ==")
sel_names = [n for n in sel_names if n not in to_drop]

# ---------- 构建全特征矩阵 ----------
plog("开始构建全特征矩阵 …")
fwd_long["date2"] = fwd_long["date"].astype(str)
base = fwd_long[["date", "asset"]].set_index(["date", "asset"]).sort_index()
w = pd.DataFrame(index=base.index)
for key in sel_names:
    pool, _, _, _ = [c for c in candidates if f"{c[0]}:{c[1]}" == key][0]
    fp   = [c for c in candidates if f"{c[0]}:{c[1]}" == key][0][2]
    fmt  = [c for c in candidates if f"{c[0]}:{c[1]}" == key][0][3]
    df = load_factor(fp, fmt, trad_set)
    if df is None:
        continue
    sub = df.set_index(["date", "asset"])["fv"]
    w[key] = sub
    cur = len(sel_names)
    if (len(w.columns)) % 10 == 0:
        plog(f"  已合并 {len(w.columns)} 列 …")
    del sub, df

w = w.reset_index()
w.to_parquet(f"{ROOT}/data/build/features_all.parquet")
plog(f"== 全特征矩阵已保存: {w.shape} ==")
plog(f"[{time.strftime('%H:%M:%S')}] ===== 完成 =====")