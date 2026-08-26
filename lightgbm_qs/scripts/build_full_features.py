# -*- coding: utf-8 -*-
"""补全特征矩阵构建 — 全量有效因子(drop=False) → 满时间线 → 补全 → 完全去重。

口径（用户 2026-08-26）：
1. data/build/rankic_all_factors.csv 中所有 drop=False（有有效 rank_ic）的因子 ~856 个。
2. 从 data/factor_pools/{fm247,fmqa,cogfull,cogshort,cogneutral,delivery,optfac,factmat}
   读各因子值，统一成 (date, asset) 因子列，对齐到全时间线 2016-01..2026-08。
   asset 限定 tradable（data/panel/vwap_trad.parquet 的列）。
3. 补全：按 asset 分组 ffill + bfill，残余 NaN 用当日截面中位数兜底；全空列才丢弃。
4. 完全去重：补全后按抽样(≤40k行)算相关，corr>0.98 簇内只留 rank_ic 最高者。
5. 输出 data/build/features_full_filled.parquet (long: date,asset,<保留因子>) 与
   data/build/selected_full.csv（保留因子清单）。
"""
import gc, os, time, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
POOLS = os.path.join(ROOT, "data", "factor_pools")
BUILD = os.path.join(ROOT, "data", "build")
OUT_PARQUET = os.path.join(BUILD, "features_full_filled.parquet")
OUT_CSV = os.path.join(BUILD, "selected_full.csv")
CORR_THRESH = 0.98
SAMPLE_N = 40000


def plog(*a):
    print(" ".join(str(x) for x in a), flush=True)


t0 = time.time()

# ---------- 1. 候选 ----------
ic = pd.read_csv(os.path.join(BUILD, "rankic_all_factors.csv"))
valid = ic[ic["drop"] == False].reset_index(drop=True)  # noqa: E712
rank_ic = dict(zip(valid["name"], valid["rank_ic"]))
valid_names = valid["name"].tolist()
plog(f"[1] 总候选 {len(ic)}，drop=False 有效候选 {len(valid_names)}")

# ---------- 2. 基准网格 ----------
vw = pd.read_parquet(os.path.join(ROOT, "data", "panel", "vwap_trad.parquet"))
vw.index = pd.to_datetime(vw.index).date
dates = vw.index.to_numpy()
assets = vw.columns.to_numpy()
grid = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]).sort_values()
plog(f"[2] 网格: {len(grid)} cells, dates {dates[0]}..{dates[-1]}, assets {len(assets)}")
del vw
gc.collect()

trad_set = set(assets)


def load_long(fp):
    """long: datetime/asset/factor_value -> series aligned to grid (float32)."""
    df = pq.read_table(fp, columns=["datetime", "asset", "factor_value"]).to_pandas()
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    s = df.set_index(pd.MultiIndex.from_arrays([df["date"], df["asset"]], names=["date", "asset"]))["factor_value"]
    s = pd.to_numeric(s, errors="coerce")
    return s.reindex(grid).astype(np.float32)


def load_wide(fp):
    """宽格式 index=date cols=asset -> aligned series float32 (only tradable cols)."""
    df = pq.read_table(fp).to_pandas()
    df.index = pd.to_datetime(df.index)
    cols = [c for c in df.columns if c in trad_set]
    if not cols:
        return None
    df = df[cols]
    s = df.stack()
    s.index = s.index.set_names(["date", "asset"])
    s = s.reindex(grid)
    return pd.to_numeric(s, errors="coerce").astype(np.float32)


# 枚举全部池文件 → (pool, name, fp, fmt)
all_candidates = []
for pool, fmt, d in [
    ("fm247", "long", "fm247"),
    ("fmqa", "long", "fmqa"),
    ("cogfull", "wide", "cogfull"),
    ("cogshort", "wide", "cogshort"),
    ("cogneutral", "wide", "cogneutral"),
    ("delivery", "wide", "delivery"),
    ("optfac", "wide", "optimized_factors"),
    ("factmat", "wide", "factor_matrices"),
    ("lqtp", "wide", "lqtp"),
]:
    base = os.path.join(POOLS, d)
    if pool == "delivery":
        files = sorted(glob.glob(os.path.join(base, "*", "factor_values_test.parquet")))
    else:
        files = sorted(glob.glob(os.path.join(base, "*.parquet")))
    for fp in files:
        if pool == "delivery":
            name = os.path.basename(os.path.dirname(fp))
        else:
            name = os.path.basename(fp)
            if name.endswith("_neu.parquet"):
                name = name[: -len("_neu.parquet")]
            elif name.endswith(".parquet"):
                name = name[: -len(".parquet")]
        all_candidates.append((pool, name, fp, fmt))

idx2cand = {f"{p}:{n}": (p, n, f, ft) for p, n, f, ft in all_candidates}

# ---------- 逐因子加载对齐 ----------
plog("[2] 开始逐因子读取并对齐到网格 …")
loaded = {}      # key -> np.array (len(grid)) float32
validated = []   # keys 成功读取顺序
missing = []
for key in valid_names:
    c = idx2cand.get(key)
    if c is None:
        missing.append((key, "no-file"))
        continue
    pool, name, fp, fmt = c
    try:
        s = load_long(fp) if fmt == "long" else load_wide(fp)
        if s is None or s.dropna().empty:
            missing.append((key, "empty"))
            continue
        loaded[key] = s.to_numpy()
        del s
        validated.append(key)
    except Exception as e:
        missing.append((key, str(e)[:80]))
        continue
    if len(loaded) % 50 == 0:
        plog(f"  已加载 {len(loaded)} / {len(valid_names)} …")
        gc.collect()

plog(f"[2] 实际读到 {len(loaded)} 个因子，缺失 {len(missing)} 个：")
for k, why in missing:
    plog(f"    缺失 {k}  ({why})")

# ---------- 3. 组装宽矩阵 ----------
plog("[3] 组装因子矩阵 …")
grid_arr = np.empty((len(grid), len(validated)), dtype=np.float32)
for j, k in enumerate(validated):
    grid_arr[:, j] = loaded[k]
del loaded
gc.collect()
plog(f"[3] 原始矩阵 {grid_arr.shape}")

cols = list(validated)
date_arr = grid.get_level_values("date").to_numpy()
asset_arr = grid.get_level_values("asset").to_numpy()

# ---------- 4. 补全 ----------
plog("[4] 补全：asset 组内 ffill+bfill，残余用当日截面中位数 …")
# 排序为 (asset, date)
order = np.lexsort((date_arr, asset_arr))
o_date = date_arr[order]
o_asset = asset_arr[order]
o_mat = grid_arr[order]

out_mat = np.empty_like(o_mat, dtype=np.float32)
ua = np.unique(o_asset)
for a in ua:
    mask = o_asset == a
    xdf = pd.DataFrame(o_mat[mask])
    f = xdf.ffill().values
    b = pd.DataFrame(f).bfill().values
    out_mat[mask] = b
del o_mat
gc.collect()

# 残余 NaN → 当日截面中位数
ud = np.unique(o_date)
for d in ud:
    mask = o_date == d
    x = out_mat[mask]
    nanc = np.isnan(x)
    if nanc.any():
        med = np.nanmedian(x, axis=0)
        out_mat[mask] = np.where(nanc, med, x)
del o_date, o_asset
gc.collect()

# 还原 (date, asset) 顺序
inv = np.empty_like(order)
inv[order] = np.arange(len(order))
out_mat = out_mat[inv]

# 最终兜底：残余 NaN 用列中位数 → 再列均值（保证 NaN 残留≈0）
resid = np.isnan(out_mat)
if resid.any():
    colmed = np.nanmedian(out_mat, axis=0)
    colmed = np.where(np.isnan(colmed), 0.0, colmed)
    out_mat = np.where(resid, colmed, out_mat)

# 全空列丢弃
allnan = np.isnan(out_mat).all(axis=0)
kept = [k for k, an in zip(cols, allnan) if not an]
out_mat = out_mat[:, ~allnan]
plog(f"[4] 补全后因子列数 {len(kept)}，丢弃全空列 {int(allnan.sum())}")

# ---------- 5. 完全去重 ----------
plog("[5] 完全去重：抽样 corr>0.98 簇内只留 rank_ic 最高 ……")
npix = len(out_mat)
step = max(1, npix // SAMPLE_N)
si = np.arange(0, npix, step)[:SAMPLE_N]
sam = out_mat[si].astype(np.float64)
std = np.nanstd(sam, axis=0)
ok = std > 0
samok = sam[:, ok]
oknames = [k for k, fl in zip(kept, ok) if fl]
corr = np.corrcoef(samok, rowvar=False)

order_ic = sorted(range(len(oknames)), key=lambda i: -rank_ic.get(oknames[i], -1))
keep_flag = np.zeros(len(oknames), dtype=bool)
for i in order_ic:
    if keep_flag.any():
        maxc = np.max(corr[i][keep_flag])
    else:
        maxc = 0.0
    if maxc < CORR_THRESH:
        keep_flag[i] = True
selected = [oknames[i] for i in range(len(oknames)) if keep_flag[i]]
plog(f"[5] 完全去重后保留 {len(selected)} 个因子")

# ---------- 6. 输出 ----------
plog("[6] 写出 features_full_filled.parquet + selected_full.csv ……")
keep_pos = np.array([kept.index(k) for k in selected])
final = out_mat[:, keep_pos].astype(np.float32)
res = pd.DataFrame({
    "date": grid.get_level_values("date"),
    "asset": grid.get_level_values("asset"),
})
for i, k in enumerate(selected):
    res[k] = final[:, i]
res["date"] = res["date"].astype("datetime64[ns]").dt.strftime("%Y-%m-%d")
res.to_parquet(OUT_PARQUET, index=False)

sel_df = valid[valid["name"].isin(selected)].copy()
sel_df.to_csv(OUT_CSV, index=False)

nan_res = float(res[selected].isna().mean().mean())
dt = time.time() - t0
plog("=" * 60)
plog(f"总有效候选        : {len(valid_names)}")
plog(f"实际读到因子      : {len(validated)}")
plog(f"补全后因子列数    : {len(kept)}")
plog(f"完全去重后保留    : {len(selected)}")
plog(f"features_full_filled shape : {res.shape}")
plog(f"NaN 残留          : {nan_res:.6f}")
plog(f"耗时              : {dt:.1f}s")
plog(f"输出              : {OUT_PARQUET}")
plog(f"清单              : {OUT_CSV}")
plog("=" * 60)
