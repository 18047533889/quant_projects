# -*- coding: utf-8 -*-
"""并行滚动前向 LightGBM 训练 — 4 worker × 8 线程, 加速至 ~25 分钟。
用法: python train_parallel.py <slice_id 0..3>;  主进程(无参数)收集4段结果合并 predictions.parquet。
折区间: 31 折 → 4 段: [0:8), [8:16), [16:24), [24:31)。
"""
import os, sys, time, gc, pandas as pd, numpy as np, lightgbm as lgb

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
N_WORKERS = 4
ROLL_MONTHS = 3
N_BOOST = 300

PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.03,
    num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
    device="cpu", num_threads=8,
)

def log(msg):
    print(msg, flush=True)

def load_data():
    df = pd.read_parquet(f"{ROOT}/data/build/features_all.parquet")
    df["date"] = pd.to_datetime(df["date"])
    fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")
    fwd.index = pd.to_datetime(fwd.index).date
    fl = fwd.stack().rename("fwd").reset_index(); fl.columns = ["date", "asset", "fwd"]
    df["date"] = df["date"].dt.date
    df = df.merge(fl, on=["date", "asset"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    FEATURES = [c for c in df.columns if c not in ("date", "asset", "fwd")]
    return df, FEATURES

def folds():
    START = pd.Timestamp("2016-01-04")
    first_cut = START + pd.DateOffset(months=36)
    rolls = []; c = first_cut
    while c <= pd.Timestamp("2026-08-10"):
        rolls.append(c); c += pd.DateOffset(months=ROLL_MONTHS)
    return rolls

def run_slice(sl):
    df, FEATURES = load_data()
    rolls = folds()
    dates = df["date"]; FEAT = FEATURES
    out = []
    # slice index range
    lo = (sl * len(rolls)) // N_WORKERS
    hi = ((sl + 1) * len(rolls)) // N_WORKERS
    for i in range(lo, hi):
        cut = rolls[i]
        next_end = cut + pd.DateOffset(months=ROLL_MONTHS)
        train_mask = dates < cut
        oos_mask = (dates >= cut) & (dates < next_end)
        Xt = df.loc[train_mask, FEAT].values.astype(np.float32)
        yt = df.loc[train_mask, "fwd"].values.astype(np.float32)
        keep = ~np.isnan(yt); Xt, yt = Xt[keep], yt[keep]
        oos_ids = df.loc[oos_mask, ["date", "asset"]]
        if len(Xt) < 1000 or len(oos_ids) < 100:
            continue
        t0 = time.time()
        m = lgb.train(PARAMS, lgb.Dataset(Xt, yt, free_raw_data=True), num_boost_round=N_BOOST)
        Xo = df.loc[oos_mask, FEAT].values.astype(np.float32)
        num_iter = m.best_iteration or N_BOOST
        pred = m.predict(Xo, num_iteration=num_iter)
        o = oos_ids.copy(); o["pred"] = pred
        out.append(o)
        log(f"[w{sl}][{i+1}/{len(rolls)}] cut={cut.date()} train={len(Xt)} oos={len(o)} {time.time()-t0:.0f}s")
        del train_mask, oos_mask, Xt, yt, Xo, m, o
        gc.collect()
    if out:
        res = pd.concat(out, ignore_index=True)
        res.to_parquet(f"{ROOT}/data/build/preds_slice_{sl}.parquet")
        log(f"[w{sl}] done {len(res)} rows -> preds_slice_{sl}.parquet")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_slice(int(sys.argv[1]))
    else:
        import glob
        parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/data/build/preds_slice_*.parquet"))]
        pred = pd.concat(parts, ignore_index=True)
        pred.to_parquet(f"{ROOT}/data/build/predictions.parquet")
        print(f"merged {len(pred)} rows -> predictions.parquet  {pred.date.min().date()}..{pred.date.max().date()}")