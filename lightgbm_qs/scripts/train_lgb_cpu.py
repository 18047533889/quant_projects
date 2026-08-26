# -*- coding: utf-8 -*-
"""Rolling-forward LightGBM (CPU, 16 threads) for 10-day forward returns on the
82-feature matrix from ALL factor pools. Retrain every 3 months, predict next block.
"""
import pandas as pd, numpy as np, lightgbm as lgb, os, time

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
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
print(f"rows={len(df)} features={len(FEATURES)} {df.date.min().date()}..{df.date.max().date()}", flush=True)

START = df["date"].min(); ROLL = 3
first_cut = START + pd.DateOffset(months=36)
rolls=[]; c=first_cut
while c <= df["date"].max(): rolls.append(c); c = c+pd.DateOffset(months=ROLL)
print(f"{len(rolls)} refits first_cut={first_cut.date()}", flush=True)

params = dict(objective="regression", metric="l2", learning_rate=0.03,
    num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, num_threads=16, verbosity=-1)

pred_frames=[]
for i,cut in enumerate(rolls):
    train=df[df["date"]<cut]; nxt=cut+pd.DateOffset(months=ROLL)
    oos=df[(df["date"]>=cut)&(df["date"]<nxt)]
    if len(oos)<200: continue
    Xt=train[FEATURES].values.astype(np.float32); yt=train["fwd"].values.astype(np.float32)
    keep=~np.isnan(yt); Xt,yt=Xt[keep],yt[keep]
    if len(Xt)<5000: continue
    t0=time.time()
    m=lgb.train(params, lgb.Dataset(Xt,yt), num_boost_round=300)
    Xo=oos[FEATURES].values.astype(np.float32)
    pred=m.predict(Xo, num_iteration=m.best_iteration or 300)
    o=oos[["date","asset"]].copy(); o["pred"]=pred
    pred_frames.append(o)
    print(f"[{i+1}/{len(rolls)}] cut={cut.date()} train={len(Xt)} oos={len(o)} {time.time()-t0:.1f}s", flush=True)

pred=pd.concat(pred_frames, ignore_index=True)
print("OOS predictions:", len(pred), pred.date.min().date(), "..", pred.date.max().date(), flush=True)
pred.to_parquet(f"{ROOT}/data/build/predictions.parquet")
print("saved predictions.parquet", flush=True)
