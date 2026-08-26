# -*- coding: utf-8 -*-
"""Rolling-forward LightGBM (GPU) for 10-day forward returns, trained on the
full 82-feature matrix built from ALL factor pools.

Design (per user spec):
  - Train window: from 2016, expanding walk-forward.
  - Retrain every 3 months (rolling), predict on the following 3-month slice.
  - Feature: all selected factors (NaN preserved — LightGBM handles it natively).
  - Label: 10-day forward return (Vwap_t+10 / Vwap_t - 1).
  - GPU training (device='gpu', max_bin=63) for speed.
  - Outputs a prediction matrix (date x asset) for the OOS window.

Rolling: initial train = first 3 years [2016, 2019]; then every 3 months refit on
all history up to the block start (expanding), predict the next 3-month block.
"""
import pandas as pd, numpy as np, lightgbm as lgb, os

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
df = pd.read_parquet(f"{ROOT}/data/build/features_all.parquet")
df["date"] = pd.to_datetime(df["date"])
# Join the 10-day forward return label from the price panel
fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")
fwd.index = pd.to_datetime(fwd.index).date
fwd_long = fwd.stack().rename("fwd").reset_index()
fwd_long.columns = ["date", "asset", "fwd"]
df["date"] = df["date"].dt.date
df = df.merge(fwd_long, on=["date", "asset"], how="left")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "asset"]).reset_index(drop=True)

FEATURES = [c for c in df.columns if c not in ("date", "asset", "fwd")]
print(f"rows={len(df)}  features={len(FEATURES)}  date {df.date.min().date()}..{df.date.max().date()}")

START = df["date"].min()
INITIAL_TRAIN_MONTHS = 36
ROLL_MONTHS = 3
OOS_END = df["date"].max()
first_cut = START + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)

rolls = []
c = first_cut
while c <= OOS_END:
    rolls.append(c)
    c = c + pd.DateOffset(months=ROLL_MONTHS)
print(f"{len(rolls)} roll-forward refits (first_cut={first_cut.date()})")

params = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.03,
    num_leaves=127,
    min_data_in_leaf=30,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    device="gpu",
    max_bin=127,
    verbosity=-1,
)

pred_frames = []
for i, cut in enumerate(rolls):
    train = df[df["date"] < cut]
    next_end = cut + pd.DateOffset(months=ROLL_MONTHS)
    oos = df[(df["date"] >= cut) & (df["date"] < next_end)]
    if len(oos) < 200:
        continue
    Xt = train[FEATURES].values.astype(np.float32)
    yt = train["fwd"].values.astype(np.float32)
    keep = ~np.isnan(yt)
    Xt, yt = Xt[keep], yt[keep]
    if len(Xt) < 5000:
        continue
    model = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=400)
    Xo = oos[FEATURES].values.astype(np.float32)
    pred = model.predict(Xo, num_iteration=model.best_iteration or 400)
    o = oos[["date", "asset"]].copy()
    o["pred"] = pred
    pred_frames.append(o)
    print(f"[{i+1}/{len(rolls)}] cut={cut.date()} train={len(Xt)} oos={len(o)}")

pred = pd.concat(pred_frames, ignore_index=True)
print("total OOS predictions:", len(pred), "range:", pred.date.min().date(), "..", pred.date.max().date())
pred.to_parquet(f"{ROOT}/data/build/predictions.parquet")
print("saved predictions.parquet")
