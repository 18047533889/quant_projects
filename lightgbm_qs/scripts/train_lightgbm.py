# -*- coding: utf-8 -*-
"""Rolling-forward LightGBM for 10-day forward returns.

Design (per user spec):
  - Train window: from 2016, expanding walk-forward.
  - Retrain every 3 months (rolling), predict on the following 3-month test slice.
  - Feature: the selected factor values (cross-sectionally ranked + z-scored per date).
  - Label: 10-day forward return (Vwap_t+10 / Vwap_t - 1).
  - Outputs a prediction matrix (date x asset) for the OOS window, saved to parquet.

Rolling scheme:
  - First model trains on data in [2016-01-05, first_cut) where first_cut = train_start + 3y (2019).
    Actually to respect "train from 2016" + "retrain every 3 months":
    - Establish an initial training window [2016, 2019-01-01] (3 years).
    - Walk forward: each 3-month block is validation/OOS; at each block boundary,
      refit on all history up to the block start (expanding window), then predict the block.
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
df = pd.read_parquet(f"{ROOT}/data/build/train_dataset.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "asset"]).reset_index(drop=True)

FEATURES = [c for c in df.columns if c not in ("date", "asset", "fwd_ret10")]
print(f"rows={len(df)}  features={len(FEATURES)}  date {df.date.min().date()}..{df.date.max().date()}")

# Cross-sectionally rank+standardize each feature per date (tree models don't need it,
# but consistent scaling helps; skip ranking to keep raw signal — LightGBM is monotone-invariant).
# We'll keep raw values but drop all-NaN feature columns per fold.

# ---- Rolling setup ----
START = pd.Timestamp("2016-01-05")
INITIAL_TRAIN_MONTHS = 36     # first 3 years = training before first OOS
ROLL_MONTHS = 3               # retrain every 3 months
OOS_END = df["date"].max()

# Build list of roll boundaries (first-of-month)
all_dates = sorted(df["date"].unique())
first_cut = START + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)

# Generate roll start dates every 3 months from first_cut until OOS_END
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
    num_leaves=63,
    min_data_in_leaf=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    verbosity=-1,
    num_threads=os.cpu_count(),
)

pred_frames = []
for i, cut in enumerate(rolls):
    # train = rows with date < cut  (expanding train from 2016)
    train = df[df["date"] < cut]
    # next block = [cut, cut + 3mo)  -> OOS
    next_end = cut + pd.DateOffset(months=ROLL_MONTHS)
    oos = df[(df["date"] >= cut) & (df["date"] < next_end)]
    if len(oos) < 200:
        continue
    # drop feature columns that are entirely NaN in train
    cols = [f for f in FEATURES if train[f].notna().sum() > 0]
    Xt = train[cols].values.astype(np.float32)
    yt = train["fwd_ret10"].values.astype(np.float32)
    # drop rows with any NaN feature in train (label-driven model)
    keep = ~np.isnan(Xt).any(axis=1) & ~np.isnan(yt)
    Xt, yt = Xt[keep], yt[keep]
    if len(Xt) < 5000:
        continue
    model = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=400)
    # predict OOS
    Xo = oos[cols].values.astype(np.float32)
    # drop rows where features NaN
    ok = ~np.isnan(Xo).any(axis=1)
    if ok.sum() == 0:
        continue
    pred = model.predict(Xo[ok], num_iteration=model.best_iteration or 400)
    o = oos[ok][["date", "asset"]].copy()
    o["pred"] = pred
    pred_frames.append(o)
    print(f"[{i+1}/{len(rolls)}] cut={cut.date()} train={len(Xt)} oos={len(o)} pred@cut")

pred = pd.concat(pred_frames, ignore_index=True)
print("total OOS predictions:", len(pred), "date range:", pred.date.min().date(), "..", pred.date.max().date())
pred.to_parquet(f"{ROOT}/data/build/predictions.parquet")
print("saved predictions.parquet")
