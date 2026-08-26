# -*- coding: utf-8 -*-
"""Final training script — Rolling-forward LightGBM on the full feature matrix.

Inputs
------
- data/build/features_all.parquet : long frame (date, asset, <pool:factor>...), NaN preserved
  (~755k rows x ~85 cols, 2016-01-04 .. 2026-08-10, 297 assets)
- data/panel/fwd_ret10.parquet    : date x asset matrix of 10-day forward returns (Vwap basis)

Pipeline
--------
  1. Merge features_all with fwd_ret10 long form -> label column `fwd` (Vwap 10d forward).
     Features = every column not in (date, asset, fwd).
  2. Walk-forward (expanding window): first train = all history < 2019-01-04 (first 3 years),
     then every 3 months retrain on all history < cut and predict the next 3-month block OOS.
  3. LightGBM regression via lgb.train: objective=regression, lr=0.03, num_leaves=127,
     min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
     num_boost_round=300. First try device='gpu' (max_bin=127); on ANY exception/bad build/OOM
     fall back to CPU (num_threads=16) for this fold and all later folds; each fold prints
     which device was used.
  4. Concatenate OOS predictions to long frame (date, asset, pred), save
     data/build/predictions.parquet.

Memory
------
Per fold only the train Xt/yt (float32) and the OOS Xo float32 arrays are alive; model, Dataset
and matrices are released (del + gc.collect) before the next fold. No multi-fold retention.

Run
---
nohup /srv/quant/envs/quantaalpha/bin/python -u scripts/train_final.py \
      > /tmp/train_final.log 2>&1 &     # log() also appends to /tmp/train_final.log
"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
LOG_PATH = "/tmp/train_final.log"
# Validation mode: ONLY_FOLD1=1 runs just the first fold (GPU path check), prints
# the per-fold line, then exits WITHOUT writing predictions.parquet.
ONLY_FOLD1 = os.environ.get("ONLY_FOLD1") == "1"
PARAMS = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.03,
    num_leaves=127,
    min_data_in_leaf=30,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    num_boost_round=300,
    verbosity=-1,
)
N_BOOST = 300
MAX_BIN = 127


def log(msg):
    line = str(msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    t_total = time.time()

    log("[train] loading features_all ...")
    df = pd.read_parquet(f"{ROOT}/data/build/features_all.parquet")
    df["date"] = pd.to_datetime(df["date"])
    log(f"[train] features_all {df.shape} range "
        f"{df['date'].min().date()}..{df['date'].max().date()}")

    log("[train] loading fwd_ret10 (Vwap 10d forward) -> long form ...")
    fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")  # DatetimeIndex named date
    fwd_long = fwd.stack().rename("fwd").reset_index()
    fwd_long.columns = ["date", "asset", "fwd"]
    log(f"[train] fwd_long {fwd_long.shape}")

    df = df.merge(fwd_long, on=["date", "asset"], how="left")
    del fwd, fwd_long
    gc.collect()
    if df["fwd"].isna().all():
        log("[train] ERROR: no label matched (merge failed) — aborting")
        sys.exit(1)
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)

    FEATURES = [c for c in df.columns if c not in ("date", "asset", "fwd")]
    log(f"[train] rows={len(df)}  n_features={len(FEATURES)}")

    # ---- rolling schedule (expanding window) ----
    START = df["date"].min()                      # 2016-01-04
    INITIAL_MONTHS = 36                           # first 3 years of history
    ROLL_MONTHS = 3                               # retrain every quarter / predict next quarter
    first_cut = START + pd.DateOffset(months=INITIAL_MONTHS)   # 2019-01-04
    oos_end = df["date"].max()
    rolls = []
    c = first_cut
    while c <= oos_end:
        rolls.append(c)
        c += pd.DateOffset(months=ROLL_MONTHS)
    log(f"[train] folds={len(rolls)} first_cut={first_cut.date()} roll={ROLL_MONTHS}m")

    params_gpu = dict(PARAMS, device="gpu", max_bin=MAX_BIN)
    params_cpu = dict(PARAMS, device="cpu", num_threads=16)

    use_gpu = False
    pred_frames = []

    # ---- rolling forward training ----
    for i, cut in enumerate(rolls):
        next_end = cut + pd.DateOffset(months=ROLL_MONTHS)
        date = df["date"]
        train_mask = date < cut                       # expanding history
        oos_mask = (date >= cut) & (date < next_end)

        Xt = df.loc[train_mask, FEATURES].values.astype(np.float32)
        yt = df.loc[train_mask, "fwd"].values.astype(np.float32)
        keep = ~np.isnan(yt)
        Xt, yt = Xt[keep], yt[keep]
        n_train = len(Xt)

        ooss_ids = df.loc[oos_mask, ["date", "asset"]]
        n_oos = len(ooss_ids)
        if n_train < 1000 or n_oos < 100:
            log(f"[train][{i + 1}/{len(rolls)}] cut={cut.date()} train_rows={n_train} "
                f"oos_rows={n_oos} SKIP (too few rows)")
            del train_mask, oos_mask, Xt, yt, ooss_ids
            gc.collect()
            continue

        t0 = time.time()
        dset = lgb.Dataset(Xt, yt, free_raw_data=True)
        model = None
        if use_gpu:
            try:
                model = lgb.train(params_gpu, dset, num_boost_round=N_BOOST)
            except Exception as e:
                log(f"[train] GPU failed ({type(e).__name__}: {str(e)[:160]}) -> fallback CPU "
                    f"for fold {i + 1} and all later folds")
                use_gpu = False
        if model is None:
            model = lgb.train(params_cpu, dset, num_boost_round=N_BOOST)

        Xo = df.loc[oos_mask, FEATURES].values.astype(np.float32)
        num_iter = model.best_iteration if (model.best_iteration and model.best_iteration > 0) else N_BOOST
        pred = model.predict(Xo, num_iteration=num_iter)

        o = ooss_ids.copy()
        o["pred"] = pred
        pred_frames.append(o)

        dt = time.time() - t0
        dev = "GPU" if use_gpu else "CPU"
        log(f"[train][{i + 1}/{len(rolls)}] cut={cut.date()} train_rows={n_train} "
            f"oos_rows={n_oos} {dt:.1f}s dev={dev}  (iters={num_iter})")

        # release per-fold memory (no multi-fold retention)
        del date, train_mask, oos_mask, Xt, yt, Xo, dset, model, pred, o, ooss_ids
        gc.collect()

        if ONLY_FOLD1:
            log("VALIDATION ONLY_FOLD1=1: first fold completed — stopping before writing output.")
            return

    if not pred_frames:
        log("[train] ERROR: no OOS predictions produced — aborting")
        sys.exit(1)

    pred = pd.concat(pred_frames, ignore_index=True)
    log(f"[train] OOS predictions {len(pred)} range "
        f"{pred['date'].min().date()}..{pred['date'].max().date()}")
    out = f"{ROOT}/data/build/predictions.parquet"
    pred.to_parquet(out)
    log(f"[train] SAVED {out}  total={time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()