# -*- coding: utf-8 -*-
"""Step 4: rolling-forward LightGBM with per-12-month Optuna hyper-parameter re-search.

Caliber (hard rules):
  - features : data/build/features_full3_adj_prep.parquet (flip-selected, prepped)
  - label    : data/build/fwd_adj_neu.parquet (= neutralized COS TargetVwapReturnH10,
               AdjVwap[t+11]/AdjVwap[t+1]-1, cross-section demean/zscore)
  - per-fold feature list : data/build/walkforward_selection_flip.json
               (purged train-window rank_ic>0.015, NEVER full-sample)
  - purge/embargo : drop the last PURGE_TRADING_DAYS trading days before each cut
               from BOTH train tail and OOS front (10d fwd-label horizon).

New (user 2026-08-28): hyper-params re-searched every 12 months.
  - cuts every 3 months; cuts[0::4] are "re-opt cuts".
  - at each re-opt cut: run Optuna (N_TRIALS=12, capped rows) on that cut's purged
    train window using the fold's own feature subset; cache best params per cut.
  - the 3 following cuts reuse the most recent cached params.
  - first re-opt cut (2020-01-04) -> params for 2020-01/04/07/10.

Usage:
  python train_opt_adj_flip.py <slice 0..3>     # parallel rolling slices
  python train_opt_adj_flip.py                  # merge slices -> predictions_adj_flip.parquet
Env:
  FEATURES_PARQUET / LABEL_PARQUET / SELECTION_MANIFEST / SLICE_PAT / BEST_JSON_DIR
"""
import os
import sys
import time
import gc
import glob
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
N_WORKERS = 4
ROLL_MONTHS = 3
TRAIN_MONTHS = 48
PURGE_TRADING_DAYS = 10
EMBARGO_TRADING_DAYS = 0
RE_OPT_EVERY = 2            # every 2 cuts = 6 months (user-asked 2026-08-29)
N_TRIALS = 12               # Optuna trials per re-search
OPT_MAX_ROWS = 120_000
OPT_MAX_ROUND = 300
OPT_EARLY = 20

FEATURES_PARQUET = os.environ.get("FEATURES_PARQUET", f"{ROOT}/data/build/features_full3_adj_prep.parquet")
LABEL_PARQUET = os.environ.get("LABEL_PARQUET", f"{ROOT}/data/build/fwd_adj_neu.parquet")
SELECTION_MANIFEST = os.environ.get("SELECTION_MANIFEST", f"{ROOT}/data/build/walkforward_selection_flip.json")
BEST_JSON_DIR = os.environ.get("BEST_JSON_DIR", f"{ROOT}/data/build/best_params_flip")
SLICE_PAT = os.environ.get("SLICE_PAT", f"{ROOT}/data/build/preds_adj_flip_slice_%d.parquet")
LOG = "/tmp/train_opt_adj_flip.log"

DEFAULT_PARAMS = dict(objective="regression", metric="l2", learning_rate=0.03,
                      num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                      device="cpu", num_threads=4)
MAX_BOOST = 400

_logf = None
def log(m):
    print(m, flush=True)
    if _logf is not None:
        _logf.write(m + "\n")
        _logf.flush()


def load_data():
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    lab = pd.read_parquet(LABEL_PARQUET)
    lab["date"] = pd.to_datetime(lab["date"])
    df = df.merge(lab, on=["date", "asset"], how="left")
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    FEAT = [c for c in df.columns if c not in ("date", "asset", "fwd_neu")]
    return df, FEAT


def rolls():
    START = pd.Timestamp("2016-01-04")
    fc = START + pd.DateOffset(months=TRAIN_MONTHS)
    r = []
    c = fc
    while c <= pd.Timestamp("2026-08-10"):
        r.append(c)
        c += pd.DateOffset(months=ROLL_MONTHS)
    return r


def load_fold_selection():
    if not os.path.exists(SELECTION_MANIFEST):
        raise FileNotFoundError(
            f"[train] {SELECTION_MANIFEST} 不存在 —— 先运行 "
            "build_walkforward_manifest_flip.py。全样本 selected_factors*.csv 不是合法替代(选择泄漏)。")
    with open(SELECTION_MANIFEST) as f:
        payload = json.load(f)
    if payload.get("purge_trading_days") != PURGE_TRADING_DAYS:
        raise ValueError("selection manifest purge mismatch — regenerate")
    folds = {pd.Timestamp(k): list(v.get("factors", [])) for k, v in payload.get("cuts", {}).items()}
    return folds


def _lgb_params(p):
    out = dict(DEFAULT_PARAMS)
    for k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
              "bagging_fraction", "lambda_l1", "lambda_l2", "num_boost_round"):
        if k in p:
            out[k] = p[k]
    return out


def load_best_params_for(cut, rs, reopt_cache):
    """Return params for this cut: nearest re-opt cache <= cut, else DEFAULT.

    Falls back to disk (BEST_JSON_DIR/best_<date>.json) so a slice can reuse the
    re-opt results computed by an earlier slice (e.g. cut 4's search feeds cuts
    5/6/7 even though cut 6 lives in slice 1)."""
    reopt_cuts = sorted([c for c in reopt_cache])
    disk_cuts = []
    if os.path.isdir(BEST_JSON_DIR):
        for fn in glob.glob(os.path.join(BEST_JSON_DIR, "best_*.json")):
            try:
                disk_cuts.append(pd.Timestamp(os.path.basename(fn)[5:-5]))
            except Exception:
                pass
    all_cuts = sorted(set(reopt_cuts) | set(disk_cuts))
    chosen = None
    for c in all_cuts:
        if c <= cut:
            chosen = c
        else:
            break
    if chosen is not None:
        if chosen in reopt_cache:
            return _lgb_params(reopt_cache[chosen]), chosen
        fp = os.path.join(BEST_JSON_DIR, f"best_{chosen.date()}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                return _lgb_params(json.load(f)), chosen
    return dict(DEFAULT_PARAMS), None


def ensure_nearest_reopt(cut_idx, rs, reopt_cache, sl):
    """If this cut needs re-opt params owned by an EARLIER slice (re-opt index
    < lo), wait (bounded 30 min) for that slice to persist best_<date>.json so we
    don't silently fall back to DEFAULT_PARAMS at slice boundaries."""
    r = cut_idx - (cut_idx % RE_OPT_EVERY)
    if r in reopt_cache:
        return
    if r < 0 or cut_idx % RE_OPT_EVERY == 0:
        return  # this slice runs the re-opt itself in this iteration
    need_cut = rs[r]
    fp = os.path.join(BEST_JSON_DIR, f"best_{need_cut.date()}.json")
    if os.path.exists(fp):
        with open(fp) as f:
            reopt_cache[need_cut] = json.load(f)
        return
    log(f"[w{sl}] cut={need_cut.date()} 的 re-opt 由前序 slice 负责，等待落盘 …")
    waited = 0
    while not os.path.exists(fp) and waited < 1800:
        time.sleep(10)
        waited += 10
    if os.path.exists(fp):
        with open(fp) as f:
            reopt_cache[need_cut] = json.load(f)
        log(f"[w{sl}] cut={need_cut.date()} re-opt params 就绪（等 {waited}s）")
    else:
        log(f"[w{sl}] WARN: cut={need_cut.date()} re-opt 30min 未落盘，本折用 DEFAULT_PARAMS")


def run_optuna(cut, df, uniq_dates, FEAT_FOLD):
    """Optuna search on the purged train window ending at cut. Returns best params dict."""
    import optuna
    u_purge = PURGE_TRADING_DAYS + EMBARGO_TRADING_DAYS
    u_train = uniq_dates[uniq_dates < cut]
    u_train = u_train[:-u_purge] if len(u_train) > u_purge else u_train
    tr = df[df["date"].isin(u_train)]
    tr = tr.tail(OPT_MAX_ROWS)
    log(f"  [opt@{cut.date()}] rows={len(tr)} feats={len(FEAT_FOLD)} trials={N_TRIALS}")

    def _split_cv(tr_df):
        dts = np.sort(tr_df["date"].unique())
        n = len(dts)
        k = min(3, max(2, n // 60))
        edges = np.array_split(dts, k)
        folds = []
        for i in range(k):
            va_dates = set(edges[i])
            tr_va = tr_df[~tr_df["date"].isin(va_dates)]
            folds.append((tr_va, tr_df[tr_df["date"].isin(va_dates)]))
        return folds

    def _trial_score(params, tr_df):
        ics = []
        for trf, vaf in _split_cv(tr_df):
            Xt = trf[FEAT_FOLD].values.astype(np.float32)
            yt = trf["fwd_neu"].values.astype(np.float32)
            kk = ~np.isnan(yt)
            Xt, yt = Xt[kk], yt[kk]
            if len(Xt) < 3000:
                continue
            nv = max(1000, int(len(Xt) * 0.1))
            Xv, yv = Xt[-nv:], yt[-nv:]
            Xt, yt = Xt[:-nv], yt[:-nv]
            m = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=OPT_MAX_ROUND,
                          valid_sets=[lgb.Dataset(Xv, yv)],
                          callbacks=[lgb.early_stopping(OPT_EARLY, verbose=False)])
            Xo = vaf[FEAT_FOLD].values.astype(np.float32)
            pred = m.predict(Xo, num_iteration=m.best_iteration or 100)
            y = vaf["fwd_neu"].values.astype(np.float32)
            mask = ~np.isnan(y)
            if mask.sum() < 100:
                continue
            ic = np.corrcoef(pred[mask], y[mask])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
            del Xt, yt, Xv, yv, m
            gc.collect()
        return float(np.mean(ics)) if ics else -1.0

    def objective(trial):
        p = dict(DEFAULT_PARAMS)
        p["learning_rate"] = trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.03, 0.05])
        p["num_leaves"] = trial.suggest_categorical("num_leaves", [31, 63, 127, 255])
        p["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [10, 20, 30, 50])
        p["feature_fraction"] = trial.suggest_categorical("feature_fraction", [0.6, 0.7, 0.8, 0.9])
        p["bagging_fraction"] = trial.suggest_categorical("bagging_fraction", [0.7, 0.8, 0.9])
        p["lambda_l1"] = trial.suggest_categorical("lambda_l1", [0.0, 0.1, 1.0, 5.0])
        p["lambda_l2"] = trial.suggest_categorical("lambda_l2", [0.0, 0.1, 1.0, 5.0])
        p["num_boost_round"] = trial.suggest_categorical("num_boost_round", [200, 300, 400, 600])
        return _trial_score(p, tr)

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    best["num_boost_round"] = int(best["num_boost_round"])
    log(f"  [opt@{cut.date()}] best_ic={study.best_value:.5f} params={best} ({time.time()-t0:.0f}s)")
    return best


def run_slice(sl):
    df, FEAT = load_data()
    rs = rolls()
    dates = df["date"]
    fold_features = load_fold_selection()
    uniq_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    u_purge = PURGE_TRADING_DAYS + EMBARGO_TRADING_DAYS
    lo = (sl * len(rs)) // N_WORKERS
    hi = ((sl + 1) * len(rs)) // N_WORKERS
    out = []
    reopt_cache = {}
    for i in range(lo, hi):
        cut = rs[i]
        nxt = cut + pd.DateOffset(months=ROLL_MONTHS)
        if cut not in fold_features:
            raise KeyError(f"[train] cut {cut.date()} 在 selection manifest 中无条目")
        FEAT_FOLD = [f for f in fold_features[cut] if f in set(FEAT)]
        if not FEAT_FOLD:
            log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} SKIP: no fold features")
            continue

        # per-12-month re-opt: global cut index i % 4 == 0 (cut 0, 4, 8, ... = every 12 months)
        if i % RE_OPT_EVERY == 0:
            best = run_optuna(cut, df, uniq_dates, FEAT_FOLD)
            reopt_cache[cut] = best
            os.makedirs(BEST_JSON_DIR, exist_ok=True)
            with open(os.path.join(BEST_JSON_DIR, f"best_{cut.date()}.json"), "w") as f:
                json.dump(best, f, indent=2)
            log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} re-opt done")
        else:
            ensure_nearest_reopt(i, rs, reopt_cache, sl)

        params, src_cut = load_best_params_for(cut, rs, reopt_cache)

        u_train = uniq_dates[uniq_dates < cut]
        u_train_cut = u_train[:-u_purge] if len(u_train) > u_purge else u_train
        u_oos = uniq_dates[(uniq_dates >= cut) & (uniq_dates < nxt)]
        u_oos_cut = u_oos[u_purge:] if len(u_oos) > u_purge else u_oos
        train = df[dates.isin(u_train_cut)]
        oos = df[dates.isin(u_oos_cut)]
        if len(oos) < 100:
            continue
        Xt = train[FEAT_FOLD].values.astype(np.float32)
        yt = train["fwd_neu"].values.astype(np.float32)
        kk = ~np.isnan(yt)
        Xt, yt = Xt[kk], yt[kk]
        if len(Xt) < 5000:
            continue
        nv = max(2000, int(len(Xt) * 0.1))
        Xv, yv = Xt[-nv:], yt[-nv:]
        Xt, yt = Xt[:-nv], yt[:-nv]
        t0t = time.time()
        max_round = int(params.get("num_boost_round", MAX_BOOST))
        m = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=max_round,
                      valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        Xo = oos[FEAT_FOLD].values.astype(np.float32)
        pred = m.predict(Xo, num_iteration=m.best_iteration or 100)
        o = oos[["date", "asset"]].copy()
        o["pred"] = pred
        out.append(o)
        log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} feats={len(FEAT_FOLD)} src={src_cut.date() if src_cut else 'default'} "
            f"train={len(Xt)} oos={len(o)} iters={m.best_iteration or 100} {time.time()-t0t:.0f}s")
        del Xt, yt, Xv, yv, Xo, m, o
        gc.collect()
    if out:
        res = pd.concat(out, ignore_index=True)
        res.to_parquet(SLICE_PAT % sl)
        log(f"[w{sl}] done {len(res)} -> preds_adj_flip_slice_{sl}.parquet")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    _logf = open(LOG, "a")
    log(f"=== train_opt_adj_flip.py started argv={sys.argv[1:]} ===")
    if len(sys.argv) > 1:
        run_slice(int(sys.argv[1]))
    else:
        parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/data/build/preds_adj_flip_slice_*.parquet"))]
        if not parts:
            log("no slices found")
        else:
            pred = pd.concat(parts, ignore_index=True)
            pred.to_parquet(f"{ROOT}/data/build/predictions_adj_flip.parquet")
            log(f"merged {len(pred)} -> predictions_adj_flip.parquet  "
                f"{pred.date.min()}..{pred.date.max()}")
    if _logf is not None:
        _logf.close()
