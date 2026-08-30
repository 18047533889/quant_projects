# -*- coding: utf-8 -*-
"""滚动前向 LightGBM, 输入 = 预处理后特征(features_full2_prep.parquet) + 后复权中性化标签(fwd_adj_neu.parquet)。

用法(与 train_opt.py 一致):
  python train_opt_adj.py <slice 0..3>    # 并行滚动前向(4 worker)
  python train_opt_adj.py                 # 合并 preds_adj_slice_*.parquet -> predictions_adj.parquet

设计:
  - 特征: features_full2_prep.parquet (2166 列, 已 cs_winsor/rank/zscore 预处理)
  - 标签: fwd_adj_neu.parquet (后复权 Vwap 10日前向收益, 逐日截面中性化)
  - 超参: scripts/best_params_full2.json (optuna 搜索得到: lr=0.01, leaves=31, min_data=10, ff=0.7, bf=0.7, l1=5, l2=1, rounds=300)
  - 滚动: 首切 2016-01-04+48月=2020-01-04, 每 3 月 expanding 重训, 预测下 3 月, early stopping
  - 输出 pred 列名: 'pred' (可直接喂组合优化), 日期与 asset 与特征同网格

P0-B (2026-08-28) 两处防泄漏接线:
  1. 每 3 月切折时 purge 最后 PURGE_TRADING_DAYS=10 个交易日(训练尾部 + OOS 前端),
     与 train_final.py 同口径 —— 10 日前向标签窗口不再跨切点。
  2. 每折特征 = 只用 SELECTION_MANIFEST (data/build/walkforward_selection.json) 中
     该折自身训练窗选出的因子清单(factor_selection.py --folds-from-train 产出);
     全样本 selected_factors*.csv 一律不作为输入(全样本选择 = 用了未来 OOS 标签)。
     环境变量 SELECTION_MANIFEST 可覆盖清单路径。
"""
import os, sys, time, gc, glob, json
import pandas as pd, numpy as np, lightgbm as lgb

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
N_WORKERS = 4; ROLL_MONTHS = 3; TRAIN_MONTHS = 48
# P0-B purge/embargo (same semantics as train_final.py): drop the last
# (PURGE + EMBARGO) trading days of the train block before each cut and the same
# tail from the OOS front, so no 10-day fwd-label window crosses the cut.
# PURGE_TRADING_DAYS = 10 matches the fwd_ret10 (10d fwd Vwap) label horizon.
PURGE_TRADING_DAYS = 10
EMBARGO_TRADING_DAYS = 0
FEATURES_PARQUET = os.environ.get("FEATURES_PARQUET", f"{ROOT}/data/build/features_full2_prep.parquet")
LABEL_PARQUET = os.environ.get("LABEL_PARQUET", f"{ROOT}/data/build/fwd_adj_neu.parquet")
BEST_JSON = os.environ.get("BEST_JSON", f"{ROOT}/scripts/best_params_full2.json")
SLICE_PAT = os.environ.get("SLICE_PAT", f"{ROOT}/data/build/preds_adj_slice_%d.parquet")
SELECTION_MANIFEST = os.environ.get(
    "SELECTION_MANIFEST", f"{ROOT}/data/build/walkforward_selection.json")
LOG = "/tmp/train_opt_adj.log"

DEFAULT_PARAMS = dict(objective="regression", metric="l2", learning_rate=0.03,
                      num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                      device="cpu", num_threads=8)
MAX_BOOST = 400

_logf = None
def log(m):
    print(m, flush=True)
    if _logf is not None:
        _logf.write(m + "\n"); _logf.flush()

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
    START = pd.Timestamp("2016-01-04"); fc = START + pd.DateOffset(months=TRAIN_MONTHS)
    r = []; c = fc
    while c <= pd.Timestamp("2026-08-10"):
        r.append(c); c += pd.DateOffset(months=ROLL_MONTHS)
    return r

def load_best_params():
    if os.path.exists(BEST_JSON):
        with open(BEST_JSON) as f:
            d = json.load(f)
        p = dict(DEFAULT_PARAMS)
        for k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
                  "bagging_fraction", "lambda_l1", "lambda_l2", "num_boost_round"):
            if k in d:
                p[k] = d[k]
        return p
    return dict(DEFAULT_PARAMS)

def load_fold_selection():
    """每折因子清单 = walk-forward selection manifest 中该折自身的列表 (P0-B)。

    缺清单直接报错退出 —— 绝不回退到全样本 selected_factors*.csv(选择泄漏)。
    """
    if not os.path.exists(SELECTION_MANIFEST):
        raise FileNotFoundError(
            f"[train] {SELECTION_MANIFEST} 不存在 —— 先运行 "
            "`python scripts/factor_selection.py --folds-from-train`。全样本 "
            "selected_factors*.csv 不是合法替代(P0-B 选择泄漏)。")
    sys.path.insert(0, f"{ROOT}/scripts")
    from factor_selection import load_selection_manifest  # noqa: E402 (single source of truth)
    folds, meta = load_selection_manifest(path=SELECTION_MANIFEST)
    log(f"[train] selection manifest: purge={meta.get('purge_trading_days')} "
        f"embargo={meta.get('embargo_trading_days')} label_basis={meta.get('label_basis')} "
        f"cuts={len(folds)}")
    return {pd.Timestamp(c): list(v) for c, v in folds.items()}

def run_slice(sl, params):
    df, FEAT = load_data(); rs = rolls(); dates = df["date"]
    fold_features = load_fold_selection()
    available = set(FEAT)
    uniq_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    u_purge = PURGE_TRADING_DAYS + EMBARGO_TRADING_DAYS
    date = df["date"]
    lo = (sl * len(rs)) // N_WORKERS; hi = ((sl + 1) * len(rs)) // N_WORKERS
    out = []
    for i in range(lo, hi):
        cut = rs[i]; nxt = cut + pd.DateOffset(months=ROLL_MONTHS)
        # P0-B per-fold feature list: ONLY this fold's own walk-forward selection
        if cut not in fold_features:
            raise KeyError(
                f"[train] cut {cut.date()} 在 selection manifest 中无条目 —— 用相同切点表重新生成"
                "(不允许回退到全样本清单)")
        FEAT_FOLD = [f for f in fold_features[cut] if f in available]
        if not FEAT_FOLD:
            log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} SKIP: manifest 因子全部不在特征矩阵中")
            continue
        # P0-B purge: drop the last (purge+embargo) trading days before the cut from the
        # train block and the same tail from the OOS front (10d fwd label horizon).
        u_train = uniq_dates[uniq_dates < cut]
        u_train_cut = u_train[:-u_purge] if len(u_train) > u_purge else u_train
        u_oos = uniq_dates[(uniq_dates >= cut) & (uniq_dates < nxt)]
        u_oos_cut = u_oos[u_purge:] if len(u_oos) > u_purge else u_oos
        train = df[date.isin(u_train_cut)]
        oos = df[date.isin(u_oos_cut)]
        if len(oos) < 100: continue
        Xt = train[FEAT_FOLD].values.astype(np.float32); yt = train["fwd_neu"].values.astype(np.float32)
        k = ~np.isnan(yt); Xt, yt = Xt[k], yt[k]
        if len(Xt) < 5000: continue
        nv = max(2000, int(len(Xt) * 0.1)); Xv, yv = Xt[-nv:], yt[-nv:]; Xt, yt = Xt[:-nv], yt[:-nv]
        t0 = time.time()
        max_round = int(params.get("num_boost_round", MAX_BOOST))
        m = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=max_round,
                      valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        Xo = oos[FEAT_FOLD].values.astype(np.float32)
        pred = m.predict(Xo, num_iteration=m.best_iteration or 100)
        o = oos[["date", "asset"]].copy(); o["pred"] = pred; out.append(o)
        log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} feats={len(FEAT_FOLD)} purge={u_purge} train={len(Xt)} oos={len(o)} iters={m.best_iteration or 100} {time.time()-t0:.0f}s")
        del Xt, yt, Xv, yv, Xo, m, o; gc.collect()
    if out:
        res = pd.concat(out, ignore_index=True)
        res.to_parquet(SLICE_PAT % sl)
        log(f"[w{sl}] done {len(res)} -> preds_adj_slice_{sl}.parquet")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    _logf = open(LOG, "a")
    log(f"=== train_opt_adj.py started argv={sys.argv[1:]} ===")
    if len(sys.argv) > 1:
        sl = int(sys.argv[1])
        params = load_best_params()
        log(f"[w{sl}] using params: {params}")
        run_slice(sl, params)
    else:
        parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/data/build/preds_adj_slice_*.parquet"))]
        if not parts:
            log("no slices found")
        else:
            pred = pd.concat(parts, ignore_index=True)
            pred.to_parquet(f"{ROOT}/data/build/predictions_adj.parquet")
            log(f"merged {len(pred)} -> predictions_adj.parquet  {pred.date.min().date()}..{pred.date.max().date()}")
    if _logf is not None:
        _logf.close()
