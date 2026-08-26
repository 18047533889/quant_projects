# -*- coding: utf-8 -*-
"""并行滚动前向 LightGBM + 超参优化 (optuna)。

用法:
  python train_opt.py opt [n_trials] [max_opt_rows]   # 只在首个切点前的训练数据上寻优, 存 best_params.json
  python train_opt.py <slice 0..3>                    # 用最优(或默认)超参跑滚动前向切片
  python train_opt.py                                 # 无参: 合并 preds_opt_slice_*.parquet -> predictions.parquet

设计:
- 滚动前向逻辑与 train_v2.py 完全一致: 首切 2016+48月=2020-01-04, 每 3 月重训(expanding), 预测下 3 月。
- 寻优: 用 2016-2020(首切前) 的训练数据做 3 折时序(expanding) CV, 目标最大化平均 rank IC
  (同时记录 L2)。optuna TPESampler, 默认 30 trials, 控制 ~10 分钟。
- 寻优结果缓存到 best_params.json; 滚动各折均用该套最优超参(保持简单, 不做 warm-start 微调)。
- 日志写到 /tmp/train_opt.log。
"""
import os, sys, time, gc, json, glob
import pandas as pd, numpy as np, lightgbm as lgb

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
N_WORKERS = 4; ROLL_MONTHS = 3; TRAIN_MONTHS = 48
FEATURES_PARQUET = os.environ.get("FEATURES_PARQUET", f"{ROOT}/data/build/features_full_filled.parquet")
BEST_JSON = os.environ.get("BEST_JSON", f"{ROOT}/scripts/best_params.json")
SLICE_PAT = os.environ.get("SLICE_PAT", f"{ROOT}/data/build/preds_opt_slice_%d.parquet")
LOG = "/tmp/train_opt.log"

# 固定超参 (寻优失败时回退, 即 train_v2 的基准)
DEFAULT_PARAMS = dict(objective="regression", metric="l2", learning_rate=0.03,
                      num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                      device="cpu", num_threads=8)

# 寻优预算
N_TRIALS = 30           # optuna trials
MAX_OPT_ROWS = 150_000  # 寻优用数据上限(时间序靠后段)
OPT_MAX_ROUND = 300     # 寻优每折最多轮数
OPT_EARLY = 20          # 寻优 early stopping

_logf = None

def log(m):
    print(m, flush=True)
    if _logf is not None:
        _logf.write(m + "\n"); _logf.flush()

def load_data():
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet"); fwd.index = pd.to_datetime(fwd.index).date
    fl = fwd.stack().rename("fwd").reset_index(); fl.columns = ["date", "asset", "fwd"]
    df["date"] = df["date"].dt.date
    df = df.merge(fl, on=["date", "asset"], how="left"); df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    FEAT = [c for c in df.columns if c not in ("date", "asset", "fwd")]
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

# ---------------- 寻优 ----------------

def _split_cv(df, FEAT, n_folds=3):
    """时序 expanding 切分: 每折用 [所有在验证块之前] 训练, 验证 = 当前块。
    保证 fold0 至少带走日期数//n_folds 中的第一块, 若不足则让 fold0 训练集
    包含整个数据 (退化但不会空)。"""
    date_sorted = np.sort(df["date"].unique())
    # 日期数不足时退化为 n_folds-1 折偏最小(避免空训练折)
    if len(date_sorted) < n_folds * 2:
        n_folds = max(1, len(date_sorted) // 2)
    edges = np.array_split(date_sorted, n_folds)
    folds = []
    tr_pool = df
    for k in range(n_folds):
        # expanding: 训练 = 所有在验证块之前的日期; 验证 = 当前块
        tr = df[df["date"] < edges[k][0]]
        if k == 0 and len(tr) == 0 and len(tr_pool) > 0:
            # 首个切点过早导致训练空, 退化为时间前 60% 做训练(至少非空)
            n_split = int(len(date_sorted) * 0.6)
            tr = df[df["date"] <= date_sorted[min(n_split, len(date_sorted) - 1)]]
            q = date_sorted[int(len(date_sorted) * 0.6):]
            va = df[df["date"].isin(q)]
        else:
            va = df[df["date"].isin(edges[k])]
        folds.append((tr, va))
    return folds

def _trial_score(params, df, FEAT, n_folds=3):
    """返回 (mean_rank_ic, mean_l2)。"""
    ics, l2s = [], []
    for tr, va in _split_cv(df, FEAT, n_folds):
        Xt = tr[FEAT].values.astype(np.float32); yt = tr["fwd"].values.astype(np.float32)
        k = ~np.isnan(yt); Xt, yt = Xt[k], yt[k]
        if len(Xt) < 3000:
            continue
        # early stopping 用训练集末 10%
        nv = max(1000, int(len(Xt) * 0.1)); Xv, yv = Xt[-nv:], yt[-nv:]; Xt, yt = Xt[:-nv], yt[:-nv]
        m = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=OPT_MAX_ROUND,
                      valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(OPT_EARLY, verbose=False)])
        Xo = va[FEAT].values.astype(np.float32)
        pred = m.predict(Xo, num_iteration=m.best_iteration or 100)
        y = va["fwd"].values.astype(np.float32)
        mask = ~np.isnan(y)
        if mask.sum() < 100:
            continue
        ic = np.corrcoef(pred[mask], y[mask])[0, 1]
        if np.isfinite(ic):
            ics.append(ic)
        l2s.append(np.mean((pred[mask] - y[mask]) ** 2))
        del Xt, yt, Xv, yv, m
    if not ics:
        return -1.0, 1e9
    return float(np.mean(ics)), float(np.mean(l2s))

def run_opt(n_trials=N_TRIALS, max_rows=MAX_OPT_ROWS):
    import optuna
    df, FEAT = load_data()
    # 取首切前的训练数据 (2016 - 2020-01), 时间序靠后段, 上限 max_rows
    first_cut = pd.Timestamp("2020-01-04")
    opt_df = df[df["date"] < first_cut]
    opt_df = opt_df.tail(max_rows).reset_index(drop=True)
    log(f"[opt] opt data rows={len(opt_df)} date={opt_df.date.min().date()}..{opt_df.date.max().date()} feats={len(FEAT)} trials={n_trials}")

    def objective(trial):
        params = dict(DEFAULT_PARAMS)
        params["learning_rate"] = trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.03, 0.05])
        params["num_leaves"] = trial.suggest_categorical("num_leaves", [31, 63, 127, 255])
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [10, 20, 30, 50])
        params["feature_fraction"] = trial.suggest_categorical("feature_fraction", [0.6, 0.7, 0.8, 0.9])
        params["bagging_fraction"] = trial.suggest_categorical("bagging_fraction", [0.7, 0.8, 0.9])
        params["lambda_l1"] = trial.suggest_categorical("lambda_l1", [0, 0.1, 1, 5])
        params["lambda_l2"] = trial.suggest_categorical("lambda_l2", [0, 0.1, 1, 5])
        params["num_boost_round"] = trial.suggest_categorical("num_boost_round", [200, 300, 400, 600])
        ic, l2 = _trial_score(params, opt_df, FEAT)
        trial.set_user_attr("l2", l2)
        return ic

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["num_boost_round"] = int(best["num_boost_round"])
    best["lambda_l1"] = float(best["lambda_l1"]); best["lambda_l2"] = float(best["lambda_l2"])
    log(f"[opt] done in {time.time()-t0:.0f}s  best_rank_ic={study.best_value:.5f}  l2={study.best_trial.user_attrs.get('l2')}")
    log(f"[opt] best_params={best}")
    with open(BEST_JSON, "w") as f:
        json.dump(best, f, indent=2)
    return best

# ---------------- 滚动前向 ----------------

def run_slice(sl, params):
    df, FEAT = load_data(); rs = rolls(); dates = df["date"]
    lo = (sl * len(rs)) // N_WORKERS; hi = ((sl + 1) * len(rs)) // N_WORKERS
    out = []
    for i in range(lo, hi):
        cut = rs[i]; nxt = cut + pd.DateOffset(months=ROLL_MONTHS)
        train = df[dates < cut]; oos = df[(dates >= cut) & (dates < nxt)]
        if len(oos) < 100: continue
        Xt = train[FEAT].values.astype(np.float32); yt = train["fwd"].values.astype(np.float32)
        k = ~np.isnan(yt); Xt, yt = Xt[k], yt[k]
        if len(Xt) < 5000: continue
        nv = max(2000, int(len(Xt) * 0.1)); Xv, yv = Xt[-nv:], yt[-nv:]; Xt, yt = Xt[:-nv], yt[:-nv]
        t0 = time.time()
        max_round = int(params.get("num_boost_round", 600))
        m = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=max_round,
                      valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        Xo = oos[FEAT].values.astype(np.float32)
        pred = m.predict(Xo, num_iteration=m.best_iteration or 100)
        o = oos[["date", "asset"]].copy(); o["pred"] = pred; out.append(o)
        log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} train={len(Xt)} oos={len(o)} iters={m.best_iteration or 100} {time.time()-t0:.0f}s")
        del Xt, yt, Xv, yv, Xo, m, o; gc.collect()
    if out:
        res = pd.concat(out, ignore_index=True)
        res.to_parquet(SLICE_PAT % sl)
        log(f"[w{sl}] done {len(res)} -> preds_opt_slice_{sl}.parquet")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    _logf = open(LOG, "a")
    log(f"=== train_opt.py started argv={sys.argv[1:]} ===")

    if len(sys.argv) > 1 and sys.argv[1] == "opt":
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRIALS
        max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_OPT_ROWS
        run_opt(n_trials, max_rows)
    elif len(sys.argv) > 1:
        sl = int(sys.argv[1])
        params = load_best_params()
        log(f"[w{sl}] using params: {params}")
        run_slice(sl, params)
    else:
        parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/data/build/preds_opt_slice_*.parquet"))]
        if not parts:
            log("no slices found")
        else:
            pred = pd.concat(parts, ignore_index=True)
            pred.to_parquet(f"{ROOT}/data/build/predictions.parquet")
            log(f"merged {len(pred)} -> predictions.parquet  {pred.date.min().date()}..{pred.date.max().date()}")
    if _logf is not None:
        _logf.close()
