# -*- coding: utf-8 -*-
"""#22 step4: 组装 818 特征 (783 + 34) 并跑 2000fac 同配置 walk-forward 对比评估。

流程 (全程增量, 不碰 run_20260830_2000fac 任何产物):
  1. 读 783 特征 features_full3_adj_prep.parquet + 34 新列 prep features_inc35_prep.parquet
     → 合并成 features_818_adj_prep.parquet (只有新增 34 列是新的)
  2. 用 build_walkforward_manifest_flip 的语义, 但直接向量化算 27 折 per-fold rank_ic,
     为每个 cut 生成含新因子的 fold 特征清单 (785 特征池)
  3. 复用 run_20260830_2000fac 已缓存的 best_params (data/build/best_params_flip/) —
     完全同参数, 不做 Optuna
  4. 同 train_opt_adj_flip.py 的 rolling 训练 (4 slices 并行可选), 只输出 818 版本预测
  5. OOS daily rank_ic 对比 (818 vs 783 基线 csv)
Env: 无 (直接改 FEATURES_PARQUET / SLICE_PAT)
"""
import os, sys, time, json, gc
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PROG = "/tmp/fac818_progress.log"

F783 = os.path.join(BUILD, "features_full3_adj_prep.parquet")   # 783 已 prep (来自 2000fac run)
FINC = os.path.join(BUILD, "features_inc35_prep.parquet")        # 34 新列 prep
F818 = os.path.join(BUILD, "features_818_adj_prep.parquet")
LABEL = os.path.join(BUILD, "fwd_adj_neu.parquet")
WF818 = os.path.join(BUILD, "walkforward_selection_818.json")
SLICE_PAT = os.path.join(BUILD, "preds_818_slice_%d.parquet")
PRED818 = os.path.join(BUILD, "predictions_818.parquet")

PURGE = 10
ROLL_MONTHS = 3
TRAIN_MONTHS = 48
RANK_IC_THRESHOLD = 0.015


def plog(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def rolls():
    START = pd.Timestamp("2016-01-04")
    fc = START + pd.DateOffset(months=TRAIN_MONTHS)
    r = []
    c = fc
    while c <= pd.Timestamp("2026-08-10"):
        r.append(c)
        c += pd.DateOffset(months=ROLL_MONTHS)
    return r


def main():
    t0 = time.time()
    plog("===== step4: assemble 818 features + per-fold manifest + OOS rank_ic =====")

    # 1) merge
    if not os.path.exists(F818):
        a = pd.read_parquet(F783)
        b = pd.read_parquet(FINC)
        keep = [c for c in b.columns if c not in ("date", "asset") and c != "alphasage_rank_amount"]
        plog(f"merge: 783={a.shape}  inc={len(keep)} new cols")
        a["date"] = pd.to_datetime(a["date"])
        b["date"] = pd.to_datetime(b["date"])
        merged = a.merge(b[["date", "asset"] + keep], on=["date", "asset"], how="left")
        plog(f"merged shape={merged.shape}")
        merged.to_parquet(F818, index=False)
        del a, b, merged
        gc.collect()
    else:
        plog(f"F818 already exists: {F818}")
    df = pd.read_parquet(F818)
    df["date"] = pd.to_datetime(df["date"])
    FEAT = [c for c in df.columns if c not in ("date", "asset", "fwd_neu")]
    plog(f"818 matrix: {df.shape} feats={len(FEAT)}")

    # 2) per-fold manifest (向量化, 与 build_walkforward_manifest_flip 同语义)
    if not os.path.exists(WF818):
        lab = pd.read_parquet(LABEL)
        lab["date"] = pd.to_datetime(lab["date"])
        mdf = df.merge(lab, on=["date", "asset"], how="left")
        lab_wide = mdf.pivot_table(index="date", columns="asset", values="fwd_neu").sort_index().astype(np.float64)
        all_dates = lab_wide.index
        cuts = rolls()
        payload = {"purge_trading_days": PURGE, "embargo_trading_days": 0,
                   "label_basis": "vwap_to_vwap_fwd10_neu", "rank_ic_threshold": RANK_IC_THRESHOLD,
                   "cuts": {}}
        fv_cache = {}
        for fcol in FEAT:
            w = mdf.pivot_table(index="date", columns="asset", values=fcol).sort_index()
            w = w.reindex(index=all_dates, columns=lab_wide.columns).astype(np.float32)
            fv_cache[fcol] = w
        plog(f"pre-pivoted {len(fv_cache)} factors ({time.time()-t0:.0f}s)")

        def rank_rows(X):
            n, m = X.shape
            order = np.argsort(X, axis=1, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            cols = np.tile(np.arange(m), (n, 1))
            ranks[np.arange(n)[:, None], order] = cols
            return ranks

        for ci, cut in enumerate(cuts):
            sel_dates = all_dates[all_dates < cut]
            if len(sel_dates) > PURGE:
                sel_dates = sel_dates[:-PURGE]
            sel_dates = lab_wide.index.intersection(sel_dates)
            if len(sel_dates) < 100:
                payload["cuts"][str(cut.date())] = {"factors": []}
                continue
            lab_block = lab_wide.loc[sel_dates].to_numpy()
            chosen = []
            for fcol in FEAT:
                fv_block = fv_cache[fcol].loc[sel_dates].to_numpy()
                if fv_block.shape != lab_block.shape:
                    continue
                ok = np.isfinite(fv_block) & np.isfinite(lab_block)
                rf = rank_rows(np.where(ok, fv_block, np.inf))
                rl = rank_rows(np.where(ok, lab_block, np.inf))
                rf[~ok] = np.nan; rl[~ok] = np.nan
                rf_d = rf - np.nanmean(rf, axis=1, keepdims=True)
                rl_d = rl - np.nanmean(rl, axis=1, keepdims=True)
                num = np.nansum(rf_d * rl_d, axis=1)
                denom = np.sqrt(np.nansum(rf_d ** 2, axis=1) * np.nansum(rl_d ** 2, axis=1))
                with np.errstate(invalid="ignore", divide="ignore"):
                    ics = num / denom
                ics = ics[np.isfinite(ics)]
                if len(ics) >= 200 and ics.mean() > RANK_IC_THRESHOLD:
                    chosen.append(fcol)
            payload["cuts"][str(cut.date())] = {"factors": chosen}
            plog(f"  cut {cut.date()} selected {len(chosen)} ({time.time()-t0:.0f}s)")
            del lab_block
        with open(WF818, "w") as f:
            json.dump(payload, f, indent=2)
        plog(f"WROTE {WF818}")
    else:
        plog(f"WF818 already exists")

    # 3) OOS daily rank_ic comparison: single fold pass
    #    reuse the 2000fac cached best_params (same config RE_OPT_EVERY=2)
    import lightgbm as lgb
    lab = pd.read_parquet(LABEL)
    lab["date"] = pd.to_datetime(lab["date"])
    df = df.merge(lab, on=["date", "asset"], how="left")
    FEAT = [c for c in df.columns if c not in ("date", "asset", "fwd_neu")]
    best_dir = os.path.join(BUILD, "best_params_flip")
    with open(WF818) as f:
        wf = json.load(f)
    fold_features = {pd.Timestamp(k): v.get("factors", []) for k, v in wf["cuts"].items()}
    rs = rolls()
    uniq_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    oos_list = []
    for i, cut in enumerate(rs):
        nxt = cut + pd.DateOffset(months=ROLL_MONTHS)
        ff = fold_features.get(cut, [])
        if not ff:
            continue
        ff = [f for f in ff if f in set(FEAT)]
        if not ff:
            continue
        u_train = uniq_dates[uniq_dates < cut]
        u_train = u_train[:-PURGE] if len(u_train) > PURGE else u_train
        u_oos = uniq_dates[(uniq_dates >= cut) & (uniq_dates < nxt)]
        u_oos = u_oos[PURGE:] if len(u_oos) > PURGE else u_oos
        train = df[df["date"].isin(u_train)]
        oos = df[df["date"].isin(u_oos)]
        if len(oos) < 100:
            continue
        Xt = train[ff].values.astype(np.float32)
        yt = train["fwd_neu"].values.astype(np.float32)
        kk = ~np.isnan(yt)
        Xt, yt = Xt[kk], yt[kk]
        if len(Xt) < 5000:
            continue
        nv = max(2000, int(len(Xt) * 0.1))
        Xv, yv = Xt[-nv:], yt[-nv:]
        Xt, yt = Xt[:-nv], yt[:-nv]
        # params from cached best (nearest re-opt cut <= this cut)
        params = dict(objective="regression", metric="l2", learning_rate=0.03,
                      num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                      device="cpu", num_threads=31)
        reopt_idx = i - (i % 2)
        bp = os.path.join(best_dir, f"best_{rs[reopt_idx].date()}.json")
        if not os.path.exists(bp):
            for cand in sorted(os.listdir(best_dir)):
                cd = pd.Timestamp(cand[5:-5])
                if cd <= cut:
                    bp = os.path.join(best_dir, cand)
        if os.path.exists(bp):
            with open(bp) as f:
                p = json.load(f)
            for k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
                      "bagging_fraction", "lambda_l1", "lambda_l2", "num_boost_round"):
                if k in p:
                    params[k] = p[k]
        max_round = int(params.get("num_boost_round", 400))
        m = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=max_round,
                      valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        Xo = oos[ff].values.astype(np.float32)
        pred = m.predict(Xo, num_iteration=m.best_iteration or 100)
        o = oos[["date", "asset"]].copy()
        o["pred"] = pred
        oos_list.append(o)
        plog(f"  fold {cut.date()} feats={len(ff)} iters={m.best_iteration or 100} oos={len(o)} ({time.time()-t0:.0f}s)")
        del Xt, yt, Xv, yv, Xo, m, o
        gc.collect()
    pred = pd.concat(oos_list, ignore_index=True)
    pred.to_parquet(PRED818, index=False)
    plog(f"WROTE {PRED818} rows={len(pred)}")

    # 4) OOS daily rank_ic vs baseline
    lab = pd.read_parquet(LABEL)
    lab["date"] = pd.to_datetime(lab["date"])
    m = pred.merge(lab, on=["date", "asset"], how="inner")
    rows = []
    for d, g in m.groupby("date"):
        v = g.dropna(subset=["pred", "fwd_neu"])
        if len(v) < 10:
            continue
        ric = pd.Series(v["pred"].to_numpy()).rank().corr(pd.Series(v["fwd_neu"].to_numpy()).rank())
        if np.isfinite(ric):
            rows.append((d, float(ric)))
    ic818 = pd.DataFrame(rows, columns=["date", "ric"]).set_index("date").sort_index()
    ic818.to_csv(os.path.join(BUILD, "oos_daily_rank_ic_818.csv"))
    base = pd.read_csv(os.path.join(ROOT, "outputs/run_20260830_2000fac/oos_daily_rank_ic_2000fac.csv"))
    base.columns = ["date", "ric_base"]
    base["date"] = pd.to_datetime(base["date"])
    j = ic818.join(base.set_index("date"), how="inner")
    ic818_all = pd.Series(ic818["ric"], index=ic818.index)
    base_all = pd.Series(j["ric_base"], index=j.index)
    # mean over common window
    res = {
        "n_oos_days": len(ic818),
        "mean_rank_ic_818": float(ic818["ric"].mean()),
        "ic_ir_818": float(ic818["ric"].mean() / ic818["ric"].std() * np.sqrt(252)),
        "ic_positive_frac_818": float((ic818["ric"] > 0).mean()),
        "n_common_days": len(j),
        "mean_rank_ic_base_common": float(j["ric_base"].mean()),
        "mean_rank_ic_818_common": float(j["ric"].mean()),
        "diff_common": float(j["ric"].mean() - j["ric_base"].mean()),
        "n_folds": len(oos_list),
    }
    with open(os.path.join(BUILD, "fac818_oos_compare.json"), "w") as f:
        json.dump(res, f, indent=2)
    plog(f"OOS compare: {res}")
    plog(f"DONE total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
