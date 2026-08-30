# -*- coding: utf-8 -*-
"""因子筛选重跑（后复权标签版）。

- 因子：data/factor_pools/*（fm247/fmqa/cogfull/cogshort/cogneutral/delivery/optfac/factmat/lqtp）
- 标签：data/panel/fwd_ret10_adj.parquet（后复权 Vwap 10 日前向收益，vwap-to-vwap）
- 资产：data/panel/vwap_trad_adj.parquet 的列（tradable 297）
- 筛选：rank_ic > 0.015，逐日截面秩相关，与 merge_all_factors.py 同口径
- 去重：按因子值相关性 >0.98 簇内保留 rank_ic 最高（完全重复去除）
- 输出：data/build/selection_adj/rankic_adj.csv + selected_adj.csv
"""
import os, glob, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
OUT_DIR = os.path.join(ROOT, "data/build/selection_adj")
os.makedirs(OUT_DIR, exist_ok=True)
LOG = "/tmp/rebuild_selection_adj.log"
CORR_THRESH = 0.98
IC_THRESH = 0.015

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    t0 = time.time()
    trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad_adj.parquet").columns)
    fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10_adj.parquet")
    fwd_long = fwd.copy()
    fwd_long["date"] = pd.to_datetime(fwd_long["date"]).dt.date
    trad_set = set(trad)
    plog(f"[{time.strftime('%H:%M:%S')}] tradable={len(trad)} fwd_range={fwd_long.date.min()}..{fwd_long.date.max()} fwd_rows={len(fwd_long)}")

    def load_factor(full_path, fmt):
        if fmt == "long":
            df = pq.read_table(full_path, columns=["datetime", "asset", "factor_value"]).to_pandas()
            df = df[df.asset.isin(trad_set)]
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
            return df[["date", "asset", "factor_value"]].rename(columns={"factor_value": "fv"})
        df = pq.read_table(full_path).to_pandas()
        df.index = pd.to_datetime(df.index).date
        cols = [c for c in df.columns if c in trad_set]
        if not cols:
            return None
        w = df[cols].stack().rename("fv").reset_index()
        w.columns = ["date", "asset", "fv"]
        w["fv"] = pd.to_numeric(w["fv"], errors="coerce")
        return w

    pools = [
        ("fm247", "long", "data/factor_pools/fm247"),
        ("fmqa", "long", "data/factor_pools/fmqa"),
        ("cogfull", "wide", "data/factor_pools/cogfull"),
        ("cogshort", "wide", "data/factor_pools/cogshort"),
        ("cogneutral", "wide", "data/factor_pools/cogneutral"),
        ("delivery", "wide", "data/factor_pools/delivery"),
        ("optfac", "wide", "data/factor_pools/optimized_factors"),
        ("factmat", "wide", "data/factor_pools/factor_matrices"),
        ("lqtp", "wide", "data/factor_pools/lqtp"),
        ("new026", "wide", "data/factor_pools/new_20260830"),
    ]
    candidates = []
    for pool, fmt, d in pools:
        base = f"{ROOT}/{d}"
        if pool == "delivery":
            files = sorted(glob.glob(f"{base}/*/factor_values_test.parquet"))
        else:
            files = sorted(glob.glob(f"{base}/*.parquet"))
        for fp in files:
            if pool == "delivery":
                name = os.path.basename(os.path.dirname(fp))
            else:
                name = os.path.basename(fp)
                name = name[:-len("_neu.parquet")] if name.endswith("_neu.parquet") else name[:-len(".parquet")]
            candidates.append((pool, name, fp, fmt))
    plog(f"候选因子: {len(candidates)}")

    ic_rows = []
    feats = {}
    for pool, name, fp, fmt in candidates:
        key = f"{pool}:{name}"
        df = load_factor(fp, fmt)
        if df is None or len(df) < 500:
            ic_rows.append((key, pool, np.nan, 0, True)); continue
        m = df.merge(fwd_long, on=["date", "asset"], how="inner")
        if len(m) < 1500:
            ic_rows.append((key, pool, np.nan, 0, True)); continue
        try:
            ic = m.groupby("date").apply(
                lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False)
            ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            plog(f"  rank_ic 失败 {key}: {e}")
            ic_rows.append((key, pool, np.nan, 0, True)); continue
        if len(ic) < 50:
            ic_rows.append((key, pool, np.nan, 0, True)); continue
        rank_ic = float(ic.mean())
        ic_rows.append((key, pool, rank_ic, len(ic), False))
        feats[key] = df
        del m, ic
    icdf = pd.DataFrame(ic_rows, columns=["name", "pool", "rank_ic", "n_dates", "drop"])
    icdf.to_csv(f"{OUT_DIR}/rankic_adj.csv", index=False)
    # P0-B (2026-08-28): this is the FULL-SAMPLE diagnostic list (rank_IC over ALL dates
    # = future OOS labels). It is written for coverage/dedup bookkeeping ONLY and must
    # never feed training. The research-correct per-fold lists live in
    # data/build/walkforward_selection.json (factor_selection.py --folds-from-train,
    # purged train-window stats). Here we additionally emit the walk-forward lists
    # restricted to this adj label so the adj trainer can consume them per fold.
    sel = icdf[(icdf["rank_ic"] > IC_THRESH) & (~icdf["drop"])].sort_values("rank_ic", ascending=False)
    plog(f"rank_ic>0.015 选中(全样本诊断, 严禁直接喂训练): {len(sel)}  /  {len(icdf)}")

    WF_SELECTION_JSON = os.environ.get(
        "SELECTION_MANIFEST", f"{ROOT}/data/build/walkforward_selection.json")
    if os.path.exists(WF_SELECTION_JSON):
        import sys as _sys
        _sys.path.insert(0, f"{ROOT}/scripts")
        from factor_selection import load_selection_manifest  # noqa: E402
        _folds, _meta = load_selection_manifest(path=WF_SELECTION_JSON)
        plog(f"walk-forward 每折清单可用: cuts={len(_folds)} purge={_meta.get('purge_trading_days')} "
             f"—— 训练侧只允许用该清单, 不用本脚本的全样本 selected_adj.csv")
    else:
        plog("!! 缺 walkforward_selection.json —— 训练侧会拒绝启动(无全样本回退)。"
             "请跑 factor_selection.py --folds-from-train")

    # 完全重复去重
    samples = {}
    for key in sel["name"].tolist():
        if key not in feats:
            continue
        smp = feats[key].sample(n=min(40000, len(feats[key])), random_state=42)
        samples[key] = smp.set_index(["date", "asset"])["fv"]
    skeys = list(samples)
    to_drop = set()
    for i in range(len(skeys)):
        for j in range(i + 1, len(skeys)):
            a, b = skeys[i], skeys[j]
            s = pd.concat([samples[a], samples[b]], axis=1, join="inner").dropna()
            if len(s) < 500:
                continue
            corr = s.iloc[:, 0].corr(s.iloc[:, 1])
            if corr >= CORR_THRESH:
                ic_a = sel.loc[sel["name"] == a, "rank_ic"].iloc[0]
                ic_b = sel.loc[sel["name"] == b, "rank_ic"].iloc[0]
                to_drop.add(a if ic_a <= ic_b else b)
    plog(f"完全重复去重后去除 {len(to_drop)}，剩 {len(sel) - len(to_drop)}")
    sel_names = [n for n in sel["name"].tolist() if n not in to_drop]
    sel = sel[sel["name"].isin(sel_names)].copy()
    sel.to_csv(f"{OUT_DIR}/selected_adj.csv", index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] DONE {len(sel_names)} selected total={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
