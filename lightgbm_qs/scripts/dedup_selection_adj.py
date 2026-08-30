# -*- coding: utf-8 -*-
"""完全重复去重（独立步骤，内存安全分块）：读取 rankic_adj.csv 的选中因子，
逐对抽样 corr>0.98 聚类去重，输出 selected_adj.csv。与 merge_all_factors.py 同口径。
"""
import os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
OUT_DIR = os.path.join(ROOT, "data/build/selection_adj")
LOG = "/tmp/dedup_selection_adj.log"
CORR_THRESH = 0.98
IC_THRESH = 0.015

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    t0 = time.time()
    trad = set(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad_adj.parquet").columns)
    icdf = pd.read_csv(f"{OUT_DIR}/rankic_adj.csv")
    sel = icdf[(icdf["rank_ic"] > IC_THRESH) & (icdf["drop"] == False)].sort_values("rank_ic", ascending=False)
    plog(f"[{time.strftime('%H:%M:%S')}] 去重候选: {len(sel)}")

    # 因子路径索引
    def pool_for_name(name):
        pool = name.split(":", 1)[0]
        return pool, name.split(":", 1)[1]

    def load_factor_sample(full_path, fmt, n=40000, seed=42):
        if fmt == "long":
            df = pq.read_table(full_path, columns=["datetime", "asset", "factor_value"]).to_pandas()
            df = df[df.asset.isin(trad)]
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
            s = df[["date", "asset", "factor_value"]].rename(columns={"factor_value": "fv"})
        else:
            df = pq.read_table(full_path).to_pandas()
            df.index = pd.to_datetime(df.index).date
            cols = [c for c in df.columns if c in trad]
            if not cols:
                return None
            s = df[cols].stack().rename("fv").reset_index()
            s.columns = ["date", "asset", "fv"]
            s["fv"] = pd.to_numeric(s["fv"], errors="coerce")
        return s.sample(n=min(n, len(s)), random_state=seed).set_index(["date", "asset"])["fv"]

    pool_dirs = {
        "fm247": ("long", f"{ROOT}/data/factor_pools/fm247"),
        "fmqa": ("long", f"{ROOT}/data/factor_pools/fmqa"),
        "cogfull": ("wide", f"{ROOT}/data/factor_pools/cogfull"),
        "cogshort": ("wide", f"{ROOT}/data/factor_pools/cogshort"),
        "cogneutral": ("wide", f"{ROOT}/data/factor_pools/cogneutral"),
        "delivery": ("wide", f"{ROOT}/data/factor_pools/delivery"),
        "optfac": ("wide", f"{ROOT}/data/factor_pools/optimized_factors"),
        "factmat": ("wide", f"{ROOT}/data/factor_pools/factor_matrices"),
        "lqtp": ("wide", f"{ROOT}/data/factor_pools/lqtp"),
    }
    def resolve_path(pool, name):
        if pool == "delivery":
            p = os.path.join(pool_dirs[pool][1], name, "factor_values_test.parquet")
            return p if os.path.exists(p) else None
        base = pool_dirs[pool][1]
        for cand in (os.path.join(base, f"{name}.parquet"), os.path.join(base, f"{name}_neu.parquet")):
            if os.path.exists(cand):
                return cand
        return None

    samples = {}
    for key in sel["name"].tolist():
        pool, name = pool_for_name(key)
        fmt, _ = pool_dirs[pool]
        p = resolve_path(pool, name)
        if p is None:
            plog(f"  路径缺失 {key}"); continue
        try:
            smp = load_factor_sample(p, fmt)
        except Exception as e:
            plog(f"  读取失败 {key}: {type(e).__name__} {str(e)[:60]}"); continue
        if smp is None:
            continue
        samples[key] = smp

    plog(f"  抽样就绪: {len(samples)}")
    skeys = list(samples)
    ic_lookup = dict(zip(sel["name"], sel["rank_ic"]))
    to_drop = set()
    # 分块两两比较（每块 40 因子，内存受控；块内两两 + 跨块由块重叠覆盖）
    for i in range(len(skeys)):
        a = skeys[i]
        for b in skeys[i+1:]:
            s = pd.concat([samples[a], samples[b]], axis=1, join="inner").dropna()
            if len(s) < 500:
                continue
            corr = s.iloc[:, 0].corr(s.iloc[:, 1])
            if corr >= CORR_THRESH:
                to_drop.add(a if ic_lookup[a] <= ic_lookup[b] else b)
        if (i+1) % 40 == 0:
            plog(f"  {i+1}/{len(skeys)} 已丢 {len(to_drop)} {time.time()-t0:.0f}s")
    plog(f"完全重复去重去除 {len(to_drop)}，剩 {len(sel) - len(to_drop)}")
    sel_names = [n for n in sel["name"].tolist() if n not in to_drop]
    sel_final = sel[sel["name"].isin(sel_names)].copy()
    sel_final.to_csv(f"{OUT_DIR}/selected_adj.csv", index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] WROTE selected_adj.csv {len(sel_final)} total={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
