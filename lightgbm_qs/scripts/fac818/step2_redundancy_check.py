# -*- coding: utf-8 -*-
"""#22 step2: 增量 35 列 vs 现有 783 列 的相关性冗余检查 + 独立 rank_ic 报告。

方法 (对齐 merge_flip_combine 的去重口径):
  - 对每个新列取 40k 行抽样 (date,asset 对齐), 与 783 现有列逐对计算 Pearson |rho|
    (用位置交集, 等价于 inner-join dropna corr)
  - max |rho| > 0.9 标记冗余(与现有列冲突); 否则为增量信号
  - 同时输出每个新列 vs fwd_adj_neu 的 per-date rank_ic (pool 标签口径)
进度: /tmp/fac818_progress.log
"""
import os, sys, time, json
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PROG = "/tmp/fac818_progress.log"
OUT_CSV = os.path.join(BUILD, "fac818_inc35_redundancy.csv")
OUT_JSON = os.path.join(BUILD, "fac818_inc35_summary.json")


def plog(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    plog("===== step2: redundancy check 35 new vs 783 existing =====")
    base = pd.read_parquet(os.path.join(BUILD, "features_full3_adj.parquet"))
    inc = pd.read_parquet(os.path.join(BUILD, "features_inc35.parquet"))
    lab = pd.read_parquet(os.path.join(BUILD, "fwd_adj_neu.parquet"))
    lab["date"] = pd.to_datetime(lab["date"]).dt.date

    existing_cols = [c for c in base.columns if c not in ("date", "asset")]
    new_cols = [c for c in inc.columns if c not in ("date", "asset")]
    plog(f"existing={len(existing_cols)} new={len(new_cols)}")

    # long pivot for correlation sampling: build one long frame with all cols on (date,asset)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    base = base.set_index(["date", "asset"])
    inc2 = inc.copy()
    inc2["date"] = pd.to_datetime(inc2["date"]).dt.date
    inc2 = inc2.set_index(["date", "asset"])

    # sample: for each new col take 40k rows that intersect existing data
    rng = np.random.RandomState(42)
    rows = []
    new_ic = {}
    lab2 = lab.copy()
    lab2["date"] = pd.to_datetime(lab2["date"]).dt.date
    inc3 = inc.copy()
    inc3["date"] = pd.to_datetime(inc3["date"]).dt.date
    m = inc3.merge(lab2, on=["date", "asset"], how="inner")
    for c in new_cols:
        # rank_ic vs fwd_adj_neu (pool label) on full grid
        ic = m.groupby("date").apply(lambda g: g[c].rank().corr(g["fwd_neu"].rank()), include_groups=False)
        ic = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        new_ic[c] = (float(ic.mean()), int(len(ic)))

        # correlation vs existing
        idx = inc2.index
        sample_pos = rng.choice(np.arange(len(idx)), size=min(40000, len(idx)), replace=False)
        s_new = inc2.iloc[sample_pos][c]
        best = (0.0, None)
        for ec in existing_cols:
            s_ex = base[ec].iloc[sample_pos]
            ok = s_new.notna() & s_ex.notna()
            if ok.sum() < 500:
                continue
            corr = float(np.corrcoef(s_new[ok].to_numpy(), s_ex[ok].to_numpy())[0, 1])
            if abs(corr) > abs(best[0]):
                best = (corr, ec)
        redundant = abs(best[0]) > 0.9
        rows.append({"new": c, "max_abs_rho": round(abs(best[0]), 4),
                     "best_corr": round(best[0], 4), "best_existing": best[1],
                     "redundant": redundant, "rank_ic_fwd10": round(new_ic[c][0], 4),
                     "n_ic_days": new_ic[c][1]})
        plog(f"  {c}: max|rho|={abs(best[0]):.4f} vs {best[1]} redun={redundant} ic={new_ic[c][0]:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    n_redun = int(df["redundant"].sum())
    plog(f"redundant(>0.9): {n_redun} / {len(df)}")
    summary = {"n_new": len(df), "n_redundant": n_redun,
               "redundant_names": df.loc[df["redundant"], "new"].tolist()}
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    plog(f"DONE -> {OUT_CSV} total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
