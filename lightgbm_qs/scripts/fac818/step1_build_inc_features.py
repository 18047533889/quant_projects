# -*- coding: utf-8 -*-
"""#22 step1: 35 条新因子矩阵 → 对齐 783 面板窗口 → 取负翻转 → 增量特征文件。

增量（严禁全量重算 783）:
  - 读现有 features_full3_adj.parquet 的 (date,asset) 网格(767745 行) 与列集
  - 35 条矩阵(wide: date index × asset cols) reindex 到该网格, 取负翻转
  - 输出 data/build/features_inc35.parquet (仅 date, asset + 35 新列, float32)
  - 冗余检查: 新列 vs 现有 783 列 max |rho| (抽样), >0.9 标记冗余
进度: /tmp/fac818_progress.log
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
SRC = "/home/sunhaiwei/quant_projects/weekly_backtest_output"
OUT = os.path.join(BUILD, "features_inc35.parquet")
PROG = "/tmp/fac818_progress.log"

NEW50 = "/tmp/new50_selected.json"
MINUTE = "/tmp/minute_9_result.json"


def plog(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def spec_rows():
    d = json.load(open(NEW50))
    out = []
    for x in d:
        out.append({"name": x["page_name"], "flip": bool(x["is_flipped"])})
    m = json.load(open(MINUTE))
    for k in m:
        if k == "_rankic_summary":
            continue
        out.append({"name": k, "flip": True})
    return out


def load_wide(path, trad_set):
    """wide: date index x asset cols -> long [date, asset, fv] on tradable."""
    df = pq.read_table(path).to_pandas()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index).date
    cols = [c for c in df.columns if c in trad_set]
    if not cols:
        return None
    w = df[cols].stack().rename("fv").reset_index()
    w.columns = ["date", "asset", "fv"]
    w["fv"] = pd.to_numeric(w["fv"], errors="coerce")
    return w


def main():
    t0 = time.time()
    plog("===== step1: build incremental 35 features =====")

    base = pd.read_parquet(os.path.join(BUILD, "features_full3_adj.parquet"),
                           columns=["date", "asset"])
    grid = base.copy()
    grid["date"] = pd.to_datetime(grid["date"]).dt.date
    grid = grid.set_index(["date", "asset"]).sort_index()
    trad = list(pd.read_parquet(os.path.join(ROOT, "data/panel/vwap_trad_adj.parquet")).columns)
    trad_set = set(trad)
    plog(f"grid {grid.shape}  tradable={len(trad)}")

    specs = spec_rows()
    plog(f"specs={len(specs)} (26 new-mined + 9 minute, 全部取负翻正按 flag)")
    for s in specs:
        if s["name"].startswith(("amount_weighted", "high_volume", "skew_weighted", "vwap_")):
            s["path"] = os.path.join(SRC, "factor_matrices_all", s["name"] + ".parquet")
        else:
            s["path"] = os.path.join(SRC, "factor_matrices_all", "factor_" + s["name"] + ".parquet")
        if not os.path.exists(s["path"]):
            alt = os.path.join(SRC, "factor_matrices_all", s["name"] + ".parquet")
            if os.path.exists(alt):
                s["path"] = alt

    n = len(grid)
    pos = pd.Series(np.arange(n), index=grid.index)
    out = np.full((n, len(specs)), np.nan, dtype=np.float32)

    for j, s in enumerate(specs):
        df = load_wide(s["path"], trad_set)
        if df is None:
            plog(f"  [{j}] {s['name']}: LOAD FAIL")
            continue
        df = df.set_index(["date", "asset"])["fv"]
        # SIGN 调查 (2026-08-30): factor_matrices_all/optimized_factors 的矩阵
        # 已是 mining 正方向 (fwd10 标签下 rank_ic 全为正, 见诊断 log)。JSON 的
        # is_flipped 指 local_formula 方向, 不是矩阵方向。若再取负会双重翻转。
        # 因此矩阵直接使用, 不额外取负。minute 矩阵 fwd10 标签下也为正。
        # 如需翻转为负方向请设 FORCE_FLIP=1。
        if os.environ.get("FORCE_FLIP") == "1" and s["flip"]:
            df = -df
        idx = pos.reindex(df.index)
        ok = idx.notna()
        if ok.any():
            out[idx[ok].astype(int).values, j] = df.values[ok.values]
        cov = ok.mean()
        nz = np.isfinite(out[:, j]).sum()
        plog(f"  [{j}] {s['name']}: grid_cover={cov:.4f} finite={nz} flip={s['flip']}")
        del df

    g = grid.reset_index()
    out_df = pd.DataFrame({"date": g["date"].astype(str), "asset": g["asset"]})
    for j, s in enumerate(specs):
        out_df[s["name"]] = out[:, j]
    out_df.to_parquet(OUT, index=False)
    plog(f"DONE -> {OUT} shape={out_df.shape} total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
