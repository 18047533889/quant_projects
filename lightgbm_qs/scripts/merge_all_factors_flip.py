# -*- coding: utf-8 -*-
"""Step 1 of the lightgbm_qs final chain: rank-IC + sign-flip + full dedup.

User mandate (2026-08-28, verbatim intent):
  - 只先算 rankic，别的指标先别算。
  - rank_ic < 0 的因子自动取负（fv × -1），公式外层加 `-`，方向统一为正
    （因子值越大越好）。rank_ic>0 的因子保持不动。
  - 完全去重：corr > 0.98 的簇内只留 rank_ic 最高者。
  - 产出 data/build/rankic_all_factors_flipped.csv（含 flip 标志）+ 更新 selected。

Caliber (hard rules):
  - label = data/panel/fwd_ret10_adj.parquet (= COS TargetVwapReturnH10 =
    AdjVwap[t+11]/AdjVwap[t+1] - 1, verified exact).
  - universe = data/panel/vwap_trad_adj.parquet columns (297 tradable).
  - rank_ic = mean of per-date Spearman (rank-rank corr) — same口径 as
    rebuild_selection_adj.py; used here ONLY for the user's requested
    diagnostic + flip + dedup. No other metric is computed.

Dedup sampling: for lqtp files that are 5460-wide full-market, we only read the
297 tradable columns (column pruning) to keep memory bounded.

Usage:
  python merge_all_factors_flip.py [--limit N]     # N = cap factor count for smoke
Env:
  POOL_EXCLUDE_LQTP_STALE=1 (default) -> drop the 39 legacy 297-wide lqtp files.
"""
import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PANEL = os.path.join(ROOT, "data/panel")
os.makedirs(BUILD, exist_ok=True)
LOG = "/tmp/merge_all_factors_flip.log"

CORR_THRESH = 0.98
IC_THRESH = 0.015
LQTP_FORMULA_MAP = os.path.join(BUILD, "lqtp_formula_map.json")
LQTP_MIN_WIDTH = 5000          # full-market 5460-wide threshold
FRESH_CUTOFF = pd.Timestamp("2026-08-28").timestamp()


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ data
def load_fwd():
    fwd = pd.read_parquet(os.path.join(PANEL, "fwd_ret10_adj.parquet"))
    fwd["date"] = pd.to_datetime(fwd["date"]).dt.date
    return fwd


def load_trad():
    return list(pd.read_parquet(os.path.join(PANEL, "vwap_trad_adj.parquet")).columns)


def _is_wide_lqtp(path):
    """True if the lqtp parquet is the full-market 5460-wide fresh panel."""
    try:
        if os.path.getmtime(path) < FRESH_CUTOFF:
            return False
        if len(pq.ParquetFile(path).schema.names) < LQTP_MIN_WIDTH:
            return False
        return True
    except Exception:
        return False


def load_factor(full_path, fmt, trad_set, keep_cols=None, sample_rows=0):
    """Unify a factor to long DataFrame[date, asset, fv] on tradable universe.

    fmt: 'long' (datetime,asset,factor_value) | 'wide' (asset cols, maybe timestamp col)
    keep_cols: for lqtp 5460-wide files, the tradable column subset (pruned read).
    sample_rows: if >0 and fmt='wide', sample that many DATE ROWS from the wide frame
      BEFORE melting (keeps the melt small). The sampled rows are spread uniformly.
    """
    try:
        if fmt == "long":
            df = pq.read_table(full_path, columns=["datetime", "asset", "factor_value"]).to_pandas()
            df = df[df.asset.isin(trad_set)].copy()
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
            return df[["date", "asset", "factor_value"]].rename(columns={"factor_value": "fv"})
        # wide
        if keep_cols is not None:
            # lqtp full-market: timestamp column + tradable subset.
            # NOTE: factor_engine writes these parquet with 'timestamp' as the INDEX;
            # when selected, pyarrow lifts it into the DatetimeIndex (not a column).
            cols = [c for c in keep_cols if c in pq.ParquetFile(full_path).schema.names]
            df = pq.read_table(full_path, columns=["timestamp"] + cols).to_pandas()
            if "timestamp" in df.columns:
                datecol = "timestamp"
            else:
                # timestamp came back as the DatetimeIndex
                df = df.reset_index()
                datecol = "timestamp"
        else:
            df = pq.read_table(full_path).to_pandas()
            # wide files may use 'timestamp' (fresh lqtp), 'date' (legacy/optfac/factmat), or
            # a DatetimeIndex (cog/delivery).
            if "timestamp" in df.columns:
                datecol = "timestamp"
            elif "date" in df.columns:
                datecol = "date"
            else:
                df.index = pd.to_datetime(df.index)
                df = df.reset_index()
                datecol = df.columns[0]
            cols = [c for c in df.columns if c in trad_set]
            if not cols:
                return None
        if sample_rows and len(df) > sample_rows:
            # random row sample before melt (keeps melt cheap; ~120 rows x 297 cols)
            df = df.sample(n=sample_rows, random_state=42)
        df = df[[datecol] + cols].copy()
        df.columns = ["date"] + cols
        df["date"] = pd.to_datetime(df["date"]).dt.date
        w = df[["date"] + cols].melt(id_vars="date", var_name="asset", value_name="fv")
        w = w[w.asset.isin(trad_set)]
        w["fv"] = pd.to_numeric(w["fv"], errors="coerce")
        return w
    except Exception as e:
        plog(f"    load失败 {os.path.basename(full_path)}: {e}")
        return None


def enumerate_candidates(trad_set, limit=None):
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
        base = os.path.join(ROOT, d)
        if pool == "delivery":
            files = sorted(glob.glob(os.path.join(base, "*", "factor_values_test.parquet")))
        else:
            files = sorted(glob.glob(os.path.join(base, "*.parquet")))
        for fp in files:
            if pool == "delivery":
                name = os.path.basename(os.path.dirname(fp))
            else:
                name = os.path.basename(fp)
                if name.endswith("_neu.parquet"):
                    name = name[:-len("_neu.parquet")]
                elif name.endswith(".parquet"):
                    name = name[:-len(".parquet")]
            if pool == "lqtp":
                # exclude the 39 stale 297-wide legacy files that are NOT in the formula map
                if name not in _lqtp_formula_names():
                    continue
            candidates.append((pool, name, fp, fmt))
    if limit:
        candidates = candidates[:limit]
    return candidates


_lqtp_map_names = None
def _lqtp_formula_names():
    global _lqtp_map_names
    if _lqtp_map_names is None:
        with open(LQTP_FORMULA_MAP) as f:
            _lqtp_map_names = set(json.load(f).keys())
    return _lqtp_map_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] ===== merge_all_factors_flip 开始 limit={args.limit} =====")
    fwd_long = load_fwd()
    trad = load_trad()
    trad_set = set(trad)
    plog(f"tradable={len(trad)} fwd={fwd_long.date.min()}..{fwd_long.date.max()} rows={len(fwd_long)}")

    candidates = enumerate_candidates(trad_set, limit=args.limit)
    plog(f"候选因子总数: {len(candidates)}")

    # ---------------- pass 1: rank_ic ----------------
    ic_rows = []
    for pool, name, fp, fmt in candidates:
        key = f"{pool}:{name}"
        keep = None
        if pool == "lqtp" and fmt == "wide" and _is_wide_lqtp(fp):
            keep = trad_set
        df = load_factor(fp, fmt, trad_set, keep_cols=keep)
        if df is None or len(df) < 500:
            ic_rows.append((key, pool, np.nan, 0, True, False))
            continue
        m = df.merge(fwd_long, on=["date", "asset"], how="inner")
        if len(m) < 1500:
            ic_rows.append((key, pool, np.nan, 0, True, False))
            continue
        try:
            ic = m.groupby("date").apply(
                lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False)
            ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            plog(f"  rank_ic失败 {key}: {e}")
            ic_rows.append((key, pool, np.nan, 0, True, False))
            continue
        if len(ic) < 50:
            ic_rows.append((key, pool, np.nan, 0, True, False))
            continue
        raw_ic = float(ic.mean())
        flipped = bool(raw_ic < 0)
        rank_ic = float(-raw_ic) if flipped else raw_ic
        ic_rows.append((key, pool, rank_ic, len(ic), False, flipped))
        del m, ic
    icdf = pd.DataFrame(ic_rows, columns=["name", "pool", "rank_ic", "n_dates", "drop", "flipped"])
    icdf["rank_ic_orig"] = np.where(icdf["flipped"], -icdf["rank_ic"], icdf["rank_ic"])
    plog(f"== rank_ic 计算完成: {len(icdf)} 因子, flipped={int(icdf['flipped'].sum())} ==")

    # ---------------- selection: rank_ic>0.015 (after flip) ----------------
    sel = icdf[(icdf["rank_ic"] > IC_THRESH) & (~icdf["drop"])].sort_values("rank_ic", ascending=False)
    plog(f"== rank_ic>0.015 选中(flip后, 全样本诊断, 严禁直接喂训练): {len(sel)} ==")

    # ---------------- pass 2: full dedup (corr > 0.98, keep higher rank_ic) ----------------
    plog("开始完全重复去重 (corr>0.98 簇内留 rank_ic 最高) …")
    samples = {}
    for key in sel["name"].tolist():
        pool, name = key.split(":", 1)
        cand = next((c for c in candidates if f"{c[0]}:{c[1]}" == key), None)
        if cand is None:
            continue
        _, _, fp, fmt = cand
        keep = None
        if pool == "lqtp" and fmt == "wide" and _is_wide_lqtp(fp):
            keep = trad_set
        df = load_factor(fp, fmt, trad_set, keep_cols=keep)
        if df is None:
            continue
        smp = df.sample(n=min(40000, len(df)), random_state=42)
        samples[key] = smp.set_index(["date", "asset"])["fv"]
        del df

    skeys = list(samples)
    ic_lookup = dict(zip(sel["name"], sel["rank_ic"]))
    to_drop = set()
    for i in range(len(skeys)):
        a = skeys[i]
        for b in skeys[i+1:]:
            s = pd.concat([samples[a], samples[b]], axis=1, join="inner").dropna()
            if len(s) < 500:
                continue
            corr = s.iloc[:, 0].corr(s.iloc[:, 1])
            if corr >= CORR_THRESH:
                to_drop.add(a if ic_lookup[a] <= ic_lookup[b] else b)
        if (i + 1) % 60 == 0:
            plog(f"  去重进度 {i+1}/{len(skeys)} 已丢 {len(to_drop)} {time.time()-t0:.0f}s")
    plog(f"== 完全重复去重去除 {len(to_drop)}，剩 {len(sel) - len(to_drop)} ==")

    sel_names = [n for n in sel["name"].tolist() if n not in to_drop]
    sel_final = sel[sel["name"].isin(sel_names)].copy()
    sel_final.to_csv(os.path.join(BUILD, "selected_factors_flipped.csv"), index=False)
    icdf.to_csv(os.path.join(BUILD, "rankic_all_factors_flipped.csv"), index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] DONE {len(sel_final)} selected total={time.time()-t0:.0f}s")
    print("WROTE data/build/rankic_all_factors_flipped.csv + data/build/selected_factors_flipped.csv")


if __name__ == "__main__":
    main()
