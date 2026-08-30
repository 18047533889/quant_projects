# -*- coding: utf-8 -*-
"""Build comprehensive factor matrix across ALL pools (fm247, fmqa, cogfull, cogshort, cogneutral).
Unifies long-format (datetime, asset, factor_value) and wide-format (date x asset) factors,
restricts to tradable universe, computes rank_ic per factor vs 10d forward return,
keeps rank_ic > 0.015, and builds the final (date x asset) x [selected factors] feature matrix
WITH NaN preserved (LightGBM handles NaN natively; we do not drop NaN-feature rows).
Cog factors start 2018 -> NaN before then, which is fine for the model.
"""
import os, glob, sys, numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
os.makedirs(f"{ROOT}/data/build", exist_ok=True)

trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad.parquet").columns)
fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")  # date x asset
fwd.index = pd.to_datetime(fwd.index)
fwd_long = fwd.stack().rename("fwd").reset_index()
fwd_long.columns = ["date", "asset", "fwd"]
fwd_long["date"] = fwd_long["date"].dt.date

def load_factor_long(f, fmt):
    if fmt == "long":
        df = pq.read_table(f, columns=["datetime", "asset", "factor_value"]).to_pandas()
        df = df[df.asset.isin(trad)].copy()
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        return df[["date", "asset", "factor_value"]].rename(columns={"factor_value": "fv"})
    else:  # wide
        df = pq.read_table(f).to_pandas()
        df.index = pd.to_datetime(df.index).date
        cols = [c for c in trad if c in df.columns]
        if not cols:
            return None
        w = df[cols].stack().rename("fv").reset_index()
        w.columns = ["date", "asset", "fv"]
        return w

POOLS = [
    ("fm247", "long"), ("fmqa", "long"),
    ("cogfull", "wide"), ("cogshort", "wide"), ("cogneutral", "wide"),
]

results = []
feature_frames = {}

for pool, fmt in POOLS:
    d = f"{ROOT}/data/factor_pools/{pool}"
    for f in sorted(os.listdir(d)):
        if not f.endswith(".parquet"):
            continue
        path = os.path.join(d, f)
        name = f.replace("_neu.parquet", "").replace(".parquet", "")
        try:
            df = load_factor_long(path, fmt)
        except Exception as e:
            print(f"  skip {pool}/{name}: {e}"); continue
        if df is None or len(df) < 1000:
            continue
        m = df.merge(fwd_long, on=["date", "asset"], how="inner")
        if len(m) < 2000:
            continue
        ic = m.groupby("date").apply(
            lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False
        )
        ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(ic) < 50:
            continue
        rank_ic = float(ic.mean()); n_dates = len(ic)
        results.append({"name": f"{pool}:{name}", "pool": pool, "rank_ic": rank_ic, "n_dates": n_dates})
        feature_frames[f"{pool}:{name}"] = df
        print(f"  {pool}:{name} ic={rank_ic:.4f} ndays={n_dates}")

res = pd.DataFrame(results)
print("\n== total factors evaluated ==", len(res))
# P0-B (2026-08-28): this full-sample list is a DIAGNOSTIC ONLY — its rank_IC was
# computed over ALL dates, i.e. with future OOS labels (selection leakage). It must
# never be the feature set for walk-forward training. The feature matrix below is
# built from the UNION of the per-fold walk-forward lists in
# data/build/walkforward_selection.json (factor_selection.py --folds-from-train);
# the trainer then picks each fold's own subset.
sel = res[res["rank_ic"] > 0.015].sort_values("rank_ic", ascending=False)
print("selected (rank_ic>0.015, FULL-SAMPLE DIAGNOSTIC — never feed to training):", len(sel))
sel.to_csv(f"{ROOT}/data/build/selected_factors_all.csv", index=False)

WF_SELECTION_JSON = f"{ROOT}/data/build/walkforward_selection.json"
if not os.path.exists(WF_SELECTION_JSON):
    print("!! data/build/walkforward_selection.json not found — run "
          "`python factor_selection.py --folds-from-train` first. Refusing to build the "
          "feature matrix from the full-sample list (P0-B selection leakage).")
    raise SystemExit(2)
import sys as _sys
_sys.path.insert(0, f"{ROOT}/scripts")
from factor_selection import load_selection_manifest  # noqa: E402
folds, meta = load_selection_manifest(path=WF_SELECTION_JSON)
selected_names = sorted({f for lst in folds.values() for f in lst})
absent = [n for n in selected_names if n not in feature_frames]
print(f"per-fold walk-forward lists: {len(folds)} cuts, union={len(selected_names)} factors "
      f"(purge={meta.get('purge_trading_days')}, label_basis={meta.get('label_basis')})")
if absent:
    print(f"  {len(absent)} union factors are not in this script's pools (skipped here): "
          f"{absent[:5]}{' ...' if len(absent) > 5 else ''}")
print("building feature matrix for the walk-forward UNION:", len(selected_names), "factors")
# base index = all tradable assets x all dates (fwd long defines the panel)
base = fwd_long[["date", "asset"]].set_index(["date", "asset"]).sort_index()
w = base.copy()
w = w.rename(columns={"fwd": "_fwd"})
for name in selected_names:
    if name not in feature_frames:
        print(f"  skip {name}: not in evaluated pools (walk-forward union member)")
        continue
    df = feature_frames[name].set_index(["date", "asset"])["fv"]
    w[name] = df
    print(f"  added {name} -> {w.shape}")

w = w.reset_index()
w.to_parquet(f"{ROOT}/data/build/features_all.parquet")
print("\nsaved features_all.parquet:", w.shape)
