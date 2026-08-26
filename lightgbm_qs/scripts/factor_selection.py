# -*- coding: utf-8 -*-
"""Factor selection: merge fm247 + fmqa full-coverage factors, dedup by factor name,
compute rank_ic vs 10-day forward return on the investable/tradable universe, keep rank_ic>0.015.
Outputs final feature matrix (date x asset) x [selected factors] and saves the selected factor list.
"""
import glob, os, duckdb, pyarrow.parquet as pq, pandas as pd, numpy as np

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
con = duckdb.connect()

trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad.parquet").columns)
fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")  # date x asset, 10d fwd ret

def load_full(pool):
    """Return list of (factor_name, long df) for factors with full coverage in a pool."""
    out = []
    for f in glob.glob(f"{ROOT}/data/factor_pools/{pool}/*.parquet"):
        try:
            df = pq.read_table(f, columns=["datetime", "asset", "factor_value"]).to_pandas()
        except Exception:
            continue
        dmin, dmax = str(df.datetime.min())[:10], str(df.datetime.max())[:10]
        if not (dmin <= "2016-03-01" and dmax >= "2024-01-01"):
            continue
        name = os.path.basename(f).replace("_neu.parquet", "").replace(".parquet", "")
        out.append((name, df[df.asset.isin(trad)]))
    return out

fm247 = load_full("fm247")
fmqa = load_full("fmqa")
print(f"full-coverage: fm247={len(fm247)} fmqa={len(fmqa)}")

# Dedup by factor name across pools (union; names differ in scheme so low collision)
seen = set(); factors = []
for name, df in fm247 + fmqa:
    if name in seen:
        continue
    seen.add(name)
    factors.append((name, df))
print("unique factors after cross-pool dedup:", len(factors))

# Compute rank_IC per factor against fwd_ret10 (cross-sectional rank corr on each date, then mean)
fwd_dt = fwd.stack().rename("fwd").reset_index().rename(columns={"level_0": "date", "level_1": "asset"})
fwd_dt["date"] = pd.to_datetime(fwd_dt["date"]).dt.date

ic_results = []
for name, df in factors:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df = df.rename(columns={"factor_value": "fv"})[["date", "asset", "fv"]]
    m = df.merge(fwd_dt, on=["date", "asset"], how="inner")
    if len(m) < 1000:
        continue
    ic = m.groupby("date").apply(
        lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False
    )
    ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    rank_ic = ic.mean() if len(ic) else np.nan
    ic_results.append({"factor": name, "rank_ic": rank_ic, "n_dates": len(ic)})

res = pd.DataFrame(ic_results)
# Require dense coverage (real full-panel factor) to avoid single-period flukes
res = res[res["n_dates"] >= 1500].copy()
print("factors with full panel coverage (n_dates>=1500):", len(res))
sel = res[res["rank_ic"] > 0.015].sort_values("rank_ic", ascending=False)
print("selected (rank_ic>0.015):", len(sel))
sel.to_csv(f"{ROOT}/data/build/selected_factors.csv", index=False)
print("top selected:")
print(sel.head(15).to_string())
