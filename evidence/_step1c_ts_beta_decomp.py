# -*- coding: utf-8 -*-
"""Step 1c: attribute the ts_beta delegate residual (measured ~+75 ms vs a ~7 ms kernel)."""
from __future__ import annotations
import glob, json, os, statistics, time, warnings
import polars as pl

warnings.filterwarnings("ignore")
REPO = "/home/sunhaiwei/quant_projects"
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.common._polars_bridge import to_pandas_panel
load_all()

DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")
files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
files = [f for f in files if "2019-01-01" <= os.path.basename(f)[:10] <= "2022-12-31"]
syms = (pl.read_parquet(files[-1]).sort("AdjAmount", descending=True).head(30)["Symbol"].to_list())
long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", "AdjClose", "Volume"])
          .filter(pl.col("Symbol").is_in(syms)).collect())
P = {}
for c in ("AdjClose", "Volume"):
    w = (long.select(["TradeDate", "Symbol", c]).pivot(values=c, index="TradeDate", on="Symbol").sort("TradeDate"))
    pdf = w.to_pandas().set_index("TradeDate"); pdf.index.name = "date"
    P[c] = (pdf, pl.from_pandas(pdf.reset_index()))


def t(fn, target=0.5, maxi=100, mini=5):
    fn(); t0 = time.perf_counter(); fn(); one = time.perf_counter() - t0
    n = int(max(mini, min(maxi, target / one if one > 0 else maxi)))
    s = [(_s := time.perf_counter(), fn(), time.perf_counter() - _s)[2] for _ in range(n)]
    return statistics.median(s)


pd_c, pl_c = P["AdjClose"]; pd_v, pl_v = P["Volume"]
from factor_engine.cleaned_operators.price_volume.polars_price_volume import (
    _rolling_beta_polars, _aligned_beta_pandas)
from factor_engine.cleaned_operators.price_volume.beta_helpers import compute_rolling_beta
from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge

pop = OperatorRegistry.get("ts_beta", "pandas_numpy")
plop = OperatorRegistry.get("ts_beta", "polars", mode="any")
r = {}
r["pandas_reference_s"] = t(lambda: pop.calculate(pd_c, pd_v, window=20))
r["aligned_beta_pandas_s"] = t(lambda: _aligned_beta_pandas(pl_c, pl_v))
y_pd, x_pd = _aligned_beta_pandas(pl_c, pl_v)
r["compute_rolling_beta_only_s"] = t(lambda: compute_rolling_beta(y_pd, x_pd, 20, min_periods=5))
r["panel_pandas_bridge_only_s"] = t(lambda: panel_pandas_bridge(
    pl_c, lambda _: compute_rolling_beta(y_pd, x_pd, 20, min_periods=5)))
r["rolling_beta_polars_s"] = t(lambda: _rolling_beta_polars(pl_c, pl_v, window=20, min_periods=5))
r["delegate_end_to_end_s"] = t(lambda: plop._calculate_series(pl_c, pl_v, window=20))
r["to_pandas_panel_x2_s"] = t(lambda: (to_pandas_panel(pl_c), to_pandas_panel(pl_v)))
# how expensive is the date-index identity check inside _aligned_beta_pandas?
yi = pl_c["date"].to_pandas(); xi = pl_v["date"].to_pandas()
r["date_key_set_compare_s"] = t(lambda: set(yi) != set(xi))
r["date_to_pandas_only_s"] = t(lambda: pl_c["date"].to_pandas())

# --- attribute the residual to the SHARED bridge helper (base_polars.panel_pandas_bridge) ---
import numpy as np
from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
cols = [c for c in pl_c.columns if c not in PANEL_SKIP_COLUMNS]
pdf_c = pl_c.select(cols).to_pandas()
r["bridge_identity_only_s"] = t(lambda: panel_pandas_bridge(pl_c, lambda d: d))
r["bridge_select_to_pandas_s"] = t(lambda: pl_c.select(cols).to_pandas())


def _per_column_rebuild():
    return pl_c.with_columns([
        pl.Series(name=c, values=np.asarray(pdf_c[c], dtype=np.float64)) for c in cols])


def _bulk_rebuild():
    return pl_c.with_columns(pl.from_pandas(pdf_c))


r["per_column_series_rebuild_s"] = t(_per_column_rebuild)
r["bulk_from_pandas_rebuild_s"] = t(_bulk_rebuild)
r["n_numeric_cols"] = len(cols)

r["n_rows"] = len(pl_c)
r["n_cols"] = len(pl_c.columns) - 1
print(json.dumps(r, indent=1))
json.dump(r, open(f"{REPO}/evidence/_step1c_ts_beta_decomp.json", "w"), indent=1)
