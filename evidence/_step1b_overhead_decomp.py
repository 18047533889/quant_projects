# -*- coding: utf-8 -*-
"""Step 1b: decompose the delegate overhead.

The per-op A/B flags ts_beta as a 10.3x outlier.  Break the delegate's cost into
  (i)  panel marshaling  pl -> pd
  (ii) pandas reference compute
  (iii) rebuild       pd -> pl
  (iv) framework glue  (verify_frames_share_identity / numeric_cols / copies)
and report single-thread evidence (CPU-time / wall-time) for the delegate path
vs a native polars expression on the same real panel.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

REPO = "/home/sunhaiwei/quant_projects"
DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.cleaned_operators.common._polars_bridge import (  # noqa: E402
    to_pandas_panel, from_pandas_panel, numeric_cols, verify_frames_share_identity,
)
load_all()


def load_panels():
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    files = [f for f in files if "2019-01-01" <= os.path.basename(f)[:10] <= "2022-12-31"]
    syms = (pl.read_parquet(files[-1]).sort("AdjAmount", descending=True)
              .head(30)["Symbol"].to_list())
    long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", "AdjClose", "Volume"])
              .filter(pl.col("Symbol").is_in(syms)).collect())
    out = {}
    for c in ("AdjClose", "Volume"):
        w = (long.select(["TradeDate", "Symbol", c])
                 .pivot(values=c, index="TradeDate", on="Symbol").sort("TradeDate"))
        pdf = w.to_pandas().set_index("TradeDate")
        pdf.index.name = "date"
        out[c] = (pdf, pl.from_pandas(pdf.reset_index()))
    return out


def t(fn, target=0.5, maxi=200, mini=5):
    fn()
    t0 = time.perf_counter(); fn(); one = time.perf_counter() - t0
    n = int(max(mini, min(maxi, target / one if one > 0 else maxi)))
    s = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); s.append(time.perf_counter() - t0)
    return statistics.median(s), min(s), n


def cpu_ratio(fn, n=5):
    fn()
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        fn()
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    return cpu / wall if wall > 0 else float("nan"), wall / n


def main():
    P = load_panels()
    pd_c, pl_c = P["AdjClose"]
    pd_v, pl_v = P["Volume"]
    res = {"panel": {"rows": int(pd_c.shape[0]), "cols": int(pd_c.shape[1])}}

    out = {}
    for canon, args_pd, args_pl, kw in [
        ("ts_beta", (pd_c, pd_v), (pl_c, pl_v), {"window": 20}),
        ("composition_normalized_entropy",
         (pd_c, pd_v, pd_c, pd_v, pd_c, pd_v, pd_c, pd_v),
         (pl_c, pl_v, pl_c, pl_v, pl_c, pl_v, pl_c, pl_v), {}),
        ("ts_cusum_pressure", (pd_c,), (pl_c,), {"reference_window": 20, "drift": 0.5}),
    ]:
        pop = OperatorRegistry.get(canon, "pandas_numpy")
        plop = OperatorRegistry.get(canon, "polars", mode="any")
        ref = pop.calculate(*args_pd, **kw)          # for rebuild timing
        t_pandas, _, n1 = t(lambda: pop.calculate(*args_pd, **kw))
        t_in, _, _ = t(lambda: [to_pandas_panel(a) for a in args_pl])
        t_out, _, _ = t(lambda: from_pandas_panel(args_pl[0], ref.copy()))
        t_glue, _, _ = t(lambda: (numeric_cols(args_pl[0]),
                                  verify_frames_share_identity(canon, *args_pl)))
        t_deleg, _, n2 = t(lambda: plop._calculate_series(*args_pl, **kw))
        # framework-level: what _call_pandas_delegate adds over (in+compute)
        try:
            from factor_engine.cleaned_operators.rolling_pack import _call_pandas_delegate
            t_cpd, _, _ = t(lambda: _call_pandas_delegate(
                canon, tuple(args_pl), {**kw, **({"q": 0.5} if canon == "cs_quantile_resid" else {})}))
        except Exception as e:
            t_cpd = None
        out[canon] = {
            "n_frames": len(args_pl),
            "pandas_compute_s": t_pandas,
            "marshal_in_s": t_in,
            "marshal_in_per_frame_s": t_in / len(args_pl),
            "rebuild_out_s": t_out,
            "framework_glue_s": t_glue,
            "delegate_total_s": t_deleg,
            "delegate_overhead_s": t_deleg - t_pandas,
            "delegate_overhead_x": t_deleg / t_pandas,
            "residual_after_marshal_s": t_deleg - t_pandas - t_in - t_out,
            "call_pandas_delegate_s": t_cpd,
            "compute_share_of_delegate_pct": round(100.0 * t_pandas / t_deleg, 2),
        }
        print(canon, json.dumps(out[canon], indent=1), flush=True)

    # single-thread evidence on the same real panel
    c0 = pl_c.columns[0]
    e = pl.col(c0)
    for _ in range(6):
        e = (e + 1.0).log().abs()
    native = lambda: pl_c.lazy().with_columns(e.alias("__n")).collect()
    for canon, args_pl, kw in [
        ("ts_cusum_pressure", (pl_c,), {"reference_window": 20, "drift": 0.5}),
        ("composition_normalized_entropy",
         (pl_c, pl_v, pl_c, pl_v, pl_c, pl_v, pl_c, pl_v), {}),
    ]:
        plop = OperatorRegistry.get(canon, "polars", mode="any")
        r, per = cpu_ratio(lambda: plop._calculate_series(*args_pl, **kw))
        print(f"CPU_RATIO delegate {canon} = {r:.2f}  ({per*1000:.1f} ms/call)", flush=True)
        out[canon]["delegate_cpu_per_wall"] = round(r, 3)

    # concat-style native op for a fair "polars can use cores" reference:
    wide = pl_c.lazy().with_columns([(pl.col(c) * 1.0001).alias(c) for c in pl_c.columns])
    r, per = cpu_ratio(lambda: wide.collect(), n=20)
    print(f"CPU_RATIO native 30-col polars lazy = {r:.2f}  ({per*1000:.3f} ms/call)", flush=True)
    res["operators"] = out
    res["cpu_per_wall"] = {"native_30col_polars": round(r, 3)}
    with open(f"{REPO}/evidence/_step1b_overhead_decomp.json", "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE _step1b_overhead_decomp.json")


if __name__ == "__main__":
    main()
