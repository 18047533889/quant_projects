#!/usr/bin/env python3
"""Smoke tests for extended factor evaluation metrics."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_extensions import (  # noqa: E402
    compute_extended_eval,
    enrich_post_analysis,
    ic_autocorrelation,
    ic_half_life,
    ic_monthly_heatmap,
    ic_significance,
    merge_extended_into_analysis,
    rolling_ic_series,
)


def test_ic_diagnostics() -> None:
    ics = [0.05, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03]
    dates = [20200102, 20200103, 20200106, 20200107, 20200108, 20200109, 20200110]
    hm = ic_monthly_heatmap(dates, ics)
    assert hm["years"] == [2020]
    assert len(hm["matrix"][0]) == 12
    ac = ic_autocorrelation(ics)
    assert "ic_autocorr_lag1" in ac
    hl = ic_half_life(ics, max_lag=5)
    assert "ic_half_life_days" in hl


def test_extended_eval_synthetic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fac = root / "factor.parquet"
        fwd = root / "fwd.parquet"
        ind = root / "industry.parquet"
        mcap = root / "mcap.parquet"

        rows = []
        for td in (20200102, 20200103, 20200106, 20200107, 20200108):
            for i, sym in enumerate(["A", "B", "C", "D", "E"]):
                rows.append({"trade_date": td, "symbol": sym, "value": float(i + td % 7)})
        pd.DataFrame(rows).to_parquet(fac)

        fwd_rows = []
        for td in (20200102, 20200103, 20200106, 20200107):
            sig = td
            for sym in ["A", "B", "C", "D", "E"]:
                fwd_rows.append({"signal_date": sig, "symbol": sym, "value": 0.001 * (hash(sym) % 5)})
        pd.DataFrame(fwd_rows).to_parquet(fwd)

        ind_rows = []
        mcap_rows = []
        for td in (20200102, 20200103, 20200106, 20200107, 20200108):
            for j, sym in enumerate(["A", "B", "C", "D", "E"]):
                ind_rows.append(
                    {"trade_date": td, "symbol": sym, "industry_code": "I1" if j < 3 else "I2"}
                )
                mcap_rows.append({"trade_date": td, "symbol": sym, "log_market_cap": float(j + 1)})
        pd.DataFrame(ind_rows).to_parquet(ind)
        pd.DataFrame(mcap_rows).to_parquet(mcap)

        con = duckdb.connect()
        con.execute(
            f"CREATE TABLE fwd AS SELECT * FROM read_parquet('{fwd.as_posix()}')"
        )
        ext = compute_extended_eval(
            factor_path=fac,
            fwd_returns_path=fwd,
            industry_path=ind,
            market_cap_path=mcap,
            min_names=3,
            con=con,
        )
        merged = merge_extended_into_analysis({"mean_rank_ic": 0.01}, ext)
        assert "industry_neutral" in ext
        assert "size_neutral" in ext
        assert "ic_monthly_heatmap" in ext
        assert merged.get("industry_neutral_mean_rank_ic") is not None


def test_enrich_post_analysis() -> None:
    dates = [20200102, 20200103, 20200106, 20200107, 20200108, 20200109, 20200110]
    ics = [0.05, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03]
    base = {
        "trade_dates": dates,
        "daily_rank_ic": ics,
        "group_mean_returns": [0.001, 0.002, 0.003, 0.004, 0.005],
        "daily_ls_returns": [0.01, -0.005, 0.008, 0.002, -0.001, 0.004, 0.003],
        "extended_eval": {"ic_half_life_days": 12.0},
    }
    out = enrich_post_analysis(base)
    assert "ic_t_stat" in out
    assert out["extended_eval"].get("rolling_ic_20d")
    assert out["extended_eval"].get("ic_autocorr_curve")
    sig = ic_significance(ics)
    assert sig["ic_n_days"] == 7.0


if __name__ == "__main__":
    test_ic_diagnostics()
    test_extended_eval_synthetic()
    test_enrich_post_analysis()
    print("ok eval_extensions tests passed")
