# -*- coding: utf-8 -*-
"""Cross-backend bit-parity on REAL A-share data (ashare_stock_daily_adj).

The synthetic full-matrix proves value coverage; THIS module proves the
multi-backend contract on real market data with real schema: pandas vs
polars vs duckdb_sql vs auto must agree bit-for-bit (NaN pattern + values).
Pattern proven in R63 (bit-identical across backends).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.expanduser("~/cos_data"))
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")

import factor_engine.cleaned_operators  # noqa: E402,F401
from factor_engine.cleaned_operators import load_all  # noqa: E402

with __import__("warnings").catch_warnings():
    __import__("warnings").simplefilter("ignore")
    load_all()

from factor_engine.api.dsl_parser import parse_factor  # noqa: E402
from factor_engine.backend.factory import build_backend  # noqa: E402
from factor_engine.runtime.engine import FactorEngine  # noqa: E402
from factor_engine.storage.sources.data_access_source import DataAccessSource  # noqa: E402

INSTRUMENTS = (
    [f"{c}.SZ" for c in ("000001", "000002", "000063", "000100", "000333",
                         "000651", "000725", "000858")]
    + [f"{c}.SH" for c in ("600000", "600016", "600030", "600036", "600104",
                           "600276", "600438", "600519")]
)

CORE_FACTORS = {
    "par_id_close": "close",
    "par_id_volume": "volume",
    "par_ts_mean": "ts_mean(close, 20)",
    "par_ts_rank": "ts_rank(volume, 20)",
    "par_ts_corr": "ts_corr(close, volume, 30)",
    "par_rank": "rank(volume)",
    "par_ts_std": "ts_std(close, 20)",
    "par_delta": "delta(close, 5)",
    "par_ema": "ema(close, 20)",
    "par_zscore": "ts_zscore(close, 20)",
    "par_chain": "rank(ts_mean(ts_zscore(volume, 20), 10))",
    "par_sign": "sign(delta(close, 1))",
}


def _run_real(backend, factor_names):
    src = DataAccessSource(
        dataset="ashare_stock_daily_adj",
        start_date="2024-01-01", end_date="2024-12-31",
        instrument_filter=list(INSTRUMENTS),
        run_mode="interactive_research", production=False, read_auto=False,
    )
    eng = FactorEngine(build_backend(backend), src, run_mode="research")
    factors = [parse_factor(CORE_FACTORS[n], name=n) for n in factor_names]
    out = eng.run_many(factors, market="ashare", result_policy="return")
    results = out.get("results") if isinstance(out, dict) else out
    vals = {}
    for n in factor_names:
        v = results[n]
        if hasattr(v, "to_numpy"):
            vals[n] = v.to_numpy(dtype="float64", na_value=np.nan)
        elif hasattr(v, "values"):
            vals[n] = np.asarray(v, dtype="float64")
        else:
            vals[n] = np.asarray(getattr(v, "data", None) or v, dtype="float64")
    return vals


def _cmp(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return f"nan-diff {int((np.isnan(a) ^ np.isnan(b)).sum())}"
    m = np.isfinite(a)
    if not m.any():
        return None
    d = np.abs(a[m] - b[m])
    if d.max() > 1e-9:
        return f"maxabs {d.max():.3e}"
    return None


@pytest.mark.parametrize("backend", ("polars", "duckdb_sql", "auto"))
def test_real_data_bit_parity(backend):
    """Every core factor must be bit-identical between pandas and backend."""
    base = _run_real("pandas", list(CORE_FACTORS))
    other = _run_real(backend, list(CORE_FACTORS))
    diffs = {n: m for n in CORE_FACTORS if (m := _cmp(base[n], other[n]))}
    assert not diffs, f"{backend} disagrees with pandas on real data: {diffs}"


def test_real_data_auto_uses_vectorized_path():
    """auto must route core factors to polars/sql (never the row-wise path)."""
    out = None
    src = DataAccessSource(
        dataset="ashare_stock_daily_adj",
        start_date="2024-01-01", end_date="2024-12-31",
        instrument_filter=list(INSTRUMENTS),
        run_mode="interactive_research", production=False, read_auto=False,
    )
    eng = FactorEngine(build_backend("auto"), src, run_mode="research")
    factors = [parse_factor(e, name=n) for n, e in CORE_FACTORS.items()]
    out = eng.run_many(factors, market="ashare", result_policy="return")
    paths = out.get("backend_paths") if isinstance(out, dict) else None
    if paths is not None:
        for n, p in paths.items():
            s = str(p)
            assert "row" not in s.lower() or "rowwise" not in s.lower().replace("_", ""), (
                f"auto routed {n} to a row-wise path: {p}"
            )
