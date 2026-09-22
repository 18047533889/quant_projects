# -*- coding: utf-8 -*-
"""Synthetic in-memory panel source for full-operator test matrix.

Contract: load_column(name) -> MultiIndex Series indexed by (timestamp, instrument),
mirroring ParquetSource/DataAccessSource. Deterministic (seeded), includes NaN
gaps and a zero-volume "suspend" day so operators exercise missing-data paths.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from factor_engine.storage.sources.datasource import DataSource

DATES = pd.date_range("2024-01-01", periods=180, freq="B")
INSTRUMENTS = [f"T{i:03d}.SZ" for i in range(24)] + [f"T{i:03d}.SH" for i in range(24)]


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def _panel(name: str) -> pd.Series:
    """Deterministic float panel for a logical field name."""
    rng = np.random.default_rng(_seed(name))
    n_d, n_i = len(DATES), len(INSTRUMENTS)

    if name in ("close", "open", "high", "low", "adjclose", "adj_high", "adj_low"):
        # correlated random walk family so TS ops have signal
        base = {"close": 100.0, "open": 99.5, "high": 101.0, "low": 98.5,
                "adjclose": 100.0, "adj_high": 101.0, "adj_low": 98.5}.get(name, 100.0)
        drift = rng.normal(0, 0.01, size=(n_i,))
        steps = rng.normal(0, 0.02, size=(n_d, n_i)) + drift[None, :]
        vals = base * np.cumprod(1.0 + steps, axis=0)
        if name in ("high", "adj_high"):
            vals *= 1.01
        if name in ("low", "adj_low"):
            vals *= 0.99
        if name == "open":
            vals *= 1.0 + rng.normal(0, 0.002, size=vals.shape)
    elif name in ("volume", "adj_volume", "amount", "adj_amount"):
        vals = np.exp(rng.normal(14.0, 0.6, size=(n_d, n_i)))  # lognormal ~ 1.2M
        if name in ("amount", "adj_amount"):
            vals *= 100.0
    elif name in ("returns", "ret", "pct_change"):
        vals = rng.normal(0, 0.02, size=(n_d, n_i))
    elif name in ("vwap", "adj_vwap"):
        vals = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.015, size=(n_d, n_i)), axis=0)
    elif name in ("turnover", "turnover_rate"):
        vals = np.abs(rng.normal(0.02, 0.01, size=(n_d, n_i)))
    elif name in ("shares_outstanding", "float_shares", "free_float_shares"):
        vals = np.exp(rng.normal(19.0, 0.3, size=(n_d, n_i)))
    elif name in ("is_suspended", "suspended", "tradable", "st_flag"):
        vals = np.zeros((n_d, n_i))
        vals[40:45, 3] = 1.0
        vals[80:83, 7] = 1.0
    else:
        # generic deterministic fallback: any unknown logical field still yields
        # a well-formed panel so signature-probing never dies on data shape
        vals = rng.normal(0.0, 1.0, size=(n_d, n_i))

    # inject NaNs deterministically (missing-data paths): ~2% cells + 3 full
    # tail rows so window ops hit min_periods/warmup boundaries
    mask = rng.random(vals.shape) < 0.02
    vals = vals.astype("float64")
    vals[mask] = np.nan
    vals[-1, :6] = np.nan

    idx = pd.MultiIndex.from_product([DATES, INSTRUMENTS], names=["timestamp", "instrument"])
    return pd.Series(vals.reshape(-1), index=idx, name=name)


_CACHE: dict[str, pd.Series] = {}
_PL_CACHE: dict[str, object] = {}


class SyntheticPanelSource(DataSource):
    """In-memory deterministic panel; zero I/O, safe for parallel test shards."""

    def load_column(self, name: str):
        if name not in _CACHE:
            _CACHE[name] = _panel(name)
        return _CACHE[name]

    def scan_polars_long(self, columns):
        """Long LazyFrame view (timestamp, instrument, <cols>) for the
        polars/SQL/auto backends -- mirrors LongTableDataSource semantics."""
        import polars as pl
        from factor_engine.storage.factor_format import series_to_long_table

        merged = None
        for name in sorted(set(columns)):
            if name not in _PL_CACHE:
                import polars as _pl
                pdf = series_to_long_table(
                    self.load_column(name),
                    timestamp_col="timestamp",
                    asset_col="instrument",
                    value_col=name,
                )
                _PL_CACHE[name] = _pl.from_pandas(pdf).lazy()
            part = _PL_CACHE[name]
            merged = part if merged is None else merged.join(
                part, on=["timestamp", "instrument"], how="inner")
        return merged.sort(["instrument", "timestamp"])

    def execution_spec(self):
        return {
            "kind": "synthetic_panel",
            "dates": str(DATES[0].date()) + ".." + str(DATES[-1].date()),
            "n_dates": len(DATES),
            "n_instruments": len(INSTRUMENTS),
        }


WARM_FIELDS = ["close", "volume", "amount", "turnover", "returns", "vwap",
               "high", "low", "open", "industry", "is_suspended",
               "shares_outstanding", "adjclose", "adj_vwap", "st_flag"]


def warm_polars_cache(fields=None):
    """Build polars long frames in the PARENT so forked children share them."""
    src = SyntheticPanelSource()
    for f in (fields or WARM_FIELDS):
        if f not in _PL_CACHE:
            src.scan_polars_long([f])
