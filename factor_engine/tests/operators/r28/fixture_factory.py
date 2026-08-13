# -*- coding: utf-8 -*-
"""R28 §一百八十一..一百八十三: fixture factory.

Generates semantically appropriate synthetic panels for operator families so
tests exercise the intended input domain (returns for GARCH, price for OHLC,
PIT filing series for fundamental, sparse event process for event ops) instead
of feeding every operator the same ``close`` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def panel_from(values: np.ndarray, *, instruments: int = 3, start: str = "2024-01-02") -> pd.DataFrame:
    """Multi-instrument daily panel from a shape-(n, instruments) array."""
    dates = pd.bdate_range(start, periods=values.shape[0])
    idx = pd.MultiIndex.from_product([dates, [f"I{k}" for k in range(values.shape[1])]], names=["timestamp", "instrument"])
    return pd.DataFrame(values.reshape(-1, 1), index=idx, columns=["x"])


def price_panel(n: int = 240, instruments: int = 3, seed: int = 0) -> pd.DataFrame:
    """Smooth geometric-random-walk price levels (never a bare white-noise series)."""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((n, instruments)) * 0.02
    price = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return panel_from(price)


def return_panel(n: int = 240, instruments: int = 3, seed: int = 1) -> pd.DataFrame:
    """Daily returns (GARCH / AR / volatility input domain)."""
    rng = np.random.default_rng(seed)
    return panel_from(rng.standard_normal((n, instruments)) * 0.02)


def ohlcv(n: int = 240, instruments: int = 3, seed: int = 2) -> dict[str, pd.DataFrame]:
    """OHLCV with open->close consistency (open/pre_close/close relationships)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal((n, instruments)) * 0.015, axis=0))
    open_ = close * np.exp(rng.standard_normal((n, instruments)) * 0.01)
    high = np.maximum(open_, close) * np.exp(rng.uniform(0, 0.005, (n, instruments)))
    low = np.minimum(open_, close) * np.exp(-rng.uniform(0, 0.005, (n, instruments)))
    vol = rng.lognormal(15.0, 0.3, (n, instruments))
    return {
        "close": panel_from(close),
        "open": panel_from(open_),
        "high": panel_from(high),
        "low": panel_from(low),
        "volume": panel_from(vol),
    }


def event_panel(n: int = 240, instruments: int = 3, seed: int = 3, density: float = 0.08) -> pd.DataFrame:
    """Sparse event process (0/1)."""
    rng = np.random.default_rng(seed)
    ev = (rng.uniform(0, 1, (n, instruments)) < density).astype(float)
    return panel_from(ev)


def group_panel(n: int = 240, instruments: int = 3) -> pd.DataFrame:
    return panel_from(np.tile(np.arange(instruments), (n, 1)).astype(float))


def two_return_panels(n: int = 240, instruments: int = 3, seed: int = 5):
    """A y panel and correlated x panels for regression-family ops."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n, 1))
    y = panel_from(common + rng.standard_normal((n, instruments)) * 0.1)
    x1 = panel_from(0.5 * common + rng.standard_normal((n, instruments)) * 0.5)
    x2 = panel_from(0.3 * common + rng.standard_normal((n, instruments)) * 0.7)
    return y, x1, x2


def pit_filing_series(n: int = 120, instruments: int = 3, seed: int = 7) -> pd.DataFrame:
    """A slow-moving fundamental PIT-style series (quarterly step function)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 2.0, (1, instruments))
    out = np.zeros((n, instruments))
    for i in range(n):
        out[i] = base + (i // 20) * 0.05 + rng.normal(0, 0.01, (1, instruments))
    return panel_from(out)


def minute_panel(n: int = 240, instruments: int = 3, seed: int = 11) -> pd.DataFrame:
    """Intraday (bar-level) feature input."""
    rng = np.random.default_rng(seed)
    return panel_from(rng.standard_normal((n, instruments)))


def fixture_for(family: str, n: int = 240, instruments: int = 3, seed: int = 42) -> dict:
    """Return keyword fixture kwargs for a model/stat family."""
    if family in ("garch", "har", "volatility"):
        return {"x": return_panel(n, instruments, seed)}
    if family in ("kalman",):
        return {"x": price_panel(n, instruments, seed)}
    if family in ("pca", "panel_model"):
        y, x1, x2 = two_return_panels(n, instruments, seed)
        return {"y": y, "x1": x1, "x2": x2}
    if family in ("ar",):
        return {"x": return_panel(n, instruments, seed)}
    if family in ("rqa", "matrix_profile", "entropy", "spectral", "dmd", "signature", "state", "kernel"):
        return {"x": return_panel(n, instruments, seed)}
    if family in ("regression",):
        y, x1, x2 = two_return_panels(n, instruments, seed)
        return {"y": y, "x": x1, "x1": x1, "x2": x2}
    if family in ("event",):
        return {"x": event_panel(n, instruments, seed)}
    return {"x": return_panel(n, instruments, seed)}
