# -*- coding: utf-8 -*-
"""Unified OHLC domain contract (review #5 R5-49).

Candle / technical / volatility / spread / A-share-limit families each assume
their OHLC inputs are well-formed.  A bad vendor bar (High < Low, a negative
price, a zero range) manufactures fake shadows / ranges / hammer / spreads /
volatility rather than failing.  This module is the single place that checks
the invariant, so it can be applied at the source -> typed-IR boundary (or as a
defense-in-depth guard inside kernels) instead of each module re-encoding its
own partial assumption.

Contract per bar (all four of O/H/L/C finite):
  * O, H, L, C > 0            (prices are strictly positive)
  * H >= L                    (high is at least the low)
  * H >= max(O, C)            (high caps the body)
  * L <= min(O, C)            (low floors the body)

One bar with a valid row mask == True is a *violation* of the domain, not a
missing observation — a suspension bar is all-NaN (mask False) and is allowed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class OHLCContractViolation(ValueError):
    """Raised when a panel violates the OHLC domain invariant."""


def _frame(name: str, value: Any, template: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if not value.index.equals(template.index) or not value.columns.equals(template.columns):
            raise OHLCContractViolation(f"{name} panel is not aligned with the base OHLC panel")
        return value
    return pd.DataFrame(value, index=template.index, columns=template.columns)


def ohlc_valid_mask(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Boolean mask of bars satisfying the OHLC invariant (NaN -> False)."""
    o = _frame("open", open, high)
    c = _frame("close", close, high)
    l = _frame("low", low, high)
    ov, hv, lv, cv = (f.to_numpy(dtype=float, copy=False) for f in (o, high, l, c))
    all_finite = np.isfinite(ov) & np.isfinite(hv) & np.isfinite(lv) & np.isfinite(cv)
    positive = (ov > 0.0) & (hv > 0.0) & (lv > 0.0) & (cv > 0.0)
    high_ok = hv >= lv
    high_body = hv >= np.maximum(ov, cv)
    low_body = lv <= np.minimum(ov, cv)
    valid = all_finite & positive & high_ok & high_body & low_body
    return pd.DataFrame(valid, index=high.index, columns=high.columns)


def assert_valid_ohlc(
    open: pd.DataFrame | None,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    name: str = "ohlc",
) -> None:
    """Fail-closed guard: raise ``OHLCContractViolation`` listing the offending
    bar count when any jointly-finite bar breaks the invariant."""
    if open is None:
        return  # callers that do not supply open (e.g. H/L/close only) opt out
    mask = ohlc_valid_mask(open, high, low, close)
    invalid = int((~mask.to_numpy(dtype=bool)).sum())
    if invalid:
        raise OHLCContractViolation(
            f"{name}: {invalid} jointly-finite bar(s) violate the OHLC invariant "
            "(H>=L, H>=max(O,C), L<=min(O,C), O,H,L,C>0); a bad vendor bar must "
            "fail closed, not manufacture fake shadows/ranges (R5-49)"
        )
