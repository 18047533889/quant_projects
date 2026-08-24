"""R2-P0-019: Polars ts_cov parameter-domain parity."""
from __future__ import annotations

import pytest

import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_ts_cov_rejects_min_periods_above_window() -> None:
    load_all()
    operator = OperatorRegistry.get("ts_cov", "polars", mode="research")
    frame = pl.DataFrame({"asset": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="min_periods must (?:be <= window|not exceed window)"):
        operator.calculate(frame, frame, window=2, min_periods=3)
