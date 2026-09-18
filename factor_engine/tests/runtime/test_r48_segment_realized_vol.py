from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _oracle(index: pd.DatetimeIndex, values: np.ndarray, segment: str) -> float:
    lo, hi = {"morning": (570, 690), "afternoon": (780, 900)}[segment]
    total = 0.0
    for i in range(1, len(values)):
        minute = index[i].hour * 60 + index[i].minute
        prev, cur = float(values[i - 1]), float(values[i])
        if lo <= minute <= hi and np.isfinite(prev) and np.isfinite(cur) and prev > 0.0 and cur > 0.0:
            total += math.log(cur / prev) ** 2
    return math.sqrt(total)


def _run(frame: pl.DataFrame, segment: str, session_tz: str | None = None) -> pl.DataFrame:
    op = OperatorRegistry.get("intra_segment_realized_vol", backend="polars")
    return op.calculate(frame, segment=segment, session_tz=session_tz)


def _full_session() -> pd.DatetimeIndex:
    day = pd.Timestamp("2024-01-03")
    minutes = list(range(571, 691)) + list(range(781, 901))
    return pd.DatetimeIndex([day + pd.Timedelta(minutes=minute) for minute in minutes])


@pytest.mark.parametrize("segment", ["morning", "afternoon"])
def test_segment_realized_vol_uses_full_physical_axis_without_bridging_nonfinite(segment: str) -> None:
    index = _full_session()
    values = 100.0 * np.exp(np.arange(len(index), dtype=float) * 0.001)
    values[5] = np.nan
    values[125] = np.inf
    frame = pl.DataFrame({"QuoteTime": pl.Series(index), "B": values * 2.0, "A": values})

    got = _run(frame, segment)

    assert got.columns == ["date", "B", "A"]
    expected = _oracle(index, values, segment)
    assert got["A"][0] == pytest.approx(expected)
    assert got["B"][0] == pytest.approx(expected)


def test_segment_realized_vol_applies_requested_session_timezone() -> None:
    local = _full_session()
    utc = local.tz_localize("Asia/Shanghai").tz_convert("UTC")
    values = 100.0 * np.exp(np.arange(len(local), dtype=float) * 0.001)
    frame = pl.DataFrame({"QuoteTime": pl.Series(utc), "A": values})

    for segment in ("morning", "afternoon"):
        got = _run(frame, segment, session_tz="Asia/Shanghai")
        assert got["date"].to_list() == [pd.Timestamp("2024-01-03").date()]
        assert got["A"][0] == pytest.approx(_oracle(local, values, segment))


def test_segment_realized_vol_preserves_axes_when_segment_has_no_rows() -> None:
    index = pd.DatetimeIndex(["2024-01-03 16:00", "2024-01-04 16:00"])
    frame = pl.DataFrame({"QuoteTime": pl.Series(index), "B": [100.0, 101.0], "A": [200.0, 202.0]})

    got = _run(frame, "morning")

    assert got.columns == ["date", "B", "A"]
    assert got["date"].to_list() == [pd.Timestamp("2024-01-03").date(), pd.Timestamp("2024-01-04").date()]
    assert np.isnan(got.select("B", "A").to_numpy()).all()


def test_segment_realized_vol_sparse_session_matches_formal_coverage_nan() -> None:
    index = pd.DatetimeIndex([
        "2024-01-03 09:29", "2024-01-03 09:30", "2024-01-03 09:32",
        "2024-01-03 11:30", "2024-01-03 13:01", "2024-01-03 15:00",
    ])
    values = np.array([100.0, 101.0, 140.0, 150.0, 165.0, 198.0])
    frame = pl.DataFrame({"QuoteTime": pl.Series(index), "B": values * 2.0, "A": values})

    for segment in ("morning", "afternoon"):
        got = _run(frame, segment)
        assert got.columns == ["date", "B", "A"]
        assert np.isnan(got.select("B", "A").to_numpy()).all()
