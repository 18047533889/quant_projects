# -*- coding: utf-8 -*-
"""Golden parquet 回归：frozen fixture + 手算对照。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry
from tests.fixtures.golden_panel import (
    build_golden_close_panel,
    expected_ts_delay,
    expected_ts_mean,
)

pd = pytest.importorskip("pandas")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
FIXTURE_PATH = FIXTURE_DIR / "close_panel.parquet"


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


@pytest.fixture(scope="module")
def golden_close_parquet() -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if not FIXTURE_PATH.exists():
        panel = build_golden_close_panel()
        frame = panel.reset_index()
        frame.columns = ["timestamp", "instrument", "close"]
        frame.to_parquet(FIXTURE_PATH, index=False)
    return FIXTURE_PATH


def _op(name: str):
    return OperatorRegistry.get(name)


def test_golden_parquet_ts_mean(golden_close_parquet):
    frame = pd.read_parquet(golden_close_parquet)
    idx = pd.MultiIndex.from_arrays(
        [frame["timestamp"], frame["instrument"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series(frame["close"].values, index=idx)
    panel = close.unstack("instrument")
    out = _op("ts_mean").calculate(panel, window=2)
    exp = expected_ts_mean(close, 2).unstack("instrument")
    pd.testing.assert_frame_equal(out, exp, check_names=False)


def test_golden_parquet_ts_delay(golden_close_parquet):
    frame = pd.read_parquet(golden_close_parquet)
    idx = pd.MultiIndex.from_arrays(
        [frame["timestamp"], frame["instrument"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series(frame["close"].values, index=idx)
    panel = close.unstack("instrument")
    out = _op("ts_delay").calculate(panel, window=1)
    exp = expected_ts_delay(close, 1).unstack("instrument")
    pd.testing.assert_frame_equal(out, exp, check_names=False)
