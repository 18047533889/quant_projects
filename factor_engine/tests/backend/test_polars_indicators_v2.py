# -*- coding: utf-8 -*-
"""Parity tests for native Polars parameterized technical indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(20260805)
    index = pd.date_range("2022-01-01", periods=120, freq="D")
    close = pd.DataFrame(
        {
            "A": 80.0 + np.cumsum(rng.normal(0.1, 1.2, len(index))),
            "B": 45.0 + np.cumsum(rng.normal(-0.02, 0.7, len(index))),
        },
        index=index,
    )
    spread = pd.DataFrame(rng.uniform(0.5, 2.5, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    volume = pd.DataFrame(rng.integers(10_000, 2_000_000, close.shape), index=index, columns=close.columns).astype(float)
    return high, low, close, volume


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None
    assert polars_op is not None
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("name", "inputs", "kwargs"),
    [
        ("DMI_plus", (0, 1, 2), {"window": 14}),
        ("DMI_minus", (0, 1, 2), {"window": 14}),
        ("DX", (0, 1, 2), {"window": 14}),
        ("NATR", (0, 1, 2), {"window": 14}),
        ("PPO", (2,), {"fast_window": 12, "slow_window": 26}),
        ("PPO_signal", (2,), {"fast_window": 12, "slow_window": 26, "signal_window": 9}),
        ("PPO_hist", (2,), {"fast_window": 12, "slow_window": 26, "signal_window": 9}),
        ("PVO", (3,), {"fast_window": 12, "slow_window": 26}),
        ("PVO_signal", (3,), {"fast_window": 12, "slow_window": 26, "signal_window": 9}),
        ("PVO_hist", (3,), {"fast_window": 12, "slow_window": 26, "signal_window": 9}),
        ("CMO", (2,), {"window": 14}),
        ("VortexPlus", (0, 1, 2), {"window": 14}),
        ("VortexMinus", (0, 1, 2), {"window": 14}),
        ("TSI", (2,), {"long_window": 25, "short_window": 13}),
        ("TSI_signal", (2,), {"long_window": 25, "short_window": 13, "signal_window": 7}),
        (
            "UltimateOscillator",
            (0, 1, 2),
            {"short_window": 7, "medium_window": 14, "long_window": 28, "short_weight": 4.0, "medium_weight": 2.0, "long_weight": 1.0},
        ),
        ("DEMA", (2,), {"window": 15}),
        ("TEMA", (2,), {"window": 15}),
        ("CMF", (0, 1, 2, 3), {"window": 20}),
        ("MFI", (0, 1, 2, 3), {"window": 14}),
    ],
)
def test_native_polars_indicator_matches_pandas(panels, name, inputs, kwargs):
    _assert_parity(name, tuple(panels[index] for index in inputs), kwargs)


def test_polars_indicator_edge_values_are_shape_preserving():
    index = pd.date_range("2025-01-01", periods=12, freq="D")
    close = pd.DataFrame({"A": [1.0, 1.0, np.nan, 2.0, 0.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0]}, index=index)
    high = close + 1.0
    low = close - 1.0
    volume = pd.DataFrame({"A": [0.0, 10.0, np.nan, 20.0, 30.0, 0.0, 40.0, 50.0, 60.0, 0.0, 70.0, 80.0]}, index=index)
    for name, args, kwargs in (
        ("DMI_plus", (high, low, close), {"window": 3}),
        ("DMI_minus", (high, low, close), {"window": 3}),
        ("DX", (high, low, close), {"window": 3}),
        ("NATR", (high, low, close), {"window": 3}),
        ("CMO", (close,), {"window": 3}),
        ("VortexPlus", (high, low, close), {"window": 3}),
        ("VortexMinus", (high, low, close), {"window": 3}),
        ("TSI", (close,), {"long_window": 5, "short_window": 3}),
        ("UltimateOscillator", (high, low, close), {"short_window": 2, "medium_window": 3, "long_window": 5, "short_weight": 4.0, "medium_weight": 2.0, "long_weight": 1.0}),
        ("DEMA", (close,), {"window": 3}),
        ("TEMA", (close,), {"window": 3}),
        ("CMF", (high, low, close, volume), {"window": 3}),
        ("MFI", (high, low, close, volume), {"window": 3}),
    ):
        _assert_parity(name, args, kwargs)


def test_polars_indicator_nan_warmup_matches_pandas():
    # 完全随机的 NaN 位置，验证所有窗口算子在 NaN 前的 warmup 语义一致。
    rng = np.random.default_rng(7)
    index = pd.date_range("2023-06-01", periods=60, freq="D")
    mask = rng.random((60, 2)) < 0.15
    base = pd.DataFrame(rng.normal(10, 3, (60, 2)), index=index, columns=["A", "B"])
    base[mask] = np.nan
    close = base
    high = close + 1.0
    low = close - 1.0
    volume = pd.DataFrame(rng.uniform(1e3, 1e5, (60, 2)), index=index, columns=["A", "B"])
    volume[mask] = np.nan
    for name, args, kwargs in (
        ("DMI_plus", (high, low, close), {"window": 5}),
        ("DMI_minus", (high, low, close), {"window": 5}),
        ("DX", (high, low, close), {"window": 5}),
        ("NATR", (high, low, close), {"window": 5}),
        ("PPO", (close,), {"fast_window": 3, "slow_window": 6}),
        ("PPO_hist", (close,), {"fast_window": 3, "slow_window": 6, "signal_window": 2}),
        ("PVO", (volume,), {"fast_window": 3, "slow_window": 6}),
        ("CMO", (close,), {"window": 5}),
        ("VortexPlus", (high, low, close), {"window": 5}),
        ("TSI", (close,), {"long_window": 8, "short_window": 4}),
        ("UltimateOscillator", (high, low, close), {"short_window": 3, "medium_window": 5, "long_window": 9, "short_weight": 4.0, "medium_weight": 2.0, "long_weight": 1.0}),
        ("DEMA", (close,), {"window": 5}),
        ("TEMA", (close,), {"window": 5}),
        ("CMF", (high, low, close, volume), {"window": 5}),
        ("MFI", (high, low, close, volume), {"window": 5}),
    ):
        _assert_parity(name, args, kwargs)
