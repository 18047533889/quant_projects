"""核心算子数学正确性回归测试（手算基准）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

pd = pytest.importorskip("pandas")


def _panel(values, col="A"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({col: list(values)}, index=idx)


def _op(name):
    ensure_cleaned_loaded()
    return OperatorRegistry.get(name)


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    ensure_cleaned_loaded()


class TestMathRegression:
    def test_ts_mean_window_3(self):
        x = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _op("ts_mean").calculate(x, 3)
        # 本引擎 ts_mean 默认 min_periods=1
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        for i, exp in enumerate(expected):
            assert out.iloc[i]["A"] == pytest.approx(exp)

    def test_ts_std_uses_sample_ddof(self):
        x = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _op("ts_std").calculate(x, window=3)
        manual = x["A"].rolling(3, min_periods=1).std(ddof=1)
        for i in range(len(x)):
            assert out.iloc[i]["A"] == pytest.approx(manual.iloc[i], rel=1e-9, nan_ok=True)

    def test_ts_delay_1(self):
        x = _panel([10.0, 20.0, 30.0, 40.0])
        out = _op("ts_delay").calculate(x, 1)
        assert pd.isna(out.iloc[0]["A"])
        assert out.iloc[1]["A"] == pytest.approx(10.0)
        assert out.iloc[3]["A"] == pytest.approx(30.0)

    def test_ts_delta_1(self):
        x = _panel([1.0, 4.0, 3.0, 8.0])
        out = _op("ts_delta").calculate(x, 1)
        assert pd.isna(out.iloc[0]["A"])
        assert out.iloc[1]["A"] == pytest.approx(3.0)
        assert out.iloc[2]["A"] == pytest.approx(-1.0)

    def test_rank_cross_section_two_cols(self):
        x = pd.DataFrame(
            {"A": [1.0, 100.0], "B": [2.0, 50.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )
        out = _op("rank").calculate(x)
        # 第一行：A 最小 → rank 较低（pct）
        assert out.loc[out.index[0], "A"] < out.loc[out.index[0], "B"]
        # 第二行截面 rank 仅依赖当行，与第一行无关
        assert out.loc[out.index[1], "A"] == pytest.approx(1.0)
        assert out.loc[out.index[1], "B"] == pytest.approx(0.5)

    def test_cs_rank_prefix_invariant(self):
        """截断未来日期后，历史日截面 rank 不变。"""
        x = pd.DataFrame(
            {
                "A": [1.0, 10.0, 3.0, 30.0],
                "B": [2.0, 20.0, 4.0, 40.0],
            },
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
        )
        full = _op("rank").calculate(x)
        part = _op("rank").calculate(x.iloc[:2])
        assert full.iloc[0]["A"] == pytest.approx(part.iloc[0]["A"])
        assert full.iloc[1]["A"] == pytest.approx(part.iloc[1]["A"])
