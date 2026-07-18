# -*- coding: utf-8
"""保护性算子与价量 golden。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy protected/price-volume primitives were removed from the active surface")

from cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged


def _run(canon: str, panel: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None
    out = op.calculate(panel.copy(), *args, **kwargs)
    assert_panel_shape_unchanged(panel, out)
    return out


def test_protected_log_non_positive_clamps_to_epsilon(golden_panel, loaded):
    panel = golden_panel.copy()
    panel.loc["2024-01-01", "A"] = -1.0
    out = _run("protected_log", panel)
    eps = 1e-12
    assert out.loc["2024-01-01", "A"] == pytest.approx(np.log(eps), rel=1e-6)
    assert out.loc["2024-01-02", "A"] == pytest.approx(np.log(2.0), rel=1e-9)


def test_protected_sqrt_negative_returns_zero(golden_panel, loaded):
    panel = golden_panel.copy()
    panel.loc["2024-01-01", "A"] = -4.0
    out = _run("protected_sqrt", panel)
    assert out.loc["2024-01-01", "A"] == 0.0
    assert out.loc["2024-01-02", "A"] == pytest.approx(np.sqrt(2.0), rel=1e-9)


def test_rolling_beta_constant_benchmark(golden_panel, loaded):
    idx = pd.date_range("2024-01-01", periods=6)
    ret = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}, index=idx)
    bm = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}, index=idx)
    out = _run("rolling_beta", ret, bm, 3)
    assert out.loc[idx[2], "A"] == pytest.approx(1.0, rel=1e-6)


def test_vwap_rolling_per_column(golden_panel, loaded):
    idx = pd.date_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=idx)
    vol = pd.DataFrame({"A": [100.0, 200.0, 100.0, 200.0]}, index=idx)
    out = _run("vwap", close, vol, 2)
    w0 = 10 * 100 + 11 * 200
    w1 = 100 + 200
    assert out.loc[idx[1], "A"] == pytest.approx(w0 / w1, rel=1e-9)
