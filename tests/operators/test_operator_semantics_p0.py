# -*- coding: utf-8
"""P0 算子语义 golden tests（shape 不变 + 截面广播 + rank 语义）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy P0 semantics reference removed canonicals")

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.common.cs_broadcast import broadcast_row_stat, cs_rank_01
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def _panel():
    idx = pd.date_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, 5.0],
            "B": [2.0, 2.0, 2.0, np.nan, 10.0],
            "C": [np.nan, 1.0, 3.0, 4.0, 5.0],
        },
        index=idx,
    )


def test_broadcast_row_stat_preserves_shape():
    x = _panel()
    out = broadcast_row_stat(x, x.mean(axis=1))
    assert out.shape == x.shape
    assert list(out.columns) == list(x.columns)
    assert out.loc["2024-01-01", "A"] == pytest.approx(out.loc["2024-01-01", "B"])


def test_c_mean_not_all_zero(_loaded):
    op = OperatorRegistry.get("c_mean")
    x = _panel()
    out = op.calculate(x)
    assert out.shape == x.shape
    assert not (out.fillna(-999) == 0).all().all()
    row_mean = x.mean(axis=1).iloc[0]
    assert out.loc["2024-01-01", "A"] == pytest.approx(row_mean)


def test_c_count_broadcast(_loaded):
    op = OperatorRegistry.get("c_count")
    out = op.calculate(_panel())
    assert out.loc["2024-01-01", "A"] == pytest.approx(2.0)


def test_c_percentile_uses_p_parameter(_loaded):
    op = OperatorRegistry.get("c_percentile")
    x = _panel()
    out = op.calculate(x, p=0.5)
    med = x.quantile(0.5, axis=1).iloc[0]
    assert out.loc["2024-01-01", "A"] == pytest.approx(med)


def test_rank_is_0_1(_loaded):
    op = OperatorRegistry.get("rank")
    x = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]})
    out = op.calculate(x)
    assert out.loc[0, "A"] == pytest.approx(0.0)
    assert out.loc[0, "B"] == pytest.approx(1.0)


def test_rank_pct_matches_pandas_pct(_loaded):
    op = OperatorRegistry.get("rank_pct")
    x = _panel()
    expected = x.rank(pct=True, axis=1)
    out = op.calculate(x)
    pd.testing.assert_frame_equal(out, expected)


def test_cs_rank_01_single_value_row():
    x = pd.DataFrame({"A": [5.0], "B": [np.nan]})
    out = cs_rank_01(x)
    assert out.loc[0, "A"] == pytest.approx(0.5)
    assert pd.isna(out.loc[0, "B"])


def test_vwap_zero_volume_is_nan(_loaded):
    op = OperatorRegistry.get("vwap")
    price = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    volume = pd.DataFrame({"A": [0.0, 10.0, 10.0]})
    out = op.calculate(price, volume, window=2)
    assert pd.isna(out.iloc[0, 0])


def test_dropna_not_production_allowed(_loaded):
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec

    spec = build_operator_spec("dropna")
    assert spec is not None
    assert spec.allow_in_production is False


def test_rolling_beta_to_market_not_production_allowed(_loaded):
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec

    spec = build_operator_spec("rolling_beta_to_market")
    assert spec is not None
    assert spec.status == "experimental"
    assert spec.allow_in_production is False
