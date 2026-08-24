# -*- coding: utf-8
"""对照审查清单的 golden / 边界测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy audit inventory references operators removed from the active registry")

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_c_sum_all_nan_row_is_null(_loaded):
    op = OperatorRegistry.get("c_sum")
    x = pd.DataFrame({"A": [np.nan, 1.0], "B": [np.nan, 2.0]})
    out = op.calculate(x)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(3.0)


def test_rank_single_column_is_half(_loaded):
    op = OperatorRegistry.get("rank")
    x = pd.DataFrame({"A": [42.0]})
    out = op.calculate(x)
    assert out.iloc[0, 0] == pytest.approx(0.5)


def test_cs_quantile_matches_c_percentile(_loaded):
    c_pct = OperatorRegistry.get("c_percentile")
    cs_q = OperatorRegistry.get("cs_quantile")
    x = pd.DataFrame({"A": [1.0, 4.0], "B": [3.0, 6.0], "C": [5.0, 8.0]})
    out1 = c_pct.calculate(x, p=0.5)
    out2 = cs_q.calculate(x, p=0.5)
    pd.testing.assert_frame_equal(out1, out2)


def test_cs_pct_rank_differs_from_cs_quantile(_loaded):
    rank_op = OperatorRegistry.get("cs_pct_rank")
    q_op = OperatorRegistry.get("cs_quantile")
    x = pd.DataFrame({"A": [1.0, 10.0], "B": [2.0, 20.0], "C": [3.0, 30.0]})
    rank_out = rank_op.calculate(x)
    q_out = q_op.calculate(x, p=0.5)
    assert not np.isclose(rank_out.iloc[0, 0], q_out.iloc[0, 0], equal_nan=True)


def test_idio_vol_requires_benchmark(_loaded):
    op = OperatorRegistry.get("idio_vol")
    ret = pd.DataFrame({"A": np.linspace(0.01, 0.05, 20)})
    with pytest.raises(ValueError, match="benchmark"):
        op.calculate(ret, window=10)


def test_idio_vol_with_benchmark(_loaded):
    op = OperatorRegistry.get("idio_vol")
    idx = pd.date_range("2024-01-01", periods=30)
    ret = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.02, 30)}, index=idx)
    bench = pd.DataFrame({"M": np.random.default_rng(1).normal(0, 0.015, 30)}, index=idx)
    out = op.calculate(ret, bench, window=20)
    assert out.iloc[-1, 0] == pytest.approx(out.iloc[-1, 0])
    assert pd.notna(out.iloc[-1, 0])


def test_aroon_insufficient_data_is_nan(_loaded):
    op = OperatorRegistry.get("AROON")
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    out = op.calculate(close, window=25)
    assert out.isna().all().all()


def test_real_turnover_rate_zero_float_shares(_loaded):
    op = OperatorRegistry.get("real_turnover_rate")
    vol = pd.DataFrame({"A": [1000.0]})
    shares = pd.DataFrame({"A": [0.0]})
    out = op.calculate(vol, shares)
    assert pd.isna(out.iloc[0, 0])


def test_removed_bfill_has_no_runtime(_loaded):
    assert OperatorRegistry.get("bfill") is None


def test_downside_beta_requires_benchmark(_loaded):
    op = OperatorRegistry.get("downside_beta")
    ret = pd.DataFrame({"A": np.linspace(0.01, 0.05, 20)})
    with pytest.raises(ValueError, match="benchmark"):
        op.calculate(ret, window=10)


def test_open_gap(_loaded):
    op = OperatorRegistry.get("open_gap")
    open_ = pd.DataFrame({"A": [110.0, 105.0]})
    close = pd.DataFrame({"A": [100.0, 110.0]})
    out = op.calculate(open_, close)
    assert pd.isna(out.iloc[0, 0])
    assert out.iloc[1, 0] == pytest.approx(105.0 / 100.0 - 1.0)


def test_cs_mad_zscore_robust(_loaded):
    op = OperatorRegistry.get("cs_mad_zscore")
    x = pd.DataFrame({"A": [1.0, 100.0], "B": [2.0, 200.0], "C": [3.0, 300.0]})
    out = op.calculate(x)
    assert out.loc[x.index[0], "A"] == pytest.approx(-1.0)
    assert out.loc[x.index[0], "C"] == pytest.approx(1.0)


def test_capm_param_contracts_complete(_loaded):
    from factor_engine.cleaned_operators.operator_spec import check_capm_param_contracts

    assert not check_capm_param_contracts(), check_capm_param_contracts()


def test_micro_realized_vol_respects_session(_loaded):
    from factor_engine.cleaned_operators.microstructure.session import pct_change_by_session

    idx = pd.DatetimeIndex(["2024-01-02 09:31", "2024-01-02 09:32", "2024-01-03 09:31"])
    s = pd.Series([100.0, 101.0, 200.0], index=idx)
    out = pct_change_by_session(s)
    assert pd.isna(out.iloc[2])
