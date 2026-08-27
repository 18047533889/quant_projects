# -*- coding: utf-8 -*-
"""Operator-correctness audit fixes — regression tests.

Each test calls an operator with its CANONICAL keyword parameter names (the
surface contract that mining/Agent emits) and asserts the result matches the
pandas reference (or the mathematically correct value).  These guard the P0
keyword-call-rejection and P1 surface-drift fixes.

Backends exercised:
  - P0-1 power (polars)
  - P0-2 where (polars)
  - P0-3 rolling_beta_to_market (polars)
  - P0-4 ratio family (pandas)
  - P0-5 intra_segment_*_share (pandas)
  - P1 open_*_return (polars)
  - P1 fin_seasonal_* (polars)
  - P1 group_rank_weighted_value.fallback_policy (polars)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _pd_panel(values, col="A"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({col: list(values)}, index=idx)


def _pl_panel(values, col="A"):
    pl = pytest.importorskip("polars")
    return pl.DataFrame({col: [float(v) for v in values]})


def _get(name, backend):
    ensure_cleaned_loaded()
    return OperatorRegistry.get(name, backend=backend)


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    ensure_cleaned_loaded()


def _assert_polars_matches_pandas(name, pandas_args, operator_args=(), kw_polars_args=None):
    """Call pandas ref with pandas_args positionally, call polars with the
    SAME values but bound via canonical KEYWORD names, and assert parity.
    kw_polars_args overrides specific canonical kwargs (e.g. for polars-only
    scalar handling) when provided."""
    pd_ref = _get(name, "pandas_numpy")
    polars_op = _get(name, "polars")
    if pd_ref is None or polars_op is None:
        pytest.skip(f"{name} missing pandas_numpy/polars backend")
    kwargs = dict(kw_polars_args or {})
    pandas_result = pd_ref.calculate(*pandas_args, *operator_args)
    # Rebind panels to keyword form on the polars backend.
    if not kwargs:
        kw = dict(zip(pd_ref.metadata.param_names[: len(pandas_args)], pandas_args))
        kwargs.update({
            k: (_pl_panel(v[v.columns[0]], col=v.columns[0]) if isinstance(v, pd.DataFrame) else v)
            for k, v in kw.items()
        })
    # Scalar operator args are bound as canonical keywords (positional order on
    # the pandas ref matches declared param order after the panels).
    scalar_kw = dict(zip(pd_ref.metadata.param_names[len(pandas_args):], operator_args))
    kw_after = dict(kwargs)
    kw_after.update(scalar_kw)
    polars_result = polars_op.calculate(**kw_after)
    cols = pandas_result.columns.tolist()
    np.testing.assert_allclose(
        polars_result.select(cols).to_numpy(),
        pandas_result.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-8,
        atol=1e-8,
    )


class TestPowerPolars:
    def test_keyword_y_is_honored(self):
        # P0-1: power(x, y=3.0) must be x**3, not the default x**2.
        x = _pl_panel([2.0, 3.0, 4.0])
        out = _get("power", "polars").calculate(x, y=3.0)
        np.testing.assert_allclose(out.select("A").to_numpy()[:, 0], [8.0, 27.0, 64.0])

    def test_matches_pandas_reference(self):
        x = _pd_panel([2.0, 3.0, 4.0, 5.0])
        _assert_polars_matches_pandas("power", (x,), operator_args=(3.0,))


class TestWherePolars:
    def test_canonical_keywords(self):
        cond = _pl_panel([1.0, 0.0, 2.0])
        x = _pl_panel([10.0, 20.0, 30.0])
        y = _pl_panel([100.0, 200.0, 300.0])
        out = _get("where", "polars").calculate(condition=cond, x=x, y=y)
        np.testing.assert_allclose(out.select("A").to_numpy()[:, 0], [10.0, 200.0, 30.0])

    def test_matches_pandas_reference(self):
        cond = _pd_panel([1.0, 0.0, 2.0, 0.0])
        x = _pd_panel([10.0, 20.0, 30.0, 40.0])
        y = _pd_panel([100.0, 200.0, 300.0, 400.0])
        _assert_polars_matches_pandas("where", (cond, x, y))


class TestRollingBetaToMarketPolars:
    def test_window_3_no_min_periods_raise(self):
        # P0-3: canonical (ret, benchmark_ret, window); window=3 must not raise
        # the inherited min_periods=5 > window error.
        ret = _pl_panel([1.0, 2.0, 3.0, 4.0, 5.0])
        bench = _pl_panel([1.0, 1.0, 1.0, 1.0, 1.0])
        out = _get("rolling_beta_to_market", "polars").calculate(
            ret=ret, benchmark_ret=bench, window=3
        )
        assert out.shape[0] == 5

    def test_matches_pandas_reference(self):
        ret = _pd_panel([1.0, 2.0, 3.0, 4.0, 5.0])
        bench = _pd_panel([2.0, 1.0, 4.0, 3.0, 5.0])
        _assert_polars_matches_pandas("rolling_beta_to_market", (ret, bench), operator_args=(3,))


class TestRatioFamilyPandas:
    @pytest.mark.parametrize("name,kwargs,expected", [
        ("float_share_ratio", {"float_shares": 2.0, "total_shares": 4.0}, 0.5),
        ("free_float_share_ratio", {"free_float_shares": 1.0, "total_shares": 4.0}, 0.25),
        ("true_turnover_rate", {"volume": 10.0, "free_float_shares": 4.0}, 2.5),
        ("benchmark_relative_price", {"price": 3.0, "benchmark_price": 6.0}, 0.5),
    ])
    def test_canonical_keyword_call(self, name, kwargs, expected):
        op = _get(name, "pandas_numpy")
        panels = {
            k: _pd_panel([v]) for k, v in kwargs.items()
        }
        out = op.calculate(**panels)
        assert out.iloc[0]["A"] == pytest.approx(expected)

    def test_matches_pandas_reference(self):
        # Positional vs canonical-keyword must agree (same underlying kernel).
        op = _get("float_share_ratio", "pandas_numpy")
        a = _pd_panel([2.0, 4.0, 8.0])
        b = _pd_panel([4.0, 8.0, 16.0])
        pos = op.calculate(a, b)
        kw = op.calculate(float_shares=a, total_shares=b)
        np.testing.assert_allclose(kw.to_numpy(), pos.to_numpy(), equal_nan=True)


class TestIntraSegmentSharePandas:
    def _minute_panel(self):
        idx = pd.date_range("2020-01-01 09:30:00", periods=60, freq="min")
        return pd.DataFrame({"A": np.arange(1.0, 61.0)}, index=idx)

    def test_volume_canonical_keyword(self):
        op = _get("intra_segment_volume_share", "pandas_numpy")
        if op is None:
            pytest.skip("intra_segment_volume_share not registered")
        out = op.calculate(volume=self._minute_panel(), segment="morning", market="ashare_default")
        assert out.shape[0] == 1

    def test_amount_canonical_keyword(self):
        op = _get("intra_segment_amount_share", "pandas_numpy")
        if op is None:
            pytest.skip("intra_segment_amount_share not registered")
        out = op.calculate(amount=self._minute_panel(), segment="morning", market="ashare_default")
        assert out.shape[0] == 1


class TestOpenReturnPolars:
    @pytest.mark.parametrize("name", [
        "open_close_return", "open_to_vwap_return", "overnight_return",
    ])
    def test_canonical_open_px_keyword(self, name):
        op = _get(name, "polars")
        if op is None:
            pytest.skip(f"{name} not registered on polars")
        open_px = _pl_panel([1.0, 2.0, 3.0])
        other = _pl_panel([2.0, 3.0, 4.0])
        second = "close" if name == "open_close_return" else (
            "vwap" if name == "open_to_vwap_return" else "pre_close")
        out = op.calculate(open_px=open_px, **{second: other})
        assert out.shape[0] == 3

    def test_open_close_matches_pandas(self):
        # pandas ref enforces a shared price basis (R11/P0-60), so use
        # concept-named columns and identical column sets on both panels.
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        open_px = pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=idx)
        close = pd.DataFrame({"open": [2.0, 4.0, 6.0]}, index=idx)
        _assert_polars_matches_pandas("open_close_return", (open_px, close))


class TestFinSeasonalPolars:
    @pytest.mark.parametrize("name", ["fin_seasonal_zscore", "fin_seasonal_percentile"])
    def test_canonical_period_end_keyword(self, name):
        op = _get(name, "polars")
        if op is None:
            pytest.skip(f"{name} not registered on polars")
        x = _pl_panel([1.0, 2.0, 3.0, 4.0, 5.0])
        period_end = _pl_panel([1.0, 2.0, 3.0, 4.0, 5.0])
        fq = _pl_panel([1.0, 1.0, 1.0, 1.0, 1.0])
        out = op.calculate(x=x, period_end=period_end, fiscal_quarter=fq)
        assert out.shape[0] == 5


class TestGroupRankWeightedFallback:
    def test_fallback_policy_keep_original_live(self):
        # P1: fallback_policy is a declared param; a whole-missing group row
        # with keep_original must return x, not raise/reject the keyword.
        x = _pl_panel([1.0, 2.0, 3.0])
        group = _pl_panel([float("nan"), float("nan"), float("nan")])
        out = _get("group_rank_weighted_value", "polars").calculate(
            x=x, group=group, fallback_policy="keep_original"
        )
        np.testing.assert_allclose(out.select("A").to_numpy()[:, 0], [1.0, 2.0, 3.0])

    def test_fallback_policy_nan_default(self):
        x = _pl_panel([1.0, 2.0, 3.0])
        group = _pl_panel([float("nan"), float("nan"), float("nan")])
        out = _get("group_rank_weighted_value", "polars").calculate(x=x, group=group)
        assert np.isnan(out.select("A").to_numpy()[:, 0]).all()

    def test_global_no_group(self):
        x = _pl_panel([1.0, 2.0, 3.0])
        out = _get("group_rank_weighted_value", "polars").calculate(
            x=x, group=None, fallback_policy="global"
        )
        assert np.isfinite(out.select("A").to_numpy()[:, 0]).all()
