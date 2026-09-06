import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.common.time_series import MovingBeta
from factor_engine.cleaned_operators.price_volume.polars_price_volume import (
    DownsideBetaPolars,
    RollingBetaPolars,
    RollingBetaToMarketPolars,
    TSBetaPolars,
    TailBetaPolars,
    _conditional_beta,
)


@pytest.mark.parametrize("n", [3, 5, 20])
def test_conditional_beta_affine_oracle_and_finite_cohort(n):
    x = -np.arange(float(n), 0.0, -1.0)
    y = 7.0 + 2.0 * x
    assert _conditional_beta(y, x, n, mode="downside")[-1] == pytest.approx(2.0)
    x_bad, y_bad = x.copy(), y.copy()
    x_bad[n // 2] = np.inf
    y_bad[n // 2] = -np.inf
    got = _conditional_beta(y_bad, x_bad, n, mode="downside")[-1]
    if n - 1 >= 3:
        assert got == pytest.approx(2.0)
    else:
        assert np.isnan(got)


def test_tail_beta_normal_quantile_and_ties():
    x = np.arange(-100.0, 0.0)
    y = 3.0 + 2.0 * x
    assert _conditional_beta(y, x, 100, mode="tail", q=0.05)[-1] == pytest.approx(2.0)
    tied = np.array([-2.0, -2.0, -2.0, -1.0, 0.0])
    got = _conditional_beta(4.0 * tied + 1.0, tied, 5, mode="tail", q=0.5)
    assert np.isnan(got[-1])  # selected ties have zero benchmark variance


def test_conditional_beta_zero_variance_is_nan():
    x = -np.ones(5)
    assert np.isnan(_conditional_beta(np.arange(5.0), x, 5, mode="downside")[-1])


def _panel(values):
    return pl.DataFrame({"a": values})


@pytest.mark.parametrize(
    "cls", [TSBetaPolars, RollingBetaPolars, RollingBetaToMarketPolars]
)
def test_rolling_beta_aliases_use_paired_finite_cohort(cls):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 3.0 * x + 2.0
    y[2] = np.nan
    x[4] = np.inf
    got = cls()._calculate_series(
        _panel(y), _panel(x), window=6, min_periods=3
    )["a"].to_numpy()
    assert got[-1] == pytest.approx(3.0)


def test_rolling_beta_single_benchmark_column_broadcasts():
    ret = pl.DataFrame({"a": [3.0, 5.0, 7.0], "b": [-1.0, -4.0, -7.0]})
    benchmark = pl.DataFrame({"market": [1.0, 2.0, 3.0]})
    got = RollingBetaPolars()._calculate_series(
        ret, benchmark, window=3, min_periods=2
    )
    assert got["a"][-1] == pytest.approx(2.0)
    assert got["b"][-1] == pytest.approx(-3.0)


def test_beta_inputs_align_shuffled_dates_and_reject_duplicate_keys():
    dates = pl.Series("date", pd.date_range("2024-01-01", periods=4))
    ret = pl.DataFrame({"date": dates, "a": [-5.0, -3.0, -1.0, 3.0]})
    benchmark = pl.DataFrame({"date": dates, "a": [-3.0, -2.0, -1.0, 1.0]})
    shuffled = benchmark.reverse()
    rolling = RollingBetaPolars()._calculate_series(
        ret, shuffled, window=4, min_periods=3
    )
    assert rolling["a"][-1] == pytest.approx(2.0)
    downside = DownsideBetaPolars()._calculate_series(ret, shuffled, window=4)
    assert downside["a"][-1] == pytest.approx(2.0)

    duplicate = benchmark.with_columns(
        pl.Series("date", [dates[0], dates[0], dates[2], dates[3]])
    )
    with pytest.raises(OperatorParameterError, match="unique date"):
        RollingBetaPolars()._calculate_series(
            ret, duplicate, window=4, min_periods=3
        )


@pytest.mark.parametrize(
    "op",
    [DownsideBetaPolars(), TailBetaPolars()],
)
def test_conditional_beta_strict_parameters(op):
    args = (_panel([-3.0, -2.0, -1.0]), _panel([-3.0, -2.0, -1.0]))
    with pytest.raises(OperatorParameterError):
        op._calculate_series(*args, window=3.5)
    with pytest.raises(OperatorParameterError, match="unknown"):
        op._calculate_series(*args, mystery=1)
    with pytest.raises(OperatorParameterError, match="min_stop"):
        op._calculate_series(*args, min_stop=2)


@pytest.mark.parametrize("q", [0.0, 1.0, np.inf, np.nan, True])
def test_tail_beta_q_is_finite_open_probability(q):
    op = TailBetaPolars()
    with pytest.raises(OperatorParameterError):
        op._calculate_series(
            _panel([-3.0, -2.0, -1.0]),
            _panel([-3.0, -2.0, -1.0]),
            window=3,
            q=q,
        )


def test_ts_beta_min_stop_is_explicit_migration_error():
    y = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(OperatorParameterError, match="min_stop.*min_periods"):
        MovingBeta()._calculate_series(y, y, window=3, min_stop=2)
    with pytest.raises(OperatorParameterError, match="min_stop.*min_periods"):
        TSBetaPolars()._calculate_series(
            _panel([1.0, 2.0, 3.0]),
            _panel([1.0, 2.0, 3.0]),
            window=3,
            min_stop=2,
        )
