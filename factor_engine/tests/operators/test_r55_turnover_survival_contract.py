"""Turnover survival requires sufficiently observed traded mass, not weaker guards."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = [
    "ts_turnover_age_dispersion", "ts_turnover_cost_dispersion",
    "ts_turnover_cost_entropy", "ts_turnover_cost_entropy_vol_scaled",
    "ts_turnover_cost_mode_distance", "ts_turnover_cost_quantile_distance",
    "ts_turnover_cost_skew", "ts_turnover_holding_age",
    "ts_turnover_near_cost_mass", "ts_turnover_profit_share",
    "ts_turnover_reference_price", "ts_turnover_old_mass",
]

@pytest.mark.parametrize("canonical", NAMES)
@pytest.mark.parametrize("time_name", ["date", "timestamp", "__fe_time__", "trade_date"])
def test_turnover_survival_public_backends_and_strict_past_identity(canonical, time_name):
    load_all()
    index = pd.date_range("2024-01-01", periods=48)
    t = np.arange(48, dtype=float)
    price = pd.DataFrame({"B": 100 + .4*t + np.sin(t), "A": 80 + .3*t + np.cos(t)}, index=index)
    turnover = pd.DataFrame(.2, index=index, columns=price.columns)
    reference = OperatorRegistry.get(canonical, backend="pandas_numpy")
    native = OperatorRegistry.get(canonical, backend="polars")
    expected = reference.calculate(price, turnover, window=20)
    assert np.isfinite(expected.to_numpy()).any(), canonical
    # exp(-20*.2) < .10: observed turnover accounts for enough historical mass.
    assert np.exp(-20*.2) < .1
    plprice = pl.from_pandas(price.rename_axis(time_name).reset_index())
    plturnover = pl.from_pandas(turnover.rename_axis(time_name).reset_index())
    actual = native.calculate(plprice, plturnover, window=20)
    assert actual.columns == [time_name, "B", "A"]
    assert actual[time_name].to_list() == list(index.to_pydatetime())
    np.testing.assert_allclose(actual.select("B","A").to_numpy(), expected.to_numpy(), equal_nan=True)
    prefix = reference.calculate(price.iloc[:30], turnover.iloc[:30], window=20)
    pd.testing.assert_frame_equal(prefix, expected.iloc[:30])
    if canonical == "ts_turnover_reference_price":
        past = price["A"].iloc[-21:-1].to_numpy()
        weight = -np.expm1(-.2) * np.exp(-.2*np.arange(19,-1,-1))
        exact = np.sum(weight * past)/np.sum(weight)
        assert expected["A"].iloc[-1] == pytest.approx(exact)
        altered = price.copy()
        altered.iloc[-1,:] = 1e6
        unchanged = reference.calculate(altered, turnover, window=20)
        assert unchanged["A"].iloc[-1] == pytest.approx(exact)


@pytest.mark.parametrize("time_name", ["date", "timestamp", "__fe_time__", "trade_date"])
def test_cpt_preserves_all_standard_time_axes(time_name):
    load_all()
    index = pd.date_range("2024-01-01", periods=30)
    values = pd.DataFrame({"A": .03*np.sin(np.arange(30)), "B": .02*np.cos(np.arange(30))}, index=index)
    expected = OperatorRegistry.get("ts_cpt_value", "pandas_numpy").calculate(values, window=20)
    actual = OperatorRegistry.get("ts_cpt_value", "polars").calculate(
        pl.from_pandas(values.rename_axis(time_name).reset_index()), window=20)
    assert actual[time_name].to_list() == list(index.to_pydatetime())
    assert np.isfinite(expected.to_numpy()).any()
    np.testing.assert_allclose(actual.select("A","B").to_numpy(), expected.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("identity", ["stock_code", "instrument", "symbol", "inst"])
@pytest.mark.parametrize("canonical", ["ts_turnover_reference_price", "ts_cpt_value"])
def test_rolling_chip_kernels_reject_mixed_instrument_long_panels(identity, canonical):
    load_all()
    frame = pl.DataFrame({"timestamp": [1,1,2,2], identity: ["A","B","A","B"], "value": [1.,2.,3.,4.]})
    args = [frame, frame] if canonical.startswith("ts_turnover") else [frame]
    with pytest.raises(ValueError, match="multi-stock long input"):
        OperatorRegistry.get(canonical, "polars").calculate(*args, window=20)


def test_turnover_rejects_reordered_price_turnover_identity():
    load_all()
    frame = pl.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=20), "A": np.arange(20.)+100})
    with pytest.raises(ValueError, match="different PanelIdentity"):
        OperatorRegistry.get("ts_turnover_reference_price", "polars").calculate(frame, frame.reverse(), window=20)


@pytest.mark.parametrize("canonical", NAMES)
@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_turnover_window_cannot_be_shorter_than_five_observation_support(canonical, backend):
    load_all()
    index = pd.date_range("2024-01-01", periods=12)
    price = pd.DataFrame({"A": 100 + np.arange(12.)}, index=index)
    turns = pd.DataFrame(.5, index=index, columns=["A"])
    args = [price, turns]
    if backend == "polars":
        args = [pl.from_pandas(p.rename_axis("date").reset_index()) for p in args]
    op = OperatorRegistry.get(canonical, backend)
    for window in (2,3,4):
        with pytest.raises(ValueError, match="window"):
            op.calculate(*args, window=window)
    valid = op.calculate(*args, window=5)
    assert np.isfinite(valid["A"].to_numpy()).any()
