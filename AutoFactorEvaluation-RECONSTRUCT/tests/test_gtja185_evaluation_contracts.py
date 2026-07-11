from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.gtja185_batch import (
    BatchEvaluationConfig,
    SplitBoundaries,
    _apply_point_in_time_universe,
    _long_short_series,
    _price_forward_returns,
    _purged_split_mask,
)


def _price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=8)
    return pd.DataFrame(
        {
            "datetime": dates,
            "asset": "A",
            "vwap": np.arange(100.0, 108.0),
        }
    )


def test_forward_return_enters_on_next_bar():
    frame = _price_frame()
    result = _price_forward_returns(frame, (2,), "vwap", entry_lag=1)[2]
    # signal at t0 enters at t1=101 and exits at t3=103
    assert result.iloc[0] == pytest.approx(103.0 / 101.0 - 1.0)
    assert result.iloc[-3:].isna().all()


def test_purged_split_never_crosses_boundary():
    dates = pd.bdate_range("2024-01-02", periods=10)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "asset"])
    bounds = SplitBoundaries(
        first_date=str(dates[0].date()),
        train_end=str(dates[4].date()),
        valid_end=str(dates[7].date()),
        last_date=str(dates[-1].date()),
    )
    train = _purged_split_mask(index, "train", bounds, dates, entry_lag=1, horizon=2)
    valid = _purged_split_mask(index, "valid", bounds, dates, entry_lag=1, horizon=2)
    test = _purged_split_mask(index, "test", bounds, dates, entry_lag=1, horizon=2)
    assert list(np.flatnonzero(train)) == [0, 1]
    assert list(np.flatnonzero(valid)) == []
    assert list(np.flatnonzero(test)) == []


def test_turnover_uses_weights_and_costs_net_returns():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C", "D"]],
        names=["datetime", "asset"],
    )
    signal = pd.Series([-2, -1, 1, 2] * 2, index=index, dtype=float)
    returns = pd.Series([0.0, 0.0, 0.01, 0.01] * 2, index=index, dtype=float)
    gross, net, turnover = _long_short_series(
        signal,
        returns,
        min_assets=4,
        n_quantiles=2,
        cost_bps=10.0,
    )
    assert gross.iloc[0] == pytest.approx(0.01)
    assert gross.iloc[1] == pytest.approx(0.01)
    assert turnover.iloc[0] == pytest.approx(1.0)
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert net.iloc[0] == pytest.approx(gross.iloc[0] - 0.001)
    assert net.iloc[1] == pytest.approx(gross.iloc[1])


def test_point_in_time_universe_filters_by_date_and_tradability():
    market = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2),
            "asset": ["A", "B", "A", "B"],
            "close": [1.0, 2.0, 1.1, 2.1],
        }
    )
    universe = pd.DataFrame(
        {
            "datetime": market["datetime"],
            "asset": market["asset"],
            "is_member": [True, False, True, True],
            "is_tradable": [True, True, False, True],
        }
    )
    filtered = _apply_point_in_time_universe(market, universe)
    assert filtered[["datetime", "asset"]].to_records(index=False).tolist() == [
        (np.datetime64("2024-01-02T00:00:00.000000000"), "A"),
        (np.datetime64("2024-01-03T00:00:00.000000000"), "B"),
    ]


def test_production_requires_registered_point_in_time_universe():
    with pytest.raises(ValueError, match="require_point_in_time_universe"):
        BatchEvaluationConfig(run_mode="production").validate()
    BatchEvaluationConfig(
        run_mode="production",
        require_point_in_time_universe=True,
    ).validate()
