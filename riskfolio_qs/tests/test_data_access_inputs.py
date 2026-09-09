from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskfolio_qs.adapters.data_access_inputs import (
    load_historical_market_returns,
    load_portfolio_inputs,
)


class _Table:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


class _Store:
    def __init__(self, benchmark: pd.DataFrame, daily: pd.DataFrame) -> None:
        self.benchmark = benchmark
        self.daily = daily
        self.calls: list[tuple[str, dict[str, object]]] = []

    def read_result(self, dataset: str, **kwargs):
        self.calls.append((dataset, kwargs))
        frame = self.benchmark if dataset == "ashare_index_constituent" else self.daily
        snapshot = SimpleNamespace(
            snapshot_id=f"{dataset}-snapshot",
            registry_hash="registry",
            schema_hash="schema",
            file_manifest_hash="files",
        )
        stats = SimpleNamespace(rows=len(frame), bytes=int(frame.memory_usage().sum()))
        return SimpleNamespace(table=_Table(frame), snapshot=snapshot, stats=stats)


class _HistoricalStore:
    def __init__(self, calendar: pd.DataFrame, daily: pd.DataFrame) -> None:
        self.frames = {
            "ashare_calendar": calendar,
            "ashare_stock_daily": daily,
        }
        self.calls: list[tuple[str, dict[str, object]]] = []

    def read_result(self, dataset: str, **kwargs):
        self.calls.append((dataset, kwargs))
        frame = self.frames[dataset]
        snapshot = SimpleNamespace(
            snapshot_id=f"{dataset}-snapshot",
            registry_hash="registry",
            schema_hash="schema",
            file_manifest_hash="files",
        )
        stats = SimpleNamespace(rows=len(frame), bytes=int(frame.memory_usage().sum()))
        return SimpleNamespace(table=_Table(frame), snapshot=snapshot, stats=stats)


def test_data_access_builds_normalized_benchmark_and_conservative_tradable() -> None:
    date = pd.Timestamp("2025-12-24")
    alpha = pd.DataFrame(
        [[0.1, 0.2, 0.3]],
        index=[date],
        columns=["A", "B", "C"],
    )
    benchmark = pd.DataFrame(
        {
            "TradeDate": [date, date, date, date],
            "IndexSymbol": ["000905.SH", "000905.SH", "000905.SH", "000300.SH"],
            "Symbol": ["A", "B", "X", "A"],
            "Weight": [60.0, 39.0, 1.0, 100.0],
        }
    )
    daily = pd.DataFrame(
        {
            "TradeDate": [date, date],
            "Symbol": ["A", "B"],
            "Volume": [100, 0],
            "Amount": [1_000.0, 0.0],
        }
    )
    store = _Store(benchmark, daily)

    result = load_portfolio_inputs(
        alpha,
        benchmark_index="000905.SH",
        minimum_benchmark_weight_coverage=0.95,
        store=store,
    )

    np.testing.assert_allclose(result.benchmark.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        result.benchmark.iloc[0].to_numpy(),
        np.array([60.0 / 99.0, 39.0 / 99.0, 0.0]),
    )
    assert result.tradable.iloc[0].to_dict() == {
        "A": True,
        "B": False,
        "C": False,
    }
    assert result.provenance["benchmark"]["weight_coverage_by_date"] == {
        "2025-12-24": 0.99
    }
    assert store.calls[1][1]["instrument_filter"] == ["A", "B", "C"]


def test_data_access_rejects_low_benchmark_coverage() -> None:
    date = pd.Timestamp("2025-12-24")
    alpha = pd.DataFrame([[0.1]], index=[date], columns=["A"])
    benchmark = pd.DataFrame(
        {
            "TradeDate": [date, date],
            "IndexSymbol": ["000905.SH", "000905.SH"],
            "Symbol": ["A", "X"],
            "Weight": [10.0, 90.0],
        }
    )
    daily = pd.DataFrame(
        {
            "TradeDate": [date],
            "Symbol": ["A"],
            "Volume": [100],
            "Amount": [1_000.0],
        }
    )

    with pytest.raises(ValueError, match="does not retain enough benchmark weight"):
        load_portfolio_inputs(
            alpha,
            benchmark_index="000905.SH",
            store=_Store(benchmark, daily),
        )


def test_historical_market_loader_scales_return_and_preserves_missing() -> None:
    trade_dates = pd.bdate_range("2023-12-01", "2024-04-02")
    alpha_dates = pd.DatetimeIndex(["2024-04-01", "2024-04-02"])
    alpha = pd.DataFrame(
        [[0.1, 0.2], [0.2, 0.1]],
        index=alpha_dates,
        columns=["A", "B"],
    )
    calendar = pd.DataFrame(
        {"TradeDate": trade_dates, "IsTradeDay": True}
    )
    daily = pd.DataFrame(
        [
            {
                "TradeDate": date,
                "Symbol": asset,
                "Return": value,
            }
            for date in trade_dates[-25:]
            for asset, value in (("A", 100.0), ("B", -50.0))
            if not (date == trade_dates[-3] and asset == "B")
        ]
    )
    store = _HistoricalStore(calendar, daily)

    result = load_historical_market_returns(
        alpha,
        lookback_days=20,
        market_data_lag_periods=1,
        store=store,
    )

    assert result.returns.loc[trade_dates[-1], "A"] == pytest.approx(0.01)
    assert result.returns.loc[trade_dates[-1], "B"] == pytest.approx(-0.005)
    assert pd.isna(result.returns.loc[trade_dates[-3], "B"])
    assert result.provenance["market"]["input_type"] == "return"
    assert result.provenance["market"]["scale"] == pytest.approx(0.0001)
    assert store.calls[1][1]["instrument_filter"] == ["A", "B"]
