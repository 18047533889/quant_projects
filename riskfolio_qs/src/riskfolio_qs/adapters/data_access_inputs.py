"""Production portfolio inputs sourced from the team's ``data_access`` layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


BENCHMARK_DATASET = "ashare_index_constituent"
TRADABLE_DATASET = "ashare_stock_daily"
HISTORICAL_MARKET_DATASET = "ashare_stock_daily"
CALENDAR_DATASET = "ashare_calendar"
TRADABLE_RULE = "Volume > 0 and Amount > 0; missing rows are not tradable"
HISTORICAL_RETURN_RULE = "StockDailyBar.Return / 10000"
DEFAULT_LINEAR_COST_BPS = 5.0


@dataclass(frozen=True)
class DataAccessPortfolioInputs:
    """Matrices and audit information resolved from ``data_access``."""

    benchmark: pd.DataFrame | None
    tradable: pd.DataFrame
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DataAccessHistoricalMarket:
    """Point-in-time daily returns used by historical covariance modes."""

    returns: pd.DataFrame
    provenance: dict[str, Any]


def _result_frame(result: Any, dataset: str) -> pd.DataFrame:
    try:
        frame = result.table.to_pandas()
    except AttributeError as exc:
        raise TypeError(
            f"data_access result for {dataset!r} does not expose table.to_pandas()"
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"data_access result for {dataset!r} is not tabular")
    return frame


def _snapshot_provenance(result: Any) -> dict[str, Any]:
    snapshot = result.snapshot
    stats = result.stats
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "registry_hash": str(snapshot.registry_hash),
        "schema_hash": str(snapshot.schema_hash),
        "file_manifest_hash": str(snapshot.file_manifest_hash),
        "rows": int(stats.rows),
        "bytes": int(stats.bytes),
    }


def _resolve_store(store: Any | None) -> Any:
    if store is not None:
        return store
    try:
        from data_access import get_store
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "riskfolio_qs CLI requires the internal data_access package; "
            "install data_access and its runtime dependencies"
        ) from exc
    return get_store()


def load_portfolio_inputs(
    alpha: pd.DataFrame,
    *,
    benchmark_index: str | None,
    minimum_benchmark_weight_coverage: float = 0.95,
    include_benchmark: bool = True,
    store: Any | None = None,
) -> DataAccessPortfolioInputs:
    """Build benchmark and tradability matrices for an alpha matrix.

    The dataset names and tradability rule are intentionally code-owned.  The
    CLI is an internal production interface and does not accept substitute
    benchmark/tradability files.
    """

    if alpha.empty or len(alpha.columns) == 0:
        raise ValueError("alpha must not be empty")
    benchmark_index = str(benchmark_index or "").strip()
    if include_benchmark and not benchmark_index:
        raise ValueError("adapter.benchmark_index must be non-empty")
    coverage_floor = float(minimum_benchmark_weight_coverage)
    if not 0.0 < coverage_floor <= 1.0:
        raise ValueError(
            "adapter.minimum_benchmark_weight_coverage must be in (0, 1]"
        )

    dates = pd.DatetimeIndex(pd.to_datetime(alpha.index)).tz_localize(None).normalize()
    assets = pd.Index([str(asset) for asset in alpha.columns], name="asset")
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()

    store = _resolve_store(store)
    benchmark: pd.DataFrame | None = None
    provenance: dict[str, Any] = {}
    if include_benchmark:
        benchmark_result = store.read_result(
            BENCHMARK_DATASET,
            columns=["TradeDate", "IndexSymbol", "Symbol", "Weight"],
            time_range=(start, end),
        )
        benchmark_long = _result_frame(benchmark_result, BENCHMARK_DATASET)
        required_benchmark = {"TradeDate", "IndexSymbol", "Symbol", "Weight"}
        missing = sorted(required_benchmark - set(benchmark_long.columns))
        if missing:
            raise ValueError(f"{BENCHMARK_DATASET} is missing columns: {missing}")
        benchmark_long["TradeDate"] = (
            pd.to_datetime(benchmark_long["TradeDate"])
            .dt.tz_localize(None)
            .dt.normalize()
        )
        benchmark_long["IndexSymbol"] = benchmark_long["IndexSymbol"].astype(str)
        benchmark_long["Symbol"] = benchmark_long["Symbol"].astype(str)
        benchmark_long = benchmark_long.loc[
            benchmark_long["IndexSymbol"].eq(benchmark_index)
            & benchmark_long["TradeDate"].isin(dates)
        ].copy()
        if benchmark_long.empty:
            raise ValueError(
                f"{BENCHMARK_DATASET} has no rows for index {benchmark_index!r} "
                f"between {start} and {end}"
            )
        if benchmark_long.duplicated(["TradeDate", "Symbol"]).any():
            raise ValueError(
                f"{BENCHMARK_DATASET} has duplicate TradeDate-Symbol rows for "
                f"index {benchmark_index!r}"
            )
        benchmark_long["Weight"] = pd.to_numeric(
            benchmark_long["Weight"], errors="coerce"
        )
        if (
            benchmark_long["Weight"].isna().any()
            or not np.isfinite(benchmark_long["Weight"]).all()
            or benchmark_long["Weight"].lt(0).any()
        ):
            raise ValueError(
                f"{BENCHMARK_DATASET}.Weight must be finite and non-negative"
            )

        full_weight = benchmark_long.groupby("TradeDate")["Weight"].sum().reindex(dates)
        selected_long = benchmark_long.loc[benchmark_long["Symbol"].isin(assets)]
        selected_weight = (
            selected_long.groupby("TradeDate")["Weight"]
            .sum()
            .reindex(dates)
            .fillna(0.0)
        )
        if full_weight.isna().any() or full_weight.le(0).any():
            missing_dates = [
                date.date().isoformat()
                for date in dates[full_weight.isna() | full_weight.le(0)]
            ]
            raise ValueError(
                f"{BENCHMARK_DATASET} has no positive total weight on: {missing_dates}"
            )
        coverage = selected_weight / full_weight
        below_floor = coverage.lt(coverage_floor)
        if below_floor.any():
            details = {
                date.date().isoformat(): round(float(value), 6)
                for date, value in coverage.loc[below_floor].items()
            }
            raise ValueError(
                "alpha universe does not retain enough benchmark weight for "
                f"{benchmark_index!r}; required={coverage_floor:.2%}, actual={details}"
            )
        benchmark = (
            selected_long.pivot(index="TradeDate", columns="Symbol", values="Weight")
            .reindex(index=dates, columns=assets)
            .fillna(0.0)
        )
        benchmark = benchmark.div(benchmark.sum(axis=1), axis=0)
        benchmark.index.name = "date"
        benchmark.columns.name = "asset"
        coverage_by_date = {
            date.date().isoformat(): float(value) for date, value in coverage.items()
        }
        provenance["benchmark"] = {
            "source": "data_access",
            "dataset": BENCHMARK_DATASET,
            "index_symbol": benchmark_index,
            "minimum_weight_coverage": coverage_floor,
            "weight_coverage_by_date": coverage_by_date,
            **_snapshot_provenance(benchmark_result),
        }

    tradable_result = store.read_result(
        TRADABLE_DATASET,
        columns=["TradeDate", "Symbol", "Volume", "Amount"],
        time_range=(start, end),
        instrument_filter=assets.tolist(),
    )
    daily = _result_frame(tradable_result, TRADABLE_DATASET)
    required_daily = {"TradeDate", "Symbol", "Volume", "Amount"}
    missing = sorted(required_daily - set(daily.columns))
    if missing:
        raise ValueError(f"{TRADABLE_DATASET} is missing columns: {missing}")
    daily["TradeDate"] = (
        pd.to_datetime(daily["TradeDate"]).dt.tz_localize(None).dt.normalize()
    )
    daily["Symbol"] = daily["Symbol"].astype(str)
    daily = daily.loc[
        daily["TradeDate"].isin(dates) & daily["Symbol"].isin(assets)
    ].copy()
    if daily.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError(f"{TRADABLE_DATASET} has duplicate TradeDate-Symbol rows")
    volume = pd.to_numeric(daily["Volume"], errors="coerce")
    amount = pd.to_numeric(daily["Amount"], errors="coerce")
    daily["tradable"] = volume.gt(0) & amount.gt(0)
    tradable = (
        daily.pivot(index="TradeDate", columns="Symbol", values="tradable")
        .reindex(index=dates, columns=assets)
        .eq(True)
    )
    tradable.index.name = "date"
    tradable.columns.name = "asset"

    provenance["tradable"] = {
        "source": "data_access",
        "dataset": TRADABLE_DATASET,
        "rule": TRADABLE_RULE,
        "tradable_cells": int(tradable.to_numpy().sum()),
        "total_cells": int(tradable.size),
        **_snapshot_provenance(tradable_result),
    }
    provenance["linear_cost_bps"] = {
        "source": "riskfolio_qs_internal_constant",
        "value": DEFAULT_LINEAR_COST_BPS,
    }
    return DataAccessPortfolioInputs(
        benchmark=benchmark,
        tradable=tradable,
        provenance=provenance,
    )


def load_historical_market_returns(
    alpha: pd.DataFrame,
    *,
    lookback_days: int,
    market_data_lag_periods: int = 0,
    store: Any | None = None,
) -> DataAccessHistoricalMarket:
    """Load a point-in-time return matrix with enough warm-up history.

    ``StockDailyBar.Return`` is stored in 1/10000 units.  The returned matrix
    contains decimal daily returns and is deliberately labelled as returns;
    callers must not apply ``pct_change`` to it again.
    """

    if alpha.empty or len(alpha.columns) == 0:
        raise ValueError("alpha must not be empty")
    lookback_days = int(lookback_days)
    market_data_lag_periods = int(market_data_lag_periods)
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least 2")
    if market_data_lag_periods < 0:
        raise ValueError("market_data_lag_periods must be non-negative")

    dates = pd.DatetimeIndex(pd.to_datetime(alpha.index)).tz_localize(None).normalize()
    assets = pd.Index([str(asset) for asset in alpha.columns], name="asset")
    first_date = dates.min()
    last_date = dates.max()
    required_history = lookback_days + market_data_lag_periods + 1
    calendar_start = first_date - timedelta(
        days=max(370, required_history * 3)
    )
    store = _resolve_store(store)

    calendar_result = store.read_result(
        CALENDAR_DATASET,
        columns=["TradeDate", "IsTradeDay"],
        time_range=(
            calendar_start.date().isoformat(),
            last_date.date().isoformat(),
        ),
    )
    calendar = _result_frame(calendar_result, CALENDAR_DATASET)
    required_calendar = {"TradeDate", "IsTradeDay"}
    missing = sorted(required_calendar - set(calendar.columns))
    if missing:
        raise ValueError(f"{CALENDAR_DATASET} is missing columns: {missing}")
    calendar["TradeDate"] = (
        pd.to_datetime(calendar["TradeDate"]).dt.tz_localize(None).dt.normalize()
    )
    trade_dates = pd.DatetimeIndex(
        calendar.loc[calendar["IsTradeDay"].eq(True), "TradeDate"].unique()
    ).sort_values()
    available_history = trade_dates[trade_dates <= first_date]
    if len(available_history) < required_history:
        raise ValueError(
            f"{CALENDAR_DATASET} provides only {len(available_history)} trade "
            f"dates through {first_date.date()}, need {required_history}"
        )
    load_start = pd.Timestamp(available_history[-required_history])
    load_dates = trade_dates[
        (trade_dates >= load_start) & (trade_dates <= last_date)
    ]

    market_result = store.read_result(
        HISTORICAL_MARKET_DATASET,
        columns=["TradeDate", "Symbol", "Return"],
        time_range=(
            load_start.date().isoformat(),
            last_date.date().isoformat(),
        ),
        instrument_filter=assets.tolist(),
    )
    daily = _result_frame(market_result, HISTORICAL_MARKET_DATASET)
    required_daily = {"TradeDate", "Symbol", "Return"}
    missing = sorted(required_daily - set(daily.columns))
    if missing:
        raise ValueError(f"{HISTORICAL_MARKET_DATASET} is missing columns: {missing}")
    daily["TradeDate"] = (
        pd.to_datetime(daily["TradeDate"]).dt.tz_localize(None).dt.normalize()
    )
    daily["Symbol"] = daily["Symbol"].astype(str)
    daily = daily.loc[
        daily["TradeDate"].isin(load_dates) & daily["Symbol"].isin(assets)
    ].copy()
    if daily.duplicated(["TradeDate", "Symbol"]).any():
        raise ValueError(
            f"{HISTORICAL_MARKET_DATASET} has duplicate TradeDate-Symbol rows"
        )
    daily["Return"] = pd.to_numeric(daily["Return"], errors="coerce") / 10000.0
    finite_returns = daily["Return"].dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_returns).all():
        raise ValueError(f"{HISTORICAL_MARKET_DATASET}.Return contains infinity")

    returns = (
        daily.pivot(index="TradeDate", columns="Symbol", values="Return")
        .reindex(index=load_dates, columns=assets)
        .astype(float)
    )
    returns.index.name = "date"
    returns.columns.name = "asset"
    missing_assets = returns.columns[returns.notna().sum(axis=0).eq(0)].tolist()
    if missing_assets:
        raise ValueError(
            "historical returns have no observations for assets: "
            f"{missing_assets[:5]}"
        )
    missing_alpha_dates = dates[~dates.isin(returns.index)]
    if len(missing_alpha_dates):
        raise ValueError(
            "historical returns calendar is missing alpha dates: "
            f"{[date.date().isoformat() for date in missing_alpha_dates[:5]]}"
        )

    provenance = {
        "market": {
            "source": "data_access",
            "dataset": HISTORICAL_MARKET_DATASET,
            "field": "Return",
            "input_type": "return",
            "scale": 0.0001,
            "rule": HISTORICAL_RETURN_RULE,
            "lookback_days": lookback_days,
            "market_data_lag_periods": market_data_lag_periods,
            "load_start": load_start.date().isoformat(),
            "load_end": last_date.date().isoformat(),
            "non_missing_cells": int(returns.notna().to_numpy().sum()),
            "total_cells": int(returns.size),
            **_snapshot_provenance(market_result),
        },
        "calendar": {
            "source": "data_access",
            "dataset": CALENDAR_DATASET,
            **_snapshot_provenance(calendar_result),
        },
    }
    return DataAccessHistoricalMarket(returns=returns, provenance=provenance)


__all__ = [
    "BENCHMARK_DATASET",
    "CALENDAR_DATASET",
    "DEFAULT_LINEAR_COST_BPS",
    "DataAccessHistoricalMarket",
    "DataAccessPortfolioInputs",
    "HISTORICAL_MARKET_DATASET",
    "HISTORICAL_RETURN_RULE",
    "TRADABLE_DATASET",
    "TRADABLE_RULE",
    "load_historical_market_returns",
    "load_portfolio_inputs",
]
