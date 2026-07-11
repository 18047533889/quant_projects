#!/usr/bin/env python3
"""LQTP gRPC client helpers for auth, factor analysis, and backtest."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path
from typing import Any, Iterable

import grpc
import pandas as pd

LQTP_ROOT = Path(__file__).resolve().parents[2] / "lqtp-python-grpc-examples"
sys.path.insert(0, str(LQTP_ROOT / "protos"))

import Factor_pb2  # noqa: E402
import Factor_pb2_grpc  # noqa: E402
import Struct_pb2  # noqa: E402
import Struct_pb2_grpc  # noqa: E402


DEFAULT_SERVER = os.getenv("LQTP_SERVER", "118.89.191.220:50051")
SAFE_LQTP_SYMBOLS_LIST = [
    "000001.SZ", "000004.SZ", "000006.SZ", "000008.SZ", "000011.SZ",
    "000014.SZ", "000016.SZ", "000021.SZ", "000023.SZ", "000026.SZ",
]
SAFE_LQTP_SYMBOLS = set(SAFE_LQTP_SYMBOLS_LIST)


def fetch_lqtp_universe(
    *,
    token: str,
    begin_date: int,
    end_date: int,
    server: str = DEFAULT_SERVER,
) -> set[str]:
    """Discover tradable symbols from a lightweight RunFactor sample."""
    resp = run_factor_formula(
        token=token,
        formula="close / delay(close, 1) - 1",
        begin_date=begin_date,
        end_date=end_date,
        warmup=1,
        analyze=False,
        server=server,
    )
    symbols: set[str] = set()
    for point in resp.values:
        for value in point.values:
            symbols.add(str(value.symbol))
    return symbols


@dataclass
class LqtpAuth:
    access_token: str
    refresh_token: str
    user_id: str
    username: str


def _metadata(token: str | None) -> tuple[tuple[str, str], ...] | None:
    return (("authorization", f"Bearer {token}"),) if token else None


def _channel(server: str) -> grpc.Channel:
    return grpc.insecure_channel(
        server,
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )


def login(server: str, username: str, password: str) -> LqtpAuth:
    import Auth_pb2  # noqa: E402
    import Auth_pb2_grpc  # noqa: E402

    stub = Auth_pb2_grpc.AuthServiceStub(_channel(server))
    resp = stub.Login(
        Auth_pb2.LoginRequest(username=username, password=password),
        timeout=60,
    )
    return LqtpAuth(
        access_token=resp.access_token,
        refresh_token=resp.refresh_token,
        user_id=resp.user.user_id,
        username=resp.user.username,
    )


def _to_lqtp_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    if text.isdigit():
        return f"{text.zfill(6)}.SZ"
    return text


def long_df_to_daily_values(long_df: pd.DataFrame) -> list[Factor_pb2.FactorDailyValues]:
    work = long_df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work["trade_date"] = work["datetime"].dt.strftime("%Y%m%d").astype(int)
    work["quote_time"] = 0
    work["symbol"] = work["asset"].map(_to_lqtp_symbol)

    points: list[Factor_pb2.FactorDailyValues] = []
    for (trade_date, quote_time), group in work.groupby(["trade_date", "quote_time"], sort=True):
        values = [
            Factor_pb2.FactorValue(symbol=row.symbol, value=float(row.value))
            for row in group.itertuples(index=False)
            if pd.notna(row.value)
        ]
        if values:
            points.append(
                Factor_pb2.FactorDailyValues(
                    trade_date=int(trade_date),
                    quote_time=int(quote_time),
                    values=values,
                )
            )
    return points


def analyze_factor_values(
    *,
    token: str,
    daily_values: list[Factor_pb2.FactorDailyValues],
    begin_date: int,
    end_date: int,
    server: str = DEFAULT_SERVER,
) -> Factor_pb2.FactorAnalysisResponse:
    stub = Factor_pb2_grpc.FactorServiceStub(_channel(server))
    request = Factor_pb2.FactorAnalysisRequest(
        begin_date=begin_date,
        end_date=end_date,
        daily_values=daily_values,
    )
    return stub.AnalyzeFactor(request, metadata=_metadata(token), timeout=300)


def evaluate_factor(
    *,
    token: str,
    daily_values: list[Factor_pb2.FactorDailyValues],
    begin_date: int,
    end_date: int,
    server: str = DEFAULT_SERVER,
    returns_cache: Path | None = None,
    **_deprecated: Any,
) -> tuple[dict[str, Any], str]:
    """Evaluate uploaded factor_engine values (never re-runs our DSL on LQTP)."""
    from scripts.cogalpha_lqtp.factor_eval import evaluate_uploaded_values

    if _deprecated:
        import warnings

        warnings.warn(
            "lqtp_formula / RunFactor fallback is removed; only uploaded values are evaluated.",
            DeprecationWarning,
            stacklevel=2,
        )
    return evaluate_uploaded_values(
        token=token,
        daily_values=daily_values,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        returns_cache=returns_cache,
    )


def run_factor_formula(
    *,
    token: str,
    formula: str,
    begin_date: int,
    end_date: int,
    warmup: int = 1,
    analyze: bool = True,
    server: str = DEFAULT_SERVER,
) -> Factor_pb2.RunFactorResponse:
    stub = Factor_pb2_grpc.FactorServiceStub(_channel(server))
    request = Factor_pb2.RunFactorRequest(
        formula=formula,
        begin_date=begin_date,
        end_date=end_date,
        warmup=warmup,
        analyze=analyze,
        value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_ALL,
        value_limit=50000,
    )
    return stub.RunFactor(request, metadata=_metadata(token), timeout=300)


def analysis_to_dict(analysis: Factor_pb2.FactorAnalysis) -> dict[str, Any]:
    quote_times = (
        list(analysis.quote_times)
        if len(analysis.quote_times) == len(analysis.trade_dates)
        else [0] * len(analysis.trade_dates)
    )
    return {
        "mean_ic": analysis.mean_ic,
        "std_ic": analysis.std_ic,
        "icir": analysis.icir,
        "ic_positive_ratio": analysis.ic_positive_ratio,
        "coverage": analysis.coverage,
        "long_short_return": analysis.long_short_return,
        "long_short_sharpe": analysis.long_short_sharpe,
        "daily_ic": list(analysis.daily_ic),
        "daily_ls_returns": list(analysis.daily_ls_returns),
        "trade_dates": list(analysis.trade_dates),
        "quote_times": quote_times,
        "sample_counts": list(analysis.sample_counts),
        "coverages": list(analysis.coverages),
        "group_mean_returns": list(analysis.group_mean_returns),
        "group_pnls": [
            {"group": item.group, "pnl": list(item.pnl)}
            for item in analysis.group_pnls
        ],
    }


def factor_values_to_long_df(values: Iterable[Factor_pb2.FactorDailyValues]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for point in values:
        for value in point.values:
            rows.append(
                {
                    "trade_date": point.trade_date,
                    "quote_time": point.quote_time,
                    "symbol": value.symbol,
                    "value": value.value,
                }
            )
    return pd.DataFrame(rows)


def top_quantile_weights(
    long_df: pd.DataFrame,
    *,
    top_frac: float = 0.1,
    allowed_symbols: set[str] | None = SAFE_LQTP_SYMBOLS,
) -> list[Struct_pb2.Weight]:
    work = long_df.copy()
    work["trade_date"] = work["trade_date"].astype(int)
    if allowed_symbols is not None:
        work["symbol"] = work["symbol"].astype(str).map(_to_lqtp_symbol)
        work = work[work["symbol"].isin(allowed_symbols)]
    weights: list[Struct_pb2.Weight] = []
    for trade_date, group in work.groupby("trade_date", sort=True):
        valid = group.dropna(subset=["value"])
        if valid.empty:
            continue
        n = max(1, int(len(valid) * top_frac))
        picks = valid.nlargest(n, "value")
        w = 1.0 / len(picks)
        for row in picks.itertuples(index=False):
            weights.append(
                Struct_pb2.Weight(
                    trade_date=int(trade_date),
                    quote_time=0,
                    symbol=str(row.symbol),
                    value=w,
                )
            )
    return weights


def backtest_result_to_dict(result: Struct_pb2.Result) -> dict[str, Any]:
    return {
        "trade_date": result.trade_date,
        "bod_net_asset": result.bod_net_asset,
        "eod_net_asset": result.eod_net_asset,
        "bod_market_value": result.bod_market_value,
        "eod_market_value": result.eod_market_value,
        "bod_long_market_value": result.bod_long_market_value,
        "eod_long_market_value": result.eod_long_market_value,
        "bod_short_market_value": result.bod_short_market_value,
        "eod_short_market_value": result.eod_short_market_value,
        "bod_cash": result.bod_cash,
        "eod_cash": result.eod_cash,
        "expect_buy_amount": result.expect_buy_amount,
        "expect_sell_amount": result.expect_sell_amount,
        "buy_amount": result.buy_amount,
        "sell_amount": result.sell_amount,
        "buy_execution_ratio": result.buy_execution_ratio,
        "sell_execution_ratio": result.sell_execution_ratio,
        "execution_ratio": result.execution_ratio,
        "turnover_rate": result.turnover_rate,
        "leverage": result.leverage,
        "commission": result.commission,
        "net_asset_diff": result.net_asset_diff,
        "ret": result.ret,
    }


def run_backtest_from_weights(
    *,
    token: str,
    weights: list[Struct_pb2.Weight],
    begin_date: int,
    end_date: int,
    cash: float = 1_000_000.0,
    server: str = DEFAULT_SERVER,
) -> tuple[list[dict[str, Any]], str]:
    request = Struct_pb2.Request(
        begin_date=begin_date,
        end_date=end_date,
        weights=weights,
        cash=cash,
        commission_buy=0.03,
        commission_sell=0.03,
        return_orders=True,
        return_targets=True,
        return_positions=True,
        persistence_mode=Struct_pb2.BACKTEST_PERSISTENCE_MODE_NONE,
        price_type=Struct_pb2.CLOSE,
    )
    stub = Struct_pb2_grpc.BacktestServiceStub(_channel(server))
    rows: list[dict[str, Any]] = []
    backtest_id = ""
    for resp in stub.Backtest(request, metadata=_metadata(token), timeout=300):
        if resp.error:
            raise RuntimeError(resp.error)
        backtest_id = resp.backtest_id or backtest_id
        rows.append(backtest_result_to_dict(resp.result))
    return rows, backtest_id
