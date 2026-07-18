#!/usr/bin/env python3
"""LQTP gRPC client helpers for auth, factor analysis, and backtest."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import grpc
import numpy as np
import pandas as pd

def _ensure_lqtp_protos() -> Path:
    """Prefer sibling kit protos, then official examples under quant_projects."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "ashare_lqtp_kit" / "protos",
        here.parents[2] / "lqtp-python-grpc-examples" / "protos",
    ]
    for path in candidates:
        if path.is_dir():
            p = str(path)
            if p not in sys.path:
                sys.path.insert(0, p)
            return path
    raise FileNotFoundError(
        "LQTP protos not found. Expected ashare_lqtp_kit/protos or "
        "lqtp-python-grpc-examples/protos under quant_projects."
    )


LQTP_PROTOS = _ensure_lqtp_protos()

import Factor_pb2  # noqa: E402
import Factor_pb2_grpc  # noqa: E402
import Struct_pb2  # noqa: E402
import Struct_pb2_grpc  # noqa: E402


DEFAULT_SERVER = os.getenv("LQTP_SERVER", "110.42.223.26:50051")
DEFAULT_TOKEN_REFRESH_SECONDS = 10 * 60
DEFAULT_BACKTEST_AUTH_RETRIES = 12
DEFAULT_BACKTEST_CASH = 10_000_000.0  # 1000万本金；图中 NAV 另归一化到 start=1
DEFAULT_LQTP_LOGIN_RETRIES = 12
DEFAULT_LQTP_LOGIN_WAIT_SEC = 15.0
# Platform may return large Backtest streams; default gRPC limit is 4MB.
GRPC_CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
)
_MISSING_QUOTE_RE = re.compile(r"行情不存在:\s*(\S+)")
SAFE_LQTP_SYMBOLS_LIST = [
    "000001.SZ", "000004.SZ", "000006.SZ", "000008.SZ", "000011.SZ",
    "000014.SZ", "000016.SZ", "000021.SZ", "000023.SZ", "000026.SZ",
]
SAFE_LQTP_SYMBOLS = set(SAFE_LQTP_SYMBOLS_LIST)


def env_username() -> str:
    """LQTP username from env (no hardcoded account in source)."""
    return os.getenv("LQTP_USERNAME", "").strip()


def env_password() -> str:
    """LQTP password from env (never hardcode secrets)."""
    return os.getenv("LQTP_PASSWORD", "")


def require_credentials(*, username: str | None = None, password: str | None = None) -> tuple[str, str]:
    """Resolve username/password; raise SystemExit-friendly ValueError if missing."""
    user = (username if username is not None else env_username()).strip()
    pwd = password if password is not None else env_password()
    if not user or not pwd:
        raise ValueError(
            "Missing LQTP credentials. Set LQTP_USERNAME / LQTP_PASSWORD "
            "or pass --username/--password."
        )
    return user, pwd


def make_channel(server: str | None = None) -> grpc.Channel:
    """Create insecure gRPC channel with large message limits."""
    return grpc.insecure_channel(server or DEFAULT_SERVER, options=list(GRPC_CHANNEL_OPTIONS))



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
    return make_channel(server)


def _is_transient_grpc_error(error: BaseException) -> bool:
    if isinstance(error, grpc.RpcError):
        code = error.code()
        return code in {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.INTERNAL,
        }
    text = str(error).lower()
    return "connection refused" in text or "failed to connect" in text or "socket closed" in text


def login_with_retry(
    server: str,
    username: str,
    password: str,
    *,
    max_attempts: int = DEFAULT_LQTP_LOGIN_RETRIES,
    wait_sec: float = DEFAULT_LQTP_LOGIN_WAIT_SEC,
) -> LqtpAuth:
    """Login with backoff when LQTP gRPC is temporarily down (Connection refused / UNAVAILABLE)."""
    last_error: BaseException | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            return login(server, username, password)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts or not _is_transient_grpc_error(exc):
                raise
            delay = min(wait_sec * attempt, 120.0)
            print(
                f"LQTP login attempt {attempt}/{max_attempts} failed ({exc}); "
                f"retry in {delay:.0f}s ..."
            )
            time.sleep(delay)
    raise RuntimeError(f"LQTP login failed after {max_attempts} attempts: {last_error}")


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


def refresh_access_token(server: str, refresh_token: str) -> LqtpAuth:
    """Exchange refresh_token for a new access/refresh token pair."""
    import Auth_pb2  # noqa: E402
    import Auth_pb2_grpc  # noqa: E402

    stub = Auth_pb2_grpc.AuthServiceStub(_channel(server))
    resp = stub.RefreshToken(
        Auth_pb2.RefreshTokenRequest(refresh_token=refresh_token),
        timeout=60,
    )
    return LqtpAuth(
        access_token=resp.access_token,
        refresh_token=resp.refresh_token,
        user_id=resp.user.user_id,
        username=resp.user.username,
    )


@dataclass
class LqtpTokenManager:
    """Proactively refresh LQTP access tokens before they expire."""

    server: str
    username: str
    password: str
    refresh_interval_seconds: int = DEFAULT_TOKEN_REFRESH_SECONDS
    _auth: LqtpAuth | None = None
    _last_refresh_monotonic: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def login(
        cls,
        server: str,
        username: str,
        password: str,
        *,
        refresh_interval_seconds: int = DEFAULT_TOKEN_REFRESH_SECONDS,
        login_retries: int = DEFAULT_LQTP_LOGIN_RETRIES,
        login_wait_sec: float = DEFAULT_LQTP_LOGIN_WAIT_SEC,
    ) -> LqtpTokenManager:
        mgr = cls(
            server=server,
            username=username,
            password=password,
            refresh_interval_seconds=refresh_interval_seconds,
        )
        mgr._login_retries = login_retries
        mgr._login_wait_sec = login_wait_sec
        mgr._do_login()
        return mgr

    def _do_login(self) -> None:
        retries = getattr(self, "_login_retries", DEFAULT_LQTP_LOGIN_RETRIES)
        wait_sec = getattr(self, "_login_wait_sec", DEFAULT_LQTP_LOGIN_WAIT_SEC)
        self._auth = login_with_retry(
            self.server,
            self.username,
            self.password,
            max_attempts=retries,
            wait_sec=wait_sec,
        )
        self._last_refresh_monotonic = time.monotonic()
        print(f"LQTP login ok user={self._auth.username}")

    def _do_refresh(self) -> None:
        if self._auth is None:
            self._do_login()
            return
        try:
            self._auth = refresh_access_token(self.server, self._auth.refresh_token)
            self._last_refresh_monotonic = time.monotonic()
            print(f"LQTP token refreshed user={self._auth.username}")
        except grpc.RpcError as error:
            print(f"LQTP refresh failed ({error.code()}), re-login ...")
            self._do_login()

    def maybe_refresh(self, *, force: bool = False) -> None:
        with self._lock:
            if self._auth is None:
                self._do_login()
                return
            elapsed = time.monotonic() - self._last_refresh_monotonic
            if force or elapsed >= self.refresh_interval_seconds:
                self._do_refresh()

    @property
    def token(self) -> str:
        with self._lock:
            if self._auth is None:
                self._do_login()
            elif time.monotonic() - self._last_refresh_monotonic >= self.refresh_interval_seconds:
                self._do_refresh()
            if self._auth is None:
                raise RuntimeError("LQTP token manager is not authenticated")
            return self._auth.access_token


def is_lqtp_auth_error(exc: BaseException) -> bool:
    """True when LQTP rejected the call due to an expired/invalid access token."""
    if isinstance(exc, grpc.RpcError):
        try:
            if exc.code() == grpc.StatusCode.UNAUTHENTICATED:
                return True
        except Exception:  # noqa: BLE001
            pass
    msg = str(exc).lower()
    needles = (
        "unauthenticated",
        "authorization token",
        "token 无效",
        "token 已过期",
        "无效或已过期",
        "invalid token",
        "expired token",
    )
    return any(n in msg for n in needles)


def run_backtest_auth_safe(
    token_mgr: LqtpTokenManager,
    *,
    max_auth_retries: int = DEFAULT_BACKTEST_AUTH_RETRIES,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Run backtest; on token expiry refresh/re-login and retry (never skip auth errors)."""
    last_err = ""
    for attempt in range(max_auth_retries):
        token_mgr.maybe_refresh(force=(attempt > 0))
        try:
            return run_backtest_from_weights(token=token_mgr.token, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_lqtp_auth_error(exc):
                raise
            last_err = str(exc)
            print(f"  LQTP token expired → refresh & retry ({attempt + 1}/{max_auth_retries})")
            token_mgr.maybe_refresh(force=True)
    raise RuntimeError(f"LQTP backtest auth failed after {max_auth_retries} refresh retries: {last_err}")


def safe_backtest(
    *,
    token_mgr: LqtpTokenManager | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Backtest wrapper: auto-refresh token via *token_mgr*; skip only non-auth failures."""
    if token_mgr is not None:
        try:
            return run_backtest_auth_safe(token_mgr, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_lqtp_auth_error(exc):
                raise
            print(f"  backtest skipped: {exc}")
            return [], ""
    if token is None:
        raise ValueError("safe_backtest requires token_mgr or token")
    try:
        return run_backtest_from_weights(token=token, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if is_lqtp_auth_error(exc):
            raise RuntimeError(
                "LQTP token expired; pass token_mgr=LqtpTokenManager for auto-refresh"
            ) from exc
        print(f"  backtest skipped: {exc}")
        return [], ""


def _to_lqtp_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    if text.isdigit():
        return f"{text.zfill(6)}.SZ"
    return text


def long_df_to_daily_values(long_df: pd.DataFrame) -> list[Factor_pb2.FactorDailyValues]:
    work = long_df.copy()
    if "trade_date" in work.columns:
        work["trade_date"] = work["trade_date"].astype(int)
        work["quote_time"] = 0
        if "symbol" not in work.columns:
            work["symbol"] = work["asset"].astype(str).map(_to_lqtp_symbol)
        else:
            work["symbol"] = work["symbol"].astype(str).map(_to_lqtp_symbol)
    else:
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
    try:
        from scripts.cogalpha_lqtp.factor_eval import evaluate_uploaded_values
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "evaluate_factor requires scripts.cogalpha_lqtp.factor_eval "
            "(full quant_projects tree)."
        ) from exc

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
    factor_name: str = "",
    value_limit: int = 0,
) -> Factor_pb2.RunFactorResponse:
    stub = Factor_pb2_grpc.FactorServiceStub(_channel(server))
    request = Factor_pb2.RunFactorRequest(
        formula=formula,
        begin_date=begin_date,
        end_date=end_date,
        warmup=warmup,
        analyze=analyze,
        value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_ALL,
        value_limit=value_limit,
        factor_name=factor_name,
    )
    return stub.RunFactor(request, metadata=_metadata(token), timeout=600)


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


# Modern low broker fee: 0.01 = 0.01% each side (LQTP unit: 0.1 = 0.1%).
DEFAULT_COMMISSION_BUY = 0.01
DEFAULT_COMMISSION_SELL = 0.01


def _trade_date_shift_map(dates: Iterable[int], shift: int = 1) -> dict[int, int]:
    ordered = sorted({int(d) for d in dates})
    out: dict[int, int] = {}
    for i, d in enumerate(ordered):
        j = i + shift
        if j < len(ordered):
            out[d] = ordered[j]
    return out


def build_tradable_by_exec_date(returns_long: pd.DataFrame) -> dict[int, set[str]]:
    """Map exec trade_date -> symbols with LQTP open bar on that date (point-in-time).

    Used for TopK: signal on T → trade open(T+1); only pick names quoteable on exec date.
    """
    if returns_long.empty:
        return {}
    work = returns_long.copy()
    work["trade_date"] = work["trade_date"].astype(int)
    work["symbol"] = work["symbol"].astype(str).map(_to_lqtp_symbol)
    return {
        int(d): set(g["symbol"].astype(str))
        for d, g in work.groupby("trade_date", sort=True)
    }


def top_quantile_weights(
    long_df: pd.DataFrame,
    *,
    top_frac: float = 0.1,
    max_picks_per_day: int = 50,
    allowed_symbols: set[str] | None = SAFE_LQTP_SYMBOLS,
    tradable_by_exec_date: dict[int, set[str]] | None = None,
    signal_to_trade_lag: int = 1,
) -> list[Struct_pb2.Weight]:
    """Build TopK long-only weights.

    Factor known after close on signal date T cannot trade at T close.
    With *signal_to_trade_lag*=1 and OPEN price backtest, weights are placed on T+1
    so LQTP trades at open(T+1).
    """
    work = long_df.copy()
    work["trade_date"] = work["trade_date"].astype(int)
    if allowed_symbols is not None:
        work["symbol"] = work["symbol"].astype(str).map(_to_lqtp_symbol)
        work = work[work["symbol"].isin(allowed_symbols)]
    shift_map = _trade_date_shift_map(work["trade_date"], shift=signal_to_trade_lag)
    weights: list[Struct_pb2.Weight] = []
    for trade_date, group in work.groupby("trade_date", sort=True):
        exec_date = shift_map.get(int(trade_date))
        if exec_date is None:
            continue
        valid = group.dropna(subset=["value"])
        if tradable_by_exec_date is not None:
            ok = tradable_by_exec_date.get(int(exec_date))
            if ok is not None:
                valid = valid[valid["symbol"].isin(ok)]
        if valid.empty:
            continue
        n = max(1, min(max_picks_per_day, int(len(valid) * top_frac)))
        picks = valid.nlargest(n, "value")
        w = 1.0 / len(picks)
        for row in picks.itertuples(index=False):
            weights.append(
                Struct_pb2.Weight(
                    trade_date=int(exec_date),
                    quote_time=0,
                    symbol=str(row.symbol),
                    value=w,
                )
            )
    return weights


def run_topk_backtest_for_long_df(
    token_mgr: LqtpTokenManager,
    long_df: pd.DataFrame,
    *,
    begin_date: int,
    end_date: int,
    server: str = DEFAULT_SERVER,
    allowed_symbols: set[str] | None = None,
    open_returns_long: pd.DataFrame | None = None,
    pre_excluded: set[str] | None = None,
    max_symbol_excludes: int = 200,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], set[str]]:
    """TopK OPEN backtest on LQTP BacktestService.

    Weights are rebuilt after each missing-quote exclusion. When *open_returns_long* is
    provided, TopK picks are filtered point-in-time: only symbols with an open bar on
    the exec date (T+1 open) are eligible — no global retroactive delisting filter.
    """
    tradable_by_exec = (
        build_tradable_by_exec_date(open_returns_long) if open_returns_long is not None else None
    )
    excluded: set[str] = set(pre_excluded or ())
    for _ in range(max_symbol_excludes + 1):
        ok_symbols = None if allowed_symbols is None else (allowed_symbols - excluded)
        weights = top_quantile_weights(
            long_df,
            allowed_symbols=ok_symbols,
            tradable_by_exec_date=tradable_by_exec,
        )
        if not weights:
            return [], "", summarize_backtest([]), excluded
        try:
            rows, bt_id = run_backtest_auth_safe(
                token_mgr,
                weights=weights,
                begin_date=begin_date,
                end_date=end_date,
                server=server,
                max_quote_retries=0,
            )
            if excluded:
                print(f"  backtest ok after excluding {len(excluded)} missing-quote symbols")
            health = diagnose_backtest_nav_health(rows)
            if not health.get("healthy"):
                print(
                    f"  WARNING: backtest has {health['frozen_tail_days']} frozen tail days "
                    f"(last active {health.get('last_active_trade_date')}); {health.get('hint', '')}"
                )
            return rows, bt_id, summarize_backtest(rows), excluded
        except Exception as exc:  # noqa: BLE001
            if is_lqtp_auth_error(exc):
                raise
            match = _MISSING_QUOTE_RE.search(str(exc))
            if not match:
                print(f"  backtest skipped: {exc}")
                return [], "", summarize_backtest([]), excluded
            bad = _to_lqtp_symbol(match.group(1))
            if bad in excluded:
                print(f"  backtest skipped: {exc}")
                return [], "", summarize_backtest([]), excluded
            excluded.add(bad)
            print(f"  backtest exclude missing quote {bad}")
    print("  backtest skipped: too many missing-quote symbols")
    return [], "", summarize_backtest([]), excluded


def long_short_decile_weights(
    long_df: pd.DataFrame,
    *,
    top_frac: float = 0.1,
    bottom_frac: float = 0.1,
    max_picks_per_side: int = 50,
    allowed_symbols: set[str] | None = SAFE_LQTP_SYMBOLS,
    signal_to_trade_lag: int = 1,
) -> list[Struct_pb2.Weight]:
    """Equal-weight long top / short bottom decile; negative weight = short."""
    work = long_df.copy()
    work["trade_date"] = work["trade_date"].astype(int)
    if allowed_symbols is not None:
        work["symbol"] = work["symbol"].astype(str).map(_to_lqtp_symbol)
        work = work[work["symbol"].isin(allowed_symbols)]
    shift_map = _trade_date_shift_map(work["trade_date"], shift=signal_to_trade_lag)
    weights: list[Struct_pb2.Weight] = []
    for trade_date, group in work.groupby("trade_date", sort=True):
        exec_date = shift_map.get(int(trade_date))
        if exec_date is None:
            continue
        valid = group.dropna(subset=["value"])
        if len(valid) < 20:
            continue
        n_long = max(1, min(max_picks_per_side, int(len(valid) * top_frac)))
        n_short = max(1, min(max_picks_per_side, int(len(valid) * bottom_frac)))
        longs = valid.nlargest(n_long, "value")
        shorts = valid.nsmallest(n_short, "value")
        w_l = 0.5 / len(longs)
        w_s = -0.5 / len(shorts)
        for row in longs.itertuples(index=False):
            weights.append(
                Struct_pb2.Weight(
                    trade_date=int(exec_date),
                    quote_time=0,
                    symbol=str(row.symbol),
                    value=w_l,
                )
            )
        for row in shorts.itertuples(index=False):
            weights.append(
                Struct_pb2.Weight(
                    trade_date=int(exec_date),
                    quote_time=0,
                    symbol=str(row.symbol),
                    value=w_s,
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


def diagnose_backtest_nav_health(
    rows: list[dict[str, Any]], *, min_frozen_days: int = 20
) -> dict[str, Any]:
    """Detect LQTP zombie-portfolio tail: flat NAV with zero turnover at end of stream."""
    if len(rows) < min_frozen_days + 1:
        return {"healthy": True, "frozen_tail_days": 0, "last_active_trade_date": None}
    frame = pd.DataFrame(rows).sort_values("trade_date")
    nav = pd.to_numeric(frame["eod_net_asset"], errors="coerce")
    turnover = pd.to_numeric(frame.get("turnover_rate", 0), errors="coerce").fillna(0)
    frozen = 0
    for i in range(len(frame) - 1, 0, -1):
        if abs(float(nav.iloc[i]) - float(nav.iloc[i - 1])) < 1e-4 and float(turnover.iloc[i]) == 0.0:
            frozen += 1
        else:
            break
    last_active_idx = max(0, len(frame) - frozen - 1)
    return {
        "healthy": frozen < min_frozen_days,
        "frozen_tail_days": int(frozen),
        "last_active_trade_date": int(frame["trade_date"].iloc[last_active_idx]),
        "hint": (
            "TopK weights were not rebuilt after excluding missing-quote symbols; "
            "use run_topk_backtest_for_long_df instead of run_backtest_auth_safe on raw weights."
            if frozen >= min_frozen_days
            else ""
        ),
    }


def save_backtest_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist daily TopK backtest stream for debugging / report regen."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def summarize_backtest(rows: list[dict[str, Any]], *, initial_cash: float = DEFAULT_BACKTEST_CASH) -> dict[str, Any]:
    """Extra metrics from LQTP daily backtest stream.

    LQTP ``Result.ret`` is in basis points (10000 = 100%); ``turnover_rate`` is in percent
    (100 = 100%).
    """
    if not rows:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe": float("nan"),
            "volatility": float("nan"),
            "calmar": float("nan"),
            "win_rate": float("nan"),
            "avg_turnover": float("nan"),
            "total_commission": float("nan"),
            "trading_days": 0,
            "final_nav": float("nan"),
        }
    frame = pd.DataFrame(rows).sort_values("trade_date")
    nav = pd.to_numeric(frame["eod_net_asset"], errors="coerce")
    # Prefer NAV-implied daily returns (robust); fall back to ret/10000 (bps → decimal)
    if len(nav) >= 2:
        rets = nav.pct_change().fillna(0.0)
    else:
        rets = pd.to_numeric(frame["ret"], errors="coerce").fillna(0.0) / 10000.0
    turnover = pd.to_numeric(frame.get("turnover_rate", 0.0), errors="coerce").fillna(0.0) / 100.0
    commission = pd.to_numeric(frame.get("commission", 0.0), errors="coerce").fillna(0.0)

    final_nav = float(nav.iloc[-1]) if len(nav) else float("nan")
    start_nav = float(nav.iloc[0]) if len(nav) else initial_cash
    if "bod_net_asset" in frame.columns and pd.notna(frame["bod_net_asset"].iloc[0]):
        bod0 = float(frame["bod_net_asset"].iloc[0])
        if bod0 > 0:
            start_nav = bod0
    total_return = final_nav / start_nav - 1.0 if start_nav else float("nan")
    n = max(len(frame), 1)
    years = n / 252.0
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and total_return > -1 else float("nan")

    peak = nav.cummax()
    dd = nav / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")
    vol = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else float("nan")
    sharpe = (
        float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
        if len(rets) > 1 and rets.std(ddof=1) > 1e-12
        else 0.0
    )
    calmar = float(annualized / abs(max_dd)) if max_dd == max_dd and abs(max_dd) > 1e-12 else float("nan")
    win_rate = float((rets > 0).mean()) if len(rets) else float("nan")

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized) if annualized == annualized else float("nan"),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "volatility": vol,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_turnover": float(turnover.mean()) if len(turnover) else float("nan"),
        "total_commission": float(commission.sum()) if len(commission) else float("nan"),
        "trading_days": int(n),
        "final_nav": final_nav,
        "initial_cash": float(start_nav),
    }


def run_backtest_from_weights(
    *,
    token: str,
    weights: list[Struct_pb2.Weight],
    begin_date: int,
    end_date: int,
    cash: float = DEFAULT_BACKTEST_CASH,
    server: str = DEFAULT_SERVER,
    return_details: bool = False,
    max_quote_retries: int = 32,
    commission_buy: float = DEFAULT_COMMISSION_BUY,
    commission_sell: float = DEFAULT_COMMISSION_SELL,
    price_type: int | None = None,
    enable_short_selling: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Run LQTP backtest at OPEN by default (signal lagged to next open).

    Commission unit: 0.01 = 0.01%. Drops missing-quote symbols and retries.
    """
    work = list(weights)
    excluded: set[str] = set()
    last_error = ""
    if price_type is None:
        price_type = Struct_pb2.OPEN

    for _attempt in range(max_quote_retries + 1):
        if not work:
            raise RuntimeError("no backtest weights left after excluding missing-quote symbols")

        request = Struct_pb2.Request(
            begin_date=begin_date,
            end_date=end_date,
            weights=work,
            cash=cash,
            commission_buy=commission_buy,
            commission_sell=commission_sell,
            return_orders=return_details,
            return_targets=return_details,
            return_positions=return_details,
            persistence_mode=Struct_pb2.BACKTEST_PERSISTENCE_MODE_NONE,
            price_type=price_type,
            enable_short_selling=enable_short_selling,
        )
        stub = Struct_pb2_grpc.BacktestServiceStub(_channel(server))
        rows: list[dict[str, Any]] = []
        backtest_id = ""
        try:
            for resp in stub.Backtest(request, metadata=_metadata(token), timeout=300):
                if resp.error:
                    raise RuntimeError(resp.error)
                backtest_id = resp.backtest_id or backtest_id
                rows.append(backtest_result_to_dict(resp.result))
            if excluded:
                print(f"  backtest ok after excluding symbols: {sorted(excluded)}")
            return rows, backtest_id
        except RuntimeError as exc:
            last_error = str(exc)
            match = _MISSING_QUOTE_RE.search(last_error)
            if not match:
                raise
            bad = _to_lqtp_symbol(match.group(1))
            if bad in excluded:
                raise RuntimeError(f"backtest retry loop on repeated missing symbol {bad}: {last_error}") from exc
            excluded.add(bad)
            work = [w for w in work if _to_lqtp_symbol(w.symbol) != bad]
            print(f"  backtest retry excluding missing quote symbol {bad}")

    raise RuntimeError(last_error or "backtest failed")
