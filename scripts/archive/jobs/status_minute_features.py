#!/usr/bin/env python3
"""Pure, local minute-bar status features (F06 family).

Each function computes one F06 feature from a single (symbol, day) slice of
StockMinuteBar bars. QuoteTime is strictly converted from UTC to
Asia/Shanghai, and only the continuous-auction session
(09:31-11:30, 13:01-15:00) is kept.

Deliberately NOT implemented:
- ``limit_touch_*`` -- StockMinuteBar has no ``HighLimit``/``LowLimit``
  columns (confirmed against the COS parquet schema; those columns exist only
  on StockDailyBar), so limit-touch paths cannot be measured reliably.
- ``auction_imbalance`` -- requires 09:15-09:25 auction / Level-2 order data
  and is registered as EXTERNAL_GAP in the registry.

The functions are pure: they accept an iterable of bar row mappings (or a
PyArrow Table) and return floats, so they can be unit-tested without any COS
download. A day whose total volume is zero (suspended stock) yields NaN for
every feature.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CN_TZ = dt.timezone(dt.timedelta(hours=8))
UTC_TZ = dt.timezone.utc

BUCKET = "quantsociety-cold-data-1425188104"
SOURCE_ROOT = f"cos://{BUCKET}/clean_data/ashare/lqtp_data"

MORNING_START = 9 * 60 + 31  # 09:31
MORNING_END = 11 * 60 + 30  # 11:30
AFTERNOON_START = 13 * 60 + 1  # 13:01
AFTERNOON_END = 15 * 60  # 15:00

# Public function name -> registry canonical feature id.
FEATURE_IDS: dict[str, str] = {
    "open_30m_ret": "open30_ret",
    "close_30m_ret": "close30_ret",
    "morning_afternoon_spread": "morning_afternoon_gap",
    "intraday_rv": "intraday_rv",
    "intraday_trend_score": "intraday_trend_score",
    "intraday_reversal": "intraday_reversal",
    "volume_u_shape": "volume_u_shape",
    "last_hour_amount_share": "last_hour_amount_share",
    "vwap_path_pressure": "vwap_path_pressure",
    "intraday_vol_of_vol": "intraday_vol_of_vol",
}

_ALIASES = {
    "time": ("QuoteTime", "quote_time", "quoteTime", "Time", "time"),
    "symbol": ("Symbol", "symbol", "Code", "code"),
    "trade_date": ("TradeDate", "trade_date", "Date", "date"),
    "open": ("Open", "open"),
    "close": ("Close", "close"),
    "volume": ("Volume", "volume", "Vol", "vol"),
    "amount": ("Amount", "amount"),
    "vwap": ("Vwap", "vwap", "VWAP"),
}


def _value(row: Mapping[str, Any], field: str) -> Any:
    for name in _ALIASES[field]:
        if name in row:
            return row[name]
    return None


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _as_rows(data: Any) -> list[Mapping[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_pylist"):
        return list(data.to_pylist())
    if isinstance(data, Mapping):
        return [data]
    return list(data)


def in_continuous_session(minutes_of_day: float) -> bool:
    """Return whether a minute-of-day is in the A-share continuous session."""
    return (MORNING_START <= minutes_of_day <= MORNING_END) or (
        AFTERNOON_START <= minutes_of_day <= AFTERNOON_END
    )


def minute_of_day(value: dt.datetime) -> float:
    return value.hour * 60 + value.minute + value.second / 60.0


def to_asia_shanghai(value: Any) -> dt.datetime | None:
    """Convert a QuoteTime to Asia/Shanghai.

    StockMinuteBar stores QuoteTime as UTC. Naive datetimes are assumed to be
    UTC and shifted by +8 hours; epoch milliseconds are also accepted.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        local = value if value.tzinfo is not None else value.replace(tzinfo=UTC_TZ)
        return local.astimezone(CN_TZ)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return dt.datetime.fromtimestamp(value / 1000.0, tz=UTC_TZ).astimezone(CN_TZ)
    if isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC_TZ)
        return parsed.astimezone(CN_TZ)
    raise TypeError(f"unsupported QuoteTime value: {value!r}")


@dataclass(frozen=True)
class MinuteSeries:
    """Session-filtered, time-sorted minute bars for one (symbol, day)."""

    times: list[dt.datetime]
    minutes: np.ndarray
    open: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    vwap: np.ndarray

    @property
    def day_open(self) -> float:
        return float(self.open[0])

    @property
    def day_close(self) -> float:
        return float(self.close[-1])

    @property
    def total_volume(self) -> float:
        return float(np.nansum(self.volume))


def prepare_bars(data: Any) -> MinuteSeries | None:
    """Localize QuoteTime, drop non-session bars, sort by time."""
    raw: list[tuple[dt.datetime, float, float, float, float, float]] = []
    for row in _as_rows(data):
        quote = to_asia_shanghai(_value(row, "time"))
        if quote is None:
            continue
        minutes = minute_of_day(quote)
        if not in_continuous_session(minutes):
            continue
        raw.append(
            (
                quote,
                _number(_value(row, "open")),
                _number(_value(row, "close")),
                _number(_value(row, "volume")),
                _number(_value(row, "amount")),
                _number(_value(row, "vwap")),
            )
        )
    if not raw:
        return None
    raw.sort(key=lambda item: item[0])
    times = [item[0] for item in raw]
    minutes = np.array([minute_of_day(t) for t in times], dtype=float)
    arrays = [np.array([item[k] for item in raw], dtype=float) for k in range(1, 6)]
    return MinuteSeries(times, minutes, arrays[0], arrays[1], arrays[2], arrays[3], arrays[4])


def _tradable(d: MinuteSeries) -> bool:
    return bool(np.isfinite(d.total_volume) and d.total_volume > 0)


def _bar_at(d: MinuteSeries, minute: int, mode: str = "last_before") -> int | None:
    if mode == "exact":
        idx = np.flatnonzero(np.isclose(d.minutes, minute))
    else:
        idx = np.flatnonzero(d.minutes <= minute)
    return int(idx[-1]) if len(idx) else None


def _window_mask(d: MinuteSeries, start_min: float, end_min: float) -> np.ndarray:
    return (d.minutes >= start_min) & (d.minutes <= end_min)


def _session_return(d: MinuteSeries, start_min: int, end_min: int) -> float:
    """Close-to-close return over the bars between two minute labels."""
    i = _bar_at(d, start_min, "last_before")
    j = _bar_at(d, end_min, "last_before")
    if i is None or j is None or j <= i:
        return math.nan
    c0, c1 = d.close[i], d.close[j]
    if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0:
        return math.nan
    return float(c1 / c0 - 1.0)


def _window_vwap(d: MinuteSeries, start_min: float, end_min: float) -> float:
    mask = _window_mask(d, start_min, end_min)
    total_volume = np.nansum(d.volume[mask])
    total_amount = np.nansum(d.amount[mask])
    if not np.isfinite(total_volume) or total_volume <= 0:
        return math.nan
    return float(total_amount / total_volume)


# ---------------------------------------------------------------------------
# Internal per-feature implementations (single MinuteSeries in, float out).
# ---------------------------------------------------------------------------


def _open_30m_ret(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    vwap = _window_vwap(d, MORNING_START, 10 * 60)  # 09:31-10:00
    if math.isnan(vwap) or not np.isfinite(d.day_open) or d.day_open <= 0:
        return math.nan
    return float(vwap / d.day_open - 1.0)


def _close_30m_ret(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    i = _bar_at(d, 14 * 60 + 30, "last_before")
    j = _bar_at(d, AFTERNOON_END, "last_before")
    if i is None or j is None or j <= i:
        return math.nan
    c0, c1 = d.close[i], d.close[j]
    if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0:
        return math.nan
    return float(c1 / c0 - 1.0)


def _morning_afternoon_spread(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    morning = _session_return(d, MORNING_START, MORNING_END)
    afternoon = _session_return(d, AFTERNOON_START, AFTERNOON_END)
    return morning - afternoon


def _intraday_reversal(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    morning = _session_return(d, MORNING_START, MORNING_END)
    afternoon = _session_return(d, AFTERNOON_START, AFTERNOON_END)
    return afternoon - morning


def _intraday_rv(d: MinuteSeries, block: int = 5) -> float:
    if not _tradable(d):
        return math.nan
    close = d.close[np.isfinite(d.close)]
    if len(close) < 2 * block:
        return math.nan
    block_ends = close[block - 1 :: block]
    if len(block_ends) < 2:
        return math.nan
    logret = np.diff(np.log(block_ends))
    return float(np.sum(logret**2))


def _intraday_trend_score(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    close = d.close
    y = np.log(close)
    finite = np.isfinite(y)
    x = np.arange(len(close))
    xf, yf = x[finite], y[finite]
    if len(xf) < 2:
        return math.nan
    xm, ym = xf.mean(), yf.mean()
    varx = float(np.sum((xf - xm) ** 2))
    if varx <= 0:
        return math.nan
    cov = float(np.sum((xf - xm) * (yf - ym)))
    vary = float(np.sum((yf - ym) ** 2))
    slope = cov / varx
    r2 = (cov * cov) / (varx * vary) if vary > 0 else 0.0
    return float(slope * r2)


def _volume_u_shape(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    total = np.nansum(d.volume)
    if not np.isfinite(total) or total <= 0:
        return math.nan
    first = np.nansum(d.volume[_window_mask(d, MORNING_START, 10 * 60 + 30)])
    last = np.nansum(d.volume[_window_mask(d, 14 * 60 + 1, AFTERNOON_END)])
    return float((first + last) / total)


def _last_hour_amount_share(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    total = np.nansum(d.amount)
    if not np.isfinite(total) or total <= 0:
        return math.nan
    last = np.nansum(d.amount[_window_mask(d, 14 * 60 + 1, AFTERNOON_END)])
    return float(last / total)


def _vwap_path_pressure(d: MinuteSeries) -> float:
    if not _tradable(d):
        return math.nan
    volume = d.volume
    amount = d.amount
    valid = np.isfinite(amount) & np.isfinite(volume) & (volume > 0)
    if np.count_nonzero(valid) == 0:
        return math.nan
    cum_amount = np.cumsum(np.where(valid, amount, 0.0))
    cum_volume = np.cumsum(np.where(valid, volume, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = np.where(cum_volume > 0, cum_amount / cum_volume, np.nan)
    dev = d.close - cum_vwap
    ref = d.day_close
    if not np.isfinite(ref) or ref <= 0:
        ref = np.nanmedian(d.close)
    return float(np.nansum(dev[valid]) / ref)


def _intraday_vol_of_vol(d: MinuteSeries, window: int = 5) -> float:
    if not _tradable(d):
        return math.nan
    close = d.close[np.isfinite(d.close)]
    if len(close) < window + 2:
        return math.nan
    logret = np.diff(np.log(close))
    local = []
    for i in range(window - 1, len(logret)):
        chunk = logret[i - window + 1 : i + 1]
        local.append(float(np.std(chunk)))
    if len(local) < 2:
        return math.nan
    return float(np.std(np.array(local), ddof=1))


# ---------------------------------------------------------------------------
# Public per-feature functions (raw bars in, float out).
# ---------------------------------------------------------------------------


def open_30m_ret(data: Any) -> float:
    d = prepare_bars(data)
    return _open_30m_ret(d) if d is not None else math.nan


def close_30m_ret(data: Any) -> float:
    d = prepare_bars(data)
    return _close_30m_ret(d) if d is not None else math.nan


def morning_afternoon_spread(data: Any) -> float:
    d = prepare_bars(data)
    return _morning_afternoon_spread(d) if d is not None else math.nan


def intraday_rv(data: Any) -> float:
    d = prepare_bars(data)
    return _intraday_rv(d) if d is not None else math.nan


def intraday_trend_score(data: Any) -> float:
    d = prepare_bars(data)
    return _intraday_trend_score(d) if d is not None else math.nan


def intraday_reversal(data: Any) -> float:
    d = prepare_bars(data)
    return _intraday_reversal(d) if d is not None else math.nan


def volume_u_shape(data: Any) -> float:
    d = prepare_bars(data)
    return _volume_u_shape(d) if d is not None else math.nan


def last_hour_amount_share(data: Any) -> float:
    d = prepare_bars(data)
    return _last_hour_amount_share(d) if d is not None else math.nan


def vwap_path_pressure(data: Any) -> float:
    d = prepare_bars(data)
    return _vwap_path_pressure(d) if d is not None else math.nan


def intraday_vol_of_vol(data: Any) -> float:
    d = prepare_bars(data)
    return _intraday_vol_of_vol(d) if d is not None else math.nan


def intraday_features(data: Any) -> dict[str, float]:
    """Compute all F06 minute features, keyed by registry canonical id."""
    d = prepare_bars(data)
    names = list(FEATURE_IDS.values())
    if d is None or not _tradable(d):
        return {name: math.nan for name in names}
    values = [
        _open_30m_ret(d),
        _close_30m_ret(d),
        _morning_afternoon_spread(d),
        _intraday_rv(d),
        _intraday_trend_score(d),
        _intraday_reversal(d),
        _volume_u_shape(d),
        _last_hour_amount_share(d),
        _vwap_path_pressure(d),
        _intraday_vol_of_vol(d),
    ]
    return dict(zip(names, values))


def group_intraday_features(data: Any) -> list[dict[str, Any]]:
    """Group rows by (symbol, trade date) and compute the full feature set."""
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _as_rows(data):
        key = (_value(row, "symbol"), _value(row, "trade_date"))
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for (symbol, trade_date), bars in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        features = intraday_features(bars)
        features["symbol"] = symbol
        features["trade_date"] = trade_date
        result.append(features)
    return result


def to_pyarrow_table(rows: Sequence[Mapping[str, Any]]):
    import pyarrow as pa

    return pa.Table.from_pylist(list(rows))


def download_partition(
    dataset: str, date: dt.date, destination: str | Path, **retry_kwargs: Any
) -> None:
    """Download one minute-bar partition through the approved COS wrapper.

    Reuses the retrying wrapper from :mod:`status_partition_runner`, which
    waits 120 seconds and retries up to 3 times on transient HTTP 400/524
    failures. The lazy import keeps this module importable and the feature
    functions free of any COS dependency.
    """
    from status_partition_runner import cos_cp

    key = f"{SOURCE_ROOT}/{dataset}/{date.isoformat()}.parquet"
    cos_cp(key, destination, **retry_kwargs)


__all__ = [
    "CN_TZ",
    "BUCKET",
    "SOURCE_ROOT",
    "FEATURE_IDS",
    "MinuteSeries",
    "prepare_bars",
    "to_asia_shanghai",
    "in_continuous_session",
    "open_30m_ret",
    "close_30m_ret",
    "morning_afternoon_spread",
    "intraday_rv",
    "intraday_trend_score",
    "intraday_reversal",
    "volume_u_shape",
    "last_hour_amount_share",
    "vwap_path_pressure",
    "intraday_vol_of_vol",
    "intraday_features",
    "group_intraday_features",
    "to_pyarrow_table",
    "download_partition",
]
