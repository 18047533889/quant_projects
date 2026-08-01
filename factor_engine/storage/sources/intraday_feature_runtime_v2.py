# -*- coding: utf-8 -*-
"""Session-clock and execution-local cache for intraday→daily aggregation.

Minute timestamp semantics are explicit:

- ``bar_end``: regular bars are labelled ``(session_open, session_close]``;
- ``bar_start``: regular bars are labelled ``[session_open, session_close)``.

This prevents the common 391-observation/79-five-minute-bar off-by-one error for
US sessions and prevents A-share lunch boundaries from sharing a bucket.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import intraday_feature_extension as base

_TIMESTAMP_CONVENTIONS = frozenset({"bar_end", "bar_start"})
_PROFILE_FEATURES = frozenset({
    "profile_zscore",
    "profile_deviation",
    "abnormal_volume_profile",
    "abnormal_return_profile",
    "abnormal_vol_profile",
})
_FEATURE_MIN_BARS: dict[str, int] = {
    "realized_skew": 4,
    "realized_kurtosis": 5,
    "realized_quarticity": 5,
    "bipower_variation": 3,
    "jump_variation": 3,
    "jump_ratio": 3,
    "trend_slope": 3,
    "trend_r2": 3,
    "return_autocorr": 4,
    "return_volume_corr": 4,
    "abs_return_volume_corr": 4,
    "vwap_slope": 3,
    "volume_profile_slope": 3,
}


def _hm(text: str) -> int:
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid HH:MM {text!r}")
    hour, minute = (int(value) for value in parts)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid HH:MM {text!r}")
    return hour * 60 + minute


def _session_segments(
    dataset: str,
    session_open: str,
    session_close: str,
) -> list[tuple[int, int]]:
    if dataset == "ashare_stock_minute":
        return [(_hm("09:30"), _hm("11:30")), (_hm("13:00"), _hm("15:00"))]
    start, stop = _hm(session_open), _hm(session_close)
    if stop <= start:
        raise ValueError("overnight regular sessions require an explicit calendar adapter")
    return [(start, stop)]


def _timestamp_convention(source: Any, params: dict[str, Any]) -> str:
    inner_params = dict(getattr(source.inner, "params", {}) or {})
    value = str(
        params.get("timestamp_convention")
        or inner_params.get("timestamp_convention")
        or "bar_end"
    ).strip().lower()
    if value not in _TIMESTAMP_CONVENTIONS:
        raise ValueError(
            "timestamp_convention must be 'bar_end' or 'bar_start'"
        )
    return value


def _effective_minutes(
    dataset: str,
    session_open: str,
    session_close: str,
    cutoff: str,
) -> int:
    end = _hm(cutoff)
    total = 0
    for start, stop in _session_segments(dataset, session_open, session_close):
        if end > start:
            total += max(0, min(end, stop) - start)
    return max(0, total)


def _ordinal(
    timestamps: pd.Series,
    dataset: str,
    session_open: str,
    session_close: str,
    timestamp_convention: str = "bar_end",
) -> pd.Series:
    convention = str(timestamp_convention).lower()
    if convention not in _TIMESTAMP_CONVENTIONS:
        raise ValueError("unsupported timestamp convention")

    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    output = pd.Series(np.nan, index=timestamps.index, dtype=float)
    offset = 0
    for start, stop in _session_segments(dataset, session_open, session_close):
        if convention == "bar_end":
            mask = (minute_of_day > start) & (minute_of_day <= stop)
            local = minute_of_day.loc[mask] - start - 1
        else:
            mask = (minute_of_day >= start) & (minute_of_day < stop)
            local = minute_of_day.loc[mask] - start
        output.loc[mask] = offset + local
        offset += stop - start
    return output


def _clock_bars(
    group: pd.DataFrame,
    bar_minutes: int,
    dataset: str,
    session_open: str,
    session_close: str,
    timestamp_convention: str = "bar_end",
) -> pd.DataFrame:
    width = int(bar_minutes)
    if width <= 0:
        raise ValueError("bar_minutes must be positive")
    group = group.sort_values("timestamp").copy()
    ordinal = _ordinal(
        group["timestamp"],
        dataset,
        session_open,
        session_close,
        timestamp_convention,
    )
    valid = ordinal.notna()
    group = group.loc[valid].copy()
    if group.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "amount"]
        )
    group["_slot"] = (ordinal.loc[valid].astype(int) // width).to_numpy()

    rows: list[dict[str, float | pd.Timestamp]] = []
    for _, bars in group.groupby("_slot", sort=True):
        rows.append(
            {
                "timestamp": bars["timestamp"].iloc[-1],
                "open": float(bars["open"].iloc[0]),
                "high": float(bars["high"].max()),
                "low": float(bars["low"].min()),
                "close": float(bars["close"].iloc[-1]),
                "volume": float(
                    pd.to_numeric(bars["volume"], errors="coerce").fillna(0).sum()
                ),
                "amount": float(
                    pd.to_numeric(bars["amount"], errors="coerce").fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cache(source: Any, name: str) -> dict[Any, Any]:
    attribute = f"_intraday_{name}_cache"
    cache = getattr(source, attribute, None)
    if cache is None:
        cache = {}
        setattr(source, attribute, cache)
    return cache


def _safe_index_signature(index: Any) -> tuple[Any, ...]:
    try:
        length = len(index)
    except Exception:
        return ("unknown",)
    if length == 0:
        return (0, None, None)
    try:
        timestamps = pd.to_datetime(index.get_level_values(0))
        instruments = index.get_level_values(1)
        return (
            length,
            str(timestamps.min()),
            str(timestamps.max()),
            int(pd.Index(instruments).nunique()),
        )
    except Exception:
        return (length, str(index[0]), str(index[-1]))


def _source_signature(source: Any, dataset: str, history_days: int) -> tuple[Any, ...]:
    inner = source.inner
    instrument_filter = getattr(inner, "instrument_filter", None)
    if instrument_filter is None:
        instrument_key: tuple[str, ...] | None = None
    elif isinstance(instrument_filter, str):
        instrument_key = (instrument_filter,)
    else:
        instrument_key = tuple(sorted(str(value) for value in instrument_filter))
    return (
        str(getattr(source, "_execution_id", "")),
        dataset,
        int(history_days),
        str(getattr(inner, "data_snapshot_id", "") or ""),
        str(getattr(inner, "start_date", "") or ""),
        str(getattr(inner, "end_date", "") or ""),
        instrument_key,
        _safe_index_signature(source._anchor_index()),
    )


def _grouped_bars(
    source: Any,
    dataset: str,
    session_open: str,
    session_close: str,
    cutoff: str,
    bar_minutes: int,
    min_coverage: float,
    history_days: int,
    timestamp_convention: str,
    minimum_bars: int,
):
    identity = _source_signature(source, dataset, history_days)
    key = (
        identity,
        session_open,
        session_close,
        cutoff,
        int(bar_minutes),
        float(min_coverage),
        timestamp_convention,
        int(minimum_bars),
    )
    grouped_cache = _cache(source, "bars")
    if key in grouped_cache:
        return grouped_cache[key]

    source_child = base._child(source, dataset, history_days)
    frame_key = identity
    frame_cache = _cache(source, "frame")
    if frame_key not in frame_cache:
        frame_cache[frame_key] = base._wide_frame(source_child)
    frame = frame_cache[frame_key].copy()
    frame = base._hhmm_filter(frame, session_open, cutoff)
    frame["date"] = frame["timestamp"].dt.normalize()

    usable_minutes = _effective_minutes(dataset, session_open, session_close, cutoff)
    if usable_minutes <= 0:
        raise ValueError("cutoff_time leaves no completed regular-session minutes")
    expected = max(1, int(math.ceil(usable_minutes / int(bar_minutes))))
    required = max(int(minimum_bars), int(math.ceil(expected * min_coverage)))
    if required > expected:
        raise ValueError(
            f"minimum bar requirement {required} exceeds expected completed bars {expected}"
        )

    grouped: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for (date, instrument), raw_group in frame.groupby(
        ["date", "instrument"], sort=True
    ):
        bars = _clock_bars(
            raw_group,
            bar_minutes,
            dataset,
            session_open,
            session_close,
            timestamp_convention,
        )
        # Duplicate/auction timestamps cannot inflate coverage beyond 100%.
        observed = min(len(bars), expected)
        coverage = observed / expected
        if observed >= required and coverage >= min_coverage:
            grouped.append((pd.Timestamp(date), str(instrument), bars.iloc[:expected]))

    grouped_cache[key] = (source_child, grouped, expected)
    return grouped_cache[key]


def load_intraday_feature(source: Any, params: dict[str, Any]):
    feature = str(params.get("feature") or "").strip()
    if not feature:
        raise ValueError("intraday_feature requires feature")

    bar_minutes = int(params.get("bar_minutes", 5))
    min_coverage = float(params.get("min_coverage", 0.8))
    history_days = int(params.get("history_days", 0))
    user_minimum = int(params.get("min_bars", 2))
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if history_days < 0:
        raise ValueError("history_days must be non-negative")
    if user_minimum < 2:
        raise ValueError("min_bars must be >= 2")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0,1]")

    dataset, session_open, session_close, _ = base._dataset_and_session(source, params)
    convention = _timestamp_convention(source, params)
    cutoff = str(params.get("cutoff_time") or "session_close")
    cutoff = session_close if cutoff == "session_close" else cutoff
    if _hm(cutoff) < _hm(session_open) or _hm(cutoff) > _hm(session_close):
        raise ValueError("cutoff_time must lie inside configured regular session")

    minimum_bars = max(user_minimum, _FEATURE_MIN_BARS.get(feature, 2))
    source_child, grouped, _ = _grouped_bars(
        source,
        dataset,
        session_open,
        session_close,
        cutoff,
        bar_minutes,
        min_coverage,
        history_days,
        convention,
        minimum_bars,
    )

    values: dict[tuple[pd.Timestamp, str], float] = {}
    if feature in _PROFILE_FEATURES:
        if history_days < 2:
            raise ValueError("profile features require history_days >= 2")
        values = base._profile_scores(grouped, feature, history_days)
    else:
        previous_close: dict[str, float] = {}
        for date, instrument, bars in grouped:
            values[(date, instrument)] = base._calc(
                feature,
                bars,
                params,
                prev_close=previous_close.get(instrument, np.nan),
            )
            previous_close[instrument] = float(bars["close"].iloc[-1])

    if hasattr(source, "_record_dependency"):
        source._record_dependency(
            dataset,
            kind="minute_to_daily",
            snapshot_id=getattr(source_child, "data_snapshot_id", None),
            feature=feature,
            bar_minutes=bar_minutes,
            timestamp_convention=convention,
            cutoff_time=cutoff,
            min_coverage=min_coverage,
            min_bars=minimum_bars,
            join_policy="exact_date",
        )

    anchor = source._anchor_index()
    if not values:
        return pd.Series(np.nan, index=anchor, name=f"intraday_{feature}")
    index = pd.MultiIndex.from_tuples(
        list(values), names=["timestamp", "instrument"]
    )
    series = pd.Series(
        list(values.values()), index=index, name=f"intraday_{feature}"
    ).sort_index()
    return source._align_exact_by_instrument(anchor, series)


def install_intraday_feature_runtime() -> None:
    from .lqtp_logical_source_v2 import LQTPLogicalDataSource

    if getattr(LQTPLogicalDataSource, "_intraday_feature_v2_installed", False):
        return
    original = LQTPLogicalDataSource._minute_daily

    def patched(self, field, transform, params):
        if transform == "intraday_feature":
            return load_intraday_feature(self, dict(params))
        return original(self, field, transform, params)

    LQTPLogicalDataSource._minute_daily = patched
    LQTPLogicalDataSource._intraday_feature_v2_installed = True
