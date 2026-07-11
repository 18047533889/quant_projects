"""Leakage-safe, vectorized metrics for GTJA185 evaluation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .gtja185_models import (
    BatchEvaluationConfig,
    FactorRunRecord,
    SplitBoundaries,
)


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "asset", "open", "high", "low", "close", "volume", "vwap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market frame missing required columns: {missing}")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="raise")
    result["asset"] = result["asset"].astype(str)
    if result[["datetime", "asset"]].isna().any().any():
        raise ValueError("market frame contains null datetime/asset keys")
    duplicated = result.duplicated(["datetime", "asset"], keep=False)
    if duplicated.any():
        sample = result.loc[duplicated, ["datetime", "asset"]].head(10).to_dict("records")
        raise ValueError(f"market frame contains duplicate keys: {sample}")
    return result.sort_values(["asset", "datetime"]).reset_index(drop=True)


def split_boundaries(frame: pd.DataFrame, config: BatchEvaluationConfig) -> SplitBoundaries:
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(frame["datetime"]).dt.normalize().unique())
    )
    if len(dates) < 30:
        raise ValueError("evaluation requires at least 30 distinct dates")
    train_index = max(1, min(len(dates) - 2, int(len(dates) * config.train_fraction)))
    valid_index = max(
        train_index + 1,
        min(len(dates) - 1, int(len(dates) * (config.train_fraction + config.valid_fraction))),
    )
    return SplitBoundaries(
        first_date=str(dates[0].date()),
        train_end=str(dates[train_index - 1].date()),
        valid_end=str(dates[valid_index - 1].date()),
        last_date=str(dates[-1].date()),
    )


def price_forward_returns(
    frame: pd.DataFrame,
    horizons: Sequence[int],
    price_field: str,
    *,
    entry_lag: int,
) -> dict[int, pd.Series]:
    """Return next-entry holding-period returns.

    A signal observed at ``t`` enters at ``t + entry_lag`` and exits after
    ``horizon`` additional trading observations. Same-bar fills are impossible.
    """
    if price_field not in frame.columns:
        raise ValueError(f"forward price field not found: {price_field}")
    if entry_lag < 1:
        raise ValueError("entry_lag must be >= 1")
    panel = (
        frame.set_index(["datetime", "asset"])[price_field]
        .astype(float)
        .sort_index()
    )
    grouped = panel.groupby(level="asset", group_keys=False)
    entry = grouped.shift(-entry_lag)
    output: dict[int, pd.Series] = {}
    for horizon in sorted(set(int(value) for value in horizons)):
        exit_price = grouped.shift(-(entry_lag + horizon))
        output[horizon] = (
            (exit_price / entry - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .rename(f"forward_return_{horizon}")
        )
    return output


def purged_split_mask(
    index: pd.MultiIndex,
    split: str,
    bounds: SplitBoundaries,
    trading_dates: pd.DatetimeIndex,
    *,
    entry_lag: int,
    horizon: int,
) -> np.ndarray:
    """Return a chronological split mask whose labels remain inside the split."""
    dates = pd.DatetimeIndex(pd.to_datetime(index.get_level_values("datetime"))).normalize()
    calendar = pd.DatetimeIndex(trading_dates).normalize().unique().sort_values()
    positions = calendar.get_indexer(dates)
    exit_positions = positions + int(entry_lag) + int(horizon)
    valid_exit = (positions >= 0) & (exit_positions >= 0) & (exit_positions < len(calendar))
    exit_values = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    exit_values[valid_exit] = calendar_values[exit_positions[valid_exit]]
    exit_dates = pd.DatetimeIndex(exit_values)
    train_end = pd.Timestamp(bounds.train_end)
    valid_end = pd.Timestamp(bounds.valid_end)
    last_date = pd.Timestamp(bounds.last_date)
    if split == "train":
        return np.asarray(valid_exit & (dates <= train_end) & (exit_dates <= train_end))
    if split == "valid":
        return np.asarray(
            valid_exit
            & (dates > train_end)
            & (dates <= valid_end)
            & (exit_dates <= valid_end)
        )
    if split == "test":
        return np.asarray(valid_exit & (dates > valid_end) & (exit_dates <= last_date))
    if split == "all":
        return np.asarray(valid_exit)
    raise ValueError(f"unknown split: {split}")


def apply_point_in_time_universe(
    market_frame: pd.DataFrame,
    universe_frame: pd.DataFrame,
    *,
    universe_id: str | None = None,
) -> pd.DataFrame:
    required = {"datetime", "asset", "is_member"}
    missing = sorted(required - set(universe_frame.columns))
    if missing:
        raise ValueError(f"universe frame missing required columns: {missing}")
    universe = universe_frame.copy()
    universe["datetime"] = pd.to_datetime(universe["datetime"], errors="raise")
    universe["asset"] = universe["asset"].astype(str)
    if universe_id is not None and "universe_id" in universe.columns:
        universe = universe.loc[universe["universe_id"].astype(str) == str(universe_id)].copy()
    if universe.duplicated(["datetime", "asset"]).any():
        raise ValueError("universe frame contains duplicate (datetime, asset) keys")
    columns = ["datetime", "asset", "is_member"]
    if "is_tradable" in universe.columns:
        columns.append("is_tradable")
    merged = market_frame.merge(
        universe[columns],
        on=["datetime", "asset"],
        how="left",
        validate="one_to_one",
    )
    eligible = merged["is_member"].fillna(False).astype(bool)
    if "is_tradable" in merged.columns:
        eligible &= merged["is_tradable"].fillna(False).astype(bool)
    filtered = merged.loc[eligible, market_frame.columns].copy()
    if filtered.empty:
        raise ValueError("point-in-time universe filter removed all market rows")
    return filtered.sort_values(["asset", "datetime"]).reset_index(drop=True)


def purify_series(
    series: pd.Series,
    *,
    mad_multiplier: float,
    min_assets: int,
) -> pd.Series:
    """Vectorized daily MAD winsorization followed by cross-sectional z-scoring."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError("factor purification requires MultiIndex(datetime, asset)")
    if values.index.has_duplicates:
        raise ValueError("factor purification requires unique (datetime, asset) keys")
    values.index = values.index.set_names(["datetime", "asset"])
    panel = values.unstack("asset").sort_index()
    counts = panel.notna().sum(axis=1)
    medians = panel.median(axis=1, skipna=True)
    mad = panel.sub(medians, axis=0).abs().median(axis=1, skipna=True)
    scale = 1.4826 * mad
    valid_scale = np.isfinite(scale) & (scale > 0)
    lower = (medians - float(mad_multiplier) * scale).where(valid_scale, -np.inf)
    upper = (medians + float(mad_multiplier) * scale).where(valid_scale, np.inf)
    clipped = panel.clip(lower=lower, upper=upper, axis=0)
    means = clipped.mean(axis=1, skipna=True)
    stds = clipped.std(axis=1, skipna=True, ddof=1)
    eligible = (counts >= int(min_assets)) & np.isfinite(stds) & (stds > 1e-12)
    standardized = clipped.sub(means, axis=0).div(stds, axis=0)
    standardized.loc[~eligible, :] = np.nan
    stacked = standardized.stack(dropna=False, future_stack=True).rename(series.name)
    stacked.index = stacked.index.set_names(["datetime", "asset"])
    return stacked.reindex(values.index).sort_index()


def _matrix_row_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = left.align(right, join="outer", axis=0)
    left, right = left.align(right, join="outer", axis=1)
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.sum(axis=1).astype(np.int64)
    safe_counts = np.maximum(counts, 1)
    x_mean = np.where(valid, x, 0.0).sum(axis=1) / safe_counts
    y_mean = np.where(valid, y, 0.0).sum(axis=1) / safe_counts
    x_centered = np.where(valid, x - x_mean[:, None], 0.0)
    y_centered = np.where(valid, y - y_mean[:, None], 0.0)
    numerator = (x_centered * y_centered).sum(axis=1)
    denominator = np.sqrt(
        (x_centered * x_centered).sum(axis=1)
        * (y_centered * y_centered).sum(axis=1)
    )
    correlations = np.full(len(counts), np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=correlations,
        where=(counts >= 2) & np.isfinite(denominator) & (denominator > 0),
    )
    return correlations, counts


def daily_ic(signal: pd.Series, returns: pd.Series, *, min_assets: int) -> pd.DataFrame:
    joined = pd.concat(
        [
            pd.to_numeric(signal, errors="coerce").rename("signal"),
            pd.to_numeric(returns, errors="coerce").rename("return"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    signal_panel = joined["signal"].unstack("asset").sort_index()
    return_panel = joined["return"].unstack("asset").reindex(signal_panel.index)
    ic, counts = _matrix_row_correlation(signal_panel, return_panel)
    rank_signal = signal_panel.rank(axis=1, method="average", na_option="keep")
    rank_return = return_panel.rank(axis=1, method="average", na_option="keep")
    rank_ic, _ = _matrix_row_correlation(rank_signal, rank_return)
    eligible = counts >= int(min_assets)
    if not eligible.any():
        return pd.DataFrame(columns=["datetime", "n", "ic", "rank_ic"])
    return pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(signal_panel.index[eligible]),
            "n": counts[eligible].astype(int),
            "ic": ic[eligible],
            "rank_ic": rank_ic[eligible],
        }
    )


def summary_stats(values: pd.Series, *, hac_lags: int = 0) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "ir": None,
            "t_stat": None,
            "hac_t_stat": None,
            "hac_p_value": None,
            "positive_ratio": None,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    t_stat = mean / (std / math.sqrt(len(clean))) if math.isfinite(std) and std > 0 else float("nan")
    array = clean.to_numpy(dtype=float)
    demeaned = array - mean
    n = len(array)
    max_lag = min(max(int(hac_lags), 0), max(n - 1, 0))
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(long_run_variance, 0.0)
    hac_se = math.sqrt(long_run_variance / n) if n else float("nan")
    hac_t = mean / hac_se if math.isfinite(hac_se) and hac_se > 0 else float("nan")
    hac_p = math.erfc(abs(hac_t) / math.sqrt(2.0)) if math.isfinite(hac_t) else float("nan")
    return {
        "count": int(len(clean)),
        "mean": mean,
        "std": std if math.isfinite(std) else None,
        "ir": ir if math.isfinite(ir) else None,
        "t_stat": t_stat if math.isfinite(t_stat) else None,
        "hac_t_stat": hac_t if math.isfinite(hac_t) else None,
        "hac_p_value": hac_p if math.isfinite(hac_p) else None,
        "positive_ratio": float((clean > 0).mean()),
    }


def long_short_series(
    signal: pd.Series,
    returns: pd.Series,
    *,
    min_assets: int,
    n_quantiles: int,
    cost_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    joined = pd.concat(
        [
            pd.to_numeric(signal, errors="coerce").rename("signal"),
            pd.to_numeric(returns, errors="coerce").rename("return"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    signal_panel = joined["signal"].unstack("asset").sort_index()
    return_panel = joined["return"].unstack("asset").reindex(signal_panel.index)
    valid = signal_panel.notna() & return_panel.notna()
    counts = valid.sum(axis=1)
    ranks = signal_panel.where(valid).rank(
        axis=1,
        method="average",
        pct=True,
        na_option="keep",
    )
    low_cut = 1.0 / int(n_quantiles)
    high_cut = 1.0 - low_cut
    bottom = (ranks <= low_cut) & valid
    top = (ranks > high_cut) & valid
    top_counts = top.sum(axis=1)
    bottom_counts = bottom.sum(axis=1)
    eligible = (
        (counts >= max(int(min_assets), int(n_quantiles) * 2))
        & (top_counts > 0)
        & (bottom_counts > 0)
    )
    if not bool(eligible.any()):
        empty = pd.DatetimeIndex([], name="datetime")
        return (
            pd.Series(index=empty, dtype=float, name="long_short_gross"),
            pd.Series(index=empty, dtype=float, name="long_short_net"),
            pd.Series(index=empty, dtype=float, name="turnover"),
        )
    weights = top.astype(float).div(top_counts.replace(0, np.nan), axis=0)
    weights -= bottom.astype(float).div(bottom_counts.replace(0, np.nan), axis=0)
    weights = weights.loc[eligible].fillna(0.0)
    realized = return_panel.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    gross = (weights * realized).sum(axis=1).astype(float)
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    turnover = 0.5 * changes.abs().sum(axis=1)
    net = gross - turnover * float(cost_bps) / 10_000.0
    for value in (gross, net, turnover):
        value.index.name = "datetime"
    return (
        gross.rename("long_short_gross"),
        net.rename("long_short_net"),
        turnover.rename("turnover"),
    )


def performance_stats(
    returns: pd.Series,
    *,
    annualization: float,
    hac_lags: int = 0,
    holding_period: int = 1,
) -> dict[str, float | int | None]:
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "hac_sharpe": None,
            "hac_long_run_volatility": None,
            "max_drawdown": None,
            "max_drawdown_non_overlapping_worst": None,
            "hit_rate": None,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    annual_return = mean * float(annualization)
    annual_vol = std * math.sqrt(float(annualization)) if math.isfinite(std) else float("nan")
    sharpe = annual_return / annual_vol if math.isfinite(annual_vol) and annual_vol > 0 else float("nan")
    array = clean.to_numpy(dtype=float)
    demeaned = array - mean
    n = len(array)
    max_lag = min(max(int(hac_lags), 0), max(n - 1, 0))
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(long_run_variance, 0.0)
    long_run_std = math.sqrt(long_run_variance)
    hac_sharpe = (
        mean / long_run_std * math.sqrt(float(annualization))
        if math.isfinite(long_run_std) and long_run_std > 0
        else float("nan")
    )
    hac_long_run_vol = long_run_std * math.sqrt(float(annualization))
    period = max(int(holding_period), 1)
    drawdowns: list[float] = []
    for offset in range(min(period, n)):
        sleeve = clean.iloc[offset::period]
        if sleeve.empty:
            continue
        wealth = (1.0 + sleeve.clip(lower=-0.999999)).cumprod()
        drawdowns.append(float((wealth / wealth.cummax() - 1.0).min()))
    worst_drawdown = min(drawdowns) if drawdowns else float("nan")
    return {
        "count": int(len(clean)),
        "mean": mean,
        "annual_return": annual_return,
        "annual_volatility": annual_vol if math.isfinite(annual_vol) else None,
        "sharpe": sharpe if math.isfinite(sharpe) else None,
        "hac_sharpe": hac_sharpe if math.isfinite(hac_sharpe) else None,
        "hac_long_run_volatility": hac_long_run_vol if math.isfinite(hac_long_run_vol) else None,
        "max_drawdown": worst_drawdown if math.isfinite(worst_drawdown) else None,
        "max_drawdown_non_overlapping_worst": worst_drawdown if math.isfinite(worst_drawdown) else None,
        "hit_rate": float((clean > 0).mean()),
    }


def primary_horizon_metrics(record: FactorRunRecord) -> Mapping[str, Any]:
    return record.metrics.get("21") or record.metrics.get("5") or record.metrics.get("1") or {}


def route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    """Route with validation only; test remains a locked final holdout."""
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    validation = primary.get("valid", {})
    rank_stats = validation.get("rank_ic") or {}
    performance = validation.get("long_short_net") or {}
    rank_ic = rank_stats.get("mean")
    icir = rank_stats.get("ir")
    positive = rank_stats.get("positive_ratio")
    sharpe = performance.get("hac_sharpe")
    if sharpe is None:
        sharpe = performance.get("sharpe")
    if coverage < 0.35 or rank_ic is None or float(rank_ic) <= 0:
        return "rejected"
    if (
        float(rank_ic) >= 0.03
        and float(icir or 0.0) >= 0.45
        and float(sharpe or 0.0) >= 0.8
        and float(positive or 0.0) >= 0.55
    ):
        return "tier3a_core"
    if (
        float(rank_ic) >= 0.015
        and float(icir or 0.0) >= 0.20
        and float(sharpe or 0.0) >= 0.25
    ):
        return "tier3b_satellite"
    return "tier2_research"


def apply_validation_fdr(records: Sequence[FactorRunRecord], *, alpha: float) -> None:
    candidates: list[tuple[int, float]] = []
    for index, record in enumerate(records):
        if record.status != "success":
            continue
        validation = primary_horizon_metrics(record).get("valid", {})
        p_value = ((validation.get("rank_ic") or {}).get("hac_p_value"))
        if p_value is None:
            continue
        value = float(p_value)
        if math.isfinite(value) and 0 <= value <= 1:
            candidates.append((index, value))
    ordered = sorted(candidates, key=lambda item: item[1])
    family_size = len(ordered)
    q_values: dict[int, float] = {}
    running = 1.0
    for reverse_position, (record_index, p_value) in enumerate(reversed(ordered), start=1):
        rank = family_size - reverse_position + 1
        running = min(running, min(1.0, p_value * family_size / max(rank, 1)))
        q_values[record_index] = running
    for index, record in enumerate(records):
        q_value = q_values.get(index)
        record.metrics["multiple_testing"] = {
            "method": "benjamini_hochberg",
            "family_size": family_size,
            "alpha": float(alpha),
            "validation_rank_ic_q_value": q_value,
            "passed": q_value is not None and q_value <= alpha,
        }
        if (
            record.status == "success"
            and record.route in {"tier3a_core", "tier3b_satellite"}
            and (q_value is None or q_value > alpha)
        ):
            record.route = "tier2_research"


def ranking_rows(records: Sequence[FactorRunRecord]) -> list[dict[str, Any]]:
    route_score = {
        "tier3a_core": 3,
        "tier3b_satellite": 2,
        "tier2_research": 1,
        "rejected": 0,
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        primary = primary_horizon_metrics(record)
        validation = primary.get("valid", {})
        test = primary.get("test", {})
        multiple_testing = record.metrics.get("multiple_testing", {})
        rows.append(
            {
                "factor_name": record.factor_name,
                "status": record.status,
                "route": record.route,
                "route_score": route_score.get(record.route, -1),
                "coverage": record.coverage,
                "direction": record.direction,
                "validation_rank_ic": ((validation.get("rank_ic") or {}).get("mean")),
                "validation_rank_ic_ir": ((validation.get("rank_ic") or {}).get("ir")),
                "validation_rank_ic_hac_t": ((validation.get("rank_ic") or {}).get("hac_t_stat")),
                "validation_rank_ic_q_value": multiple_testing.get("validation_rank_ic_q_value"),
                "validation_long_short_net_hac_sharpe": ((validation.get("long_short_net") or {}).get("hac_sharpe")),
                "test_rank_ic": ((test.get("rank_ic") or {}).get("mean")),
                "test_rank_ic_ir": ((test.get("rank_ic") or {}).get("ir")),
                "test_rank_ic_positive_ratio": ((test.get("rank_ic") or {}).get("positive_ratio")),
                "test_long_short_net_hac_sharpe": ((test.get("long_short_net") or {}).get("hac_sharpe")),
                "test_long_short_net_annual_return": ((test.get("long_short_net") or {}).get("annual_return")),
                "test_long_short_net_max_drawdown": ((test.get("long_short_net") or {}).get("max_drawdown")),
                "turnover_mean": test.get("turnover_mean"),
                "compile_seconds": record.compile_seconds,
                "execute_seconds": record.execute_seconds,
                "error": record.error,
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("route_score", -1),
            -row["validation_rank_ic_q_value"]
            if row.get("validation_rank_ic_q_value") is not None
            else -999.0,
            row["validation_rank_ic_ir"]
            if row.get("validation_rank_ic_ir") is not None
            else -999.0,
            row["validation_long_short_net_hac_sharpe"]
            if row.get("validation_long_short_net_hac_sharpe") is not None
            else -999.0,
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
