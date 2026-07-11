"""Leakage-safe, vectorized metrics for generic factor-pack evaluation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .batch_models import FactorRunRecord, SplitBoundaries


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "asset", "open", "high", "low", "close", "volume", "vwap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market frame missing required columns: {missing}")
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="raise")
    out["asset"] = out["asset"].astype(str)
    if out[["datetime", "asset"]].isna().any().any():
        raise ValueError("market frame contains null datetime/asset keys")
    if out.duplicated(["datetime", "asset"]).any():
        raise ValueError("market frame contains duplicate (datetime, asset) keys")
    return out.sort_values(["asset", "datetime"]).reset_index(drop=True)


def split_boundaries(frame: pd.DataFrame, config: Any) -> SplitBoundaries:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["datetime"]).dt.normalize().unique()))
    if len(dates) < 30:
        raise ValueError("evaluation requires at least 30 distinct dates")
    train_count = max(1, min(len(dates) - 2, int(len(dates) * config.train_fraction)))
    valid_count = max(
        train_count + 1,
        min(len(dates) - 1, int(len(dates) * (config.train_fraction + config.valid_fraction))),
    )
    return SplitBoundaries(
        train_end=str(dates[train_count - 1].date()),
        valid_end=str(dates[valid_count - 1].date()),
        first_date=str(dates[0].date()),
        last_date=str(dates[-1].date()),
    )


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
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError("factor purification requires MultiIndex(datetime, asset)")
    values.index = values.index.set_names(["datetime", "asset"])
    if values.index.has_duplicates:
        raise ValueError("factor purification requires unique keys")
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
    stacked = standardized.stack(future_stack=True).rename(series.name)
    stacked.index = stacked.index.set_names(["datetime", "asset"])
    return stacked.reindex(values.index).sort_index()


def price_forward_returns(
    frame: pd.DataFrame,
    horizons: Sequence[int],
    price_field: str,
    *,
    entry_lag: int,
) -> dict[int, pd.Series]:
    if price_field not in frame.columns:
        raise ValueError(f"forward price field not found: {price_field}")
    panel = frame.set_index(["datetime", "asset"])[price_field].astype(float).sort_index()
    grouped = panel.groupby(level="asset", group_keys=False)
    entry = grouped.shift(-int(entry_lag))
    output: dict[int, pd.Series] = {}
    for horizon in sorted(set(int(value) for value in horizons)):
        exit_price = grouped.shift(-(int(entry_lag) + horizon))
        result = (exit_price / entry - 1.0).replace([np.inf, -np.inf], np.nan)
        output[horizon] = result.rename(f"forward_return_{horizon}")
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
    dates = pd.DatetimeIndex(pd.to_datetime(index.get_level_values("datetime"))).normalize()
    calendar = pd.DatetimeIndex(trading_dates).normalize().unique().sort_values()
    positions = calendar.get_indexer(dates)
    exits = positions + int(entry_lag) + int(horizon)
    valid = (positions >= 0) & (exits >= 0) & (exits < len(calendar))
    exit_dates = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_dates[valid] = calendar.to_numpy(dtype="datetime64[ns]")[exits[valid]]
    exit_index = pd.DatetimeIndex(exit_dates)
    train_end = pd.Timestamp(bounds.train_end)
    valid_end = pd.Timestamp(bounds.valid_end)
    last_date = pd.Timestamp(bounds.last_date)
    if split == "train":
        return np.asarray(valid & (dates <= train_end) & (exit_index <= train_end))
    if split == "valid":
        return np.asarray(valid & (dates > train_end) & (dates <= valid_end) & (exit_index <= valid_end))
    if split == "test":
        return np.asarray(valid & (dates > valid_end) & (exit_index <= last_date))
    if split == "all":
        return np.asarray(valid)
    raise ValueError(f"unknown split: {split}")


def _row_corr(left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    left, right = left.align(right, join="outer", axis=0)
    left, right = left.align(right, join="outer", axis=1)
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.sum(axis=1).astype(int)
    safe = np.maximum(counts, 1)
    x_mean = np.where(valid, x, 0.0).sum(axis=1) / safe
    y_mean = np.where(valid, y, 0.0).sum(axis=1) / safe
    xc = np.where(valid, x - x_mean[:, None], 0.0)
    yc = np.where(valid, y - y_mean[:, None], 0.0)
    numerator = (xc * yc).sum(axis=1)
    denominator = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    corr = np.full(len(counts), np.nan, dtype=float)
    np.divide(numerator, denominator, out=corr, where=(counts >= 2) & (denominator > 0))
    return corr, counts


def daily_ic(signal: pd.Series, returns: pd.Series, *, min_assets: int) -> pd.DataFrame:
    joined = pd.concat(
        [pd.to_numeric(signal, errors="coerce").rename("signal"), pd.to_numeric(returns, errors="coerce").rename("return")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    signal_panel = joined["signal"].unstack("asset").sort_index()
    return_panel = joined["return"].unstack("asset").reindex(signal_panel.index)
    ic, counts = _row_corr(signal_panel, return_panel)
    rank_ic, _ = _row_corr(
        signal_panel.rank(axis=1, method="average", na_option="keep"),
        return_panel.rank(axis=1, method="average", na_option="keep"),
    )
    eligible = counts >= int(min_assets)
    return pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(signal_panel.index[eligible]),
            "n": counts[eligible],
            "ic": ic[eligible],
            "rank_ic": rank_ic[eligible],
        }
    )


def _hac_variance(values: np.ndarray, lags: int) -> float:
    if len(values) == 0:
        return float("nan")
    demeaned = values - float(values.mean())
    n = len(values)
    variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, min(max(int(lags), 0), n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
    return max(variance, 0.0)


def summary_stats(values: pd.Series, *, hac_lags: int = 0) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {"count": 0, "mean": None, "std": None, "ir": None, "t_stat": None, "hac_t_stat": None, "hac_p_value": None, "positive_ratio": None}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    t_stat = mean / (std / math.sqrt(len(clean))) if math.isfinite(std) and std > 0 else float("nan")
    variance = _hac_variance(clean.to_numpy(dtype=float), hac_lags)
    hac_se = math.sqrt(variance / len(clean)) if math.isfinite(variance) else float("nan")
    hac_t = mean / hac_se if math.isfinite(hac_se) and hac_se > 0 else float("nan")
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(hac_t))) if math.isfinite(hac_t) else float("nan")
    return {
        "count": int(len(clean)),
        "mean": mean,
        "std": std if math.isfinite(std) else None,
        "ir": ir if math.isfinite(ir) else None,
        "t_stat": t_stat if math.isfinite(t_stat) else None,
        "hac_t_stat": hac_t if math.isfinite(hac_t) else None,
        "hac_p_value": p_value if math.isfinite(p_value) else None,
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
    joined = pd.concat([signal.rename("signal"), returns.rename("return")], axis=1).replace([np.inf, -np.inf], np.nan)
    signal_panel = joined["signal"].unstack("asset").sort_index()
    return_panel = joined["return"].unstack("asset").reindex(signal_panel.index)
    valid = signal_panel.notna() & return_panel.notna()
    counts = valid.sum(axis=1)
    ranks = signal_panel.where(valid).rank(axis=1, method="average", pct=True, na_option="keep")
    low = (ranks <= 1.0 / n_quantiles) & valid
    high = (ranks > 1.0 - 1.0 / n_quantiles) & valid
    high_count = high.sum(axis=1)
    low_count = low.sum(axis=1)
    eligible = (counts >= max(int(min_assets), int(n_quantiles) * 2)) & (high_count > 0) & (low_count > 0)
    if not bool(eligible.any()):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="datetime"))
        return empty.rename("long_short_gross"), empty.rename("long_short_net"), empty.rename("turnover")
    weights = high.astype(float).div(high_count.replace(0, np.nan), axis=0)
    weights -= low.astype(float).div(low_count.replace(0, np.nan), axis=0)
    weights = weights.loc[eligible].fillna(0.0)
    realized = return_panel.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    gross = (weights * realized).sum(axis=1)
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    turnover = 0.5 * changes.abs().sum(axis=1)
    net = gross - turnover * float(cost_bps) / 10_000.0
    return gross.rename("long_short_gross"), net.rename("long_short_net"), turnover.rename("turnover")


def performance_stats(
    returns: pd.Series,
    *,
    annualization: float,
    hac_lags: int = 0,
    holding_period: int = 1,
) -> dict[str, float | int | None]:
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {"count": 0, "mean": None, "annual_return": None, "annual_volatility": None, "sharpe": None, "hac_sharpe": None, "max_drawdown": None, "hit_rate": None}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    annual_return = mean * annualization
    annual_vol = std * math.sqrt(annualization) if math.isfinite(std) else float("nan")
    sharpe = annual_return / annual_vol if math.isfinite(annual_vol) and annual_vol > 0 else float("nan")
    hac_std = math.sqrt(_hac_variance(clean.to_numpy(dtype=float), hac_lags))
    hac_sharpe = mean / hac_std * math.sqrt(252.0) if math.isfinite(hac_std) and hac_std > 0 else float("nan")
    period = max(int(holding_period), 1)
    drawdowns: list[float] = []
    for offset in range(min(period, len(clean))):
        sleeve = clean.iloc[offset::period]
        wealth = (1.0 + sleeve.clip(lower=-0.999999)).cumprod()
        drawdowns.append(float((wealth / wealth.cummax() - 1.0).min()))
    return {
        "count": int(len(clean)),
        "mean": mean,
        "annual_return": annual_return,
        "annual_volatility": annual_vol if math.isfinite(annual_vol) else None,
        "sharpe": sharpe if math.isfinite(sharpe) else None,
        "hac_sharpe": hac_sharpe if math.isfinite(hac_sharpe) else None,
        "max_drawdown": min(drawdowns) if drawdowns else None,
        "hit_rate": float((clean > 0).mean()),
    }


def route_factor(metrics: Mapping[str, Any], *, coverage: float) -> str:
    primary = metrics.get("21") or metrics.get("5") or metrics.get("1") or {}
    validation = primary.get("valid", {})
    rank_ic = ((validation.get("rank_ic") or {}).get("mean"))
    icir = ((validation.get("rank_ic") or {}).get("ir"))
    positive = ((validation.get("rank_ic") or {}).get("positive_ratio"))
    hac_sharpe = ((validation.get("long_short_net") or {}).get("hac_sharpe"))
    if coverage < 0.35 or rank_ic is None or rank_ic <= 0:
        return "rejected"
    if rank_ic >= 0.03 and (icir or 0) >= 0.45 and (hac_sharpe or 0) >= 0.8 and (positive or 0) >= 0.55:
        return "tier3a_core"
    if rank_ic >= 0.015 and (icir or 0) >= 0.20 and (hac_sharpe or 0) >= 0.25:
        return "tier3b_satellite"
    return "tier2_research"


def _bh_qvalues(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    q = np.ones(n, dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = n - reverse_rank + 1
        candidate = min(1.0, float(p_values[index]) * n / rank)
        running = min(running, candidate)
        q[index] = running
    return q.tolist()


def apply_validation_fdr(records: Sequence[FactorRunRecord], *, alpha: float) -> None:
    successful = [record for record in records if record.status == "success"]
    p_values: list[float] = []
    for record in successful:
        primary = record.metrics.get("21") or record.metrics.get("5") or record.metrics.get("1") or {}
        raw = ((primary.get("valid", {}).get("rank_ic") or {}).get("hac_p_value"))
        p_values.append(float(raw) if raw is not None and math.isfinite(float(raw)) else 1.0)
    q_values = _bh_qvalues(p_values)
    for record, p_value, q_value in zip(successful, p_values, q_values, strict=True):
        record.metrics["multiple_testing"] = {
            "family_size": len(successful),
            "validation_rank_ic_p_value": p_value,
            "validation_rank_ic_q_value": q_value,
            "fdr_alpha": float(alpha),
            "passed": bool(q_value <= alpha),
        }
        if q_value > alpha and record.route in {"tier3a_core", "tier3b_satellite"}:
            record.route = "tier2_research"


def ranking_rows(records: Sequence[FactorRunRecord]) -> list[dict[str, Any]]:
    score = {"tier3a_core": 3, "tier3b_satellite": 2, "tier2_research": 1, "rejected": 0}
    rows: list[dict[str, Any]] = []
    for record in records:
        primary = record.metrics.get("21") or record.metrics.get("5") or record.metrics.get("1") or {}
        validation = primary.get("valid", {})
        test = primary.get("test", {})
        multiple = record.metrics.get("multiple_testing", {})
        rows.append(
            {
                "factor_name": record.factor_name,
                "status": record.status,
                "route": record.route,
                "route_score": score.get(record.route, -1),
                "coverage": record.coverage,
                "direction": record.direction,
                "validation_rank_ic": ((validation.get("rank_ic") or {}).get("mean")),
                "validation_rank_ic_ir": ((validation.get("rank_ic") or {}).get("ir")),
                "validation_rank_ic_q_value": multiple.get("validation_rank_ic_q_value"),
                "validation_long_short_net_hac_sharpe": ((validation.get("long_short_net") or {}).get("hac_sharpe")),
                "test_rank_ic": ((test.get("rank_ic") or {}).get("mean")),
                "test_long_short_net_hac_sharpe": ((test.get("long_short_net") or {}).get("hac_sharpe")),
                "compile_seconds": record.compile_seconds,
                "execute_seconds": record.execute_seconds,
                "error": record.error,
            }
        )
    rows.sort(
        key=lambda row: (
            row["route_score"],
            row["validation_rank_ic_ir"] if row["validation_rank_ic_ir"] is not None else -999,
            row["validation_long_short_net_hac_sharpe"] if row["validation_long_short_net_hac_sharpe"] is not None else -999,
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
