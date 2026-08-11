# -*- coding: utf-8 -*-
"""Prediction evaluation with explicit OOS and cross-sectional conventions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

__all__ = [
    "EvaluationContractError", "ICMetricConvention", "MetricValue",
    "per_date_rank_ic", "ic_series", "block_aware_ic", "cross_sectional_ic",
    "evaluate_predictions", "EvaluationReport",
]


class EvaluationContractError(ValueError):
    """Evaluation input violates an auditable metric contract."""


@dataclass(frozen=True)
class ICMetricConvention:
    correlation: str = "spearman"
    aggregation: str = "daily_cross_sectional"
    std_ddof: int = 1
    min_pairs_per_date: int = 3

    def __post_init__(self) -> None:
        if self.correlation != "spearman" or self.aggregation != "daily_cross_sectional":
            raise EvaluationContractError("IC must be daily cross-sectional Spearman")
        if self.std_ddof != 1:
            raise EvaluationContractError("daily IC standard deviation must use ddof=1")
        if self.min_pairs_per_date < 3:
            raise EvaluationContractError("min_pairs_per_date must be >= 3")

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation": self.correlation, "aggregation": self.aggregation,
            "std_ddof": self.std_ddof, "min_pairs_per_date": self.min_pairs_per_date,
        }


DEFAULT_IC_CONVENTION = ICMetricConvention()


@dataclass(frozen=True)
class MetricValue:
    value: float = float("nan")
    status: str = "undefined"
    reason: str | None = None
    n_obs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "status": self.status, "reason": self.reason, "n_obs": self.n_obs}


def _array(name: str, values: Any, n: int | None = None) -> np.ndarray:
    out = np.asarray(values).ravel()
    if n is not None and len(out) != n:
        raise EvaluationContractError(f"{name} must be row-aligned; expected {n}, got {len(out)}")
    return out


def _weights(values: Any, n: int) -> np.ndarray | None:
    if values is None:
        return None
    out = _array("weights", values, n).astype(np.float64)
    if not np.isfinite(out).all() or np.any(out < 0) or not np.any(out > 0):
        raise EvaluationContractError("weights must be finite, non-negative, and contain positive mass")
    return out


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    xm, ym = float(np.sum(w * x)), float(np.sum(w * y))
    vx, vy = float(np.sum(w * (x - xm) ** 2)), float(np.sum(w * (y - ym) ** 2))
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sqrt(vx * vy))


def per_date_rank_ic(pred, y, dates, *, weights=None, convention=DEFAULT_IC_CONVENTION):
    """Return dates and Spearman ICs for valid daily cross-sections."""
    pred = _array("pred", pred).astype(np.float64)
    y, dates = _array("y", y, len(pred)).astype(np.float64), _array("dates", dates, len(pred))
    w = _weights(weights, len(pred))
    data = {"pred": pred, "y": y, "date": dates}
    if w is not None:
        data["weight"] = w
    frame = pd.DataFrame(data)
    frame = frame[np.isfinite(frame.pred) & np.isfinite(frame.y)]
    out_dates, out = [], []
    for date, sub in frame.groupby("date", observed=True, sort=True):
        p, yy = sub.pred.to_numpy(), sub.y.to_numpy()
        if len(p) < convention.min_pairs_per_date or np.ptp(p) == 0 or np.ptp(yy) == 0:
            continue
        if w is None:
            rho = spearmanr(p, yy)[0]
        else:
            sw = sub.weight.to_numpy()
            positive = sw > 0
            if positive.sum() < convention.min_pairs_per_date:
                continue
            rho = _weighted_corr(rankdata(p[positive]), rankdata(yy[positive]), sw[positive])
        if np.isfinite(rho):
            out_dates.append(date)
            out.append(float(rho))
    return out_dates, np.asarray(out, dtype=np.float64)


def ic_series(pred, y, dates, **kwargs):
    return per_date_rank_ic(pred, y, dates, **kwargs)


def cross_sectional_ic(y_true, y_pred, dates, *, weights=None, convention=DEFAULT_IC_CONVENTION):
    daily_dates, ics = per_date_rank_ic(y_pred, y_true, dates, weights=weights, convention=convention)
    mean_ic = float(np.mean(ics)) if len(ics) else float("nan")
    sd = float(np.std(ics, ddof=convention.std_ddof)) if len(ics) >= 2 else float("nan")
    icir = mean_ic / sd if np.isfinite(sd) and sd > 1e-12 else float("nan")
    return {"rank_ic": mean_ic, "mean_daily_rank_ic": mean_ic, "icir": icir,
            "daily_rank_ic_ir": icir, "n_dates": len(daily_dates)}


def block_aware_ic(pred, y, dates, *, overlap_horizon: int, calendar_sessions=None, weights=None):
    """Aggregate daily IC by actual calendar-session position."""
    if overlap_horizon < 1:
        raise EvaluationContractError("overlap_horizon must be >= 1")
    valid_dates, ics = per_date_rank_ic(pred, y, dates, weights=weights)
    if not valid_dates:
        return [], np.array([], dtype=np.float64)
    sessions = list(calendar_sessions) if calendar_sessions is not None else sorted(set(_array("dates", dates)))
    if len(sessions) != len(set(sessions)):
        raise EvaluationContractError("calendar_sessions must be unique")
    positions = {date: pos for pos, date in enumerate(sessions)}
    if any(date not in positions for date in valid_dates):
        raise EvaluationContractError("IC date absent from calendar snapshot")
    blocks: dict[int, list[tuple[Any, float]]] = {}
    for date, ic in zip(valid_dates, ics):
        blocks.setdefault(positions[date] // overlap_horizon, []).append((date, float(ic)))
    ordered = sorted(blocks.items())
    return ([values[0][0] for _, values in ordered],
            np.asarray([np.mean([ic for _, ic in values]) for _, values in ordered]))


@dataclass
class EvaluationReport:
    mse: float = float("nan")
    mae: float = float("nan")
    rank_ic: float = float("nan")
    pearson_ic: float = float("nan")
    icir: float = float("nan")
    ic_positive_ratio: float = float("nan")
    long_short_spread: float = float("nan")
    turnover: float = float("nan")
    coverage: float = 0.0
    subperiod_stability: float = float("nan")
    mean_daily_rank_ic: float = float("nan")
    daily_rank_ic_ir: float = float("nan")
    pooled_rank_correlation: float = float("nan")
    year_by_year: dict[str, float] = field(default_factory=dict)
    rolling_oos_ic: list[dict[str, Any]] = field(default_factory=list)
    bull_bear: dict[str, float] | None = None
    large_small_cap: dict[str, float] | None = None
    liquidity_bucket: dict[str, float] | None = None
    coverage_layers: dict[str, float] = field(default_factory=dict)
    portfolio_support: dict[str, Any] = field(default_factory=dict)
    metric_values: dict[str, MetricValue] = field(default_factory=dict)
    metric_convention: dict[str, Any] = field(default_factory=dict)
    evaluation_version: str = "r41-v1"

    def to_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "metric_values"}
        out["metric_values"] = {name: value.to_dict() for name, value in self.metric_values.items()}
        return out


def _pooled_rank_ic(pred, y):
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(pred[mask]) == 0 or np.ptp(y[mask]) == 0:
        return float("nan")
    return float(spearmanr(pred[mask], y[mask])[0])


def _portfolio_metrics(frame: pd.DataFrame, top=0.1):
    spreads, top_sets = [], []
    skipped = {"insufficient_finite_pairs": 0}
    for date, sub in frame.groupby("date", observed=True, sort=True):
        sub = sub[np.isfinite(sub.pred) & np.isfinite(sub.y)]
        if len(sub) < 10:
            skipped["insufficient_finite_pairs"] += 1
            continue
        ranked = sub.sort_values(["pred", "security_id"], kind="mergesort")
        k = max(1, int(round(len(ranked) * top)))
        top_rows, bottom_rows = ranked.iloc[-k:], ranked.iloc[:k]
        spreads.append(float(top_rows.y.mean() - bottom_rows.y.mean()))
        top_sets.append((date, set(top_rows.security_id)))
    turnovers = []
    for (_, previous), (_, current) in zip(top_sets, top_sets[1:]):
        union = previous | current
        turnovers.append(1.0 - len(previous & current) / len(union) if union else 0.0)
    support = {"n_valid_dates": len(spreads), "skipped_dates_by_reason": skipped}
    return (float(np.mean(spreads)) if spreads else float("nan"),
            float(np.mean(turnovers)) if turnovers else float("nan"), support)


def _grouped_mean_ic(pred, y, dates, labels, asof, name):
    labels, asof = _array(name, labels, len(pred)), _array(f"{name}_asof_dates", asof, len(pred))
    try:
        anchor = pd.to_datetime(pd.Series(dates), errors="raise")
        known = pd.to_datetime(pd.Series(asof), errors="raise")
    except Exception as exc:
        raise EvaluationContractError(f"{name} PIT timestamps are invalid") from exc
    if pd.isna(labels).any() or known.isna().any() or (known > anchor).any():
        raise EvaluationContractError(f"{name} contains missing or future-known labels")
    frame = pd.DataFrame({"pred": pred, "y": y, "date": dates, "group": labels})
    out = {}
    for group, sub in frame.groupby("group", observed=True, sort=True):
        _, ics = per_date_rank_ic(sub.pred, sub.y, sub.date)
        if len(ics):
            out[str(group)] = float(np.mean(ics))
    return out or None


def _year_by_year(pred, y, dates):
    try:
        parsed = pd.to_datetime(pd.Series(dates), errors="raise")
    except Exception as exc:
        raise EvaluationContractError("dates must be valid calendar timestamps") from exc
    if parsed.isna().any():
        raise EvaluationContractError("dates must not contain missing timestamps")
    frame = pd.DataFrame({"pred": pred, "y": y, "date": dates, "year": parsed.dt.year})
    out = {}
    for year, sub in frame.groupby("year", observed=True, sort=True):
        _, ics = per_date_rank_ic(sub.pred, sub.y, sub.date)
        if len(ics):
            out[str(year)] = float(np.mean(ics))
    return out


def evaluate_predictions(pred, y, dates, security_ids=None, *, date_col=None, stock_col=None,
                         weights=None, eligible=None, extra: Mapping[str, np.ndarray] | None = None,
                         convention=DEFAULT_IC_CONVENTION):
    """Compute OOS metrics; stable security identity is required for portfolio metrics."""
    del date_col, stock_col
    pred = _array("pred", pred).astype(np.float64)
    y, dates = _array("y", y, len(pred)).astype(np.float64), _array("dates", dates, len(pred))
    w = _weights(weights, len(pred))
    ids = None if security_ids is None else _array("security_ids", security_ids, len(pred))
    if ids is not None:
        identity = pd.DataFrame({"date": dates, "security_id": ids})
        if identity.security_id.isna().any() or identity.duplicated().any():
            raise EvaluationContractError("security_ids must be non-missing and unique within each date")
    finite = np.isfinite(pred) & np.isfinite(y)
    metric_w = np.ones(len(pred)) if w is None else w
    mse = float(np.average((pred[finite] - y[finite]) ** 2, weights=metric_w[finite])) if metric_w[finite].sum() else float("nan")
    mae = float(np.average(np.abs(pred[finite] - y[finite]), weights=metric_w[finite])) if metric_w[finite].sum() else float("nan")
    pearson = float("nan")
    if finite.sum() >= 3 and np.ptp(pred[finite]) > 0 and np.ptp(y[finite]) > 0:
        pearson = float(pearsonr(pred[finite], y[finite])[0]) if w is None else _weighted_corr(pred[finite], y[finite], w[finite])
    daily_dates, ics = per_date_rank_ic(pred, y, dates, weights=w, convention=convention)
    mean_daily = float(np.mean(ics)) if len(ics) else float("nan")
    sd = float(np.std(ics, ddof=1)) if len(ics) >= 2 else float("nan")
    daily_ir = mean_daily / sd if np.isfinite(sd) and sd > 1e-12 else float("nan")
    spread = turnover = float("nan")
    support = {"n_valid_dates": 0, "skipped_dates_by_reason": {"missing_security_ids": 1}}
    if ids is not None:
        spread, turnover, support = _portfolio_metrics(pd.DataFrame(
            {"pred": pred, "y": y, "date": dates, "security_id": ids}))
    if eligible is None:
        eligible_mask = np.ones(len(pred), dtype=bool)
    else:
        eligible_mask = _array("eligible", eligible, len(pred))
        if eligible_mask.dtype != np.bool_:
            raise EvaluationContractError("eligible must be a boolean mask")
    n_eligible = int(eligible_mask.sum())
    coverage = float((eligible_mask & np.isfinite(pred)).sum() / n_eligible) if n_eligible else 0.0
    report = EvaluationReport(
        mse=mse, mae=mae, rank_ic=mean_daily, pearson_ic=pearson, icir=daily_ir,
        ic_positive_ratio=float(np.mean(ics > 0)) if len(ics) else float("nan"),
        long_short_spread=spread, turnover=turnover, coverage=coverage,
        subperiod_stability=sd, mean_daily_rank_ic=mean_daily, daily_rank_ic_ir=daily_ir,
        pooled_rank_correlation=_pooled_rank_ic(pred, y), year_by_year=_year_by_year(pred, y, dates),
        rolling_oos_ic=[{"date": str(date), "rank_ic": float(ic)} for date, ic in zip(daily_dates, ics)],
        coverage_layers={
            "eligible": float(n_eligible / len(pred)) if len(pred) else 0.0,
            "feature_available": float(n_eligible / len(pred)) if len(pred) else 0.0,
            "scored": coverage,
            "label_mature": float((eligible_mask & np.isfinite(y)).sum() / n_eligible) if n_eligible else 0.0,
            "evaluated": float((eligible_mask & finite).sum() / n_eligible) if n_eligible else 0.0,
        },
        portfolio_support=support,
        metric_values={
            "daily_rank_ic_ir": MetricValue(daily_ir, "defined" if np.isfinite(daily_ir) else "undefined",
                None if np.isfinite(daily_ir) else "fewer than two non-constant daily cross-sections", len(ics)),
            "turnover": MetricValue(turnover, "defined" if np.isfinite(turnover) else "unavailable",
                None if np.isfinite(turnover) else "stable security_ids and at least two investable dates required",
                max(0, support["n_valid_dates"] - 1)),
        },
        metric_convention=convention.to_dict(),
    )
    if extra:
        for labels_key, asof_key, attr in (
            ("regime_labels", "regime_label_asof_dates", "bull_bear"),
            ("cap_labels", "cap_label_asof_dates", "large_small_cap"),
            ("liquidity_labels", "liquidity_label_asof_dates", "liquidity_bucket"),
        ):
            if extra.get(labels_key) is not None:
                if extra.get(asof_key) is None:
                    raise EvaluationContractError(f"{labels_key} requires {asof_key} PIT evidence")
                setattr(report, attr, _grouped_mean_ic(pred, y, dates, extra[labels_key], extra[asof_key], labels_key))
    return report
