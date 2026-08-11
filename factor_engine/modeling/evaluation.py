# -*- coding: utf-8 -*-
"""Prediction evaluation (§26 / §27 / §72 / §75).

Provides per-date rank IC, the pooled :class:`EvaluationReport` (§26), grouped
(§72) and block-aware (§75) IC variants.  Every metric is NaN-safe: missing
predictions / labels never raise; they are excluded pairwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

__all__ = [
    "per_date_rank_ic",
    "ic_series",
    "block_aware_ic",
    "cross_sectional_ic",
    "evaluate_predictions",
    "EvaluationReport",
]


def per_date_rank_ic(
    pred: np.ndarray, y: np.ndarray, dates
) -> tuple[list, np.ndarray]:
    """§27 — Spearman rank IC per date across stocks.

    Returns ``(dates_sorted, ics)``.  Dates with fewer than 3 finite pairs are
    dropped (a cross-sectional rank correlation is meaningless below 3).
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dates = np.asarray(dates).ravel()
    if len(pred) != len(y) or len(y) != len(dates):
        raise ValueError("pred, y, dates must be row-aligned")

    df = pd.DataFrame({"pred": pred, "y": y, "date": dates})
    df = df[np.isfinite(df["pred"]) & np.isfinite(df["y"])]
    if df.empty:
        return [], np.array([], dtype=np.float64)

    out_dates: list = []
    ics: list[float] = []
    for d, sub in df.groupby("date", observed=True):
        p = sub["pred"].to_numpy(dtype=np.float64)
        yy = sub["y"].to_numpy(dtype=np.float64)
        if len(p) < 3:
            continue
        # All-equal pred or y -> spearmanr returns NaN; skip (not evidence).
        if p.min() == p.max() or yy.min() == yy.max():
            continue
        rho, _ = spearmanr(p, yy)
        if not np.isfinite(rho):
            continue
        out_dates.append(d)
        ics.append(float(rho))
    return out_dates, np.asarray(ics, dtype=np.float64)


def ic_series(
    pred: np.ndarray, y: np.ndarray, dates
) -> tuple[list, np.ndarray]:
    """Convenience alias of :func:`per_date_rank_ic`."""
    return per_date_rank_ic(pred, y, dates)


def cross_sectional_ic(
    y_true: np.ndarray, y_pred: np.ndarray, dates
) -> dict[str, float]:
    """Cross-sectional IC bundle consumed by the trainer's validation scorer.

    Argument order is ``(y_true, y_pred, dates)`` (the trainer calls it with
    ``cross_sectional_ic(yv, pred, dv)``).  Returns a dict with ``rank_ic``
    (pooled Spearman) and ``icir`` (mean/std of per-date IC), both NaN-safe.
    """
    d, ics = per_date_rank_ic(y_pred, y_true, dates)
    rank_ic = _pooled_rank_ic(y_pred, y_true)
    if len(ics) >= 2:
        sd = float(np.std(ics))
        icir = float(np.mean(ics) / sd) if sd > 1e-12 else 0.0
    else:
        icir = float("nan")
    return {"rank_ic": rank_ic, "icir": icir, "n_dates": len(d)}


def block_aware_ic(
    pred: np.ndarray, y: np.ndarray, dates, *, overlap_horizon: int
) -> tuple[list, np.ndarray]:
    """§75 — block-aware IC.

    Per-date ICs are averaged within non-overlapping blocks of
    ``overlap_horizon`` consecutive dates.  With overlapping forward labels
    (horizon H) consecutive dates are NOT independent evidence; a block of size
    H treats each H-window as one observation, so the block count is the honest
    independent sample count.
    """
    if overlap_horizon < 1:
        raise ValueError("overlap_horizon must be >= 1")
    dates, ics = per_date_rank_ic(pred, y, dates)
    if len(dates) == 0:
        return [], np.array([], dtype=np.float64)
    block_dates: list = []
    block_ics: list[float] = []
    for i in range(0, len(dates), overlap_horizon):
        chunk = ics[i : i + overlap_horizon]
        block_dates.append(dates[i])
        block_ics.append(float(np.mean(chunk)))
    return block_dates, np.asarray(block_ics, dtype=np.float64)


@dataclass
class EvaluationReport:
    """§26 evaluation metrics.  All fields default to NaN-safe placeholders."""

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
    year_by_year: dict[str, float] = field(default_factory=dict)
    bull_bear: dict[str, float] | None = None
    large_small_cap: dict[str, float] | None = None
    liquidity_bucket: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "rank_ic": self.rank_ic,
            "pearson_ic": self.pearson_ic,
            "icir": self.icir,
            "ic_positive_ratio": self.ic_positive_ratio,
            "long_short_spread": self.long_short_spread,
            "turnover": self.turnover,
            "coverage": self.coverage,
            "subperiod_stability": self.subperiod_stability,
            "year_by_year": self.year_by_year,
            "bull_bear": self.bull_bear,
            "large_small_cap": self.large_small_cap,
            "liquidity_bucket": self.liquidity_bucket,
        }


def _pooled_rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    p, yy = pred[mask], y[mask]
    if p.min() == p.max() or yy.min() == yy.max():
        return float("nan")
    rho, _ = spearmanr(p, yy)
    return float(rho)


def _long_short_spread(df: pd.DataFrame, top: float = 0.1) -> float:
    """Per-date top-decile vs bottom-decile mean label, averaged over dates."""
    spreads: list[float] = []
    for _, sub in df.groupby("date", observed=True):
        sub = sub[np.isfinite(sub["pred"]) & np.isfinite(sub["y"])]
        if len(sub) < 10:
            continue
        order = np.argsort(sub["pred"].to_numpy(dtype=np.float64))
        k = max(1, int(round(len(sub) * top)))
        top_idx = order[-k:]
        bot_idx = order[:k]
        yv = sub["y"].to_numpy(dtype=np.float64)
        spreads.append(float(yv[top_idx].mean() - yv[bot_idx].mean()))
    return float(np.mean(spreads)) if spreads else float("nan")


def _turnover(df: pd.DataFrame, top: float = 0.1) -> float:
    """Mean 1 - overlap of top-decile membership between consecutive dates."""
    per_date = {}
    for d, sub in df.groupby("date", observed=True):
        sub = sub[np.isfinite(sub["pred"])]
        if len(sub) < 10:
            continue
        order = np.argsort(sub["pred"].to_numpy(dtype=np.float64))
        k = max(1, int(round(len(sub) * top)))
        per_date[d] = set(order[-k:].tolist())
    dates = sorted(per_date)
    if len(dates) < 2:
        return 0.0
    tov: list[float] = []
    for a, b in zip(dates, dates[1:]):
        s_a, s_b = per_date[a], per_date[b]
        inter = len(s_a & s_b)
        union = len(s_a | s_b)
        tov.append(1.0 - (inter / union if union else 1.0))
    return float(np.mean(tov)) if tov else 0.0


def _grouped_mean_ic(pred: np.ndarray, y: np.ndarray, dates, labels) -> dict[str, float] | None:
    """§72 — mean per-date rank IC within each group of ``labels``."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dates = np.asarray(dates).ravel()
    labels = np.asarray(labels).ravel()
    if len(labels) != len(pred):
        return None
    df = pd.DataFrame({"pred": pred, "y": y, "date": dates, "g": labels})
    out: dict[str, float] = {}
    for g, sub in df.groupby("g", observed=True):
        ds_, ics = per_date_rank_ic(
            sub["pred"].to_numpy(), sub["y"].to_numpy(), sub["date"].to_numpy()
        )
        if len(ics):
            out[str(g)] = float(np.nanmean(ics))
    return out or None


def _year_by_year(pred: np.ndarray, y: np.ndarray, dates) -> dict[str, float]:
    """Mean per-date IC per calendar year of the anchor date."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dates = np.asarray(dates).ravel()
    try:
        years = pd.to_datetime(pd.Series(dates)).dt.year.to_numpy()
    except Exception:
        return {}
    df = pd.DataFrame({"pred": pred, "y": y, "date": dates, "yr": years})
    out: dict[str, float] = {}
    for yr, sub in df.groupby("yr", observed=True):
        ds_, ics = per_date_rank_ic(
            sub["pred"].to_numpy(), sub["y"].to_numpy(), sub["date"].to_numpy()
        )
        if len(ics):
            out[str(yr)] = float(np.nanmean(ics))
    return out


def evaluate_predictions(
    pred: np.ndarray,
    y: np.ndarray,
    dates,
    *,
    date_col: str | None = None,
    stock_col: str | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> EvaluationReport:
    """§26 / §27 / §72 — compute the full evaluation report.

    ``pred``, ``y``, ``dates`` are row-aligned arrays.  ``date_col`` /
    ``stock_col`` are accepted for interface symmetry (the dates are already
    passed as an array).  ``extra`` may carry ``regime_labels``,
    ``cap_labels``, ``liquidity_labels`` arrays (row-aligned) for the §72
    grouped metrics; without them those fields stay ``None``.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dates = np.asarray(dates).ravel()
    if not (len(pred) == len(y) == len(dates)):
        raise ValueError("pred, y, dates must be row-aligned")

    mask = np.isfinite(pred) & np.isfinite(y)
    n_pairs = int(mask.sum())

    mse = float(np.mean((pred[mask] - y[mask]) ** 2)) if n_pairs else float("nan")
    mae = float(np.mean(np.abs(pred[mask] - y[mask]))) if n_pairs else float("nan")

    pearson_ic = float("nan")
    if n_pairs >= 3:
        pr, _ = pearsonr(pred[mask], y[mask])
        pearson_ic = float(pr)

    rank_ic = _pooled_rank_ic(pred, y)

    dates_sorted, ics = per_date_rank_ic(pred, y, dates)
    icir = float("nan")
    ic_pos = float("nan")
    if len(ics) >= 2:
        sd = float(np.std(ics))
        icir = float(np.mean(ics) / sd) if sd > 1e-12 else 0.0
        ic_pos = float(np.mean(ics > 0)) if len(ics) else float("nan")
    elif len(ics) == 1:
        ic_pos = float(ics[0] > 0)

    df = pd.DataFrame({"pred": pred, "y": y, "date": dates})
    ls_spread = _long_short_spread(df)
    tov = _turnover(df)
    coverage = float(np.isfinite(pred).mean()) if len(pred) else 0.0
    subperiod_stability = float(np.std(ics)) if len(ics) >= 2 else float("nan")

    yby = _year_by_year(pred, y, dates)

    report = EvaluationReport(
        mse=mse,
        mae=mae,
        rank_ic=rank_ic,
        pearson_ic=pearson_ic,
        icir=icir,
        ic_positive_ratio=ic_pos,
        long_short_spread=ls_spread,
        turnover=tov,
        coverage=coverage,
        subperiod_stability=subperiod_stability,
        year_by_year=yby,
    )

    if extra:
        if extra.get("regime_labels") is not None:
            report.bull_bear = _grouped_mean_ic(
                pred, y, dates, extra["regime_labels"]
            )
        if extra.get("cap_labels") is not None:
            report.large_small_cap = _grouped_mean_ic(
                pred, y, dates, extra["cap_labels"]
            )
        if extra.get("liquidity_labels") is not None:
            report.liquidity_bucket = _grouped_mean_ic(
                pred, y, dates, extra["liquidity_labels"]
            )
    return report
