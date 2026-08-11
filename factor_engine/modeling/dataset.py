# -*- coding: utf-8 -*-
"""Pooled panel dataset representation + telemetry (Model Layer Major Redesign
taskbook §4).

A predictive learner trains on pooled ``(stock, date)`` observations::

    X_{i,t} -> y_{i,t+H}

Training telemetry (§4.1) must report far more than ``n_rows`` because 4,000
stocks x 1,000 dates is not 4,000,000 independent samples (same-day stocks are
highly correlated).  The split MUST be date-authoritative (§4.2): all rows of a
date belong to one split.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["PanelDataset", "panel_telemetry"]


@dataclass
class PanelDataset:
    """A long-format pooled panel.

    ``frame`` has one row per ``(stock, date)`` observation with columns for
    features, the label, and ``date_col`` / ``stock_col``.  ``label_col`` may
    be ``None`` when scoring (features only).

    Enterprise data constraints (§4):

    * ``stock_col`` is REQUIRED — a pooled panel without an instrument id
      cannot be audited, PIT-filtered or de-duplicated;
    * ``(stock, date)`` must be UNIQUE — a duplicated ``(stock, date)`` row
      would otherwise be counted as an independent training observation
      (inflating ``raw_obs``/weights/loss).  Production defaults to
      ``duplicate_policy="error"``; pass ``duplicate_policy="aggregate"`` to
      explicitly dedupe keeping the first occurrence per ``(stock, date)``.
    """

    frame: pd.DataFrame
    date_col: str = "date"
    stock_col: str = "stock"
    feature_cols: list[str] = field(default_factory=list)
    label_col: str | None = None
    duplicate_policy: str = "error"  # "error" | "aggregate"

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("PanelDataset.frame must be a pandas DataFrame")
        if self.date_col not in self.frame.columns:
            raise ValueError(f"date column {self.date_col!r} missing from frame")
        if self.stock_col not in self.frame.columns:
            raise ValueError(
                f"stock column {self.stock_col!r} missing from frame — a pooled "
                "panel requires an instrument id (audit/PIT/dedup are impossible without it)"
            )
        if not self.feature_cols:
            self.feature_cols = [
                c for c in self.frame.columns if c not in (self.date_col, self.stock_col, self.label_col)
            ]
        missing = [c for c in self.feature_cols if c not in self.frame.columns]
        if missing:
            raise ValueError(f"feature columns missing from frame: {missing}")
        if self.label_col is not None and self.label_col not in self.frame.columns:
            raise ValueError(f"label column {self.label_col!r} missing from frame")
        if self.duplicate_policy not in ("error", "aggregate"):
            raise ValueError(
                f"duplicate_policy must be 'error' or 'aggregate', got {self.duplicate_policy!r}"
            )
        # (stock, date) uniqueness — duplicated rows would double-count samples.
        n_dups = int(
            self.frame.duplicated(subset=[self.date_col, self.stock_col]).sum()
        )
        if n_dups:
            if self.duplicate_policy == "error":
                raise ValueError(
                    f"PanelDataset has {n_dups} duplicated (date, stock) rows — "
                    "each (stock, date) must be a unique observation; pass "
                    "duplicate_policy='aggregate' to dedupe explicitly"
                )
            self.frame = self.frame.drop_duplicates(
                subset=[self.date_col, self.stock_col], keep="first"
            ).reset_index(drop=True)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        date_col: str = "date",
        stock_col: str = "stock",
        feature_cols: list[str] | None = None,
        label_col: str | None = None,
        duplicate_policy: str = "error",
    ) -> "PanelDataset":
        return cls(
            frame=frame,
            date_col=date_col,
            stock_col=stock_col,
            feature_cols=list(feature_cols) if feature_cols is not None else [],
            label_col=label_col,
            duplicate_policy=duplicate_policy,
        )

    # -- slicing -------------------------------------------------------------
    def filter_dates(self, start: Any = None, end: Any = None) -> "PanelDataset":
        """Date-authoritative slice (both bounds inclusive)."""
        dates = self.frame[self.date_col]
        mask = pd.Series(True, index=self.frame.index)
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates <= end
        return PanelDataset(
            frame=self.frame.loc[mask].reset_index(drop=True),
            date_col=self.date_col,
            stock_col=self.stock_col,
            feature_cols=list(self.feature_cols),
            label_col=self.label_col,
        )

    def filter_stocks(self, stocks: Any) -> "PanelDataset":
        """Restrict to a (legally available) universe."""
        return PanelDataset(
            frame=self.frame.loc[self.frame[self.stock_col].isin(stocks)].reset_index(drop=True),
            date_col=self.date_col,
            stock_col=self.stock_col,
            feature_cols=list(self.feature_cols),
            label_col=self.label_col,
        )

    # -- matrix view ---------------------------------------------------------
    def as_matrix(
        self,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, y, dates, stocks, finite_mask)``.

        ``finite_mask`` marks rows where all features (and the label, when
        present) are finite.  Consumers must never impute with future data.
        """
        X = self.frame[self.feature_cols].to_numpy(dtype=np.float64)
        y = (
            self.frame[self.label_col].to_numpy(dtype=np.float64)
            if self.label_col is not None
            else None
        )
        dates = self.frame[self.date_col].to_numpy()
        stocks = self.frame[self.stock_col].to_numpy()
        finite = np.isfinite(X).all(axis=1)
        if y is not None:
            finite &= np.isfinite(y)
        return X, y, dates, stocks, finite

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def telemetry(self) -> dict[str, Any]:
        """§4.1 training-set statistics."""
        return panel_telemetry(self)


def panel_telemetry(ds: PanelDataset) -> dict[str, Any]:
    """§4.1 / §5.1 — raw and effective sample statistics of a pooled panel."""
    if ds.n_rows == 0:
        return {
            "n_stock_date_obs": 0, "n_unique_dates": 0, "n_unique_stocks": 0,
            "n_industries": None, "median_stocks_per_date": 0.0,
            "min_stocks_per_date": 0, "cross_sectional_coverage": 0.0,
            "label_coverage": 0.0, "effective_date_count": 0,
            "raw_obs": 0, "finite_obs": 0,
            "missing_fraction": 0.0, "date_coverage": 0.0,
        }
    f = ds.frame
    n_stock_date = len(f)
    n_dates = int(f[ds.date_col].nunique())
    n_stocks = int(f[ds.stock_col].nunique())
    per_date = f.groupby(ds.date_col, observed=True).size()
    median_per_date = float(per_date.median())
    min_per_date = int(per_date.min())
    coverage = median_per_date / max(1, n_stocks)
    label_coverage = 1.0
    if ds.label_col is not None:
        label_coverage = float(f[ds.label_col].notna().mean())
    _, y, _, _, finite = ds.as_matrix()
    finite_obs = int(finite.sum()) if y is not None else int(np.isfinite(f[ds.feature_cols].to_numpy()).all(axis=1).sum())
    missing_fraction = (1.0 - finite_obs / n_stock_date) if n_stock_date else 0.0
    return {
        "n_stock_date_obs": int(n_stock_date),
        "n_unique_dates": n_dates,
        "n_unique_stocks": n_stocks,
        "n_industries": None,  # industry membership is DataAccess-provided (§12)
        "median_stocks_per_date": median_per_date,
        "min_stocks_per_date": min_per_date,
        "cross_sectional_coverage": float(coverage),
        "label_coverage": label_coverage,
        "effective_date_count": int(n_dates),
        "raw_obs": int(n_stock_date),
        "finite_obs": finite_obs,
        "missing_fraction": float(missing_fraction),
        "date_coverage": float(_panel_date_coverage(f, ds.date_col)),
    }


def _panel_date_coverage(f: pd.DataFrame, date_col: str) -> float:
    """Fraction of the calendar span covered by unique dates.

    ``1.0`` when the date column carries no calendar notion (e.g. integer bar
    ordinals) — there is no missing-trading-day concept to measure — and for
    any real datetime column the fraction of the calendar span covered by the
    unique trade dates in the panel (bounded in ``[0, 1]``).
    """
    if len(f) == 0:
        return 0.0
    dates = f[date_col]
    try:
        is_dt = pd.api.types.is_datetime64_any_dtype(dates)
    except Exception:
        is_dt = False
    if not is_dt:
        sample = dates.iloc[0] if len(dates) else None
        if not isinstance(sample, (pd.Timestamp, _dt.datetime, _dt.date, _dt.time)):
            return 1.0
        try:
            dts = pd.to_datetime(dates, errors="coerce")
        except Exception:
            return 1.0
        if dts.isna().all():
            return 1.0
        dates = dts
    n_unique = int(dates.nunique())
    if n_unique == 0:
        return 0.0
    try:
        span_days = (dates.max() - dates.min()).days + 1
    except Exception:
        return 1.0
    if span_days <= 0:
        return 1.0
    return float(n_unique) / float(span_days)
