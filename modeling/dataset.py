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
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["FeatureField", "FeatureSchema", "PanelDataset", "panel_telemetry"]


@dataclass(frozen=True)
class FeatureField:
    """One ordered model input, including its semantic (not just column) identity."""

    name: str
    value_ref: str
    recipe_ref: str
    state_ref: str
    dtype: str
    mask_ref: str | None = None
    cluster_version_ref: str | None = None
    missing_reason_ref: str | None = None
    missing_age_ref: str | None = None

    def __post_init__(self) -> None:
        for attr in ("name", "value_ref", "recipe_ref", "state_ref", "dtype"):
            if not isinstance(getattr(self, attr), str) or not getattr(self, attr):
                raise ValueError(f"FeatureField.{attr} is required")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name, "value_ref": self.value_ref,
            "recipe_ref": self.recipe_ref, "state_ref": self.state_ref,
            "dtype": self.dtype, "mask_ref": self.mask_ref,
            "missing_reason_ref": self.missing_reason_ref,
            "missing_age_ref": self.missing_age_ref,
            "cluster_version_ref": self.cluster_version_ref,
        }


@dataclass(frozen=True)
class FeatureSchema:
    """Explicit ordered feature contract required by production datasets."""

    columns: tuple[str, ...]
    fields: tuple[FeatureField, ...] = ()
    consumer_profile: str = ""
    feature_set_version_ref: str = ""
    manifest_version: str = "feature-manifest-v1"

    def __post_init__(self) -> None:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("FeatureSchema columns must be non-empty and unique")
        if self.fields:
            if tuple(field.name for field in self.fields) != self.columns:
                raise ValueError("FeatureSchema fields must match ordered columns exactly")
            if not self.consumer_profile or not self.feature_set_version_ref:
                raise ValueError("complete FeatureSchema requires consumer_profile and feature_set_version_ref")
            if any(not field.cluster_version_ref for field in self.fields):
                raise ValueError("complete FeatureSchema requires cluster_version_ref per field")

    @property
    def is_complete(self) -> bool:
        return bool(self.fields and self.consumer_profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "consumer_profile": self.consumer_profile,
            "feature_set_version_ref": self.feature_set_version_ref,
            "fields": [field.to_dict() for field in self.fields],
            "columns": list(self.columns),
            "complete": self.is_complete,
        }

    def fingerprint(self) -> str:
        """Canonical full-manifest hash; legacy column-only schemas are namespaced."""
        payload = self.to_dict()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        prefix = "full:" if self.is_complete else "legacy-columns:"
        return prefix + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    feature_schema: FeatureSchema | None = None
    run_mode: str = "production"
    missing_reason_plane: Any = None
    label_missing_reasons: Any = None

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
        if self.run_mode not in {"production", "research"}:
            raise ValueError("run_mode must be 'production' or 'research'")
        if self.run_mode == "production" and self.feature_schema is None and not self.feature_cols:
            raise ValueError("production PanelDataset requires explicit feature_schema")
        if self.feature_schema is not None:
            schema_cols = list(self.feature_schema.columns)
            if self.feature_cols and self.feature_cols != schema_cols:
                raise ValueError("feature_cols differ from explicit feature_schema")
            self.feature_cols = schema_cols
        elif not self.feature_cols:
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
            if self.missing_reason_plane is not None or self.label_missing_reasons is not None:
                raise ValueError(
                    "duplicate aggregation with aligned missingness evidence requires an explicit reducer"
                )
            self.frame = self.frame.drop_duplicates(
                subset=[self.date_col, self.stock_col], keep="first"
            ).reset_index(drop=True)
        if self.missing_reason_plane is not None:
            plane = self.missing_reason_plane
            required = ("reasons", "original_missing", "filled", "usable", "age", "coverage")
            if any(not hasattr(plane, name) for name in required) or not callable(plane.coverage):
                raise ValueError("missing_reason_plane is not an authoritative reason plane")
            expected = (self.n_rows, len(self.feature_cols))
            if any(np.asarray(getattr(plane, name)).shape != expected for name in required[:-1]):
                raise ValueError(f"missing_reason_plane must align to panel feature shape {expected}")
            try:
                self.missing_reason_plane = type(plane)(
                    reasons=plane.reasons, original_missing=plane.original_missing,
                    filled=plane.filled, usable=plane.usable, age=plane.age,
                )
            except Exception as exc:
                raise ValueError(f"invalid missing_reason_plane: {exc}") from exc
        if self.label_missing_reasons is not None:
            labels = np.asarray(self.label_missing_reasons, dtype=str)
            if labels.shape != (self.n_rows,):
                raise ValueError("label_missing_reasons must align to panel rows")
            labels = np.array(labels, copy=True)
            labels.flags.writeable = False
            self.label_missing_reasons = labels

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
            feature_schema=self.feature_schema,
            run_mode=self.run_mode,
            missing_reason_plane=self._slice_missingness(mask.to_numpy()),
            label_missing_reasons=(
                None if self.label_missing_reasons is None else self.label_missing_reasons[mask.to_numpy()]
            ),
        )

    def filter_stocks(self, stocks: Any) -> "PanelDataset":
        """Restrict to a (legally available) universe."""
        return PanelDataset(
            frame=self.frame.loc[self.frame[self.stock_col].isin(stocks)].reset_index(drop=True),
            date_col=self.date_col,
            stock_col=self.stock_col,
            feature_cols=list(self.feature_cols),
            label_col=self.label_col,
            feature_schema=self.feature_schema,
            run_mode=self.run_mode,
            missing_reason_plane=self._slice_missingness(
                self.frame[self.stock_col].isin(stocks).to_numpy()
            ),
            label_missing_reasons=(
                None if self.label_missing_reasons is None
                else self.label_missing_reasons[self.frame[self.stock_col].isin(stocks).to_numpy()]
            ),
        )

    def _slice_missingness(self, row_mask: np.ndarray) -> Any:
        if self.missing_reason_plane is None:
            return None
        plane = self.missing_reason_plane
        return type(plane)(
            reasons=plane.reasons[row_mask], original_missing=plane.original_missing[row_mask],
            filled=plane.filled[row_mask], usable=plane.usable[row_mask], age=plane.age[row_mask],
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
        if self.missing_reason_plane is not None:
            X = np.array(X, copy=True)
            X[~np.asarray(self.missing_reason_plane.usable, dtype=bool)] = np.nan
        y = (
            self.frame[self.label_col].to_numpy(dtype=np.float64)
            if self.label_col is not None
            else None
        )
        dates = self.frame[self.date_col].to_numpy()
        stocks = self.frame[self.stock_col].to_numpy()
        finite = np.isfinite(X).all(axis=1)
        if self.missing_reason_plane is not None:
            finite &= np.asarray(self.missing_reason_plane.usable, dtype=bool).all(axis=1)
        if y is not None:
            finite &= np.isfinite(y)
            if self.label_missing_reasons is not None:
                finite &= self.label_missing_reasons != "label_not_yet_mature"
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
    label_nan_count = 0
    label_inf_count = 0
    if ds.label_col is not None:
        labels = f[ds.label_col].to_numpy(dtype=np.float64)
        label_nan_count = int(np.isnan(labels).sum())
        label_inf_count = int(np.isinf(labels).sum())
        label_coverage = float(np.isfinite(labels).mean())
    _, y, _, _, finite = ds.as_matrix()
    finite_obs = int(finite.sum()) if y is not None else int(np.isfinite(f[ds.feature_cols].to_numpy()).all(axis=1).sum())
    missing_fraction = (1.0 - finite_obs / n_stock_date) if n_stock_date else 0.0
    semantic_coverage = (
        ds.missing_reason_plane.coverage()
        if ds.missing_reason_plane is not None else None
    )
    return {
        "n_stock_date_obs": int(n_stock_date),
        "n_unique_dates": n_dates,
        "n_unique_stocks": n_stocks,
        "n_industries": None,  # industry membership is DataAccess-provided (§12)
        "median_stocks_per_date": median_per_date,
        "min_stocks_per_date": min_per_date,
        "cross_sectional_coverage": float(coverage),
        "label_coverage": label_coverage,
        "label_nan_count": label_nan_count,
        "label_inf_count": label_inf_count,
        "effective_date_count": int(n_dates),
        "raw_obs": int(n_stock_date),
        "finite_obs": finite_obs,
        "missing_fraction": float(missing_fraction),
        "date_coverage": float(_panel_date_coverage(f, ds.date_col)),
        "missingness_coverage": semantic_coverage,
        "label_not_yet_mature_count": (
            0 if ds.label_missing_reasons is None
            else int(np.sum(ds.label_missing_reasons == "label_not_yet_mature"))
        ),
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
