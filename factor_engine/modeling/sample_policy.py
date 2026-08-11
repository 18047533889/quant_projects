# -*- coding: utf-8 -*-
"""Sample-adequacy telemetry and policy (Model Layer Major Redesign taskbook
§5.1 / §5.2).

Measures the §5.1 chain from a pooled :class:`modeling.dataset.PanelDataset`
into a JSON-able :class:`SampleTelemetry` and gates it against a
:class:`modeling.contracts.SampleAdequacyContract`::

    raw_obs -> finite_obs -> mature_label_obs -> post_purge_obs
            -> post_regime_obs -> effective_obs

Every step is a *conservative* narrowing; ``effective_obs`` is what the model
training pipeline may actually fit on.  The report is deliberately honest:
missing / immature / purged observations are never hidden.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from modeling.contracts import LabelContract, SampleAdequacyContract, sample_adequacy_met
from modeling.dataset import PanelDataset, panel_telemetry

__all__ = [
    "SampleTelemetry",
    "measure_train_telemetry",
    "adequacy_report",
    "telemetry_summary",
]


@dataclass
class SampleTelemetry:
    """§5.1 chain telemetry — every count is a narrowing of the previous one."""

    raw_obs: int = 0
    finite_obs: int = 0
    mature_label_obs: int = 0
    post_purge_obs: int = 0
    post_regime_obs: int = 0
    effective_obs: int = 0
    free_parameter_count: int = 1
    obs_per_parameter: float = 0.0
    unique_dates: int = 0
    unique_stocks: int = 0
    date_coverage: float = 0.0
    missing_fraction: float = 0.0


def _is_datetime_like(dates: pd.Series) -> bool:
    """Best-effort datetime detection for calendar-coverage computation."""
    try:
        if pd.api.types.is_datetime64_any_dtype(dates):
            return True
    except Exception:
        pass
    sample = dates.iloc[0] if len(dates) else None
    return isinstance(sample, (pd.Timestamp, _dt.datetime, _dt.date, _dt.time))


def _date_coverage(ds: PanelDataset, date_col: str) -> float:
    """Fraction of the train calendar span covered by unique trade dates.

    Only meaningful when ``date_col`` is datetime-like; otherwise 1.0 (no
    calendar notion in the data).
    """
    if ds.n_rows == 0:
        return 0.0
    dates = ds.frame[date_col]
    n_unique = int(dates.nunique())
    if n_unique == 0:
        return 0.0
    if not _is_datetime_like(dates):
        return 1.0
    try:
        dts = pd.to_datetime(dates)
        span_days = (dts.max() - dts.min()).days + 1
    except Exception:
        return 1.0
    if span_days <= 0:
        return 1.0
    return float(n_unique) / float(span_days)


def _purged_obs(
    train_ds: PanelDataset,
    validation_ds: PanelDataset | None,
    label_contract: LabelContract | None,
    finite_obs: int,
) -> int:
    """§9 purge — finite rows whose label interval overlaps the validation
    window are dropped.  Without a validation set nothing is purged.

    The horizon is applied in *bar space*: we build an ordinal index over the
    union of train and validation dates and drop any train row whose label
    exit bar (``anchor_bar + horizon``) is at/after the validation start bar.
    """
    if validation_ds is None or validation_ds.n_rows == 0:
        return finite_obs
    horizon = label_contract.horizon_bars if label_contract is not None else 1
    if horizon < 1:
        return finite_obs
    try:
        train_dates = pd.to_datetime(train_ds.frame[train_ds.date_col])
        val_dates = pd.to_datetime(validation_ds.frame[validation_ds.date_col])
    except Exception:
        return finite_obs
    union = sorted(set(train_dates) | set(val_dates))
    ordinal = {d: i for i, d in enumerate(union)}
    v_start_idx = ordinal[val_dates.min()]
    _, _, _, _, finite = train_ds.as_matrix()
    anchor_idx = train_dates.map(ordinal.get).to_numpy(dtype=float)
    keep = (
        (~np.isnan(anchor_idx))
        & (anchor_idx + horizon < v_start_idx)
        & finite
    )
    return int(keep.sum())


def measure_train_telemetry(
    train_ds: PanelDataset,
    validation_ds: PanelDataset | None = None,
    label_contract: LabelContract | None = None,
    free_parameter_count: int = 1,
    *,
    date_col: str = "date",
) -> SampleTelemetry:
    """Compute the §5.1 chain for a training panel.

    * ``raw_obs`` — every pooled row;
    * ``finite_obs`` — rows where all features (and the label when present)
      are finite;
    * ``mature_label_obs`` — rows whose forward label is matured (not NaN);
      when ``label_contract is None`` falls back to the finite rows;
    * ``post_purge_obs`` — finite rows surviving the §9 purge against the
      validation window (no-op when ``validation_ds is None``);
    * ``post_regime_obs`` — placeholder equal to ``post_purge_obs`` until
      regime telemetry is wired in;
    * ``effective_obs`` = ``post_purge_obs``.
    """
    tele = panel_telemetry(train_ds)
    raw = int(tele.get("raw_obs", 0))
    finite = int(tele.get("finite_obs", 0))

    mature = finite
    if label_contract is not None and train_ds.label_col is not None:
        mature = int(train_ds.frame[train_ds.label_col].notna().sum())

    post_purge = _purged_obs(train_ds, validation_ds, label_contract, finite)
    post_regime = post_purge
    effective = post_purge

    free = max(1, int(free_parameter_count))
    obs_per_param = effective / free if effective else 0.0
    missing = (1.0 - finite / raw) if raw else 0.0

    return SampleTelemetry(
        raw_obs=raw,
        finite_obs=finite,
        mature_label_obs=mature,
        post_purge_obs=post_purge,
        post_regime_obs=post_regime,
        effective_obs=effective,
        free_parameter_count=int(free_parameter_count),
        obs_per_parameter=obs_per_param,
        unique_dates=int(tele.get("n_unique_dates", 0)),
        unique_stocks=int(tele.get("n_unique_stocks", 0)),
        date_coverage=_date_coverage(train_ds, date_col),
        missing_fraction=missing,
    )


def adequacy_report(
    contract: SampleAdequacyContract,
    telemetry: SampleTelemetry,
) -> tuple[bool, list[str]]:
    """Gate the measured telemetry against a :class:`SampleAdequacyContract`."""
    return sample_adequacy_met(
        contract=contract,
        raw_obs=telemetry.raw_obs,
        effective_obs=telemetry.effective_obs,
        unique_dates=telemetry.unique_dates,
        unique_stocks=telemetry.unique_stocks,
        free_parameter_count=telemetry.free_parameter_count,
        missing_fraction=telemetry.missing_fraction,
        date_coverage=telemetry.date_coverage,
    )


def telemetry_summary(telemetry: SampleTelemetry) -> dict[str, Any]:
    """JSON-able dict for evidence ledgers."""
    return asdict(telemetry)
