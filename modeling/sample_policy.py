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
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from modeling.contracts import LabelContract, SampleAdequacyContract, sample_adequacy_met
from modeling.dataset import PanelDataset
from market.exchange_session_calendar import ExchangeSessionCalendar

__all__ = [
    "SampleWeightPolicy",
    "sample_weights",
    "SampleTelemetry",
    "measure_train_telemetry",
    "adequacy_report",
    "telemetry_summary",
    "resolve_sample_contract",
    "adequacy_failures",
]


class SampleWeightPolicy(str, Enum):
    """Auditable weighting policies for pooled panel observations."""

    EQUAL_ROW = "equal_row"
    EQUAL_DATE = "equal_date"
    EQUAL_DATE_THEN_STOCK = "equal_date_then_stock"
    TIME_DECAY_EQUAL_DATE = "time_decay_equal_date"


def sample_weights(
    ds: PanelDataset,
    policy: SampleWeightPolicy | str,
    *,
    half_life_dates: float | None = None,
) -> np.ndarray:
    """Return positive weights with mean one, aligned to ``ds.frame``.

    Date-balanced policies give every date equal aggregate mass.  The
    ``EQUAL_DATE_THEN_STOCK`` spelling makes the within-date equal-stock rule
    explicit; with unique ``(date, stock)`` panel rows it is mathematically the
    same as ``EQUAL_DATE``.  Time decay is applied to date masses, never rows.
    """
    policy = SampleWeightPolicy(policy)
    n = ds.n_rows
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if policy is SampleWeightPolicy.EQUAL_ROW:
        return np.ones(n, dtype=np.float64)

    dates = ds.frame[ds.date_col]
    counts = dates.groupby(dates, observed=True).transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    if policy is SampleWeightPolicy.TIME_DECAY_EQUAL_DATE:
        if half_life_dates is None or not np.isfinite(half_life_dates) or half_life_dates <= 0:
            raise ValueError("TIME_DECAY_EQUAL_DATE requires positive half_life_dates")
        ordered = pd.Index(dates.drop_duplicates()).sort_values()
        rank = {date: i for i, date in enumerate(ordered)}
        age = np.array([len(ordered) - 1 - rank[date] for date in dates], dtype=float)
        weights *= np.exp2(-age / float(half_life_dates))
    mean = float(weights.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("sample weight policy produced invalid weights")
    return weights / mean


@dataclass
class SampleTelemetry:
    """§5.1 chain telemetry — every count is a narrowing of the previous one.

    The regime/MoE specific fields (``regime_obs`` / ``expert_obs`` /
    ``state_transitions`` / ``cross_section_peers``) default to ``None`` —
    meaning "not measured" — and are filled in by the regime/MoE telemetry
    wiring.  Under :func:`~modeling.contracts.sample_adequacy_met` a ``None``
    for a contract-required field FAILS closed rather than silently passing.
    """

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
    # regime / MoE specific telemetry — None == not measured (fail closed).
    regime_obs: int | None = None
    expert_obs: int | None = None
    state_transitions: int | None = None
    cross_section_peers: int | None = None


def _is_datetime_like(dates: pd.Series) -> bool:
    """Best-effort datetime detection for calendar-coverage computation."""
    try:
        if pd.api.types.is_datetime64_any_dtype(dates):
            return True
    except Exception:
        pass
    sample = dates.iloc[0] if len(dates) else None
    return isinstance(sample, (pd.Timestamp, _dt.datetime, _dt.date, _dt.time))


def _date_coverage(
    ds: PanelDataset,
    date_col: str,
    calendar: ExchangeSessionCalendar | None,
) -> float:
    """Observed expected sessions divided by authoritative expected sessions."""
    if ds.n_rows == 0:
        return 0.0
    dates = ds.frame[date_col]
    n_unique = int(dates.nunique())
    if n_unique == 0:
        return 0.0
    if not _is_datetime_like(dates) or calendar is None:
        return 1.0
    dts = pd.to_datetime(dates).dt.normalize()
    expected = calendar.expected_sessions(dts.min(), dts.max())
    if len(expected) == 0:
        return 0.0
    observed = pd.DatetimeIndex(dts.unique()).normalize()
    return float(len(observed.intersection(expected))) / float(len(expected))


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


def _calendar_condition(
    dates: pd.Series,
    cutoff: Any,
    horizon: int,
    calendar: ExchangeSessionCalendar | None,
) -> np.ndarray:
    if cutoff is None:
        return np.ones(len(dates), dtype=bool)
    if calendar is None or not _is_datetime_like(dates):
        unique = np.sort(dates.unique())
        positions = pd.Index(unique).searchsorted(dates.to_numpy(), side="left")
        cutoff_pos = int(pd.Index(unique).searchsorted(cutoff, side="right") - 1)
        return positions + horizon <= cutoff_pos
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    result = np.zeros(len(dates), dtype=bool)
    for idx, anchor in enumerate(pd.to_datetime(dates).dt.normalize()):
        try:
            result[idx] = calendar.shift_session(anchor, horizon) <= cutoff_ts
        except ValueError:
            result[idx] = False
    return result


def measure_train_telemetry(
    train_ds: PanelDataset,
    validation_ds: PanelDataset | None = None,
    label_contract: LabelContract | None = None,
    free_parameter_count: int = 1,
    *,
    date_col: str = "date",
    calendar: ExchangeSessionCalendar | None = None,
    training_cutoff: Any = None,
    support_mask: np.ndarray | None = None,
    final_fit_mask: np.ndarray | None = None,
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
    raw = train_ds.n_rows
    raw_mask = np.ones(raw, dtype=bool)
    X, y, _, _, _ = train_ds.as_matrix()
    finite_input_mask = raw_mask & np.isfinite(X).all(axis=1)
    if y is not None:
        finite_input_mask &= np.isfinite(y)
    finite = int(finite_input_mask.sum())

    horizon = int(label_contract.horizon_bars) if label_contract is not None else 0
    mature_condition = np.ones(raw, dtype=bool)
    if y is not None:
        mature_condition &= np.isfinite(y)
    if label_contract is not None:
        cutoff = training_cutoff
        if cutoff is None:
            cutoff = train_ds.frame[train_ds.date_col].max()
        mature_condition &= _calendar_condition(
            train_ds.frame[train_ds.date_col], cutoff, horizon, calendar
        )
    mature_mask = finite_input_mask & mature_condition

    purge_condition = np.ones(raw, dtype=bool)
    if validation_ds is not None and validation_ds.n_rows:
        boundary = validation_ds.frame[validation_ds.date_col].min()
        if calendar is not None and _is_datetime_like(train_ds.frame[train_ds.date_col]):
            boundary = calendar.shift_session(boundary, -1)
        purge_condition = _calendar_condition(
            train_ds.frame[train_ds.date_col], boundary, horizon, calendar
        )
    purge_mask = mature_mask & purge_condition

    if support_mask is None:
        support_condition = np.ones(raw, dtype=bool)
    else:
        support_condition = np.asarray(support_mask, dtype=bool)
        if support_condition.shape != (raw,):
            raise ValueError("support_mask must have one value per training row")
    narrowed_support_mask = purge_mask & support_condition
    if final_fit_mask is None:
        fit_condition = np.ones(raw, dtype=bool)
    else:
        fit_condition = np.asarray(final_fit_mask, dtype=bool)
        if fit_condition.shape != (raw,):
            raise ValueError("final_fit_mask must have one value per training row")
    narrowed_final_mask = narrowed_support_mask & fit_condition

    mature = int(mature_mask.sum())
    post_purge = int(purge_mask.sum())
    post_regime = int(narrowed_support_mask.sum())
    effective = int(narrowed_final_mask.sum())
    unique_dates = int(train_ds.frame.loc[narrowed_final_mask, train_ds.date_col].nunique())
    unique_stocks = int(train_ds.frame.loc[narrowed_final_mask, train_ds.stock_col].nunique())

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
        unique_dates=unique_dates,
        unique_stocks=unique_stocks,
        date_coverage=_date_coverage(train_ds, date_col, calendar),
        missing_fraction=missing,
    )


def _int_or_none(value: Any) -> int | None:
    """Coerce a telemetry scalar to ``int``, preserving ``None`` (not measured)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    """Coerce a telemetry scalar to ``float``, preserving ``None`` (not measured)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_sample_contract(learner: Any) -> SampleAdequacyContract | None:
    """Resolve the §5 default :class:`SampleAdequacyContract` for a learner.

    ``learner`` may be a learner class or an instance (class attributes take
    priority via ``getattr``).  Resolution order:

    * ``sample_contract_family`` — the learner's explicit §5 declaration
      (``"linear"`` for pcr/pls/elastic_net, ``"regime"``, ``"moe"``);
    * ``contract_family`` — legacy alias if any learner declares it;
    * ``family`` — bare family key (``"pcr"`` etc.).

    Returns ``None`` when the learner declares no usable family key — meaning
    "no adequacy gate" — never a mismatched contract.
    """
    key = (
        _family_key(learner, "sample_contract_family")
        or _family_key(learner, "contract_family")
        or _family_key(learner, "family")
    )
    if not key:
        return None
    from modeling.learners.base import default_sample_contracts

    return default_sample_contracts().get(key)


def _family_key(learner: Any, attr: str) -> str:
    """Read a family-key attribute, tolerating the class-vs-instance split.

    ``contract_family`` is a *property* on :class:`BaseLearner` — ``getattr``
    on the CLASS returns the descriptor object (truthy, non-string), which would
    poison the lookup.  Only string values are accepted here; ``None`` on a
    property is the signal to try the next key."""
    value = getattr(learner, attr, None)
    return value if isinstance(value, str) else ""


def adequacy_failures(
    contract: SampleAdequacyContract,
    telemetry: SampleTelemetry | dict[str, Any],
    free_parameters: int,
) -> list[str]:
    """Gate ``telemetry`` against ``contract`` and return the failure list.

    ``telemetry`` may be a :class:`SampleTelemetry` or a ``dict`` with the same
    keys (dataset-style keys ``n_unique_dates`` / ``n_unique_stocks`` /
    ``median_stocks_per_date`` are also accepted).

    Every :class:`SampleTelemetry` field is fed to
    :func:`~modeling.contracts.sample_adequacy_met` for real: raw/effective obs,
    unique dates/stocks, regime obs, expert obs, state transitions,
    cross-section peers, missing fraction and date coverage are all passed
    through (never stubbed to ``None``).  A required-but-unmeasured field then
    FAILS closed under the three-state rule instead of silently passing.
    ``obs_per_parameter`` is derived by ``sample_adequacy_met`` from
    ``effective_obs`` and ``free_parameters``.

    ``free_parameters`` is the learner's ``effective_parameter_count`` (the
    true number of free parameters the fit estimates).
    """
    if isinstance(telemetry, SampleTelemetry):
        t = telemetry
    else:
        t = SampleTelemetry(
            raw_obs=_int_or_none(telemetry.get("raw_obs")) or 0,
            finite_obs=_int_or_none(telemetry.get("finite_obs")) or 0,
            mature_label_obs=_int_or_none(telemetry.get("mature_label_obs")) or 0,
            post_purge_obs=_int_or_none(telemetry.get("post_purge_obs")) or 0,
            post_regime_obs=_int_or_none(telemetry.get("post_regime_obs")) or 0,
            effective_obs=_int_or_none(
                telemetry.get("effective_obs", telemetry.get("finite_obs"))
            )
            or 0,
            free_parameter_count=_int_or_none(
                telemetry.get("free_parameter_count", free_parameters)
            )
            or free_parameters,
            obs_per_parameter=float(telemetry.get("obs_per_parameter", 0.0) or 0.0),
            unique_dates=_int_or_none(
                telemetry.get("unique_dates", telemetry.get("n_unique_dates"))
            )
            or 0,
            unique_stocks=_int_or_none(
                telemetry.get("unique_stocks", telemetry.get("n_unique_stocks"))
            )
            or 0,
            # Optional measurements absent from the dict mean "not measured"
            # (None) so a contract-required field FAILS closed, never defaults
            # to a silent 0.0 pass.
            date_coverage=_float_or_none(telemetry.get("date_coverage")),
            missing_fraction=_float_or_none(telemetry.get("missing_fraction")),
            regime_obs=_int_or_none(telemetry.get("regime_obs")),
            expert_obs=_int_or_none(telemetry.get("expert_obs")),
            state_transitions=_int_or_none(telemetry.get("state_transitions")),
            cross_section_peers=_int_or_none(
                telemetry.get(
                    "cross_section_peers", telemetry.get("median_stocks_per_date")
                )
            ),
        )
    _, failures = sample_adequacy_met(
        contract=contract,
        raw_obs=t.raw_obs,
        effective_obs=t.effective_obs,
        unique_dates=t.unique_dates,
        unique_stocks=t.unique_stocks,
        free_parameter_count=free_parameters,
        regime_obs=t.regime_obs,
        expert_obs=t.expert_obs,
        state_transitions=t.state_transitions,
        cross_section_peers=t.cross_section_peers,
        missing_fraction=t.missing_fraction,
        date_coverage=t.date_coverage,
    )
    return failures


def adequacy_report(
    contract: SampleAdequacyContract,
    telemetry: SampleTelemetry,
) -> tuple[bool, list[str]]:
    """Gate the measured telemetry against a :class:`SampleAdequacyContract`."""
    failures = adequacy_failures(contract, telemetry, telemetry.free_parameter_count)
    return (not failures), failures


def telemetry_summary(telemetry: SampleTelemetry) -> dict[str, Any]:
    """JSON-able dict for evidence ledgers."""
    return asdict(telemetry)
