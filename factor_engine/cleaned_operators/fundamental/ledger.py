# -*- coding: utf-8 -*-
"""Per-period fundamental revision ledger (post-release restatement support).

Round-3 audit item 27: a ``daily x`` + ``daily period_id`` panel alone cannot
express that a value *previously released* for a report period was later
superseded/revised.  This module records, per report period, the full
observation history — the release row and every later supersede/revision row —
and exposes a point-in-time ``value_as_of`` lookup so an operator can pick the
value valid on a given decision date without ever silently blending the
original and revised values.

The row clock is the trading-day axis of the daily as-of panel (the same axis
``transforms_v2._walk_periods`` walks).  ``value_as_of(period_key,
decision_row)`` returns the value whose latest observation row is at or before
``decision_row``, i.e. the revision valid ON that decision date.

Three operators expose the ledger to an alpha search:

* ``fin_period_restated``        — 1.0 when the report period currently visible
  at a decision date has been superseded/revised at or before that date;
  0.0 when the visible period has never been revised; NaN when no report
  period (or no value) is visible.
* ``fin_period_revision_count``  — number of supersede events for the currently
  visible period as of the decision date (0 = never revised).
* ``fin_period_revision_age``    — trading rows since the current period's last
  supersede event (0 = revised today); NaN when the period was never revised.

These complement (not replace) the daily-window ``fin_revision_*`` family in
``transforms_repairs_v2``: that family detects revisions by comparing adjacent
daily rows; this ledger is period-keyed and keeps the release/revision
timestamps themselves.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_strict import period_ordinal

_EPS = 1e-12
_SURFACE: set[str] = set()


def _default_require_parseable() -> bool:
    """Fail closed by default: unparseable report periods have no fiscal-ordinal
    contract, so the ledger refuses to place them (production semantics)."""
    try:
        from runtime.production_policy import is_production_mode

        return bool(is_production_mode())
    except Exception:
        return True


def _period_key(value: Any) -> Any:
    """Normalise a report-period cell to its hashable identity or ``None``."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value) if isinstance(value, (str, np.datetime64)) else value
    except Exception:
        return value


@dataclass(frozen=True)
class PeriodObservation:
    """A value observed for a report period at a specific trading-day row."""

    observed_at: int
    value: float


@dataclass(frozen=True)
class PeriodLedgerEntry:
    """Append-only observation history for ONE report period.

    ``first_seen_at`` is the release row; each later observation whose value
    differs from the previous one is a supersede/revision event.
    """

    period_key: Any
    first_seen_at: int
    observations: tuple[PeriodObservation, ...]

    @property
    def last_revised_at(self) -> int | None:
        """Trading-day row of the latest supersede/revision, or ``None``."""
        return self.observations[-1].observed_at if len(self.observations) > 1 else None

    @property
    def revision_count(self) -> int:
        return len(self.observations) - 1

    @property
    def latest_value(self) -> float:
        return self.observations[-1].value

    @property
    def first_value(self) -> float:
        return self.observations[0].value

    @property
    def is_restated(self) -> bool:
        return len(self.observations) > 1

    def value_as_of(self, decision_row: int) -> float | None:
        """Value valid on ``decision_row``: latest observation at or before it."""
        for observation in reversed(self.observations):
            if observation.observed_at <= decision_row:
                return observation.value
        return None


def scan_period_ledger(
    values: Any,
    period_values: Iterable[Any],
    *,
    require_parseable: bool | None = None,
) -> "OrderedDict[Any, PeriodLedgerEntry]":
    """Scan one column's daily as-of values and build the append-only ledger.

    ``values`` is a 1-D numeric series/array; ``period_values`` is the
    row-aligned report-period id sequence.  Only rows with a finite value and a
    period id are placed; a value change for the same period key is recorded as
    a supersede/revision observation (the original value is never discarded).
    """
    if require_parseable is None:
        require_parseable = _default_require_parseable()
    arr = np.asarray(values, dtype=float) if not isinstance(values, pd.Series) else values.to_numpy(dtype=float)
    pv = list(period_values)
    entries: "OrderedDict[Any, PeriodLedgerEntry]" = OrderedDict()
    for row, (value, raw) in enumerate(zip(arr, pv)):
        if not np.isfinite(value):
            continue
        key = _period_key(raw)
        if key is None:
            continue
        if require_parseable and period_ordinal(key) is None:
            continue
        entry = entries.get(key)
        if entry is None:
            entries[key] = PeriodLedgerEntry(
                period_key=key,
                first_seen_at=row,
                observations=(PeriodObservation(row, float(value)),),
            )
            continue
        if abs(entry.observations[-1].value - float(value) > _EPS:
            entries[key] = PeriodLedgerEntry(
                period_key=key,
                first_seen_at=entry.first_seen_at,
                observations=entry.observations + (PeriodObservation(row, float(value)),),
            )
    return entries


# ---------------------------------------------------------------------------
# per-column kernels (inline as-of state machines over the ledger semantics)
# ---------------------------------------------------------------------------


def _period_restated_1d(xv: np.ndarray, pv: list, require_parseable: bool) -> np.ndarray:
    n = len(xv)
    out = np.full(n, np.nan, dtype=float)
    state: dict[Any, dict[str, Any]] = {}
    for row, (value, raw) in enumerate(zip(xv, pv)):
        key = _period_key(raw)
        if key is None or not np.isfinite(value):
            continue
        if require_parseable and period_ordinal(key) is None:
            continue
        st = state.get(key)
        if st is None:
            state[key] = {"latest": float(value), "revised": False}
            out[row] = 0.0
        else:
            if abs(st["latest"] - float(value) > _EPS:
                st["latest"] = float(value)
                st["revised"] = True
            out[row] = 1.0 if st["revised"] else 0.0
    return out


def _period_revision_count_1d(xv: np.ndarray, pv: list, require_parseable: bool) -> np.ndarray:
    n = len(xv)
    out = np.full(n, np.nan, dtype=float)
    state: dict[Any, dict[str, Any]] = {}
    for row, (value, raw) in enumerate(zip(xv, pv)):
        key = _period_key(raw)
        if key is None or not np.isfinite(value):
            continue
        if require_parseable and period_ordinal(key) is None:
            continue
        st = state.get(key)
        if st is None:
            st = {"latest": float(value), "count": 0}
            state[key] = st
        else:
            if abs(st["latest"] - float(value) > _EPS:
                st["latest"] = float(value)
                st["count"] += 1
        out[row] = float(st["count"])
    return out


def _period_revision_age_1d(
    xv: np.ndarray, pv: list, require_parseable: bool, max_days: int
) -> np.ndarray:
    n = len(xv)
    out = np.full(n, np.nan, dtype=float)
    state: dict[Any, dict[str, Any]] = {}
    for row, (value, raw) in enumerate(zip(xv, pv)):
        key = _period_key(raw)
        if key is None or not np.isfinite(value):
            continue
        if require_parseable and period_ordinal(key) is None:
            continue
        st = state.get(key)
        if st is None:
            st = {"latest": float(value), "last_rev": None}
            state[key] = st
        else:
            if abs(st["latest"] - float(value) > _EPS:
                st["latest"] = float(value)
                st["last_rev"] = row
        if st["last_rev"] is None:
            out[row] = np.nan
        else:
            out[row] = float(min(max_days, row - st["last_rev"]))
    return out


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def _pos_int(value, name: str, minimum: int = 1) -> int:
    # NEW-088: strict gate — ``int(3.7) -> 3`` truncation and ``True -> 1``
    # are contract violations, not coercions.  Same single authority as the
    # rest of the fundamental family (NEW-005/006).
    from cleaned_operators.common.strict_params import strict_int

    return strict_int(value, name, minimum=minimum)


def _register(
    name: str,
    params: Iterable[str],
    fn,
    description: str,
    *,
    unit: str,
    extra_tags: Iterable[str] = (),
) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "typed_v2",
            f"signature:{','.join(params)}->series",
            "domain:fundamental",
            f"unit:{unit}",
            "cost:1",
            "revision_ledger",
            *extra_tags,
        ],
        input_units={"x": "same_as:output", "period_id": "fiscal_period"},
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"FundamentalLedger_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental.ledger",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _SURFACE.add(name)
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})
    return cls


def fin_period_restated(x: pd.DataFrame, period_id: pd.DataFrame) -> pd.DataFrame:
    """1.0 where the currently visible report period has been revised."""
    period_id = period_id.reindex(index=x.index, columns=x.columns)
    require_parseable = _default_require_parseable()
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        xv = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
        pv = period_id[col].to_numpy()
        out[col] = _period_restated_1d(xv, pv, require_parseable)
    return out


def fin_period_revision_count(x: pd.DataFrame, period_id: pd.DataFrame) -> pd.DataFrame:
    """Number of supersede events for the currently visible report period."""
    period_id = period_id.reindex(index=x.index, columns=x.columns)
    require_parseable = _default_require_parseable()
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        xv = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
        pv = period_id[col].to_numpy()
        out[col] = _period_revision_count_1d(xv, pv, require_parseable)
    return out


def fin_period_revision_age(
    x: pd.DataFrame, period_id: pd.DataFrame, max_days: int = 504
) -> pd.DataFrame:
    """Trading rows since the current period's last supersede event."""
    cap = _pos_int(max_days, "max_days")
    period_id = period_id.reindex(index=x.index, columns=x.columns)
    require_parseable = _default_require_parseable()
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        xv = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
        pv = period_id[col].to_numpy()
        out[col] = _period_revision_age_1d(xv, pv, require_parseable, cap)
    return out


_register(
    "fin_period_restated",
    ["x", "period_id"],
    fin_period_restated,
    "1.0 when the report period currently visible at a decision date has been "
    "superseded/revised at or before that date; 0.0 when the visible period was "
    "never revised; NaN when no report period/value is visible (period-keyed "
    "revision ledger, round-3 item 27).",
    unit="flag",
)
_register(
    "fin_period_revision_count",
    ["x", "period_id"],
    fin_period_revision_count,
    "Number of supersede/revision events for the report period currently "
    "visible at a decision date, as of that date (0 = never revised; NaN when "
    "no report period/value is visible).",
    unit="count",
)
_register(
    "fin_period_revision_age",
    ["x", "period_id", "max_days"],
    fin_period_revision_age,
    "Trading rows since the currently visible report period's last "
    "supersede/revision (0 = revised today; NaN when never revised).",
    unit="days",
)


from cleaned_operators import operator_surface as _surface  # noqa: E402

_surface.extend_extended_only(_SURFACE)
