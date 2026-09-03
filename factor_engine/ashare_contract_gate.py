# -*- coding: utf-8 -*-
"""A-share data-contract gate (P0#10 / GO prompt §110 item 10).

The 100k-factor production run's most dangerous failure mode is not "slow" —
it is "finished, then discovered the Return unit was wrong, a financial PIT
leaked, an intraday bar bridged across sessions, or a relation panel joined on
current membership".  This module is the single, non-bypassable gate that
checks an operator's declared data contract against A-share semantics before
the run, so a bad contract fails closed (hard violation) or is surfaced as a
warning (soft violation) instead of silently manufacturing a wrong factor.

The gate is deliberately *declarative*: it validates a ``panel_meta`` dict that
describes the operator's input contract (units, index timestamps, PIT columns,
membership join keys).  It does not itself read data — the caller (an operator
preflight, the Analyzer validator chain, or a run_many hook) supplies the
metadata it already resolved from the field catalog / panel schema.

Severity model (per GO prompt §110 "fail-closed or warn per prompt"):

* **FAIL** (hard) — a violation that would silently corrupt the factor:
  * return-basis unit mislabel (a decimal 0.05 labeled ``bp``, or a percent
    labeled ``ratio``) — the repo convention is ``Vwap.pct_change().shift(-1)``
    with the Adj-table ``Return`` column in **basis points**;
  * a minute index timestamp outside the A-share session windows
    (09:30-11:30 / 13:00-15:00 Asia/Shanghai) — lunch-gap, overnight, or
    weekend bars bridge sessions and manufacture fake intraday structure;
  * a PIT violation: an as-of / announcement column whose timestamp is strictly
    AFTER the feature/decision timestamp (a future join);
  * a membership join (industry / index / relation) that uses *current*
    membership instead of point-in-time membership.

* **WARN** (soft) — a contract that is suspicious but not provably corrupting:
  * a return field whose unit is ``dimensionless`` / unlabeled when a return
    basis is expected;
  * a minute panel whose session windows are present but whose first/last bar
    is not exactly on a session boundary (e.g. a 09:31 first bar in the
    240-bar mode);
  * a membership join whose PIT column is present but not declared as the join
    key.

The gate returns a :class:`ContractVerdict` with ``failures`` and ``warnings``
lists; callers decide whether to raise (fail-closed) or log.  The helper
:func:`assert_contract` raises :class:`AShareContractViolation` on any failure
for the fail-closed path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# A-share session model (Asia/Shanghai).
# ---------------------------------------------------------------------------
# Canonical continuous-trading sessions.  The GO prompt §3.2 "240 bars/day"
# mode starts the first bar at 09:31 and resumes at 13:01; the canonical
# session *windows* are 09:30-11:30 and 13:00-15:00.  A bar whose time falls
# strictly inside the lunch gap (11:30-13:00) or outside both windows is a
# session violation.
ASHARE_TZ = "Asia/Shanghai"
MORNING_OPEN = pd.Timestamp("09:30:00").time()
MORNING_CLOSE = pd.Timestamp("11:30:00").time()
AFTERNOON_OPEN = pd.Timestamp("13:00:00").time()
AFTERNOON_CLOSE = pd.Timestamp("15:00:00").time()

#: Return-basis unit families.  The repo convention (memory: vwap-to-vwap
#: return basis) is that a *return* is ``Vwap.pct_change().shift(-1)`` and the
#: Adj-table ``Return`` column is in **basis points** (``/10000``).  A decimal
#: return (0.05) must be labeled ``ratio``; a percent return (5.0) must be
#: labeled ``percent``; a bp return (500) must be labeled ``basis_point``.
RETURN_UNITS = frozenset({"ratio", "percent", "basis_point"})
#: Unit spellings that are NOT a return basis — a return field must never be
#: labeled with these.
NON_RETURN_UNITS = frozenset({"dimensionless", "CNY", "CNY/share", "share", "text", "boolean"})

#: Column-name hints that a field is a return / return-like quantity.
_RETURN_NAME_HINTS = ("return", "ret", "pct_change", "momentum", "yield", "excess_ret")


class AShareContractViolation(ValueError):
    """Raised by :func:`assert_contract` when a hard violation is present."""


@dataclass(frozen=True)
class ContractVerdict:
    """Result of a contract check: hard ``failures`` + soft ``warnings``."""

    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    def merge(self, other: "ContractVerdict") -> "ContractVerdict":
        return ContractVerdict(
            failures=self.failures + other.failures,
            warnings=self.warnings + other.warnings,
        )


def _is_return_name(name: str) -> bool:
    low = str(name or "").lower()
    return any(hint in low for hint in _RETURN_NAME_HINTS)


def _unit_family(unit: str | None) -> str | None:
    """Map a unit spelling to its canonical family (ratio/percent/basis_point)."""
    if unit is None:
        return None
    u = str(unit).strip().lower()
    if u in {"ratio", "fraction", "1"}:
        return "ratio"
    if u in {"percent", "%", "percentage_point"}:
        return "percent"
    if u in {"bp", "bps", "basis_point"}:
        return "basis_point"
    return None


# ---------------------------------------------------------------------------
# (a) Return-basis unit contract
# ---------------------------------------------------------------------------
def _return_magnitude_ok(value: float, family: str) -> bool:
    """True iff a return ``value`` is plausible for its declared unit family.

    A return is a small fraction of price.  In the repo convention the Adj-table
    ``Return`` column is in **basis points** (``/10000``), so a daily return is
    on the order of tens-to-hundreds of bp (0.01%..10% = 1..1000 bp).  A decimal
    return (0.05) is ``ratio``; a percent return (5.0) is ``percent``.  A value
    that is implausible for its declared family is a mislabel (e.g. 0.05 labeled
    ``basis_point`` would be 0.00005% — a degenerate near-zero return).
    """
    v = float(value)
    if family == "ratio":
        # A ratio return is a small fraction: |r| < 1 (a 100% move is extreme).
        return abs(v) < 1.0
    if family == "percent":
        # A percent return is a small number of percent: |r| < 100.
        return abs(v) < 100.0
    if family == "basis_point":
        # A bp return is a small number of bp: 1..10000 (0.01%..100%).  A value
        # below 1 bp (e.g. 0.05) is a degenerate near-zero return — a decimal
        # 0.05 is a *ratio* (5%), not 0.00005%.
        return 1.0 <= abs(v) < 10000.0
    return True


def check_return_units(panel_meta: dict[str, Any]) -> ContractVerdict:
    """Reject a return field whose declared unit contradicts its magnitude basis.

    ``panel_meta["return_fields"]`` is a list of ``{name, unit, value}`` dicts
    (or a dict ``name -> unit``).  ``value`` is an optional representative sample
    of the field's magnitude (e.g. the max |return| over the panel).  A field
    whose name looks like a return must carry a return-basis unit
    (ratio / percent / basis_point); a non-return unit label (e.g.
    ``dimensionless``) on a return field is a hard failure.  When a ``value`` is
    supplied, the gate also verifies the magnitude is plausible for the declared
    unit — a decimal 0.05 labeled ``basis_point`` (or a percent 5.0 labeled
    ``ratio``) is a hard mislabel.  A return field with no declared unit is a
    warning (unlabeled, not provably wrong).
    """
    failures: list[str] = []
    warnings: list[str] = []
    raw = panel_meta.get("return_fields")
    if raw is None:
        return ContractVerdict()
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if isinstance(v, dict):
                items.append({"name": k, **v})
            else:
                items.append({"name": k, "unit": v})
    else:
        items = list(raw)
    for item in items:
        name = str(item.get("name", "?"))
        unit = item.get("unit")
        family = _unit_family(unit)
        if family is not None:
            # A return field must be a return-basis unit.
            if family not in RETURN_UNITS:
                failures.append(
                    f"return field {name!r} is labeled unit {unit!r} "
                    f"(family {family!r}) which is not a return basis; repo "
                    "convention is Vwap.pct_change().shift(-1) with the Adj "
                    "Return column in basis points (ratio/percent/basis_point)"
                )
                continue
            # Magnitude-vs-unit consistency (only when a sample value is given).
            value = item.get("value")
            if value is not None:
                try:
                    if not _return_magnitude_ok(value, family):
                        failures.append(
                            f"return field {name!r} value {value!r} is "
                            f"implausible for declared unit {unit!r} "
                            f"(family {family!r}); a decimal 0.05 is 'ratio', "
                            "a percent 5.0 is 'percent', a bp 500 is "
                            "'basis_point' — the unit label contradicts the "
                            "magnitude basis"
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"return field {name!r} value {value!r} is not numeric; "
                        "cannot verify magnitude-vs-unit consistency"
                    )
            continue
        # Unknown / non-return unit spelling.
        if unit is not None and str(unit).strip().lower() in NON_RETURN_UNITS:
            failures.append(
                f"return field {name!r} is labeled unit {unit!r} which is not a "
                "return basis (ratio/percent/basis_point)"
            )
        elif unit is None or str(unit).strip() == "":
            warnings.append(
                f"return field {name!r} has no declared unit; cannot verify the "
                "return basis (ratio/percent/basis_point)"
            )
        else:
            warnings.append(
                f"return field {name!r} has unrecognized unit {unit!r}; "
                "cannot verify the return basis"
            )
    return ContractVerdict(tuple(failures), tuple(warnings))


# ---------------------------------------------------------------------------
# (b) Minute / session contract
# ---------------------------------------------------------------------------
def _in_session(t: pd.Timestamp) -> bool:
    """True iff ``t`` (Asia/Shanghai wall-clock) is inside a trading session."""
    tt = t.time()
    morning = MORNING_OPEN <= tt <= MORNING_CLOSE
    afternoon = AFTERNOON_OPEN <= tt <= AFTERNOON_CLOSE
    return morning or afternoon


def check_minute_sessions(panel_meta: dict[str, Any]) -> ContractVerdict:
    """Reject minute-index timestamps outside the A-share session windows.

    ``panel_meta["minute_index"]`` is a sequence of timestamps (or a
    ``pd.DatetimeIndex``).  Each is converted to Asia/Shanghai wall-clock; a
    timestamp on a weekend, or whose wall-clock time falls in the lunch gap
    (11:30-13:00) or outside 09:30-11:30 / 13:00-15:00, is a hard session
    violation (it bridges sessions / manufactures fake intraday structure).
    A timestamp that is inside a session window but not exactly on a session
    boundary (e.g. a 09:31 first bar in the 240-bar mode) is a warning.
    """
    failures: list[str] = []
    warnings: list[str] = []
    idx = panel_meta.get("minute_index")
    if idx is None:
        return ContractVerdict()
    if isinstance(idx, pd.DatetimeIndex):
        stamps = idx
    else:
        stamps = pd.DatetimeIndex(pd.to_datetime(list(idx), errors="coerce"))
    if stamps.tz is None:
        stamps = stamps.tz_localize(ASHARE_TZ)
    else:
        stamps = stamps.tz_convert(ASHARE_TZ)
    boundary_times = {MORNING_OPEN, MORNING_CLOSE, AFTERNOON_OPEN, AFTERNOON_CLOSE}
    for ts in stamps:
        if pd.isna(ts):
            continue
        if ts.weekday() >= 5:
            failures.append(
                f"minute bar {ts} falls on a weekend; A-share sessions are "
                "Mon-Fri Asia/Shanghai (no overnight/weekend bridging)"
            )
            continue
        if not _in_session(ts):
            failures.append(
                f"minute bar {ts} is outside the A-share session windows "
                "(09:30-11:30 / 13:00-15:00 Asia/Shanghai); lunch-gap or "
                "overnight bars bridge sessions"
            )
        elif ts.time() not in boundary_times:
            warnings.append(
                f"minute bar {ts} is inside a session but not on a session "
                "boundary (09:30/11:30/13:00/15:00); verify the 240-bar mode "
                "first-bar convention"
            )
    return ContractVerdict(tuple(failures), tuple(warnings))


# ---------------------------------------------------------------------------
# (c) PIT / no-lookahead contract
# ---------------------------------------------------------------------------
def check_pit_no_lookahead(panel_meta: dict[str, Any]) -> ContractVerdict:
    """Reject an as-of / announcement column that is joined into the future.

    ``panel_meta["pit_columns"]`` is a list of ``{name, asof, decision}`` dicts
    (or a dict ``name -> (asof, decision)``) where ``asof`` is the event's
    announcement / as-of timestamp and ``decision`` is the feature / decision
    timestamp for the same row.  A row whose ``asof > decision`` is a hard PIT
    violation (a future join — the event was not knowable at decision time).
    """
    failures: list[str] = []
    raw = panel_meta.get("pit_columns")
    if raw is None:
        return ContractVerdict()
    if isinstance(raw, dict):
        items = [{"name": k, "asof": v[0], "decision": v[1]} for k, v in raw.items()]
    else:
        items = list(raw)
    for item in items:
        name = str(item.get("name", "?"))
        asof = pd.to_datetime(item.get("asof"), errors="coerce")
        decision = pd.to_datetime(item.get("decision"), errors="coerce")
        if pd.isna(asof) or pd.isna(decision):
            failures.append(
                f"PIT column {name!r} has a null asof/decision timestamp; "
                "cannot prove no-lookahead (fail closed)"
            )
            continue
        if asof > decision:
            failures.append(
                f"PIT violation on {name!r}: as-of/announcement {asof} is "
                f"strictly AFTER the decision/feature timestamp {decision}; "
                "a future join leaks information not knowable at decision time"
            )
    return ContractVerdict(tuple(failures))


# ---------------------------------------------------------------------------
# (d) Membership / relation PIT contract
# ---------------------------------------------------------------------------
def check_membership_pit(panel_meta: dict[str, Any]) -> ContractVerdict:
    """Reject a membership join (industry / index / relation) that uses current
    membership instead of point-in-time membership.

    ``panel_meta["membership_joins"]`` is a list of ``{name, has_pit_key}``
    dicts (or a dict ``name -> has_pit_key``).  A membership join without a
    point-in-time key (``has_pit_key=False``) is a hard failure — current
    membership retroactively reclassifies a stock's industry / index / relation
    and leaks the future.  A join with a PIT key present but not declared as the
    join key is a warning.
    """
    failures: list[str] = []
    warnings: list[str] = []
    raw = panel_meta.get("membership_joins")
    if raw is None:
        return ContractVerdict()
    if isinstance(raw, dict):
        items = [{"name": k, "has_pit_key": v} for k, v in raw.items()]
    else:
        items = list(raw)
    for item in items:
        name = str(item.get("name", "?"))
        has_pit = bool(item.get("has_pit_key", False))
        if not has_pit:
            failures.append(
                f"membership join {name!r} (industry/index/relation) has no "
                "point-in-time membership key; current membership retroactively "
                "reclassifies constituents and leaks the future"
            )
        elif item.get("pit_key_is_join_key") is False:
            warnings.append(
                f"membership join {name!r} has a PIT key but it is not the join "
                "key; verify the join uses point-in-time membership"
            )
    return ContractVerdict(tuple(failures), tuple(warnings))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def validate_ashare_return_unit_contracts(referenced_fields: dict[str, Any]) -> list[str]:
    """Field-level return-unit gate for the Analyzer compile path.

    Builds a ``return_fields`` panel_meta from the referenced ``FieldSpec``
    dicts (keyed by field name) and runs :func:`check_return_units`.  A field
    whose name looks like a return must carry a return-basis unit
    (ratio / percent / basis_point); a non-return unit label (e.g.
    ``dimensionless``) is a hard failure.  Returns a list of human-readable
    errors (empty when every return field is contracted).  This is the
    non-bypassable hook called from :meth:`Analyzer.lower` so a formula reaching
    a mislabeled return field fails at compile time (GO prompt §110 item 10).
    """
    return_fields: list[dict[str, Any]] = []
    for name, spec in referenced_fields.items():
        if not _is_return_name(name):
            continue
        unit = getattr(spec, "unit", None)
        return_fields.append({"name": name, "unit": unit})
    if not return_fields:
        return []
    verdict = check_return_units({"return_fields": return_fields})
    return list(verdict.failures)


def validate_contract(
    operator_name: str,
    panel_meta: dict[str, Any],
) -> ContractVerdict:
    """Run all A-share contract checks for one operator's data contract.

    Args:
        operator_name: Operator / factor name (for error messages).
        panel_meta: Declarative contract metadata.  Recognized keys:
            ``return_fields``, ``minute_index``, ``pit_columns``,
            ``membership_joins`` (see each check for the schema).

    Returns:
        A :class:`ContractVerdict` with hard ``failures`` and soft ``warnings``.
    """
    verdict = ContractVerdict()
    for check in (
        check_return_units,
        check_minute_sessions,
        check_pit_no_lookahead,
        check_membership_pit,
    ):
        verdict = verdict.merge(check(panel_meta))
    if verdict.failures:
        # Prefix each failure with the operator name for traceability.
        verdict = ContractVerdict(
            failures=tuple(f"{operator_name}: {f}" for f in verdict.failures),
            warnings=verdict.warnings,
        )
    return verdict


def assert_contract(operator_name: str, panel_meta: dict[str, Any]) -> ContractVerdict:
    """Fail-closed entry point: raise on any hard violation, else return verdict.

    Warnings are returned (not raised) so callers can log them; hard failures
    raise :class:`AShareContractViolation`.
    """
    verdict = validate_contract(operator_name, panel_meta)
    if verdict.failures:
        raise AShareContractViolation(
            f"A-share contract gate rejected {operator_name!r}:\n"
            + "\n".join(f"  - {f}" for f in verdict.failures)
        )
    return verdict


__all__ = [
    "ASHARE_TZ",
    "AShareContractViolation",
    "ContractVerdict",
    "assert_contract",
    "check_membership_pit",
    "check_minute_sessions",
    "check_pit_no_lookahead",
    "check_return_units",
    "validate_ashare_return_unit_contracts",
    "validate_contract",
]
