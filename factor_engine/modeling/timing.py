# -*- coding: utf-8 -*-
"""Decision-clock validation + VWAP-to-VWAP label authority (Model Layer Major
Redesign taskbook §8 / §11 / §11.2 / §13.11 / §62).

The DecisionClock is the hidden-future-function defence line: every feature
input, the label, the score write, and the execution each have an explicit
availability.  ``clock_compliant`` answers "may this feature drive a
tradeable factor produced at this decision clock?".
"""
from __future__ import annotations

from typing import Any, Iterable

from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    BEFORE_SAME_DAY_VWAP,
    DecisionClock,
    LabelContract,
    ashare_decision_clock,
)

__all__ = [
    "clock_compliant",
    "feature_available_before_decision",
    "check_same_day_target_gate",
    "label_maturity_bar",
    "vwap_to_vwap_label",
    "sample_label_interval",
    "feature_availability_problem",
]


import re


def _time_ordinal(s: str) -> float:
    """Map a decision-clock time label to a comparable ordinal.

    Base is decision-day ``t`` (0.0); ``t+1`` shifts a full day.  Within a day:
    ``prev_close`` = -0.5, ``open`` = 0.2, ``VWAP``/``mid`` = 0.5,
    ``close`` = 1.0.  ``t+H close`` = 1.0 + H.  Unknown labels sort to a
    conservative +inf so availability is never assumed when unparseable.
    """
    if s is None:
        return float("inf")
    text = str(s).strip().lower().replace("_", " ")
    mh = re.fullmatch(r"t\s*\+\s*h\s*(\d+)\s*(.*)", text)
    if mh:
        return 1.0 + float(mh.group(1))
    m = re.match(r"t\s*([+\-]\s*\d+)?(.*)$", text)
    offset = 0.0
    if m and m.group(1):
        offset = float(m.group(1).replace(" ", ""))
    suffix = (m.group(2) if m else "") or ""
    if "prev close" in suffix or "prev_close" in suffix:
        base = -0.5
    elif "open" in suffix:
        base = 0.2
    elif "vwap" in suffix or "mid" in suffix:
        base = 0.5
    elif "close" in suffix:
        base = 1.0
    else:
        base = float("inf")
    return offset + base


def feature_available_before_decision(
    clock: DecisionClock, feature: str, required_by: str | None = None
) -> bool:
    """Whether ``feature`` is available by ``required_by``.

    The deadline defaults to the clock's **execution time** — a feature is only
    usable to drive a *tradeable* factor when it is available no later than the
    execution.  Under BEFORE_SAME_DAY_VWAP, day-t close / full-day VWAP /
    full-day volume complete at ``t_close`` which is after the ``t VWAP``
    execution, so they fail the clock (§11.1 / §62).
    """
    deadline = required_by or clock.execution_at
    avail = clock.feature_available(feature)
    return _time_ordinal(avail) <= _time_ordinal(deadline)


def feature_availability_problem(
    clock: DecisionClock, feature: str, required_by: str | None = None
) -> str | None:
    """Human-readable violation, or ``None`` when compliant."""
    deadline = required_by or clock.execution_at
    avail = clock.feature_available(feature)
    if _time_ordinal(avail) <= _time_ordinal(deadline):
        return None
    return (
        f"feature {feature!r} available at {avail} but required by {deadline} "
        f"(clock={clock.decision_at}, execution={clock.execution_at})"
    )


def clock_compliant(
    clock: DecisionClock,
    features: Iterable[str],
    required_by: str | None = None,
) -> tuple[bool, list[str]]:
    """Gate a feature set against a decision clock.

    Returns ``(ok, problems)``.  Any feature whose availability is after the
    required time fails the clock — same-day full-close / full-day-volume
    features fail under BEFORE_SAME_DAY_VWAP.
    """
    problems: list[str] = []
    for f in features:
        problem = feature_availability_problem(clock, f, required_by)
        if problem is not None:
            problems.append(problem)
    return (not problems), problems


# --------------------------------------------------------------------------- #
# Same-day target gate (§11.2 / §62)
# --------------------------------------------------------------------------- #
def check_same_day_target_gate(
    scenario: str,
    *,
    target_uses_full_day: bool = False,
    features: Iterable[str] = (),
) -> tuple[bool, list[str]]:
    """Reject same-day full-day targets / features for same-day execution.

    * AFTER_CLOSE_TO_NEXT_VWAP — close / full-day VWAP / full-day volume are
      legal feature inputs (day-t close is fully known before the t+1 VWAP
      execution).
    * BEFORE_SAME_DAY_VWAP      — a same-day full-day target or any close /
      full-day VWAP / full-day-volume feature must be rejected (§11.2 / §62).
    """
    problems: list[str] = []
    if scenario == BEFORE_SAME_DAY_VWAP:
        if target_uses_full_day:
            problems.append(
                "BEFORE_SAME_DAY_VWAP cannot use a same-day full-day target "
                "(target_t = full-day return_t is not tradeable at t VWAP)"
            )
        for f in features:
            if f in ("close", "vwap", "volume", "day_vwap", "full_day_volume"):
                problems.append(
                    f"BEFORE_SAME_DAY_VWAP: feature {f!r} completes only at day close; "
                    "not available before same-day VWAP execution"
                )
    elif scenario == AFTER_CLOSE_TO_NEXT_VWAP:
        pass  # close is known before the t+1 execution (§62 allows close here)
    else:
        problems.append(f"unknown scenario {scenario!r}")
    return (not problems), problems


# --------------------------------------------------------------------------- #
# VWAP-to-VWAP label authority (§8)
# --------------------------------------------------------------------------- #
def vwap_to_vwap_label(
    label_name: str,
    horizon_bars: int,
    *,
    overlapping: bool = False,
    embargo_bars: int = 0,
) -> LabelContract:
    """Build the A-share authority label::

        label_t(H) = VWAP_{t+H} / VWAP_t - 1
    """
    return LabelContract(
        label_name=label_name,
        origin_time="t",
        start_time_rule="entry VWAP completes at t close",
        end_time_rule=f"exit VWAP completes at t+{horizon_bars} close",
        availability_time_rule=f"label matured at t+{horizon_bars} close",
        horizon_bars=horizon_bars,
        overlapping=overlapping,
        return_basis="vwap_to_vwap",
        entry_price_basis="VWAP_t",
        exit_price_basis=f"VWAP_{{t+{horizon_bars}}}",
        purge_by_interval=True,
        embargo_bars=embargo_bars,
    )


def label_maturity_bar(anchor_index: int, contract: LabelContract) -> int:
    """The first bar at which a label anchored at ``anchor_index`` is mature."""
    return anchor_index + contract.horizon_bars


def sample_label_interval(anchor_index: int, contract: LabelContract) -> tuple[int, int]:
    """§9 — the ``[entry_time, exit_time]`` interval used by purge."""
    return contract.label_interval(anchor_index)


def mature_by(contract: LabelContract, anchor_index: int) -> int:
    return label_maturity_bar(anchor_index, contract)
