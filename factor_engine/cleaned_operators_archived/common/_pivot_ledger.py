# -*- coding: utf-8 -*-
"""Streaming confirmed-pivot ledger (shared PIT-correct pivot chain).

Retrospective history-rewrite P0 fix (R11 audit findings #1/#2)
---------------------------------------------------------------
The previous pivot chains in ``structural_levels._scan_confirmed_pivots`` and
``extrema_divergence._confirmed_extrema`` kept a *single mutable* chain: when a
future same-side candidate was more extreme than the last confirmed pivot of
that side they rewrote the past by deleting the older pivot from the final
list.  Because downstream factors re-read that final list for every historical
row, future data changed already-published past factor values (future-function
/ PIT violation).

This module replaces the mutable chain with an *append-only streaming ledger*:

* at each row ``t`` exactly one candidate, ``pivot_at = t - confirmation``,
  may complete confirmation (the only row at which its ``±confirmation``
  window is fully available);
* confirmed pivots are immutable ``ConfirmedPivotEvent`` records and are never
  retroactively deleted;
* a later more-extreme same-side candidate emits a ``PivotSuperseded`` record
  and a new ``ConfirmedPivotEvent`` whose ``supersedes`` field points at the
  old pivot;
* supersession only changes the *active* pivot set for rows
  ``>= effective_at`` (the new pivot's confirmation row), so
  ``factor(full_data)[:T] == factor(data[:T])`` holds for every prefix ``T``.

Confirmation contract (finding #1/#2, item 5)
---------------------------------------------
A candidate ``j`` is confirmed at row ``t = j + confirmation`` iff the full
``±confirmation`` neighbourhood ``[j-confirmation, j+confirmation]`` lies
inside the series *and* every element is finite.  At the series start/end
where the full window is unavailable, no asymmetric window is used to
"confirm" a pivot, and a NaN anywhere inside the window blocks confirmation.

Both ``structural_levels`` and ``extrema_divergence`` route their *strictly
alternating* confirmed-pivot chain through this single implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

PEAK = 1
TROUGH = -1


@dataclass(frozen=True)
class ConfirmedPivotEvent:
    """A pivot whose full confirmation window completed at row ``confirmed_at``.

    ``side`` is ``+1`` for a peak (strict local max) and ``-1`` for a trough
    (strict local min).  ``supersedes`` references the previously-confirmed
    same-side pivot that this one replaces from ``confirmed_at`` onward;
    ``None`` for a pivot that establishes a fresh chain link.
    """

    pivot_at: int
    side: int
    value: float
    confirmed_at: int
    supersedes: Optional["ConfirmedPivotEvent"] = None


@dataclass(frozen=True)
class PivotSuperseded:
    """A previously-confirmed pivot replaced by a more-extreme same-side one.

    Only affects factor rows ``>= effective_at``; rows before that keep seeing
    ``old_pivot``.  ``effective_at`` equals the new pivot's confirmation row.
    """

    old_pivot: ConfirmedPivotEvent
    new_pivot: ConfirmedPivotEvent
    effective_at: int


@dataclass(frozen=True)
class PivotLedgerResult:
    """Append-only outcome of a ledger scan.

    ``events`` are ordered by confirmation row (equivalently, by pivot bar).
    ``active_at(t, window=None)`` returns the pivots usable at row ``t`` —
    confirmed at or before ``t`` and not superseded by a later-confirmed
    same-side pivot at or before ``t``.
    """

    events: tuple[ConfirmedPivotEvent, ...]
    superseded: tuple[PivotSuperseded, ...]
    confirmation: int
    _superseder_at: dict = field(default_factory=dict, repr=False, compare=False, init=False)

    def __post_init__(self) -> None:
        m: dict[int, int] = {}
        for ev in self.events:
            if ev.supersedes is not None:
                m[id(ev.supersedes)] = ev.confirmed_at
        object.__setattr__(self, "_superseder_at", m)

    def active_at(self, t: int, window: Optional[int] = None) -> tuple[ConfirmedPivotEvent, ...]:
        """Confirmed pivots usable at row ``t``.

        A pivot is active at ``t`` when it was confirmed at or before ``t`` and
        no later-confirmed same-side pivot has superseded it at or before ``t``.
        ``window`` optionally restricts pivots to the trailing window
        ``[t-window+1, t]`` (matching the historical ``_usable_pivots`` /
        ``_side_indices`` trailing-window contract).
        """
        i0 = t - window + 1 if window is not None else 0
        sup = self._superseder_at
        out: list[ConfirmedPivotEvent] = []
        for ev in self.events:
            if ev.confirmed_at > t:
                break
            if ev.pivot_at < i0:
                continue
            eff = sup.get(id(ev))
            if eff is not None and eff <= t:
                continue
            out.append(ev)
        return tuple(out)


class StreamingConfirmedPivotLedger:
    """Streaming strict-alternation confirmed-pivot chain (append-only).

    Parameters
    ----------
    confirmation : int
        Half-width of the confirmation window (``>= 1``).
    prominence : float
        Minimum relative step, ``prominence * |candidate|``, required for an
        opposite-side candidate to be accepted after the last accepted pivot.
    positive_only : bool
        When True, non-positive bars can never be confirmed as pivots (price
        semantics used by ``structural_levels``); the raw detection used by
        ``extrema_divergence`` leaves this False.
    """

    __slots__ = ("confirmation", "prominence", "positive_only")

    def __init__(self, confirmation: int, prominence: float, *, positive_only: bool = False) -> None:
        conf = int(confirmation)
        if conf < 1:
            raise ValueError("confirmation must be >= 1")
        prom = float(prominence)
        if not np.isfinite(prom) or prom < 0.0:
            raise ValueError("prominence must be a finite non-negative number")
        self.confirmation = conf
        self.prominence = prom
        self.positive_only = bool(positive_only)

    def scan(self, x: np.ndarray) -> PivotLedgerResult:
        """Scan one column and return the append-only event ledger."""
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        conf = self.confirmation
        prom = self.prominence
        pos_only = self.positive_only

        events: list[ConfirmedPivotEvent] = []
        superseded: list[PivotSuperseded] = []
        last: Optional[ConfirmedPivotEvent] = None

        # Row t == pivot_at + confirmation completes that candidate's window.
        # The full window requires pivot_at >= confirmation, i.e. t >= 2*conf.
        for t in range(2 * conf, n):
            j = t - conf
            window = x[j - conf : t + 1]  # exactly [j-conf, j+conf]
            if not np.all(np.isfinite(window)):
                continue  # any NaN inside the confirmation window blocks it
            xj = x[j]
            if pos_only and not (xj > 0.0):
                continue
            left = window[:conf]
            right = window[conf + 1 :]
            is_peak = bool(np.all(xj > left) and np.all(xj > right))
            is_trough = bool(np.all(xj < left) and np.all(xj < right))
            if is_peak == is_trough:
                continue  # flat neighbourhood -> not a turning point
            side = PEAK if is_peak else TROUGH
            val = float(xj)

            if last is None:
                # first pivot of the chain is accepted directly (it establishes
                # the alternating reference).
                ev = ConfirmedPivotEvent(pivot_at=j, side=side, value=val, confirmed_at=t)
                events.append(ev)
                last = ev
            elif side != last.side:
                # Opposite side: accept only when the move clears prominence.
                if side == PEAK:
                    ok = (val - last.value) > prom * abs(val)
                else:
                    ok = (last.value - val) > prom * abs(val)
                if ok:
                    ev = ConfirmedPivotEvent(pivot_at=j, side=side, value=val, confirmed_at=t)
                    events.append(ev)
                    last = ev
                # else: ignored — does not break the chain, does not change last.
            else:
                # Same side: supersede (never rewrite the past) when strictly
                # more extreme; otherwise ignored.
                if (side == PEAK and val > last.value) or (side == TROUGH and val < last.value):
                    ev = ConfirmedPivotEvent(
                        pivot_at=j,
                        side=side,
                        value=val,
                        confirmed_at=t,
                        supersedes=last,
                    )
                    events.append(ev)
                    superseded.append(
                        PivotSuperseded(old_pivot=last, new_pivot=ev, effective_at=t)
                    )
                    last = ev

        return PivotLedgerResult(
            events=tuple(events),
            superseded=tuple(superseded),
            confirmation=conf,
        )


def confirmed_pivot_events(
    x: np.ndarray,
    confirmation: int,
    prominence: float,
    *,
    positive_only: bool = False,
) -> PivotLedgerResult:
    """Convenience wrapper: scan ``x`` with a fresh streaming ledger."""
    return StreamingConfirmedPivotLedger(
        confirmation, prominence, positive_only=positive_only
    ).scan(x)
