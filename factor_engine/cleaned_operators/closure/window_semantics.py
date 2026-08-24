# -*- coding: utf-8 -*-
"""Central WindowSemantics vocabulary (Master Spec Part D-16).

The single biggest fake-alpha source in advanced estimators is a ``window=60``
that silently ends up using only 24 valid rows yet still emits a value labelled
with the same statistic.  Declaring *how the window is counted* per operator —
via the same per-canonical side registry as MissingPolicy — makes the coverage
rules machine-checkable instead of kernel-private.

See :mod:`cleaned_operators.closure.missing_policy` for the declaration API.
"""
from __future__ import annotations

import enum
from typing import Optional

from factor_engine.cleaned_operators.closure.missing_policy import (
    declare_missing_policy,
    missing_policy_for,
)


class WindowSemantics(str, enum.Enum):
    """How a trailing window is counted / what it must contain.

    ``FULL_WINDOW``             — every one of the ``window`` bars must be a
        valid (finite) observation; fewer → NaN.  This is the recommended
        default for spectral / statistical estimators whose precision is a
        function of the actual sample size (Part D-17).
    ``MIN_SUPPORT_WINDOW``      — the window needs ``min_periods`` valid values;
        a partial window is legitimate when declared coverage is met.
    ``CONTIGUOUS_FULL_WINDOW``  — the ``window`` most recent bars must ALL be
        valid AND contiguous; a gap anywhere inside (incl. a current missing
        row) → NaN.  For embedding / recurrence / spectrum / H0-H1 estimators.
    ``EVENT_COUNT_WINDOW``      — the window counts the last N *events*, not
        calendar rows (event-clock history).
    ``PERIOD_COUNT_WINDOW``     — the window counts N report/fiscal periods
        (fundamental / analyst-estimate clock).
    ``SESSION_WINDOW``          — the window counts N trading sessions (real
        calendar sessions, not raw bar indices) — minute → daily aggreators.
    ``EXPANDING``               — growing-window statistic (no fixed N).
    ``RECURSIVE_STATE``         — recursive / stateful computation (EWM,
        Wilder, Kalman) where the "window" is an effective-memory parameter,
        not a bar count.
    """

    FULL_WINDOW = "full_window"
    MIN_SUPPORT_WINDOW = "min_support_window"
    CONTIGUOUS_FULL_WINDOW = "contiguous_full_window"
    EVENT_COUNT_WINDOW = "event_count_window"
    PERIOD_COUNT_WINDOW = "period_count_window"
    SESSION_WINDOW = "session_window"
    EXPANDING = "expanding"
    RECURSIVE_STATE = "recursive_state"
    BAR_WINDOW = "bar_window"



# Canonical -> declared WindowSemantics.  ``None`` = undeclared.
_WINDOW_SEMANTICS: dict[str, WindowSemantics] = {}


def declare_window_semantics(
    canonical: str,
    semantics: "WindowSemantics | str",
    *,
    replace: bool = False,
) -> None:
    """Declare the authoritative window-counting semantics for a canonical."""
    if not isinstance(semantics, WindowSemantics):
        try:
            semantics = WindowSemantics(str(semantics).lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown WindowSemantics {semantics!r}; choose one of "
                f"{[s.value for s in WindowSemantics]}"
            ) from exc
    canon = str(canonical)
    if canon in _WINDOW_SEMANTICS and not replace:
        raise ValueError(
            f"window_semantics already declared for {canon!r} as "
            f"{_WINDOW_SEMANTICS[canon].value}; pass replace=True to override"
        )
    _WINDOW_SEMANTICS[canon] = semantics


def window_semantics_for(canonical: str) -> Optional[WindowSemantics]:
    return _WINDOW_SEMANTICS.get(str(canonical))


def semantics_value(canonical: str) -> Optional[str]:
    s = _WINDOW_SEMANTICS.get(str(canonical))
    return s.value if s is not None else None


def clear_semantics() -> None:
    _WINDOW_SEMANTICS.clear()
