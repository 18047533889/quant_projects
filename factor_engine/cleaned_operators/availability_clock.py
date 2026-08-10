# -*- coding: utf-8 -*-
"""R30 §27/§28/§29: Availability / Decision / Execution clock.

Every production factor input distinguishes:
  event_time          — when the market event happened;
  knowledge_time      — when that event became knowable;
  factor_available_time — when the factor value is fully computable;
  decision_time       — when a decision can legally be made on it;
  execution_time      — the earliest legal fill.

Invariant: ``knowledge_time <= factor_available_time <= decision_time
<= execution_time`` (equalities depend on market / order rules).

R30 §28: daily close/high/low/volume-derived factors default to
``available_at=session_close`` with ``same_bar_usable=False`` — a backtest
must NOT assume the same close can be both the last input and the fill price.
"""
from __future__ import annotations

# Panel input names whose values are only fully known at the session close.
_SESSION_END_INPUTS = frozenset({"close", "high", "low", "volume", "amount",
                                 "vwap", "open", "ret", "returns", "benchmark",
                                 "benchmark_ret", "market_ret"})

# Panel inputs that can be known DURING the session (open auction / mid-day).
_INTRADAY_KNOWN_INPUTS = frozenset({"open_price", "preopen", "auction"})


def default_available_at(input_names: tuple[str, ...], *, grain: str = "daily") -> str | None:
    """Infer a conservative ``available_at`` for a production operator.

    Daily-grain operators that consume session-end bars default to
    ``session_close``; everything else stays ``None`` (declared by the operator).
    """
    if grain == "daily":
        if any(n in _SESSION_END_INPUTS for n in input_names):
            return "session_close"
    return None


def default_same_session_usable(
    input_names: tuple[str, ...], *, declared: bool | None = None, grain: str = "daily"
) -> bool | None:
    """Infer ``same_session_usable``.

    A daily close/high/low/volume-derived factor is NOT usable in the same
    session it is computed from (the close is only known at session end); the
    earliest legal execution is the next session open.
    """
    if declared is not None:
        return declared
    if grain == "daily" and any(n in _SESSION_END_INPUTS for n in input_names):
        return False
    return None


def availability_clock_ok(
    *, available_at: str | None, same_session_usable: bool | None, input_names: tuple[str, ...]
) -> tuple[bool, str]:
    """R30 §28 hard gate: a session-close factor must not claim same-bar usability.

    Returns ``(ok, reason)``.  A factor whose ``available_at`` is session-close
    while ``same_session_usable=True`` is a same-close execution lookahead — the
    close is only knowable at session end, so it cannot also be the fill.

    An undeclared daily session-end factor is NOT a blocker: the default policy
    is conservative (``same_session_usable`` resolves to ``False`` for session-end
    inputs), so a backtest that does not opt into same-close execution is safe.
    Such factors are reported as ``needs_declaration`` so a future round can pin
    them explicitly, but they never admit same-close fills.
    """
    if available_at == "session_close" and same_session_usable is True:
        return False, (
            "available_at=session_close with same_session_usable=True is a "
            "same-close execution lookahead (R30 §28)"
        )
    return True, ""
