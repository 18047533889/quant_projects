# -*- coding: utf-8 -*-
"""Central MissingPolicy vocabulary (Master Spec Part C-10).

Every operator that consumes a panel with gaps must resolve to one of these
machine-readable policies instead of each file re-interpreting NaN in its own
way.  A declared policy is what lets the audit engine, the execution layer and
the AlphaProbe grammar agree on what a gap MEANS for a given operator without
guessing from docstrings or kernel code.

Policies are declared per canonical via :func:`declare_missing_policy`; the
audit engine reads :func:`missing_policy_for`.  ``None`` means "undeclared" —
the operator inherits the *derived* policy from the closure audit (derivation
is heuristic and must never be treated as authoritative).
"""
from __future__ import annotations

import enum
from typing import Optional


class MissingPolicy(str, enum.Enum):
    """Semantic of a missing / non-finite observation at ``t`` for this operator.

    ``CURRENT_REQUIRED``  — the output at ``t`` describes the CURRENT state; a
        non-finite input at ``t`` forces ``output[t]=NaN`` (never re-served a
        stale previous result).  Applies to current drawdown state, current
        quantile position, current structural level distance, current pattern /
        anomaly / motif / KNN-peer state, Hankel/Wavelet/Path-Signature/
        Persistence/Multifractal trailing-embedding outputs.
    ``BREAK``             — a missing observation breaks the computation: the
        state / interval is reset at the gap (e.g. event-interval distance
        across an unknown mark, drawdown duration across a suspension).
    ``PAIRWISE_VALID``    — only valid (finite) pairs are used; invalid rows are
        dropped per pair (lagged / alignment estimators).  Never compacts the
        TIME axis silently — see ``_fe_same_axis`` + physical-time embeddings.
    ``WINDOW_VALID``      — the trailing window uses only finite values, but the
        window is otherwise unchanged (rolling mean/std on valid subset).
    ``CENSOR``            — a missing mark censors the observation (the outcome
        is unknown, not zero/neutral); used by event / marked-event operators.
    ``CARRY_STATE``       — a missing input carries the previous state forward
        (legitimate for state machines over price/level series, with the caveat
        that a long silent run still produces a stale factor).
    ``CARRY_STATE_WITH_MAX_GAP`` — as ``CARRY_STATE`` but only within a declared
        max gap; beyond it the state is invalid/re-initialised.
    ``EVENT_UNKNOWN``     — a missing value means "event status unknown"
        (EventBool semantics: neither confirmed nor denied).
    ``SOURCE_UNKNOWN``    — a missing value means "the source did not publish";
        distinct from an economically-meaningful zero.
    """

    CURRENT_REQUIRED = "current_required"
    BREAK = "break"
    PAIRWISE_VALID = "pairwise_valid"
    WINDOW_VALID = "window_valid"
    CENSOR = "censor"
    CARRY_STATE = "carry_state"
    CARRY_STATE_WITH_MAX_GAP = "carry_state_with_max_gap"
    EVENT_UNKNOWN = "event_unknown"
    SOURCE_UNKNOWN = "source_unknown"


# Canonical -> declared MissingPolicy.  ``None`` = undeclared (heuristic
# derivation only, never treated as an authoritative contract).
_MISSING_POLICY: dict[str, MissingPolicy] = {}

# Expression semantic families that a CurrentRequired operator belongs to
# (Part C-11 machine field).  Used by the audit to check "is this a current-
# state factor that must not re-serve stale history".
_CURRENT_REQUIRED_FAMILIES = frozenset(
    {
        "current_drawdown", "current_quantile", "current_structural_level",
        "current_state", "current_pattern", "current_anomaly", "current_motif",
        "current_knn_peer",
    }
)


def declare_missing_policy(
    canonical: str,
    policy: "MissingPolicy | str",
    *,
    replace: bool = False,
) -> None:
    """Declare the authoritative missing-value policy for a canonical.

    The canonical must be registered (declaration happens after the operator is
    registered) unless ``replace=True``.  Passing a raw string is accepted and
    normalised to the enum.  Re-declaring without ``replace=True`` is rejected
    so a silent override cannot drift the contract.
    """
    from cleaned_operators.registry import OperatorRegistry

    if not isinstance(policy, MissingPolicy):
        try:
            policy = MissingPolicy(str(policy).lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown MissingPolicy {policy!r}; choose one of "
                f"{[p.value for p in MissingPolicy]}"
            ) from exc
    canon = str(canonical)
    if canon in _MISSING_POLICY and not replace:
        raise ValueError(
            f"missing_policy already declared for {canon!r} as "
            f"{_MISSING_POLICY[canon].value}; pass replace=True to override"
        )
    if not replace:
        try:
            if OperatorRegistry.resolve_canonical(canon) is None:
                raise KeyError(canon)
        except Exception:
            # The audit can run before a full load_all; never fail a declaration
            # because the registry is empty at declaration time.
            pass
    _MISSING_POLICY[canon] = policy


def missing_policy_for(canonical: str) -> Optional[MissingPolicy]:
    """Return the declared policy for a canonical, or ``None`` if undeclared."""
    return _MISSING_POLICY.get(str(canonical))


def policy_value(canonical: str) -> Optional[str]:
    """Return the declared policy value string (or ``None``)."""
    p = _MISSING_POLICY.get(str(canonical))
    return p.value if p is not None else None


def clear_policies() -> None:
    """Reset the policy registry (test isolation only)."""
    _MISSING_POLICY.clear()


# ---------------------------------------------------------------------------
# CurrentRequired machine field (Part C-11)
# ---------------------------------------------------------------------------

def declare_current_required_family(canonical: str, family: str) -> None:
    """Declare that ``canonical`` is a current-state factor belonging to a
    ``_CURRENT_REQUIRED_FAMILIES`` family — the closure audit then requires it
    to declare ``CURRENT_REQUIRED`` and to fail closed on a current NaN."""
    if family not in _CURRENT_REQUIRED_FAMILIES:
        raise ValueError(
            f"unknown current-required family {family!r}; choose one of "
            f"{sorted(_CURRENT_REQUIRED_FAMILIES)}"
        )
    _CURRENT_REQUIRED_FAMILY[canonical] = family


_CURRENT_REQUIRED_FAMILY: dict[str, str] = {}


def current_required_family_for(canonical: str) -> Optional[str]:
    return _CURRENT_REQUIRED_FAMILY.get(str(canonical))
