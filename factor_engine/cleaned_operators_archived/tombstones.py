# -*- coding: utf-8 -*-
"""R30 §2/§3: physical deletion tombstones for removed operator primitives.

R30 closes the "removed-but-special-category" ambiguity: names that were once
random / future-referencing / non-causal-fill / non-shape-preserving operator
primitives are NOT "special category operators" any more.  They are deleted from
the executable system.  The only remaining evidence of them is a frozen
:class:`OperatorTombstone` (no ``calculate``, no backend, no operator class, not
counted as a canonical, not mineable) that makes any attempt to call them raise
:class:`RemovedOperatorError` — never a silent execution and never a silent
substitution with ``Delay`` / ``ffill`` / 0.

Guarantees enforced by the registry:
* ``OperatorRegistry.get(<tombstoned>)`` raises ``RemovedOperatorError``.
* ``OperatorRegistry.resolve_canonical`` leaves tombstoned names unresolved so
  no DSL / miner / emitter / composite / cold-start path can lower them.
* The names are absent from every governance "operator" set (PIT_UNSAFE /
  PANDAS_ONLY / FORBIDDEN / DENIED / explicit policies / cost model).  The
  single authority is this module.

Hard gates (audit_r30):
  ACTIVE_RANDOM_FACTOR_PRIMITIVES == 0
  ACTIVE_FUTURE_REFERENCE_PRIMITIVES == 0
  ACTIVE_NONCAUSAL_FILL_PRIMITIVES == 0
"""
from __future__ import annotations

from dataclasses import dataclass

from runtime.exceptions import FactorEngineError


class RemovedOperatorError(FactorEngineError, LookupError):
    """Raised when a physically-removed operator name is requested/executed."""


@dataclass(frozen=True)
class OperatorTombstone:
    """Frozen record of a physically-removed operator primitive.

    A tombstone is NOT an operator: it carries no ``calculate`` / backend /
    operator class, is not counted as a canonical, cannot be mined, and must
    never be silently replaced by a look-alike operator.
    """

    name: str
    removed_since: str  # R30 round tag, e.g. "R30"
    risk_class: str     # "random" | "future_reference" | "noncausal_fill" | "non_shape_preserving" | "removed_duplicate"
    reason: str
    replacement: str | None = None  # migration hint only; never auto-applied


# risk_class taxonomy used by the R30 audit.
RISK_RANDOM = "random"
RISK_FUTURE = "future_reference"
RISK_NONCAUSAL_FILL = "noncausal_fill"
RISK_NON_SHAPE = "non_shape_preserving"
RISK_REMOVED_DUP = "removed_duplicate"


def _t(
    name: str,
    risk: str,
    reason: str,
    replacement: str | None = None,
) -> tuple[str, OperatorTombstone]:
    return name, OperatorTombstone(name, "R30", risk, reason, replacement)


# Random-number generation primitives (R30 §2): never usable as production
# factor operators.
RANDOM_TOMBSTONES: tuple[tuple[str, OperatorTombstone], ...] = (
    _t("rand_uniform", RISK_RANDOM, "non-deterministic factor primitive (R30 §2)"),
    _t("rand_normal", RISK_RANDOM, "non-deterministic factor primitive (R30 §2)"),
    _t("rand_lognormal", RISK_RANDOM, "non-deterministic factor primitive (R30 §2)"),
    _t("rand_poisson", RISK_RANDOM, "non-deterministic factor primitive (R30 §2)"),
    _t("rand_exp", RISK_RANDOM, "non-deterministic factor primitive (R30 §2)"),
    _t("shuffle", RISK_RANDOM, "row-permuting factor primitive (R30 §2)"),
    _t("sample", RISK_RANDOM, "random row-sampling factor primitive (R30 §2)"),
)

# Future-referencing / lead operators (R30 §2): read tomorrow's value.
FUTURE_TOMBSTONES: tuple[tuple[str, OperatorTombstone], ...] = (
    _t("Lead", RISK_FUTURE, "leads the series by one bar; reads future observations (R30 §2)"),
    _t("lead", RISK_FUTURE, "leads the series by one bar; reads future observations (R30 §2)"),
    _t("next", RISK_FUTURE, "returns the next row's value; reads future observations (R30 §2)"),
)

# Non-causal / future-endpoint fill primitives (R30 §2).
NONCAUSAL_FILL_TOMBSTONES: tuple[tuple[str, OperatorTombstone], ...] = (
    _t("bfill", RISK_NONCAUSAL_FILL, "backward fill reads future observations (R30 §2)"),
    _t("causal_bfill", RISK_NONCAUSAL_FILL, "backward fill reads future observations (R30 §2)"),
    _t("fillna_interpolate", RISK_NONCAUSAL_FILL, "interpolation may use future endpoints (R30 §2)"),
    _t("interpolate", RISK_NONCAUSAL_FILL, "interpolation may use future endpoints (R30 §2)"),
)

# Removed utility / non-shape-preserving primitives kept for migration error
# messages only (never executable, never a canonical).  NOTE: ``dropna`` is a
# real research-status operator (registered then dedupe-unregistered) — it is
# governed by NON_SHAPE_PRESERVING / PRODUCTION_DENIED instead of a tombstone,
# so ``resolve_canonical("dropna")`` during load does not raise.
OTHER_TOMBSTONES: tuple[tuple[str, OperatorTombstone], ...] = (
    _t("interpolate", RISK_NONCAUSAL_FILL, "interpolation may use future endpoints (R30 §2)"),
    _t("norm", RISK_REMOVED_DUP, "removed duplicate utility canonical", "normalize"),
    _t("norm_l1", RISK_REMOVED_DUP, "removed duplicate utility canonical", "normalize"),
    _t("norm_linf", RISK_REMOVED_DUP, "removed duplicate utility canonical", "normalize"),
)

TOMBSTONES: dict[str, OperatorTombstone] = dict(
    RANDOM_TOMBSTONES + FUTURE_TOMBSTONES + NONCAUSAL_FILL_TOMBSTONES + OTHER_TOMBSTONES
)


def is_tombstoned(name: str) -> bool:
    """Return whether ``name`` is a physically-removed operator name."""
    return name in TOMBSTONES


def tombstone_for(name: str) -> OperatorTombstone | None:
    return TOMBSTONES.get(name)


def assert_callable(name: str) -> None:
    """Raise :class:`RemovedOperatorError` when ``name`` is tombstoned."""
    tb = TOMBSTONES.get(name)
    if tb is not None:
        hint = f" (replacement hint: {tb.replacement})" if tb.replacement else ""
        raise RemovedOperatorError(
            f"operator {name!r} was removed from the executable system ({tb.risk_class}): "
            f"{tb.reason} (removed_since={tb.removed_since}){hint}. "
            "Tombstones are not callable operators and are never silently replaced."
        )


# Names that MUST never appear in any operator-governance "special category" set
# (they are not operators).  The R30 audit asserts these sets are disjoint from
# the governance sets that used to hold them.
ALL_TOMBSTONED_NAMES: frozenset[str] = frozenset(TOMBSTONES.keys())

RANDOM_TOMBSTONED_NAMES: frozenset[str] = frozenset(dict(RANDOM_TOMBSTONES).keys())
FUTURE_TOMBSTONED_NAMES: frozenset[str] = frozenset(dict(FUTURE_TOMBSTONES).keys())
NONCAUSAL_FILL_TOMBSTONED_NAMES: frozenset[str] = frozenset(dict(NONCAUSAL_FILL_TOMBSTONES).keys())
