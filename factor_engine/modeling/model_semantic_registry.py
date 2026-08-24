# -*- coding: utf-8 -*-
"""Model Semantic Registry (P0 item 52) — single semantic authority.

Audit finding: too many governance authorities (OperatorMetadata /
OperatorSurface / model_lane / model_timing / model_contract / legacy /
ModelRegistry / SampleAdequacy / ParameterSearchPolicy /
StatefulCheckpointRegistry) already produced machine conflicts:

* legacy says ``research_only`` but ``model_lane`` said ``MODEL_FEATURE_SCORE``
  for the five panel operators (now fixed — they resolve to
  ``LEGACY_LOCAL_PREDICTIVE``); and
* ``model_contract`` said Kalman is ``stateful``/``checkpoint_supported`` while
  ``StatefulCheckpointRegistry`` had no Kalman entry — closed 2026-08-11 by the
  honest degradation (Kalman ``checkpoint_supported=False``, execution model is
  full-history replay), so :data:`KNOWN_NOT_CLOSED_CANONICALS` is now empty.

This module is the *aggregation + consistency-check layer*: it reads the
existing per-canonical authorities, folds them into a single
:class:`ModelSemanticEntry`, and reports any two-authority conflict via
:meth:`ModelSemanticRegistry.consistency_check` /
:meth:`ModelSemanticRegistry.consistency_errors`.

Design constraints
------------------
* **Additive only** — it never edits ``MODEL_LANE_EXPLICIT``,
  ``MODEL_TIMING_CONTRACTS``, ``MODEL_OPERATOR_CONTRACTS``, ``legacy`` or the
  stateful checkpoint registry.  ``register()`` layers a per-field override on
  top (winning per-field) without tearing any authority down.
* **Machine-checkable** — ``consistency_errors()`` over the full model-like
  inventory must be empty modulo the documented ``KNOWN_NOT_CLOSED_CANONICALS``
  set; ``unexpected_consistency_errors()`` filters that set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from factor_engine.cleaned_operators.model_contract import get_model_operator_contract
from factor_engine.cleaned_operators.model_lane import MODEL_LANES, assign_model_lane
from factor_engine.cleaned_operators.model_timing import (
    TimingKind,
    get_model_timing_contract,
    timing_kind_for,
)
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from modeling.legacy import classification_of, classify_execution_class

__all__ = [
    "ModelSemanticEntry",
    "ModelSemanticRegistry",
    "MODEL_SEMANTIC_REGISTRY",
    "PRODUCTION_LANES",
    "KNOWN_NOT_CLOSED_CANONICALS",
]

#: Production-admissible lanes (mirrors the model_timing production gate).
PRODUCTION_LANES: tuple[str, ...] = (
    "FAST_NATIVE_ALPHA",
    "EXPENSIVE_CERTIFIED_ALPHA",
    "MODEL_FEATURE_SCORE",
    "STATE_CONDITION_EVENT",
)

#: Documented, still-open governance gaps.  Empty as of 2026-08-11: the one
#: previously-open gap (Kalman ``checkpoint_supported`` with no
#: StatefulCheckpointRegistry entry) was closed by the honest degradation —
#: ``cleaned_operators/model_contract.py`` now sets ``checkpoint_supported=False``
#: for all six Kalman canonicals (stateful=True kept; execution model is
#: full-history replay).  The full-inventory gate asserts that NO error falls
#: outside this documented set.
KNOWN_NOT_CLOSED_CANONICALS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ModelSemanticEntry:
    """Single aggregated semantic record for a model-like canonical.

    Every field is drawn from one authority and cross-checked against the
    others.  ``timing`` is a :class:`cleaned_operators.model_timing.TimingKind`
    (or a per-field override); the rest are strings / bools.
    """

    canonical: str
    execution_class: str | None = None          # legacy.classify_execution_class
    semantic_role: str | None = None            # model_feature / model_score / diagnostic
    timing: TimingKind | None = None            # cleaned_operators TimingKind
    lane: str | None = None                     # model_lane.assign_model_lane
    searchability: bool | None = None           # legacy.default_searchable (None = unknown)
    sample_contract_family: str | None = None   # timing contract model_kind
    stateful: bool | None = None                # model_contract.stateful
    checkpoint_supported: bool | None = None    # model_contract.checkpoint_supported
    typed_inputs: str | None = None             # feature_params / input_units keys
    unit: str | None = None                     # input_units mapping, k=v;k=v
    production_certification: str | None = None # derived from lane + surface


def _heuristic_role(
    canonical: str,
    mc: Any,
    lane: str | None,
    kind: TimingKind | None,
) -> str | None:
    """Semantic role from the strongest authority, then a lane/timing heuristic."""
    if mc is not None:
        return mc.role
    if lane in (
        "DIAGNOSTIC_RESEARCH",
        "DIAGNOSTIC_DESCRIPTIVE",
        "EXPENSIVE_RESEARCH_CERTIFIED",
        "LEGACY_LOCAL_PREDICTIVE",
        "DELETE_TOMBSTONE",
    ):
        return "diagnostic"
    if lane == "MODEL_FEATURE_SCORE":
        return "model_score"
    if lane == "STATE_CONDITION_EVENT":
        return "model_feature"
    if kind == TimingKind.PRIOR_FIT_PREDICTIVE:
        return "model_score"
    return "model_feature"


def _typed_inputs(canonical: str, mc: Any) -> str | None:
    """Typed input list: model-contract feature params first, else input_units."""
    if mc is not None and getattr(mc, "feature_params", None):
        return ",".join(mc.feature_params)
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        ops = OperatorRegistry._operators.get(canonical, {}) or {}
        if ops:
            meta = next(iter(ops.values())).metadata
            iu = getattr(meta, "input_units", None) or {}
            if iu:
                return ",".join(sorted(iu))
    except Exception:
        pass
    return None


def _unit(canonical: str) -> str | None:
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        ops = OperatorRegistry._operators.get(canonical, {}) or {}
        if ops:
            meta = next(iter(ops.values())).metadata
            iu = getattr(meta, "input_units", None) or {}
            if iu:
                return ";".join(f"{k}={v}" for k, v in sorted(iu.items()))
    except Exception:
        pass
    return None


def _production_certification(lane: str | None, surface: str) -> str:
    if lane in PRODUCTION_LANES:
        return "production_searchable"
    if lane == "EXPENSIVE_RESEARCH_CERTIFIED":
        return "certified_research_only"
    if lane == "LEGACY_LOCAL_PREDICTIVE":
        return "legacy_research_only"
    if lane == "DELETE_TOMBSTONE":
        return "tombstoned"
    if lane in ("DIAGNOSTIC_RESEARCH", "DIAGNOSTIC_DESCRIPTIVE"):
        return "diagnostic_research"
    return "unclassified"


class ModelSemanticRegistry:
    """Aggregate + consistency-check the per-canonical semantic authorities.

    Additive by design: ``register()`` layers per-field overrides (useful for
    what-if simulation and for recording a canonical whose authorities are known
    to disagree); nothing here mutates the underlying dicts.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # registration (additive override)
    # ------------------------------------------------------------------ #
    def register(self, canonical: str, **fields: Any) -> ModelSemanticEntry:
        """Layer per-field overrides for a canonical and return its entry.

        ``timing`` accepts a :class:`TimingKind` or its ``.value`` string.
        Unknown fields raise ``ValueError``.  This never edits the authority
        dicts — the override wins per-field in :meth:`semantic_entry`.
        """
        allowed = {f for f in ModelSemanticEntry.__dataclass_fields__} - {"canonical"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown semantic field(s): {sorted(bad)}")
        fields = dict(fields)
        if "timing" in fields and fields["timing"] is not None:
            if not isinstance(fields["timing"], TimingKind):
                fields["timing"] = TimingKind(fields["timing"])
        existing = dict(self._overrides.get(canonical, {}))
        existing.update(fields)
        self._overrides[canonical] = existing
        return self.semantic_entry(canonical)

    # ------------------------------------------------------------------ #
    # aggregation
    # ------------------------------------------------------------------ #
    def semantic_entry(self, canonical: str) -> ModelSemanticEntry:
        """Fold every authority for ``canonical`` into one entry."""
        ov = self._overrides.get(canonical, {})
        leg = classification_of(canonical)
        mc = get_model_operator_contract(canonical)
        kind = timing_kind_for(canonical)
        lane = assign_model_lane(canonical)
        surface = classify_canonical(canonical)
        tc = get_model_timing_contract(canonical)

        execution_class = ov.get("execution_class")
        if execution_class is None:
            execution_class = (
                leg.execution_class.value
                if leg is not None
                else classify_execution_class(canonical).value
            )

        semantic_role = ov.get("semantic_role")
        if semantic_role is None:
            semantic_role = _heuristic_role(canonical, mc, lane, kind)

        timing = ov.get("timing", kind)
        lane_v = ov.get("lane", lane)
        searchability = ov.get("searchability")
        if searchability is None and leg is not None:
            searchability = leg.default_searchable
        family = ov.get("sample_contract_family")
        if family is None and tc is not None:
            family = tc.model_kind
        stateful = ov.get("stateful")
        if stateful is None and mc is not None:
            stateful = mc.stateful
        checkpoint_supported = ov.get("checkpoint_supported")
        if checkpoint_supported is None and mc is not None:
            checkpoint_supported = mc.checkpoint_supported
        typed_inputs = ov.get("typed_inputs")
        if typed_inputs is None:
            typed_inputs = _typed_inputs(canonical, mc)
        unit = ov.get("unit")
        if unit is None:
            unit = _unit(canonical)
        production_certification = ov.get("production_certification")
        if production_certification is None:
            production_certification = _production_certification(lane_v, surface)

        return ModelSemanticEntry(
            canonical=canonical,
            execution_class=execution_class,
            semantic_role=semantic_role,
            timing=timing,
            lane=lane_v,
            searchability=searchability,
            sample_contract_family=family,
            stateful=stateful,
            checkpoint_supported=checkpoint_supported,
            typed_inputs=typed_inputs,
            unit=unit,
            production_certification=production_certification,
        )

    # ------------------------------------------------------------------ #
    # consistency
    # ------------------------------------------------------------------ #
    def consistency_check(self, canonical: str) -> list[str]:
        """Two-authority conflicts for a single canonical.

        Detects, at minimum:
        * legacy ``research_only`` vs a production lane (the legacy-five bug);
        * ``model_contract`` stateful/checkpoint_supported vs a missing
          ``StatefulCheckpointRegistry`` entry (the Kalman gap, NOT_CLOSED);
        * timing-vs-contract predictive/descriptive contradictions.
        """
        entry = self.semantic_entry(canonical)
        errors: list[str] = []

        if entry.lane not in MODEL_LANES:
            errors.append(f"{canonical}: lane={entry.lane!r} not in MODEL_LANES")

        leg = classification_of(canonical)
        if leg is not None and leg.research_only and entry.lane in PRODUCTION_LANES:
            errors.append(
                f"{canonical}: legacy research_only=True "
                f"(default_searchable={leg.default_searchable}) conflicts with "
                f"lane={entry.lane} (production lane)"
            )

        # checkpoint_supported=True without a StatefulCheckpointRegistry entry is
        # a REAL conflict (the contract claims checkpointability the runtime does
        # not provide).  stateful=True alone is NOT — a stateful causal filter may
        # honestly be non-checkpointable (Kalman: full-history replay).
        if entry.checkpoint_supported:
            try:
                from stateful_contract import StatefulCheckpointRegistry

                cp = StatefulCheckpointRegistry.get(canonical)
            except Exception:
                cp = None
            if cp is None:
                errors.append(
                    f"{canonical}: model_contract checkpoint_supported=True but "
                    "StatefulCheckpointRegistry has no entry"
                )

        if entry.timing is not None:
            tc = get_model_timing_contract(canonical)
            if tc is not None:
                if entry.timing in (
                    TimingKind.PRIOR_FIT_PREDICTIVE,
                    getattr(TimingKind, "FORWARD_LABEL_SUPERVISED", None),
                ) and tc.descriptive:
                    errors.append(
                        f"{canonical}: timing={entry.timing.value} (predictive) "
                        "conflicts with ModelTimingContract descriptive "
                        f"(fit_cutoff={tc.fit_cutoff_offset}, "
                        f"horizon={tc.forecast_horizon})"
                    )
                if entry.timing == TimingKind.SELF_FIT_DESCRIPTIVE and tc.predictive:
                    errors.append(
                        f"{canonical}: timing=SELF_FIT_DESCRIPTIVE conflicts with "
                        "ModelTimingContract predictive "
                        f"(fit_cutoff={tc.fit_cutoff_offset}, "
                        f"horizon={tc.forecast_horizon})"
                    )

        return errors

    def consistency_errors(self, canonicals: Iterable[str] | None = None) -> list[str]:
        """Consistency errors across ``canonicals`` (required, or empty)."""
        out: list[str] = []
        for c in sorted(canonicals or ()):
            out.extend(self.consistency_check(c))
        return out

    def unexpected_consistency_errors(
        self, canonicals: Iterable[str] | None = None
    ) -> list[str]:
        """Consistency errors outside the documented ``KNOWN_NOT_CLOSED_CANONICALS``."""
        return [
            e
            for e in self.consistency_errors(canonicals)
            if self._canonical_of_error(e) not in KNOWN_NOT_CLOSED_CANONICALS
        ]

    @staticmethod
    def _canonical_of_error(error: str) -> str:
        return error.split(":", 1)[0].strip()


#: Default module-level registry instance.
MODEL_SEMANTIC_REGISTRY = ModelSemanticRegistry()
