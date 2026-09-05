"""FA factor-intelligence provider adapters (R61-FI-014 / plan §20 E6, matrix E6).

FO consumes FA-published factor intelligence through the
:class:`~factor_optimizer.ports.factor_intelligence.FactorIntelligenceProvider`
port: taxonomy / health card / diagnoses as narrow, frozen consumer views.
**FA is the canonical authority** for taxonomy and health — this module only
*projects* FA artifacts into FO views.  No grading rule, threshold or magic
confidence number is ever written here (enforced by
``tests.test_factor_intelligence_provider``'s source scan).

Rule for FA importability
-------------------------
FO's pyproject declares only ``factor-engine`` / ``quant-evaluator`` extras;
``factor_assets`` is deliberately optional (same relationship as
:mod:`factor_optimizer.adapters.quant_evaluator`).  The adapter therefore keeps
a **try-import + duck-typing** posture:

- When the real ``factor_assets.profiling`` package is importable, it is used.
- When it is not, :func:`create_fa_intelligence_provider` raises
  :class:`~factor_optimizer.errors.OptionalDependencyMissing` (fail-closed —
  no invented FA data).  The duck-typed conversion helpers
  (:func:`taxonomy_view_from_fa`, :func:`health_view_from_fa`,
  :func:`diagnosis_view_from_fa`) still accept FA-shaped objects so tests and
  the in-memory provider can exercise the projection against the *contract*
  without requiring the FA source tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from factor_optimizer.ports.factor_intelligence import (
    DiagnosisRepairability,
    DiagnosisSeverity,
    DiagnosisView,
    FactorHealthView,
    FactorIntelligenceProvider,
    FactorIntelligenceUnknownFactorError,
    FactorTaxonomyView,
    HealthGrade,
    HealthDimension,
    unknown_factor_view,
)

__all__ = [
    "FaFactorIntelligenceProvider",
    "InMemoryFactorIntelligenceProvider",
    "create_fa_intelligence_provider",
    "create_in_memory_factor_intelligence_provider",
    "taxonomy_view_from_fa",
    "health_view_from_fa",
    "diagnosis_view_from_fa",
]


# ---------------------------------------------------------------------------
# FA-shaped duck-typing helpers
# ---------------------------------------------------------------------------


def _tag_evidence_duck(tags: Any) -> tuple[tuple[str, str, float, tuple[str, ...]], ...]:
    """Normalize a sequence of FA ``TagEvidence`` (or duck-equivalents).

    Accepts objects exposing ``tag``/``source``/``confidence``/``evidence_refs``
    attributes (the ``factor_assets.profiling.policies.TagEvidence`` contract)
    as well as plain ``(tag, source, confidence, refs)`` tuples.  Confidence is
    float-coerced and range-checked so a crafted object cannot smuggle a bogus
    value into the view.
    """
    out: list[tuple[str, str, float, tuple[str, ...]]] = []
    for item in tags or ():
        if isinstance(item, tuple) and len(item) == 4:
            tag, source, confidence, refs = item
        else:
            tag = getattr(item, "tag", None)
            source = getattr(item, "source", None)
            confidence = getattr(item, "confidence", None)
            refs = getattr(item, "evidence_refs", None)
            if tag is None or source is None or confidence is None or refs is None:
                raise TypeError(
                    "mechanism tags must be (tag, source, confidence, refs) tuples "
                    "or objects exposing those attributes"
                )
        tag = str(tag)
        source = str(source)
        conf = float(confidence)
        if conf != conf or conf in (float("inf"), float("-inf")):
            raise ValueError("mechanism tag confidence must be finite")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("mechanism tag confidence must be in [0, 1]")
        out.append((tag, source, conf, tuple(str(r) for r in (refs or ()))))
    return tuple(out)


def taxonomy_view_from_fa(
    artifact: Any, *, factor_definition_id: Optional[str] = None
) -> FactorTaxonomyView:
    """Project an FA ``FactorTaxonomyArtifact`` (or duck-shape) into the view.

    Reads only the public attribute names of the FA contract:
    ``factor_definition_id``, ``data_domains``, ``display_family`` and
    ``mechanism_tags`` (each ``TagEvidence``).  FA structure/frequency tags,
    usage refs and the artifact content-hash stay FA-side — this is a narrow
    projection, never a re-classification.
    """
    fid = factor_definition_id or getattr(artifact, "factor_definition_id", None)
    if not fid:
        raise ValueError(
            "FA taxonomy artifact must expose factor_definition_id "
            "(or pass factor_definition_id=)"
        )
    data_domains = tuple(str(d) for d in (getattr(artifact, "data_domains", None) or ()))
    display_family = str(getattr(artifact, "display_family", "") or "UNKNOWN")
    mechanism_tags = _tag_evidence_duck(getattr(artifact, "mechanism_tags", ()))
    return FactorTaxonomyView(
        factor_definition_id=str(fid),
        data_domains=data_domains,
        display_family=display_family,
        mechanism_tags=mechanism_tags,
        is_unknown=False,
    )


def health_view_from_fa(
    artifact: Any, *, factor_definition_id: Optional[str] = None, evaluation_ref: str = ""
) -> FactorHealthView:
    """Project an FA health-card artifact (or duck-shape) into the view.

    Accepts either an object exposing ``dimension_grades`` (a mapping of
    dimension name -> grade string) and ``overall_grade``, or an object whose
    ``dimension_grade(dimension)`` / ``overall_grade`` callables return grades.
    Grade strings are validated against ``HealthGrade.all()``; dimension names
    are validated against ``HealthDimension.all()``.  FA grades pass through
    verbatim — FO never maps or thresholds them.
    """
    fid = factor_definition_id or getattr(artifact, "factor_definition_id", None)
    if not fid:
        raise ValueError(
            "FA health artifact must expose factor_definition_id "
            "(or pass factor_definition_id=)"
        )

    dimension_grades: dict[str, str] = {}
    mapping = getattr(artifact, "dimension_grades", None)
    if isinstance(mapping, Mapping):
        for name, grade in mapping.items():
            name = str(name)
            grade = str(grade)
            if name not in HealthDimension.all():
                raise ValueError(f"unknown health dimension: {name!r}")
            if grade not in HealthGrade.all():
                raise ValueError(f"unknown health grade: {grade!r}")
            dimension_grades[name] = grade
    else:
        for name in HealthDimension.all():
            raw = getattr(artifact, name, None)
            if raw is None:
                try:
                    raw = artifact.dimension_grade(name)
                except (AttributeError, TypeError):
                    raw = None
            if raw is not None:
                grade = str(raw)
                if grade not in HealthGrade.all():
                    raise ValueError(f"unknown health grade for {name}: {grade!r}")
                dimension_grades[name] = grade

    overall = getattr(artifact, "overall_grade", None)
    if callable(overall):
        overall = overall()
    overall_str = str(overall or HealthGrade.NONE)
    if overall_str not in HealthGrade.all():
        raise ValueError(f"unknown overall grade: {overall_str!r}")
    return FactorHealthView(
        factor_definition_id=str(fid),
        evaluation_ref=evaluation_ref,
        dimension_grades=dimension_grades,
        overall_grade=overall_str,
        is_unknown=False,
    )


def diagnosis_view_from_fa(
    diagnosis: Any, *, factor_definition_id: Optional[str] = None, health_ref: str = ""
) -> DiagnosisView:
    """Project one FA diagnosis artifact (or duck-shape) into the view.

    Read-only projection of ``tag``/``severity``/``confidence``/
    ``repairability``/``details``.  Severity and repairability strings are
    validated against the consumer vocab so an unknown FA value fails loudly
    rather than flowing into FO policy silently.
    """
    fid = factor_definition_id or getattr(diagnosis, "factor_definition_id", None)
    if not fid:
        raise ValueError(
            "FA diagnosis artifact must expose factor_definition_id "
            "(or pass factor_definition_id=)"
        )
    tag = str(getattr(diagnosis, "tag", "") or "")
    if not tag:
        raise ValueError("FA diagnosis artifact must expose a non-empty tag")
    severity = str(getattr(diagnosis, "severity", "") or DiagnosisSeverity.UNKNOWN)
    if severity not in DiagnosisSeverity.all():
        raise ValueError(f"unknown diagnosis severity: {severity!r}")
    repairability = str(
        getattr(diagnosis, "repairability", "") or DiagnosisRepairability.UNKNOWN
    )
    if repairability not in DiagnosisRepairability.all():
        raise ValueError(f"unknown diagnosis repairability: {repairability!r}")
    details_raw = getattr(diagnosis, "details", None)
    if details_raw is None:
        evidence = getattr(diagnosis, "evidence_refs", ()) or getattr(
            diagnosis, "repair_hints", ()
        )
        details_raw = evidence or ()
    details = tuple(str(d) for d in (details_raw or ()))
    return DiagnosisView(
        factor_definition_id=str(fid),
        health_ref=health_ref,
        tag=tag,
        severity=severity,
        confidence=float(getattr(diagnosis, "confidence", 0.0) or 0.0),
        repairability=repairability,
        details=details,
    )


# ---------------------------------------------------------------------------
# Concrete provider over the real FA profiling package
# ---------------------------------------------------------------------------


class FaFactorIntelligenceProvider:
    """``FactorIntelligenceProvider`` backed by the FA ``profiling`` package.

    The provider reads FA-published artifacts directly (best-effort import;
    duck-typed against the FA contract shapes) and converts them into narrow FO
    views.  Unknown ids raise
    :class:`~factor_optimizer.ports.factor_intelligence.FactorIntelligenceUnknownFactorError`
    — it is never ``None``.

    When FA is not importable the provider is still constructible (the
    per-call resolution happens lazily), but every call re-raises
    :class:`~factor_optimizer.errors.OptionalDependencyMissing`.
    """

    def __init__(
        self,
        *,
        get_fa_taxonomy=None,
        get_fa_health=None,
        get_fa_diagnoses=None,
        fa_profiling: Optional[Any] = None,
    ) -> None:
        self._get_fa_taxonomy = get_fa_taxonomy
        self._get_fa_health = get_fa_health
        self._get_fa_diagnoses = get_fa_diagnoses
        self._fa_profiling = fa_profiling
        self._fa_unavailable = None
        if fa_profiling is None:
            try:
                from factor_assets.profiling import taxonomy as fa_taxonomy  # optional
                self._fa_taxonomy_module = fa_taxonomy
            except ImportError as exc:
                self._fa_taxonomy_module = None
                self._fa_unavailable = exc

    def _require_fa(self) -> None:
        if self._fa_taxonomy_module is None:
            from factor_optimizer.errors import OptionalDependencyMissing

            raise OptionalDependencyMissing(
                "factor-assets", "FactorIntelligenceProvider (FA profiling)"
            ) from self._fa_unavailable

    @classmethod
    def _normalize_fa_artifact(cls, artifact: Any) -> Any:
        """Unwrap the FA ``classify_factor_taxonomy`` ``(artifact,)`` result."""
        if isinstance(artifact, tuple) and artifact and hasattr(artifact[0], "data_domains"):
            return artifact[0]
        return artifact

    # -- FactorIntelligenceProvider --------------------------------------

    def get_taxonomy(self, factor_definition_id: str) -> FactorTaxonomyView:
        if self._get_fa_taxonomy is not None:
            artifact = self._get_fa_taxonomy(factor_definition_id)
            if artifact is None:
                raise FactorIntelligenceUnknownFactorError(factor_definition_id)
            return taxonomy_view_from_fa(
                self._normalize_fa_artifact(artifact),
                factor_definition_id=factor_definition_id,
            )
        self._require_fa()
        from factor_assets.profiling.policies import get_taxonomy_policy  # optional

        artifact = self._fa_taxonomy_module.classify_factor_taxonomy(
            self._factor_placeholder(factor_definition_id),
            policy=get_taxonomy_policy(),
        )
        return taxonomy_view_from_fa(
            artifact, factor_definition_id=factor_definition_id
        )

    def get_health_card(
        self, factor_definition_id: str, evaluation_ref: str
    ) -> FactorHealthView:
        if self._get_fa_health is not None:
            artifact = self._get_fa_health(factor_definition_id, evaluation_ref)
            if artifact is None:
                raise FactorIntelligenceUnknownFactorError(factor_definition_id)
            return health_view_from_fa(
                artifact,
                factor_definition_id=factor_definition_id,
                evaluation_ref=evaluation_ref,
            )
        self._require_fa()
        raise FactorIntelligenceUnknownFactorError(
            factor_definition_id
        )  # FA health engine not yet wired: explicit, never None

    def get_diagnoses(
        self, factor_definition_id: str, health_ref: str
    ) -> Sequence[DiagnosisView]:
        if self._get_fa_diagnoses is not None:
            artifacts = self._get_fa_diagnoses(factor_definition_id, health_ref)
            if artifacts is None:
                raise FactorIntelligenceUnknownFactorError(factor_definition_id)
            return tuple(
                diagnosis_view_from_fa(
                    item,
                    factor_definition_id=factor_definition_id,
                    health_ref=health_ref,
                )
                for item in (artifacts or ())
            )
        self._require_fa()
        raise FactorIntelligenceUnknownFactorError(
            factor_definition_id
        )  # FA diagnosis engine not yet wired: explicit, never None

    @staticmethod
    def _factor_placeholder(factor_definition_id: str) -> Any:
        """Minimal FE static-analysis-shaped placeholder (unused hooks only)."""

        class _Analysis:
            factor_definition_id = factor_definition_id
            canonical_dsl_hash = factor_definition_id
            operator_usages = ()
            field_usages = ()
            existing_treatment_semantic_ids = ()

        return _Analysis()


def create_fa_intelligence_provider(
    *, get_fa_taxonomy=None, get_fa_health=None, get_fa_diagnoses=None, fa_profiling=None
) -> FaFactorIntelligenceProvider:
    """Create a FA-backed :class:`FaFactorIntelligenceProvider`.

    Best-effort against the real FA profiling package; caller-supplied
    ``get_fa_*`` hooks (for test doubles / staged FA wiring) override the FA
    imports.  When neither is available the returned provider still satisfies
    the port contract and fail-closed on access.
    """
    return FaFactorIntelligenceProvider(
        get_fa_taxonomy=get_fa_taxonomy,
        get_fa_health=get_fa_health,
        get_fa_diagnoses=get_fa_diagnoses,
        fa_profiling=fa_profiling,
    )


# ---------------------------------------------------------------------------
# In-memory provider (research / tests / downstream wiring)
# ---------------------------------------------------------------------------


class InMemoryFactorIntelligenceProvider:
    """Process-local ``FactorIntelligenceProvider`` for tests and wiring.

    Stores explicit views per ``factor_definition_id``.  Unknown ids raise
    :class:`FactorIntelligenceUnknownFactorError` (fail loud); index builders
    (:meth:`with_taxonomy`, :meth:`with_health_card`, :meth:`with_diagnosis`)
    accept either typed view objects or raw FA-shaped artifacts and run them
    through the same duck-typed projection used by the FA provider.
    """

    def __init__(self) -> None:
        self._taxonomies: dict[str, FactorTaxonomyView] = {}
        self._health: dict[str, dict[str, FactorHealthView]] = {}
        self._diagnoses: dict[str, tuple[DiagnosisView, ...]] = {}

    def with_taxonomy(
        self, factor_definition_id: str, value: Any
    ) -> "InMemoryFactorIntelligenceProvider":
        self._taxonomies[factor_definition_id] = (
            value
            if isinstance(value, FactorTaxonomyView)
            else taxonomy_view_from_fa(value, factor_definition_id=factor_definition_id)
        )
        return self

    def with_health_card(
        self,
        factor_definition_id: str,
        value: Any,
        evaluation_ref: str = "ev-in-memory",
    ) -> "InMemoryFactorIntelligenceProvider":
        view = (
            value
            if isinstance(value, FactorHealthView)
            else health_view_from_fa(value, factor_definition_id=factor_definition_id)
        )
        self._health.setdefault(factor_definition_id, {})[view.evaluation_ref or evaluation_ref] = view
        return self

    def with_diagnosis(
        self,
        factor_definition_id: str,
        value: Any,
        health_ref: str = "hc-in-memory",
    ) -> "InMemoryFactorIntelligenceProvider":
        view = (
            value
            if isinstance(value, DiagnosisView)
            else diagnosis_view_from_fa(value, factor_definition_id=factor_definition_id)
        )
        current = self._diagnoses.setdefault(factor_definition_id, ())
        self._diagnoses[factor_definition_id] = current + (view,)
        return self

    # -- FactorIntelligenceProvider --------------------------------------

    def get_taxonomy(self, factor_definition_id: str) -> FactorTaxonomyView:
        view = self._taxonomies.get(factor_definition_id)
        if view is None:
            raise FactorIntelligenceUnknownFactorError(factor_definition_id)
        return view

    def get_health_card(
        self, factor_definition_id: str, evaluation_ref: str
    ) -> FactorHealthView:
        by_ref = self._health.get(factor_definition_id)
        if by_ref is None:
            raise FactorIntelligenceUnknownFactorError(factor_definition_id)
        view = by_ref.get(evaluation_ref)
        if view is None:
            raise FactorIntelligenceUnknownFactorError(
                f"{factor_definition_id}@{evaluation_ref}"
            )
        return view

    def get_diagnoses(
        self, factor_definition_id: str, health_ref: str
    ) -> Sequence[DiagnosisView]:
        if factor_definition_id not in self._taxonomies and factor_definition_id not in self._health:
            raise FactorIntelligenceUnknownFactorError(factor_definition_id)
        return self._diagnoses.get(factor_definition_id, ())


def create_in_memory_factor_intelligence_provider() -> InMemoryFactorIntelligenceProvider:
    """Create a research/test-only in-memory provider."""
    return InMemoryFactorIntelligenceProvider()