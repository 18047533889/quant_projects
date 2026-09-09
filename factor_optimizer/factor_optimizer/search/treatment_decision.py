"""Generalized 8-step treatment decision policy (R61-FI-032 / plan §21.2).

The historical :class:`TreatmentDecisionPolicy` scored candidates on a
hard-coded tuple of 8 scalar RAW metrics (``RAW_METRICS``).  R61-FI-032
GENERALIZES the policy so the same 8-step architecture consumes FA health
*dimensions* (``ports/factor_intelligence.FactorHealthView`` 14-dim grades,
admission floors, hard-gate policy ids) instead of only the 8 scalar metrics —
while keeping the existing scalar path fully backward compatible.

Pipeline (unchanged shape, generalized consumption):

    STEP 1  Integrity hard gates      (unchanged, fail-closed evidence gates)
    STEP 2  RAW-relative deltas       (now over health dimensions, higher =
                                       better-than-RAW after normalization)
    STEP 3  Desirability              (grade ordering -> unit desirability)
    STEP 4  Dimension scores          (6 balanced dims: predictive/stability/
                                       robustness/tradability/purity_exposure/
                                       data_quality)
    STEP 5  Pareto frontier
    STEP 6  Statistical uncertainty   (unchanged bootstrap machinery; a
                                       no-bootstrap candidate is POINT_ESTIMATE_ONLY)
    STEP 7  Robust utility
    STEP 8  Near-equivalence

The policy is **constructed with a FactorFitnessSpec** that names the policy
ids for each step plus the health dimensions whose floors are enforced and the
minimum evidence tier a winner must carry.  A ``decision()`` overload accepts
FA ``FactorHealthView`` maps per candidate; the legacy ``decide()`` keeps the
hard-coded scalar path (deprecated compatibility layer) so existing callers and
tests keep passing byte-for-byte.

Evidence tiers: a candidate whose health view has no computed values / no
uncertainty series carries POINT_ESTIMATE_ONLY; when the spec requires
BOOTSTRAP_CONFIDENCE the candidate is rejected before ranking (fail closed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from factor_optimizer.contracts.evidence_value import (
    EvidenceStatus,
    EvidenceTier,
)
from factor_optimizer.contracts.factor_fitness import (
    CandidateFitnessArtifact,
    FactorFitnessSpec,
)
from factor_optimizer.contracts.treatment_integrity import (
    RAW_TREATMENT_KIND,
    IntegrityCheckResult,
    TreatmentIntegrityEvidence,
    build_integrity_evidence,
    describe_integrity_problem,
    digest_value,
)
from factor_optimizer.errors import TreatmentIntegrityError
from factor_optimizer.ports.factor_intelligence import (
    DecisionProvider,
    FactorHealthView,
    HealthDimension,
    HealthGrade,
    SelectionDecisionRequestView,
    SelectionDecisionReceiptView,
    require_bound_decision_receipt,
)
from factor_optimizer.search.desirability import desirability_for
from factor_optimizer.search.pareto import ParetoFrontier, ParetoPoint
from factor_optimizer.search.uncertainty_winner import (
    UncertaintyAwareWinnerSelector,
    UncertaintyConfig,
    UncertaintyEvidence,
)
from factor_optimizer.search.winner_selector import WinnerPolicy

# Re-export the fitness adapter helpers.
from factor_optimizer.adapters.fitness import grade_to_desirability

# The raw metrics a treatment is scored on (STEP 2 deltas are relative to RAW).
# Retained verbatim for the backward-compatible scalar path (deprecated).
RAW_METRICS = (
    "rank_ic",
    "icir",
    "turnover",
    "cost_adjusted_alpha",
    "worst_slice",
    "exposure",
    "coverage",
    "stability",
)

# The six balanced dimensions (STEP 4) — kept as the Pareto objective axes.
BALANCED_DIMENSIONS = (
    "predictive",
    "stability",
    "robustness",
    "tradability",
    "purity_exposure",
    "data_quality",
)

#: Canonical FA health dimensions whose grades may be consumed as floors.
HEALTH_DIMENSION_IDS = tuple(HealthDimension.all())


@dataclass(frozen=True)
class IntegrityGate:
    """A single non-negotiable hard-reject gate (STEP 1)."""

    name: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("gate name must be a non-empty string")
        if not isinstance(self.passed, bool):
            raise TypeError("gate passed must be a boolean")


@dataclass
class TreatmentMetrics:
    """Raw metrics for a single treatment candidate (legacy scalar path).

    Retained byte-for-byte for backward compatibility (R61-FI-032 deprecates
    this as the *scoring* authority but keeps it as the accepted input of the
    legacy ``decide()``).  New callers should prefer :class:`HealthDecisionInput`
    / :class:`DecisionInput`.
    """

    trial_id: str
    rank_ic: float = 0.0
    icir: float = 0.0
    turnover: float = 0.0
    cost_adjusted_alpha: float = 0.0
    worst_slice: float = 0.0
    exposure: float = 0.0
    coverage: float = 0.0
    stability: float = 0.0
    robustness: float = 0.0
    complexity_score: float = 0.0
    n_transforms: int = 0
    compute_cost: float = 0.0
    bootstrap_samples: Dict[str, Sequence[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        for name in RAW_METRICS + ("robustness", "complexity_score", "compute_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if isinstance(self.n_transforms, bool) or not isinstance(
            self.n_transforms, int
        ):
            raise TypeError("n_transforms must be an integer")
        if self.n_transforms < 0:
            raise ValueError("n_transforms must be >= 0")
        if not 0.0 <= self.complexity_score <= 1.0:
            raise ValueError("complexity_score must be in [0, 1]")
        if self.compute_cost < 0:
            raise ValueError("compute_cost must be >= 0")
        for metric, samples in self.bootstrap_samples.items():
            if metric not in RAW_METRICS:
                raise ValueError(f"unknown bootstrap metric {metric!r}")
            if not samples:
                raise ValueError(f"bootstrap samples for {metric!r} are empty")
            for value in samples:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        f"bootstrap sample for {metric!r} must be numeric"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"bootstrap sample for {metric!r} must be finite"
                    )

    def metric(self, name: str) -> float:
        """Return the raw metric value by name."""
        if name not in RAW_METRICS:
            raise KeyError(f"unknown metric {name!r}")
        return float(getattr(self, name))


@dataclass(frozen=True)
class HealthDecisionInput:
    """One candidate's FA health view + decision-local numeric evidence.

    The generalized STEP-2/3/4 consume ``health_view`` grades (14-dim) plus a
    small set of decision-local scalars that FA health grades do not encode
    (robustness / complexity / n_transforms / compute_cost — used by the
    robust-utility and near-equivalence steps).  ``evidence_tier`` declares the
    uncertainty work behind the candidate; ``bootstrap_samples`` optionally
    carries per-dimension resampled desirabilities for STEP 6.
    """

    trial_id: str
    health_view: FactorHealthView
    robustness: float = 0.5
    complexity_score: float = 0.0
    n_transforms: int = 0
    compute_cost: float = 0.0
    evidence_tier: EvidenceTier = EvidenceTier.POINT_ESTIMATE_ONLY
    bootstrap_samples: Optional[Mapping[str, Sequence[float]]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        if not isinstance(self.health_view, FactorHealthView):
            raise TypeError("health_view must be a FactorHealthView")
        for name, value in (
            ("robustness", self.robustness),
            ("complexity_score", self.complexity_score),
            ("compute_cost", self.compute_cost),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.complexity_score <= 1.0:
            raise ValueError("complexity_score must be in [0, 1]")
        if not 0.0 <= self.robustness <= 1.0:
            raise ValueError("robustness must be in [0, 1]")
        if self.compute_cost < 0:
            raise ValueError("compute_cost must be >= 0")
        if isinstance(self.n_transforms, bool) or not isinstance(
            self.n_transforms, int
        ):
            raise TypeError("n_transforms must be an integer")
        if self.n_transforms < 0:
            raise ValueError("n_transforms must be >= 0")
        object.__setattr__(
            self,
            "evidence_tier",
            EvidenceTier.from_value(self.evidence_tier),
        )
        if self.bootstrap_samples is not None:
            cleaned = {}
            for name, samples in self.bootstrap_samples.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("bootstrap dimension name must be non-empty")
                vals = tuple(float(v) for v in samples)
                if not vals:
                    raise ValueError(
                        f"bootstrap dimension {name!r} has no samples"
                    )
                for v in vals:
                    if not math.isfinite(v):
                        raise ValueError("bootstrap sample must be finite")
                cleaned[name] = vals
            object.__setattr__(self, "bootstrap_samples", cleaned)


@dataclass(frozen=True)
class DesirabilityAnchors:
    """Per-metric desirability anchors for the scalar decision path (STEP 3)."""

    maps: Dict[str, Dict[str, Sequence[Tuple[float, float]]]]

    def __post_init__(self) -> None:
        if not isinstance(self.maps, dict) or not self.maps:
            raise ValueError("maps must be a non-empty dict")
        for metric, spec in self.maps.items():
            if spec["direction"] not in ("increasing", "decreasing"):
                raise ValueError(
                    f"unknown direction {spec['direction']!r} for {metric!r}"
                )
            if not spec["anchors"]:
                raise ValueError(f"empty anchors for {metric!r}")

    def score(self, metric: str, value: float) -> float:
        if metric not in self.maps:
            raise KeyError(f"no desirability anchors for metric {metric!r}")
        spec = self.maps[metric]
        return desirability_for(spec["direction"], spec["anchors"], value)


@dataclass
class DecisionResult:
    """The outcome of the (scalar or health-dimension) decision pipeline.

    Attributes:
        winner_trial_id: The winning treatment (may be RAW).
        step_trace: Per-step notes for auditability.
        pareto_trial_ids: Trial ids on the Pareto frontier (STEP 5).
        statistically_plausible: Trial ids surviving STEP 6.
        raw_kept: True when RAW was a candidate and survived to the end.
        outcome: The plan §22 outcome implied by the decision (added in
            R61-FI-033 — defaults to IMPROVED for legacy scalar decisions, and
            is set by the generalized path when a RAW winner is selected).
        fitness_artifacts: Per-candidate :class:`CandidateFitnessArtifact`
            records the generalized decision path produced (R61-FI-031).
    """

    winner_trial_id: str
    step_trace: Dict[str, str] = field(default_factory=dict)
    pareto_trial_ids: List[str] = field(default_factory=list)
    statistically_plausible: List[str] = field(default_factory=list)
    raw_kept: bool = False
    outcome: str = "IMPROVED"
    fitness_artifacts: Tuple[CandidateFitnessArtifact, ...] = ()
    authoritative: bool = False
    purpose: str = "SCREENING_DIAGNOSTIC"


# ---------------------------------------------------------------------------
# Scalar->balanced-dimension weights (legacy path only)
# ---------------------------------------------------------------------------

_SCALAR_DIMENSION_WEIGHTS = {
    "predictive": {"rank_ic": 0.6, "icir": 0.4},
    "stability": {"stability": 1.0},
    "robustness": {},
    "tradability": {"turnover": 0.5, "cost_adjusted_alpha": 0.5},
    "purity_exposure": {"exposure": 1.0},
    "data_quality": {"coverage": 1.0},
}

#: Health dimension -> balanced dimension membership (which of the six Pareto
#: objective axes a health grade feeds).  FA's 14 dimensions map onto the six
#: balanced axes by *semantic group*; grades are mapped through the ordered
#: grade alphabet (no numeric threshold invented on the FO side).
_HEALTH_TO_BALANCED = {
    HealthDimension.PREDICTIVE_POWER: "predictive",
    HealthDimension.STABILITY: "stability",
    HealthDimension.ROBUSTNESS: "robustness",
    HealthDimension.TURNOVER: "tradability",
    HealthDimension.CAPACITY: "tradability",
    HealthDimension.COST_DRAG: "tradability",
    HealthDimension.DRAWDOWN: "robustness",
    HealthDimension.TAIL_RISK: "robustness",
    HealthDimension.DATA_COVERAGE: "data_quality",
    HealthDimension.FRESHNESS: "data_quality",
    HealthDimension.COMPLEXITY: "complexity",
    HealthDimension.ECONOMIC_SENSE: "economic_sense",
    HealthDimension.SHAPE_QUALITY: "shape_quality",
    HealthDimension.REGIME_SENSITIVITY: "regime_sensitivity",
}


class TreatmentDecisionPolicy:
    """Treatment-selection facade with one formal and two diagnostic paths.

    Two construction modes:

    - Legacy scalar screening mode (``decide(..., screening_only=True)``):
      built with ``DesirabilityAnchors`` + ``WinnerPolicy`` and scores the
      hard-coded 8 RAW metrics.
    - Generalized health-dimension screening mode
      (``decision(..., screening_only=True)``): built with a
      :class:`FactorFitnessSpec` + :class:`WinnerPolicy`; consumes FA
      ``FactorHealthView`` 14-dim grades per candidate, applies dimension
      floors / evidence-tier gates declared by the spec, computes RAW-relative
      deltas over the health dimensions, and runs the same Pareto / uncertainty
      / robust-utility / near-equivalence steps.
    - Formal selection (``authoritative_decision(request)``): delegates the
      complete decision to the injected FA provider and verifies that its
      receipt is exactly bound to the request.
    """

    def __init__(
        self,
        anchors: Optional[DesirabilityAnchors] = None,
        policy: Optional[WinnerPolicy] = None,
        uncertainty_config: Optional[UncertaintyConfig] = None,
        *,
        fitness_spec: Optional[FactorFitnessSpec] = None,
        require_integrity_evidence: bool = True,
        decision_provider: Optional[DecisionProvider] = None,
    ):
        # Exactly one scoring authority: legacy anchors OR generalized spec.
        if (anchors is None) == (fitness_spec is None):
            raise ValueError(
                "exactly one of anchors (legacy scalar mode) or fitness_spec "
                "(generalized health-dimension mode) must be provided"
            )
        if policy is None:
            raise ValueError("a WinnerPolicy is required")
        if not isinstance(policy, WinnerPolicy):
            raise TypeError("policy must be a WinnerPolicy")
        if anchors is not None and not isinstance(anchors, DesirabilityAnchors):
            raise TypeError("anchors must be a DesirabilityAnchors")
        if fitness_spec is not None and not isinstance(
            fitness_spec, FactorFitnessSpec
        ):
            raise TypeError("fitness_spec must be a FactorFitnessSpec")
        if not isinstance(require_integrity_evidence, bool):
            raise TypeError("require_integrity_evidence must be a bool")
        self.anchors = anchors
        self.fitness_spec = fitness_spec
        self.policy = policy
        self.uncertainty_config = uncertainty_config or UncertaintyConfig()
        self.require_integrity_evidence = require_integrity_evidence
        if decision_provider is not None and not isinstance(
            decision_provider, DecisionProvider
        ):
            raise TypeError("decision_provider must implement DecisionProvider")
        self.decision_provider = decision_provider

    def authoritative_decision(
        self, request: SelectionDecisionRequestView
    ) -> SelectionDecisionReceiptView:
        """Delegate final selection to FA and verify exact request binding.

        FO deliberately performs no grade mapping, thresholding, Pareto
        pruning, or utility recomputation on this path.
        """
        if self.decision_provider is None:
            raise ValueError("an FA DecisionProvider is required for final selection")
        receipt = self.decision_provider.decide(request)
        return require_bound_decision_receipt(request, receipt)

    # -- mode helpers --------------------------------------------------------

    @property
    def _generalized(self) -> bool:
        return self.fitness_spec is not None

    # -- STEP 1: integrity hard gates (shared) -------------------------------

    def _integrity_gates(
        self,
        trial_id: str,
        coverage: float,
        rank_ic: float,
        icir: float,
        turnover: float,
        compute_cost: float,
        evidence: Optional[TreatmentIntegrityEvidence] = None,
    ) -> List[IntegrityGate]:
        """Real, fail-closed integrity gates (shared by both modes)."""
        gates: List[IntegrityGate] = [
            IntegrityGate(
                "metric_coverage_nonzero",
                coverage > 0.0,
                "metric sanity: coverage must be positive",
            ),
            IntegrityGate(
                "metric_coverage_floor",
                coverage >= 0.5,
                "metric sanity: coverage must be >= 0.5 (sufficient sample)",
            ),
            IntegrityGate(
                "metric_coverage_bounded",
                coverage <= 1.0,
                "metric sanity: coverage must be <= 1.0",
            ),
            IntegrityGate(
                "metric_numeric_finite",
                math.isfinite(rank_ic)
                and math.isfinite(icir)
                and math.isfinite(turnover),
                "metric sanity: core metrics must be finite",
            ),
            IntegrityGate(
                "metric_rank_ic_bounded",
                rank_ic > -1.0,
                "metric sanity: rank_ic must be > -1",
            ),
            IntegrityGate(
                "compute_cost_numeric_valid",
                compute_cost >= 0.0,
                "metric sanity: compute_cost must be >= 0",
            ),
        ]
        if not self.require_integrity_evidence:
            gates.append(
                IntegrityGate(
                    "integrity_evidence",
                    False,
                    "integrity evidence was not required; the metric screen "
                    "alone is NOT a production integrity verdict",
                )
            )
            return gates
        problem = describe_integrity_problem(
            trial_id, evidence, treatment_kind=None
        )
        gates.append(
            IntegrityGate(
                "integrity_evidence",
                problem is None,
                problem or (
                    f"a complete, passing TreatmentIntegrityEvidence for "
                    f"{trial_id!r} is required"
                ),
            )
        )
        return gates

    # -- scalar-mode STEP 2-4 helpers (legacy) -------------------------------

    def _raw_relative_deltas(
        self, raw: TreatmentMetrics, treatment: TreatmentMetrics
    ) -> Dict[str, float]:
        """Scalar-mode raw-relative deltas (higher-better normalized)."""
        higher_better = {
            "rank_ic",
            "icir",
            "cost_adjusted_alpha",
            "worst_slice",
            "coverage",
            "stability",
        }
        lower_better = {"turnover", "exposure"}
        deltas: Dict[str, float] = {}
        for metric in RAW_METRICS:
            raw_val = raw.metric(metric)
            treat_val = treatment.metric(metric)
            if metric in higher_better:
                deltas[metric] = treat_val - raw_val
            elif metric in lower_better:
                deltas[metric] = raw_val - treat_val
            else:
                raise ValueError(f"unknown metric {metric!r}")
        return deltas

    def _desirabilities(self, metrics: TreatmentMetrics) -> Dict[str, float]:
        return {
            metric: self.anchors.score(metric, metrics.metric(metric))
            for metric in RAW_METRICS
        }

    def _dimension_scores(
        self, metrics: TreatmentMetrics, desirabilities: Dict[str, float]
    ) -> Dict[str, float]:
        return {
            "predictive": 0.6 * desirabilities["rank_ic"]
            + 0.4 * desirabilities["icir"],
            "stability": desirabilities["stability"],
            "robustness": metrics.robustness,
            "tradability": 0.5 * desirabilities["turnover"]
            + 0.5 * desirabilities["cost_adjusted_alpha"],
            "purity_exposure": desirabilities["exposure"],
            "data_quality": desirabilities["coverage"],
        }

    # -- generalized-mode STEP 2-4 helpers -----------------------------------

    def _health_desirability_scores(
        self, candidate: HealthDecisionInput
    ) -> Dict[str, float]:
        """Map the candidate's 14 FA health grades to balanced-dimension scores.

        Every graded dimension contributes to its balanced axis via the ordered
        grade alphabet (grade -> unit desirability, monotone); a dimension with
        grade NONE (missing) contributes NOTHING to its axis (missing != 0 —
        the axis score is the mean of the graded members only, and an axis with
        no graded member falls back to the overall grade when graded).
        """
        scores: Dict[str, List[float]] = {name: [] for name in BALANCED_DIMENSIONS}
        for dimension, grade in candidate.health_view.dimension_grades.items():
            balanced = _HEALTH_TO_BALANCED.get(dimension)
            if balanced not in BALANCED_DIMENSIONS:
                continue  # complexity/economic/shape/regime are not Pareto axes
            des = grade_to_desirability(grade)
            if des is None:
                continue
            scores[balanced].append(des)
        out: Dict[str, float] = {}
        for name in BALANCED_DIMENSIONS:
            if scores[name]:
                out[name] = sum(scores[name]) / len(scores[name])
            else:
                # No graded member: fall back to the overall grade (mapped), or
                # 0.0 only when even the overall grade is missing (an ungraded
                # candidate cannot score above zero on any axis).
                overall = grade_to_desirability(candidate.health_view.overall_grade)
                out[name] = overall if overall is not None else 0.0
        return out

    def _health_floor_gates(
        self, candidate: HealthDecisionInput
    ) -> List[IntegrityGate]:
        """STEP-1 floor gates over the required health dimensions.

        ``factor_optimizer`` declares no numeric floor thresholds (FO has no
        grading authority).  The floors are encoded in the spec's policy ids;
        this method only fails closed when the health view carries NONE (missing)
        on a REQUIRED dimension — a candidate without the floor evidence cannot
        pass the floor gate.
        """
        required = self.fitness_spec.required_health_dimensions
        if not required:
            return []
        gates = []
        for dimension in required:
            grade = candidate.health_view.grade_of(dimension)
            present = grade not in (HealthGrade.NONE, "")
            gates.append(
                IntegrityGate(
                    f"health_dimension_floor::{dimension}",
                    present,
                    f"health dimension {dimension!r} must be graded (floor "
                    f"policy {self.fitness_spec.dimension_floor_policy_id!r}); "
                    "missing evidence is not a pass",
                )
            )
        return gates

    def _evidence_tier_gate(
        self, candidate: HealthDecisionInput
    ) -> IntegrityGate:
        """STEP-1 gate: the candidate's evidence tier meets the spec minimum."""
        minimum = self.fitness_spec.minimum_evidence_tier
        ok = candidate.evidence_tier.meets(minimum)
        return IntegrityGate(
            "minimum_evidence_tier",
            ok,
            (
                f"candidate tier {candidate.evidence_tier.value} meets the "
                f"spec minimum {minimum.value}"
                if ok
                else f"candidate tier {candidate.evidence_tier.value} is below "
                f"the spec minimum {minimum.value} — no-bootstrap candidates "
                "are POINT_ESTIMATE_ONLY and cannot win under an "
                "uncertainty-requiring policy"
            ),
        )

    # -- STEP 5: Pareto frontier ---------------------------------------------

    def _pareto_frontier(
        self,
        candidates,
        *,
        use_health: bool = False,
    ):
        """Remove truly dominated treatments (higher-better on all dims)."""
        frontier = ParetoFrontier()
        points = []
        for candidate in candidates:
            if use_health:
                dims = self._health_desirability_scores(candidate)
            else:
                dims = self._dimension_scores(
                    candidate, self._desirabilities(candidate)
                )
            objectives = tuple(dims[name] for name in BALANCED_DIMENSIONS)
            point = ParetoPoint(
                trial_id=candidate.trial_id,
                objectives=objectives,
                metadata={"mode": "health" if use_health else "scalar"},
            )
            points.append(point)
            frontier.add_point(point)
        survivors = {p.trial_id for p in frontier.frontier()}
        return [c for c in candidates if c.trial_id in survivors]

    # -- STEP 6: statistical uncertainty -------------------------------------

    def _statistically_plausible(
        self, candidates, *, use_health: bool = False
    ):
        """Keep candidates with uncertainty evidence (or none -> keep).

        Scalar path unchanged: candidates with no bootstrap series are kept
        (treated as point estimates).  Health path: candidates whose bootstrap
        series is present are kept only when non-degenerate; candidates with no
        bootstrap (POINT_ESTIMATE_ONLY) are kept ONLY under a spec whose
        minimum tier is POINT_ESTIMATE_ONLY (otherwise the tier gate at STEP 1
        already rejected them).
        """
        plausible = []
        for candidate in candidates:
            samples = (
                candidate.bootstrap_samples
                if use_health
                else (candidate.bootstrap_samples or None)
            )
            if not samples:
                # No bootstrap evidence.  Under the generalized health path a
                # no-bootstrap candidate is only ever kept when the spec
                # minimum tier is POINT_ESTIMATE_ONLY (a higher minimum was
                # already enforced as a STEP-1 hard gate, so a candidate that
                # declares BOOTSTRAP_CONFIDENCE without a real series cannot
                # reach STEP 6 with an empty series and survive).
                if not use_health:
                    plausible.append(candidate)
                elif (
                    self.fitness_spec.minimum_evidence_tier
                    is EvidenceTier.POINT_ESTIMATE_ONLY
                ):
                    plausible.append(candidate)
                else:
                    # Declared tier >= BOOTSTRAP_CONFIDENCE but no actual
                    # resampled series: the candidate cannot pass STEP 6.
                    continue
            else:
                for name, series in samples.items():
                    if series and all(math.isfinite(float(v)) for v in series):
                        ok = True
                        break
                if ok:
                    plausible.append(candidate)
        return plausible

    # -- STEP 7 + 8: robust utility + near-equivalence ----------------------

    def _rank_with_uncertainty(
        self,
        candidates,
        *,
        use_health: bool = False,
    ):
        """Rank by conservative utility applying near-equivalence."""
        evidence_list = []
        for candidate in candidates:
            if use_health:
                dims = self._health_desirability_scores(candidate)
                dimension_samples: Dict[str, Sequence[float]] = {}
                if candidate.bootstrap_samples:
                    for name in BALANCED_DIMENSIONS:
                        dimension_samples[name] = list(candidate.bootstrap_samples.get(name, ()))
                        if not dimension_samples[name]:
                            dimension_samples[name] = [dims[name]]
                else:
                    for name in BALANCED_DIMENSIONS:
                        dimension_samples[name] = [dims[name]]
                complexity_score = candidate.complexity_score
                turnover = 0.0
                compute_cost = candidate.compute_cost
                n_transforms = candidate.n_transforms
                trial_id = candidate.trial_id
            else:
                dims = self._dimension_scores(
                    candidate, self._desirabilities(candidate)
                )
                dimension_samples = {}
                for name in BALANCED_DIMENSIONS:
                    if candidate.bootstrap_samples:
                        dimension_samples[name] = self._aggregate_dimension_bootstrap(
                            candidate, name
                        )
                    else:
                        dimension_samples[name] = [dims[name]]
                complexity_score = candidate.complexity_score
                turnover = candidate.turnover
                compute_cost = candidate.compute_cost
                n_transforms = candidate.n_transforms
                trial_id = candidate.trial_id
            # If a dimension has no samples and the fallback [dims] was
            # applied, the sample list is a single point — the CI machinery
            # needs >= 2 to be meaningful, so use the point-only series.
            for name in BALANCED_DIMENSIONS:
                if not dimension_samples.get(name):
                    dimension_samples[name] = [dims[name]]
            evidence_list.append(
                UncertaintyEvidence(
                    trial_id=trial_id,
                    dimension_samples=dimension_samples,
                    complexity_score=complexity_score,
                    turnover=turnover,
                    compute_cost=compute_cost,
                    n_transforms=n_transforms,
                )
            )
        if use_health:
            robustness_scores = {
                c.trial_id: c.robustness for c in candidates
            }
        else:
            robustness_scores = {
                c.trial_id: c.robustness for c in candidates
            }
        selector = UncertaintyAwareWinnerSelector(
            self.policy, self.uncertainty_config
        )
        return next(
            c for c in candidates if c.trial_id == selector.select(
                evidence_list, robustness_scores
            ).trial_id
        )

    def _aggregate_dimension_bootstrap(
        self, metrics: TreatmentMetrics, dimension: str
    ) -> List[float]:
        """Legacy scalar-mode bootstrap aggregation (unchanged)."""
        weights = _SCALAR_DIMENSION_WEIGHTS
        if dimension == "robustness":
            return [metrics.robustness]
        wmap = weights[dimension]
        n = 1
        for metric in wmap:
            if metric in metrics.bootstrap_samples:
                n = max(n, len(metrics.bootstrap_samples[metric]))
        result = []
        for i in range(n):
            acc = 0.0
            for metric, weight in wmap.items():
                samples = metrics.bootstrap_samples.get(metric)
                if samples and i < len(samples):
                    des = self.anchors.score(metric, float(samples[i]))
                else:
                    des = self.anchors.score(metric, metrics.metric(metric))
                acc += weight * des
            result.append(acc)
        return result

    # -- main entry: legacy scalar mode --------------------------------------

    def decide(
        self,
        candidates: Sequence[TreatmentMetrics],
        *,
        raw_trial_id: str = "RAW",
        integrity_evidence: Optional[
            Mapping[str, TreatmentIntegrityEvidence]
        ] = None,
        screening_only: bool = False,
    ) -> DecisionResult:
        """Run the local scalar pipeline as an explicit screening diagnostic.

        The returned winner is never formal selection authority. Production
        selection must use :meth:`authoritative_decision`.
        """
        if not screening_only:
            raise ValueError(
                "legacy decide() is screening-only; formal selection requires "
                "authoritative_decision(request)"
            )
        if self._generalized:
            raise TypeError(
                "decide() is the legacy scalar path; this policy was built "
                "with a FactorFitnessSpec — use decision() instead"
            )
        if not candidates:
            raise ValueError("candidates must be non-empty")
        for metrics in candidates:
            if not isinstance(metrics, TreatmentMetrics):
                raise TypeError("candidates must be TreatmentMetrics instances")
        evidence_map: Dict[str, TreatmentIntegrityEvidence] = {}
        if integrity_evidence is not None:
            for key, value in integrity_evidence.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("integrity_evidence keys must be trial ids")
                if not isinstance(value, TreatmentIntegrityEvidence):
                    raise TypeError(
                        "integrity_evidence values must be "
                        "TreatmentIntegrityEvidence instances"
                    )
                value.verify()
                evidence_map[key] = value

        step_trace: Dict[str, str] = {}

        raw = next(
            (m for m in candidates if m.trial_id == raw_trial_id), None
        )
        if raw is None:
            raise ValueError(
                f"RAW candidate {raw_trial_id!r} must be present in candidates"
            )

        hard_rejected = set()
        rejected_without_evidence = set()
        for metrics in candidates:
            gates = self._integrity_gates(
                metrics.trial_id,
                metrics.coverage,
                metrics.rank_ic,
                metrics.icir,
                metrics.turnover,
                metrics.compute_cost,
                evidence_map.get(metrics.trial_id),
            )
            evidence_gate = next(
                (g for g in gates if g.name == "integrity_evidence"), None
            )
            if not all(gate.passed for gate in gates):
                hard_rejected.add(metrics.trial_id)
                if evidence_gate is not None and not evidence_gate.passed:
                    rejected_without_evidence.add(metrics.trial_id)
        step_trace["step1_integrity"] = (
            f"hard-rejected: {sorted(hard_rejected) or 'none'}"
            + (
                f" (no/failing integrity evidence: "
                f"{sorted(rejected_without_evidence)})"
                if rejected_without_evidence
                else ""
            )
        )
        active = [m for m in candidates if m.trial_id not in hard_rejected]
        if not active:
            if rejected_without_evidence == {m.trial_id for m in candidates}:
                raise TreatmentIntegrityError(
                    "all candidates failed the treatment integrity gates: "
                    "no candidate carries a complete, passing "
                    "TreatmentIntegrityEvidence — nothing may be scored "
                    "(fail closed)"
                )
            raise ValueError("all candidates failed integrity hard gates")

        deltas = {
            m.trial_id: self._raw_relative_deltas(raw, m)
            for m in active
            if m.trial_id != raw_trial_id
        }
        step_trace["step2_deltas"] = (
            f"computed raw-relative deltas for {sorted(deltas)}"
        )
        step_trace["step3_desirability"] = "continuous mapping applied"
        step_trace["step4_dimensions"] = "six balanced dimensions aggregated"

        pareto = self._pareto_frontier(active)
        pareto_ids = [m.trial_id for m in pareto]
        step_trace["step5_pareto"] = f"frontier: {sorted(pareto_ids)}"
        if not pareto:
            raise ValueError("Pareto frontier is empty after STEP 5")

        plausible = self._statistically_plausible(pareto)
        plausible_ids = [m.trial_id for m in plausible]
        step_trace["step6_uncertainty"] = (
            f"statistically plausible: {sorted(plausible_ids)}"
        )
        if not plausible:
            raise ValueError("no statistically plausible candidates after STEP 6")

        step_trace["step7_robust_utility"] = "conservative lower-bound utility"
        winner = self._rank_with_uncertainty(plausible)
        step_trace["step8_near_equivalence"] = (
            f"winner: {winner.trial_id} (simpler/lower-turnover preferred "
            "when not statistically distinguishable)"
        )

        return DecisionResult(
            winner_trial_id=winner.trial_id,
            step_trace=step_trace,
            pareto_trial_ids=pareto_ids,
            statistically_plausible=plausible_ids,
            raw_kept=raw_trial_id in plausible_ids,
        )

    # -- main entry: generalized health-dimension mode -----------------------

    def decision(
        self,
        candidates: Sequence[HealthDecisionInput],
        *,
        raw_trial_id: str = "RAW",
        integrity_evidence: Optional[
            Mapping[str, TreatmentIntegrityEvidence]
        ] = None,
        screening_only: bool = False,
    ) -> DecisionResult:
        """Run the local health-dimension pipeline as a screening diagnostic.

        ``candidates`` are :class:`HealthDecisionInput` objects carrying a FA
        ``FactorHealthView`` (14-dim grades).  The pipeline:

        STEP 1  integrity hard gates + health-dimension floor gates + the
                evidence-tier minimum declared by the fitness spec.
        STEP 2  RAW-relative deltas over the health dimensions.
        STEP 3  grade order -> desirability (no numeric threshold).
        STEP 4  balanced-dimension aggregation (mean of graded members).
        STEP 5  Pareto frontier.
        STEP 6  statistical uncertainty (bootstrap series when present).
        STEP 7  robust utility.
        STEP 8  near-equivalence.

        The RAW baseline is always kept as a candidate; a diagnostic RAW winner is
        reported via ``outcome`` and its fitness artifact so the caller can
        produce the RAW factor body (R61-FI-033).
        """
        if not screening_only:
            raise ValueError(
                "local health decision() is screening-only; formal selection "
                "requires authoritative_decision(request)"
            )
        if not self._generalized:
            raise TypeError(
                "decision() is the generalized health-dimension path; this "
                "policy was built with DesirabilityAnchors — use decide() for "
                "the legacy scalar path"
            )
        if not candidates:
            raise ValueError("candidates must be non-empty")
        for cand in candidates:
            if not isinstance(cand, HealthDecisionInput):
                raise TypeError("candidates must be HealthDecisionInput instances")
        evidence_map: Dict[str, TreatmentIntegrityEvidence] = {}
        if integrity_evidence is not None:
            for key, value in integrity_evidence.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("integrity_evidence keys must be trial ids")
                if not isinstance(value, TreatmentIntegrityEvidence):
                    raise TypeError(
                        "integrity_evidence values must be "
                        "TreatmentIntegrityEvidence instances"
                    )
                value.verify()
                evidence_map[key] = value

        step_trace: Dict[str, str] = {}

        raw = next((c for c in candidates if c.trial_id == raw_trial_id), None)
        if raw is None:
            raise ValueError(
                f"RAW candidate {raw_trial_id!r} must be present in candidates"
            )

        # STEP 1: integrity + health floors + evidence tier.
        hard_rejected = set()
        rejected_without_evidence = set()
        tier_rejected = set()
        for cand in candidates:
            gates = self._integrity_gates(
                cand.trial_id,
                coverage=cand.health_view.grade_of(
                    HealthDimension.DATA_COVERAGE
                )
                not in (HealthGrade.NONE, ""),
                rank_ic=0.0,
                icir=0.0,
                turnover=0.0,
                compute_cost=cand.compute_cost,
                evidence=evidence_map.get(cand.trial_id),
            )
            gates.extend(self._health_floor_gates(cand))
            tier_gate = self._evidence_tier_gate(cand)
            gates.append(tier_gate)
            evidence_gate = next(
                (g for g in gates if g.name == "integrity_evidence"), None
            )
            if not all(gate.passed for gate in gates):
                hard_rejected.add(cand.trial_id)
                if evidence_gate is not None and not evidence_gate.passed:
                    rejected_without_evidence.add(cand.trial_id)
                if not tier_gate.passed:
                    tier_rejected.add(cand.trial_id)
        step_trace["step1_integrity"] = (
            f"hard-rejected: {sorted(hard_rejected) or 'none'}"
            + (
                f" (tier below spec minimum: {sorted(tier_rejected)})"
                if tier_rejected
                else ""
            )
            + (
                f" (no/failing integrity evidence: "
                f"{sorted(rejected_without_evidence)})"
                if rejected_without_evidence
                else ""
            )
        )
        active = [c for c in candidates if c.trial_id not in hard_rejected]
        if not active:
            if rejected_without_evidence == {c.trial_id for c in candidates}:
                raise TreatmentIntegrityError(
                    "all candidates failed the treatment integrity gates: "
                    "no candidate carries a complete, passing "
                    "TreatmentIntegrityEvidence — nothing may be scored "
                    "(fail closed)"
                )
            raise ValueError(
                "all candidates failed integrity hard gates (incl. health "
                "dimension floors / minimum evidence tier)"
            )

        # STEP 2: RAW-relative deltas over the balanced dimensions.
        raw_dims = self._health_desirability_scores(raw)
        fitness_artifacts: List[CandidateFitnessArtifact] = []
        for cand in active:
            dims = self._health_desirability_scores(cand)
            # A positive delta means better-than-RAW after grade-order
            # normalization; None only when RAW itself has no graded value.
            deltas = {}
            for name in BALANCED_DIMENSIONS:
                if raw_dims.get(name) is not None and dims.get(name) is not None:
                    deltas[name] = dims[name] - raw_dims[name]
                else:
                    deltas[name] = None
            fitness_artifacts.append(
                CandidateFitnessArtifact(
                    trial_id=cand.trial_id,
                    evaluation_ref=cand.health_view.evaluation_ref,
                    health_card_ref=(
                        cand.health_view.factor_definition_id
                        + "@"
                        + cand.health_view.evaluation_ref
                    ),
                    raw_relative_deltas=deltas,
                    evidence_tier=cand.evidence_tier,
                    status=EvidenceStatus.COMPUTED,
                    # Dimension scores use the canonical FA dimension ids (the
                    # artifact contract validates them against the 14-dim set),
                    # derived from the candidate's graded FA health dims.
                    dimension_scores={
                        dim: des
                        for dim, grade in cand.health_view.dimension_grades.items()
                        for des in (grade_to_desirability(grade),)
                        if des is not None
                    },
                )
            )
        step_trace["step2_deltas"] = (
            f"computed RAW-relative health-dimension deltas for "
            f"{sorted(c.trial_id for c in active)}"
        )
        step_trace["step3_desirability"] = (
            "FA grade order -> unit desirability (monotone, no thresholds)"
        )
        step_trace["step4_dimensions"] = (
            "six balanced dimensions aggregated over graded FA health dims"
        )

        # STEP 5: Pareto over the health-desirability axes.
        pareto = self._pareto_frontier(active, use_health=True)
        pareto_ids = [c.trial_id for c in pareto]
        step_trace["step5_pareto"] = f"frontier: {sorted(pareto_ids)}"
        if not pareto:
            raise ValueError("Pareto frontier is empty after STEP 5")

        # STEP 6: statistical uncertainty.
        plausible = self._statistically_plausible(pareto, use_health=True)
        plausible_ids = [c.trial_id for c in plausible]
        step_trace["step6_uncertainty"] = (
            f"statistically plausible: {sorted(plausible_ids)}"
        )
        if not plausible:
            raise ValueError("no statistically plausible candidates after STEP 6")

        # STEP 7 + 8.
        step_trace["step7_robust_utility"] = "conservative lower-bound utility"
        winner = self._rank_with_uncertainty(plausible, use_health=True)
        step_trace["step8_near_equivalence"] = (
            f"winner: {winner.trial_id} (simpler/lower-turnover preferred "
            "when not statistically distinguishable)"
        )
        # RAW is "kept" when it survived STEP 6 as a viable candidate — even
        # when the winner is a treatment, RAW remained a candidate through the
        # whole pipeline (treated-got-worse remains discoverable).  The raw
        # candidate must also be a member of the active set at STEP 5.
        raw_kept = raw_trial_id in pareto_ids or raw_trial_id in plausible_ids
        outcome = (
            "RAW_SELECTED_NO_IMPROVEMENT"
            if winner.trial_id == raw_trial_id
            else "IMPROVED"
        )

        return DecisionResult(
            winner_trial_id=winner.trial_id,
            step_trace=step_trace,
            pareto_trial_ids=pareto_ids,
            statistically_plausible=plausible_ids,
            raw_kept=raw_kept,
            outcome=outcome,
            fitness_artifacts=tuple(fitness_artifacts),
        )


__all__ = [
    "TreatmentMetrics",
    "IntegrityGate",
    "DesirabilityAnchors",
    "DecisionResult",
    "HealthDecisionInput",
    "TreatmentDecisionPolicy",
    "RAW_METRICS",
    "BALANCED_DIMENSIONS",
    "HEALTH_DIMENSION_IDS",
    "grade_to_desirability",
    # R55 P0-9: real integrity evidence re-exports for pipeline callers.
    "RAW_TREATMENT_KIND",
    "IntegrityCheckResult",
    "TreatmentIntegrityEvidence",
    "TreatmentIntegrityError",
    "build_integrity_evidence",
    "describe_integrity_problem",
    "digest_value",
]
