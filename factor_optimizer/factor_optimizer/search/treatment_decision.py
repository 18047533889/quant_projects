"""TreatmentDecisionPolicy: formal 8-step decision pipeline (DLIB-FO-002).

The historical ``RobustBalancedUtility`` is a single weighted utility.  This
module builds a formal 8-step decision pipeline for choosing among treatment
recipes (RAW baseline vs EWMA vs KAMA vs Dual-Neutral, etc.):

    STEP 1  Integrity Hard Gates   - only truly non-negotiable hard-rejects
    STEP 2  Raw-relative deltas    - each treatment relative to RAW
    STEP 3  Metric->Desirability   - continuous mapping, no cliff thresholds
    STEP 4  Dimension aggregation  - Predictive/Stability/Robustness/
                                     Tradability/PurityExposure/DataQuality
    STEP 5  Pareto Frontier        - remove truly dominated treatments
    STEP 6  Statistical uncertainty- bootstrap CI, probability of improvement
    STEP 7  Robust Utility         - rank only within Pareto/statistically-
                                     plausible candidates
    STEP 8  Near-equivalence rule  - prefer simpler/lower-turnover/lower-cost
                                     when not statistically distinguishable

RAW is always kept as a candidate so "treated got worse" is discoverable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from factor_optimizer.contracts.treatment_integrity import (
    RAW_TREATMENT_KIND,
    IntegrityCheckResult,
    TreatmentIntegrityEvidence,
    build_integrity_evidence,
    describe_integrity_problem,
    digest_value,
)
from factor_optimizer.errors import TreatmentIntegrityError
from factor_optimizer.search.desirability import desirability_for
from factor_optimizer.search.pareto import ParetoFrontier, ParetoPoint
from factor_optimizer.search.uncertainty_winner import (
    UncertaintyAwareWinnerSelector,
    UncertaintyConfig,
    UncertaintyEvidence,
)
from factor_optimizer.search.winner_selector import WinnerPolicy


# The raw metrics a treatment is scored on (STEP 2 deltas are relative to RAW).
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

# The six balanced dimensions (STEP 4).
BALANCED_DIMENSIONS = (
    "predictive",
    "stability",
    "robustness",
    "tradability",
    "purity_exposure",
    "data_quality",
)


@dataclass(frozen=True)
class IntegrityGate:
    """A single non-negotiable hard-reject gate (STEP 1).

    Attributes:
        name: Gate name (e.g. "metric_coverage_nonzero", "integrity_evidence").
        passed: True when the treatment satisfies the gate.
        detail: Optional human-readable detail.
    """

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
    """Raw metrics for a single treatment candidate.

    Attributes:
        trial_id: Candidate identifier (e.g. "RAW", "EWMA", "KAMA").
        rank_ic: Rank IC (higher better).
        icir: ICIR (higher better).
        turnover: Turnover (lower better).
        cost_adjusted_alpha: Cost-adjusted alpha (higher better).
        worst_slice: Worst-slice return (higher better).
        exposure: Exposure (lower better).
        coverage: Coverage (higher better).
        stability: Stability (higher better).
        robustness: Robustness score (higher better).
        complexity_score: Complexity (0..1, higher = more complex = worse).
        n_transforms: Number of transforms in the recipe.
        compute_cost: Compute cost in budget units.
        bootstrap_samples: Optional per-metric bootstrap resampled values for
            STEP 6 uncertainty.  Mapping metric -> list of resampled values.
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
class DesirabilityAnchors:
    """Per-metric desirability anchors for the decision pipeline (STEP 3).

    Each metric maps to (direction, anchors) exactly like DEFAULT_MAPS.
    """

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


@dataclass(frozen=True)
class DecisionResult:
    """The outcome of the 8-step decision pipeline.

    Attributes:
        winner_trial_id: The winning treatment (may be RAW).
        step_trace: Per-step notes for auditability.
        pareto_trial_ids: Trial ids on the Pareto frontier (STEP 5).
        statistically_plausible: Trial ids surviving STEP 6.
        raw_kept: True when RAW was a candidate and survived to the end.
    """

    winner_trial_id: str
    step_trace: Dict[str, str] = field(default_factory=dict)
    pareto_trial_ids: List[str] = field(default_factory=list)
    statistically_plausible: List[str] = field(default_factory=list)
    raw_kept: bool = False


class TreatmentDecisionPolicy:
    """Formal 8-step decision pipeline for treatment selection.

    RAW is always injected as a candidate (if not already present) so a
    treatment that made things worse is discoverable — the pipeline can pick
    RAW as the winner.
    """

    def __init__(
        self,
        anchors: DesirabilityAnchors,
        policy: WinnerPolicy,
        uncertainty_config: Optional[UncertaintyConfig] = None,
        *,
        require_integrity_evidence: bool = True,
    ):
        if not isinstance(anchors, DesirabilityAnchors):
            raise TypeError("anchors must be a DesirabilityAnchors")
        if not isinstance(policy, WinnerPolicy):
            raise TypeError("policy must be a WinnerPolicy")
        self.anchors = anchors
        self.policy = policy
        self.uncertainty_config = uncertainty_config or UncertaintyConfig()
        # R55 P0-9: the integrity hard gates are real, fail-closed gates.
        # They REQUIRE a passing TreatmentIntegrityEvidence per candidate
        # (not a re-derivation from the metrics being scored).  The escape
        # hatch exists only so the historical metric-derived screen can still
        # be run explicitly; it is recorded in the step trace as such and is
        # NOT a production integrity verdict.
        if not isinstance(require_integrity_evidence, bool):
            raise TypeError("require_integrity_evidence must be a bool")
        self.require_integrity_evidence = require_integrity_evidence

    # -- STEP 1: integrity hard gates --------------------------------------

    def _integrity_gates(
        self,
        metrics: TreatmentMetrics,
        evidence: Optional[TreatmentIntegrityEvidence] = None,
    ) -> List[IntegrityGate]:
        """Real, fail-closed integrity gates (R55 P0-9).

        The historical implementation derived every verdict from the scalar
        metrics the pipeline was about to score — a fake gate that could be
        satisfied by a plausible-looking headline number without any treatment
        ever having been applied.  The gate now REQUIRES a
        :class:`TreatmentIntegrityEvidence` produced by the code that actually
        applied the treatment:

        - evidence present and bound to this ``trial_id`` (no stale/shuffled
          evidence),
        - evidence self-consistent (content hash re-verified),
        - ``overall_status`` is PASSED and every recorded check passed
          (NOT_RUN — no checks — is a rejection),
        - the measured before/after digests in the evidence are the load-bearing
          signals the pipeline no longer fabricates.

        ``metric_sanity`` keeps the historical threshold screen as a SUPPLEMENT
        (it is cheap and catches obvious garbage), but on its own it no longer
        constitutes integrity: with ``require_integrity_evidence=True`` a
        candidate whose evidence is missing, stale or failing is hard-rejected
        regardless of how good its metrics look.
        """
        gates: List[IntegrityGate] = [
            # P0-FO-001: these scalar screens are METRIC SANITY gates only —
            # coverage > 0 can never prove PIT, rank_ic > -1 can never prove the
            # absence of future leakage, compute_cost >= 0 can never prove the
            # execution was correct.  The load-bearing integrity verdicts come
            # exclusively from the TreatmentIntegrityEvidence below.
            IntegrityGate(
                "metric_coverage_nonzero",
                metrics.coverage > 0.0,
                "metric sanity: coverage must be positive",
            ),
            IntegrityGate(
                "metric_coverage_floor",
                metrics.coverage >= 0.5,
                "metric sanity: coverage must be >= 0.5 (sufficient sample)",
            ),
            IntegrityGate(
                "metric_coverage_bounded",
                metrics.coverage <= 1.0,
                "metric sanity: coverage must be <= 1.0",
            ),
            IntegrityGate(
                "metric_numeric_finite",
                math.isfinite(metrics.rank_ic)
                and math.isfinite(metrics.icir)
                and math.isfinite(metrics.turnover),
                "metric sanity: core metrics must be finite",
            ),
            IntegrityGate(
                "metric_rank_ic_bounded",
                metrics.rank_ic > -1.0,
                "metric sanity: rank_ic must be > -1",
            ),
            IntegrityGate(
                "compute_cost_numeric_valid",
                metrics.compute_cost >= 0.0,
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
            metrics.trial_id,
            evidence,
            treatment_kind=None,
        )
        gates.append(
            IntegrityGate(
                "integrity_evidence",
                problem is None,
                problem or (
                    f"a complete, passing TreatmentIntegrityEvidence for "
                    f"{metrics.trial_id!r} is required"
                ),
            )
        )
        return gates

    # -- STEP 2: raw-relative deltas ----------------------------------------

    def _raw_relative_deltas(
        self, raw: TreatmentMetrics, treatment: TreatmentMetrics
    ) -> Dict[str, float]:
        """Each treatment metric relative to RAW.

        For higher-better metrics the delta is (treatment - raw); for
        lower-better metrics (turnover, exposure) it is (raw - treatment) so a
        positive delta always means "better than RAW".
        """
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

    # -- STEP 3: metric -> desirability -------------------------------------

    def _desirabilities(self, metrics: TreatmentMetrics) -> Dict[str, float]:
        """Continuous metric->desirability mapping (no cliff thresholds)."""
        return {
            metric: self.anchors.score(metric, metrics.metric(metric))
            for metric in RAW_METRICS
        }

    # -- STEP 4: dimension aggregation --------------------------------------

    def _dimension_scores(
        self, metrics: TreatmentMetrics, desirabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate raw-metric desirabilities into the six balanced dimensions.

        Each dimension is a weighted blend of the relevant raw-metric
        desirabilities (all in [0,1], higher better).
        """
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

    # -- STEP 5: Pareto frontier --------------------------------------------

    def _pareto_frontier(
        self, candidates: List[TreatmentMetrics]
    ) -> List[TreatmentMetrics]:
        """Remove truly dominated treatments (higher-better on all dims)."""
        frontier = ParetoFrontier()
        points = []
        for metrics in candidates:
            dims = self._dimension_scores(
                metrics, self._desirabilities(metrics)
            )
            objectives = tuple(dims[name] for name in BALANCED_DIMENSIONS)
            point = ParetoPoint(
                trial_id=metrics.trial_id,
                objectives=objectives,
                metadata={"metrics": metrics.trial_id},
            )
            points.append(point)
            frontier.add_point(point)
        survivors = {p.trial_id for p in frontier.frontier()}
        return [c for c in candidates if c.trial_id in survivors]

    # -- STEP 6: statistical uncertainty -------------------------------------

    def _statistically_plausible(
        self, candidates: List[TreatmentMetrics]
    ) -> List[TreatmentMetrics]:
        """Keep candidates with bootstrap evidence (or no bootstrap -> keep).

        When bootstrap samples are present, a candidate is kept only if its
        point estimate is finite and its CI is not degenerate.  Candidates
        without bootstrap samples are kept (they are treated as point
        estimates only).
        """
        plausible = []
        for metrics in candidates:
            if not metrics.bootstrap_samples:
                plausible.append(metrics)
                continue
            # Require at least one metric with bootstrap samples to be
            # non-degenerate (finite, non-empty).
            ok = False
            for samples in metrics.bootstrap_samples.values():
                if samples and all(math.isfinite(float(v)) for v in samples):
                    ok = True
                    break
            if ok:
                plausible.append(metrics)
        return plausible

    # -- STEP 7 + 8: robust utility + near-equivalence ----------------------

    def _rank_with_uncertainty(
        self, candidates: List[TreatmentMetrics]
    ) -> TreatmentMetrics:
        """Rank candidates by conservative utility, applying near-equivalence.

        Builds UncertaintyEvidence for each candidate (using bootstrap samples
        when available, else a single point sample) and delegates to the
        UncertaintyAwareWinnerSelector.
        """
        evidence_list = []
        for metrics in candidates:
            dims = self._dimension_scores(
                metrics, self._desirabilities(metrics)
            )
            # Build per-dimension bootstrap samples.  When the candidate has
            # no bootstrap samples, use a single point sample per dimension.
            dimension_samples: Dict[str, Sequence[float]] = {}
            for name in BALANCED_DIMENSIONS:
                if metrics.bootstrap_samples:
                    # Aggregate the raw-metric bootstrap samples into the
                    # dimension using the same weights as _dimension_scores.
                    samples = self._aggregate_dimension_bootstrap(metrics, name)
                    dimension_samples[name] = samples
                else:
                    dimension_samples[name] = [dims[name]]
            evidence_list.append(
                UncertaintyEvidence(
                    trial_id=metrics.trial_id,
                    dimension_samples=dimension_samples,
                    complexity_score=metrics.complexity_score,
                    turnover=metrics.turnover,
                    compute_cost=metrics.compute_cost,
                    n_transforms=metrics.n_transforms,
                )
            )
        robustness_scores = {
            metrics.trial_id: metrics.robustness for metrics in candidates
        }
        selector = UncertaintyAwareWinnerSelector(
            self.policy, self.uncertainty_config
        )
        return next(
            m for m in candidates if m.trial_id == selector.select(
                evidence_list, robustness_scores
            ).trial_id
        )

    def _aggregate_dimension_bootstrap(
        self, metrics: TreatmentMetrics, dimension: str
    ) -> List[float]:
        """Aggregate raw-metric bootstrap samples into a dimension's samples.

        Uses the same weights as ``_dimension_scores``.  If a required raw
        metric has no bootstrap samples, the point desirability is used for
        every iteration.
        """
        weights = {
            "predictive": {"rank_ic": 0.6, "icir": 0.4},
            "stability": {"stability": 1.0},
            "robustness": {},
            "tradability": {"turnover": 0.5, "cost_adjusted_alpha": 0.5},
            "purity_exposure": {"exposure": 1.0},
            "data_quality": {"coverage": 1.0},
        }
        if dimension == "robustness":
            return [metrics.robustness]
        wmap = weights[dimension]
        # Determine the number of bootstrap iterations.
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

    # -- main entry ----------------------------------------------------------

    def decide(
        self,
        candidates: Sequence[TreatmentMetrics],
        *,
        raw_trial_id: str = "RAW",
        integrity_evidence: Optional[
            Mapping[str, TreatmentIntegrityEvidence]
        ] = None,
    ) -> DecisionResult:
        """Run the full 8-step pipeline and return the winner.

        RAW is always injected as a candidate (if not already present) so a
        treatment that made things worse is discoverable.

        R55 P0-9: ``integrity_evidence`` maps ``trial_id ->
        TreatmentIntegrityEvidence``.  When the policy was constructed with
        ``require_integrity_evidence=True`` (the default, fail-closed), a
        candidate without complete, passing evidence bound to its trial id is
        hard-rejected at STEP 1 no matter how good its metrics look.  The RAW
        baseline is treated like any other candidate: it must carry evidence
        of its own (an identity-check evidence whose before/after digests are
        equal), which ``build_integrity_evidence`` produces for
        ``treatment_kind="raw"``.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        for metrics in candidates:
            if not isinstance(metrics, TreatmentMetrics):
                raise TypeError("candidates must be TreatmentMetrics instances")
        evidence_map: Dict[str, TreatmentIntegrityEvidence] = {}
        if integrity_evidence is not None:
            if not isinstance(integrity_evidence, Mapping):
                raise TypeError(
                    "integrity_evidence must map trial_id -> "
                    "TreatmentIntegrityEvidence"
                )
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

        # Ensure RAW is a candidate.
        raw = next(
            (m for m in candidates if m.trial_id == raw_trial_id), None
        )
        if raw is None:
            raise ValueError(
                f"RAW candidate {raw_trial_id!r} must be present in candidates"
            )

        # STEP 1: integrity hard gates.
        hard_rejected = set()
        rejected_without_evidence = set()
        for metrics in candidates:
            gates = self._integrity_gates(
                metrics, evidence_map.get(metrics.trial_id)
            )
            evidence_gate = next(
                (
                    gate
                    for gate in gates
                    if gate.name == "integrity_evidence"
                ),
                None,
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

        # STEP 2: raw-relative deltas (audit only).
        deltas = {
            m.trial_id: self._raw_relative_deltas(raw, m)
            for m in active
            if m.trial_id != raw_trial_id
        }
        step_trace["step2_deltas"] = (
            f"computed raw-relative deltas for {sorted(deltas)}"
        )

        # STEP 3: metric->desirability (implicit in dimension scoring).
        step_trace["step3_desirability"] = "continuous mapping applied"

        # STEP 4: dimension aggregation (implicit in Pareto + ranking).
        step_trace["step4_dimensions"] = "six balanced dimensions aggregated"

        # STEP 5: Pareto frontier.
        pareto = self._pareto_frontier(active)
        pareto_ids = [m.trial_id for m in pareto]
        step_trace["step5_pareto"] = f"frontier: {sorted(pareto_ids)}"
        if not pareto:
            raise ValueError("Pareto frontier is empty after STEP 5")

        # STEP 6: statistical uncertainty.
        plausible = self._statistically_plausible(pareto)
        plausible_ids = [m.trial_id for m in plausible]
        step_trace["step6_uncertainty"] = (
            f"statistically plausible: {sorted(plausible_ids)}"
        )
        if not plausible:
            raise ValueError("no statistically plausible candidates after STEP 6")

        # STEP 7: robust utility (rank within plausible candidates).
        step_trace["step7_robust_utility"] = "conservative lower-bound utility"

        # STEP 8: near-equivalence rule (prefer simpler/lower-turnover).
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


__all__ = [
    "TreatmentMetrics",
    "IntegrityGate",
    "DesirabilityAnchors",
    "DecisionResult",
    "TreatmentDecisionPolicy",
    "RAW_METRICS",
    "BALANCED_DIMENSIONS",
    # R55 P0-9: real integrity evidence re-exports for pipeline callers.
    "RAW_TREATMENT_KIND",
    "IntegrityCheckResult",
    "TreatmentIntegrityEvidence",
    "TreatmentIntegrityError",
    "build_integrity_evidence",
    "describe_integrity_problem",
    "digest_value",
]
