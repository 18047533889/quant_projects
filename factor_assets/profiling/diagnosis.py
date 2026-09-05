"""Deterministic factor diagnosis engine (R61-FI-026, plan §9/§13/§15-§17/§53).

The engine maps *evidence* + a versioned :class:`DiagnosisPolicy` to a
deterministic list of :class:`DiagnosisTag` records — no LLM, no randomness,
fixed rule order, same input ⇒ same output.

Tag set (first deterministic batch, plan E2 / §53 FI-026):

- integrity / data quality: INTEGRITY_FAILURE, DATA_QUALITY_FAILURE,
  POOR_COVERAGE, HIGH_TIE_RATIO, SPARSE_FACTOR, NUMERICAL_INSTABILITY,
  STALE_DATA
- predictive / stability / generalization: LOW_PREDICTIVE, UNSTABLE_IC,
  RECENT_DEGRADATION, OVERFIT_GENERALIZATION, LOW_STATISTICAL_CONFIDENCE
- shape: U_SHAPE, INVERTED_U, TOP_TAIL_COLLAPSE, BOTTOM_TAIL_COLLAPSE,
  NONSTATIONARY_SHAPE
- portfolio economics: HIGH_TURNOVER, HIGH_COST_DRAG, HIGH_DRAWDOWN,
  LONG_UNDERWATER, REGIME_DEPENDENT
- exposure: SIZE_EXPOSURE, INDUSTRY_EXPOSURE, MULTI_STYLE_EXPOSURE
- novelty / complexity: SEMANTIC_DUPLICATE, VALUE_NEAR_DUPLICATE,
  LOW_NOVELTY, HIGH_COMPLEXITY

Evidence-status legality (plan §8 iron law)
-------------------------------------------
Numerical-threshold diagnosis rules fire **only** on evidence whose status is
``COMPUTED``.  Bad evidence (``FAILED / INVALID / STALE / UNKNOWN /
NOT_COMPUTED_STAGE / UNAVAILABLE_INPUT / NOT_APPLICABLE``) never triggers a
numeric diagnosis — missing evidence is not evidence of a defect.  The
integrity / data-quality failure tags fire on *typed* bad-status signals
(e.g. an evidence ref with status ``FAILED`` and reason code
``timing_contract_violation``), which is a distinct, evidence-bearing rule.

Unknown / insufficient evidence ⇒ the tag is not produced (or an explicit
``UNKNOWN`` severity record is produced only by the caller-facing helper
:func:`diagnose_factor` when *no* tag fired and the caller requested it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from factor_assets.profiling.metric_grading import (
    BAD_EVIDENCE_STATUSES,
    EvidenceStatusVocabulary,
    resolve_evidence_status,
)
from factor_assets.profiling.policies import (
    DiagnosisPolicy,
    get_diagnosis_policy,
)

__all__ = [
    "DiagnosisTag",
    "DiagnosisRuleRef",
    "DIAGNOSIS_TAGS",
    "detect_diagnoses",
    "diagnose_factor",
]

#: Canonical order of the deterministic tag set (fixed rule order ⇒ stable
#: output ordering).
DIAGNOSIS_TAGS: tuple[str, ...] = (
    # integrity / data quality
    "INTEGRITY_FAILURE",
    "DATA_QUALITY_FAILURE",
    "POOR_COVERAGE",
    "HIGH_TIE_RATIO",
    "SPARSE_FACTOR",
    "NUMERICAL_INSTABILITY",
    "STALE_DATA",
    # predictive / stability / generalization
    "LOW_PREDICTIVE",
    "UNSTABLE_IC",
    "RECENT_DEGRADATION",
    "OVERFIT_GENERALIZATION",
    "LOW_STATISTICAL_CONFIDENCE",
    # shape
    "U_SHAPE",
    "INVERTED_U",
    "TOP_TAIL_COLLAPSE",
    "BOTTOM_TAIL_COLLAPSE",
    "NONSTATIONARY_SHAPE",
    # portfolio economics
    "HIGH_TURNOVER",
    "HIGH_COST_DRAG",
    "HIGH_DRAWDOWN",
    "LONG_UNDERWATER",
    "REGIME_DEPENDENT",
    # exposure / purity
    "SIZE_EXPOSURE",
    "INDUSTRY_EXPOSURE",
    "MULTI_STYLE_EXPOSURE",
    # novelty / redundancy / complexity
    "SEMANTIC_DUPLICATE",
    "VALUE_NEAR_DUPLICATE",
    "LOW_NOVELTY",
    "HIGH_COMPLEXITY",
)


@dataclass(frozen=True)
class DiagnosisTag:
    """One deterministic diagnosis record.

    ``tag`` is the canonical diagnosis kind (upper-snake).  ``severity`` /
    ``repairability`` / base ``confidence`` come from the versioned
    :class:`DiagnosisPolicy`; the engine may raise ``confidence`` with a
    deterministic margin factor but never above 1.0 and never below 0.
    ``evidence_refs`` are the exact evidence keys the rule consumed (each is a
    dotted evidence path such as ``metrics.rank_ic.value``).
    ``triggered_policy_rule`` names the rule that fired (stable id).
    """

    factor_definition_id: str
    health_ref: str
    tag: str
    severity: str
    confidence: float
    repairability: str
    evidence_refs: tuple[str, ...]
    triggered_policy_rule: str = ""

    def __post_init__(self) -> None:
        if not self.factor_definition_id:
            raise ValueError("factor_definition_id is required")
        if self.tag not in DIAGNOSIS_TAGS:
            raise ValueError(
                f"unknown diagnosis tag {self.tag!r}; expected one of "
                f"{DIAGNOSIS_TAGS}"
            )
        conf = float(self.confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", conf)
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.severity not in (
            "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "UNKNOWN",
        ):
            raise ValueError(f"unknown severity {self.severity!r}")
        if self.repairability not in (
            "REPAIRABLE", "PARTIALLY_REPAIRABLE", "UNREPAIRABLE", "UNKNOWN",
        ):
            raise ValueError(f"unknown repairability {self.repairability!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_definition_id": self.factor_definition_id,
            "health_ref": self.health_ref,
            "tag": self.tag,
            "severity": self.severity,
            "confidence": self.confidence,
            "repairability": self.repairability,
            "evidence_refs": list(self.evidence_refs),
            "triggered_policy_rule": self.triggered_policy_rule,
        }


@dataclass(frozen=True)
class DiagnosisRuleRef:
    """Reference describing one deterministic rule that fired (for auditing)."""

    tag: str
    rule_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tag or not self.rule_id:
            raise ValueError("tag and rule_id are required")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))


# ---------------------------------------------------------------------------
# Evidence access helpers
# ---------------------------------------------------------------------------


class _Evidence:
    """Typed access over a flat/dotted evidence mapping.

    Evidence keys are dotted paths (``metrics.rank_ic.value``,
    ``metrics.rank_ic.evidence_status``, ``integrity.pit_valid``, ...).  A
    ``value`` field is only numeric-trusted when its sibling
    ``evidence_status`` is ``COMPUTED``; otherwise ``value`` reads as
    ``None`` for the numeric rules (the status gates separately).
    """

    def __init__(self, evidence: Mapping[str, object]):
        self._raw = dict(evidence or {})

    def get(self, key: str, default: Any = None) -> Any:
        value = self._raw.get(key, default)
        # allow nested dicts by dotted traversal too
        if isinstance(value, Mapping):
            return value
        return value

    def has(self, key: str) -> bool:
        return key in self._raw

    def status_of(self, metric_key: str) -> str:
        """Canonical status token for a metric subtree key."""
        raw = self._raw.get(f"{metric_key}.evidence_status")
        if raw is None:
            raw = self._raw.get(metric_key, {}).get("evidence_status") if isinstance(
                self._raw.get(metric_key), Mapping
            ) else None
        if raw is None:
            return EvidenceStatusVocabulary.NONE
        try:
            return resolve_evidence_status(raw)
        except (TypeError, ValueError):
            return EvidenceStatusVocabulary.INVALID

    def value_of(self, metric_key: str) -> float | None:
        """Numeric value of a metric subtree — None unless status is COMPUTED."""
        raw = self._raw.get(f"{metric_key}.value")
        if raw is None and isinstance(self._raw.get(metric_key), Mapping):
            raw = self._raw[metric_key].get("value")
        status = self.status_of(metric_key)
        if status != EvidenceStatusVocabulary.COMPUTED:
            return None
        return self._coerce_float(raw, metric_key)

    @staticmethod
    def _coerce_float(raw: Any, key: str) -> float | None:
        if raw is None or isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            f = float(raw)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        try:
            f = float(raw)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    def num(self, key: str) -> float | None:
        return self._coerce_float(self._raw.get(key), key)


# ---------------------------------------------------------------------------
# Deterministic rule engine
# ---------------------------------------------------------------------------


def _mk(
    evidence: _Evidence,
    policy: DiagnosisPolicy,
    *,
    tag: str,
    health_ref: str,
    factor_definition_id: str,
    rule_id: str,
    refs: Sequence[str],
    confidence_boost: float = 0.0,
) -> DiagnosisTag:
    base = policy.confidence(tag)
    conf = min(1.0, max(0.0, base + confidence_boost))
    return DiagnosisTag(
        factor_definition_id=factor_definition_id,
        health_ref=health_ref,
        tag=tag,
        severity=policy.severity(tag),
        confidence=conf,
        repairability=policy.repairability(tag),
        evidence_refs=tuple(refs),
        triggered_policy_rule=rule_id,
    )


def _triggered(
    out: list[DiagnosisTag],
    rule: DiagnosisRuleRef,
    *,
    evidence: _Evidence,
    policy: DiagnosisPolicy,
    health_ref: str,
    factor_definition_id: str,
    confidence_boost: float = 0.0,
) -> None:
    out.append(
        _mk(
            evidence,
            policy,
            tag=rule.tag,
            health_ref=health_ref,
            factor_definition_id=factor_definition_id,
            rule_id=rule.rule_id,
            refs=rule.evidence_refs,
            confidence_boost=confidence_boost,
        )
    )


def _num_lt(evidence: _Evidence, key: str, threshold: float) -> bool:
    v = evidence.num(key)
    return v is not None and v < threshold


def _num_gt(evidence: _Evidence, key: str, threshold: float) -> bool:
    v = evidence.num(key)
    return v is not None and v > threshold


def _num_abs_gt(evidence: _Evidence, key: str, threshold: float) -> bool:
    v = evidence.num(key)
    return v is not None and abs(v) > threshold


def detect_diagnoses(
    *,
    factor_definition_id: str,
    health_ref: str = "",
    evidence: Mapping[str, object] | None = None,
    policy: DiagnosisPolicy | None = None,
) -> tuple[DiagnosisTag, ...]:
    """Deterministically detect diagnosis tags from evidence + policy.

    The engine iterates a fixed rule order and emits each tag at most once.
    Bad/unknown evidence never fires a numeric-threshold rule (COMPUTED-only);
    integrity / data-quality failure rules fire on typed bad-status signals
    with a machine-readable reason code.  Same inputs ⇒ same output.
    """
    if policy is None:
        policy = get_diagnosis_policy()
    ev = _Evidence(evidence or {})
    th = policy.thresholds
    out: list[DiagnosisTag] = []

    # ------------------------------------------------------------------ #
    # Integrity / data quality (typed status + reason signals)           #
    # ------------------------------------------------------------------ #
    for gate_id, reason in (
        ("pit_valid", "PIT_INVALID"),
        ("label_maturity_ok", "LABEL_TIMING_VIOLATION"),
        ("snapshot_identity_ok", "SNAPSHOT_MISMATCH"),
        ("universe_identity_ok", "UNIVERSE_MISMATCH"),
        ("finite_shape_schema_ok", "SPLIT_CONTAMINATION"),
        ("unit_scale_ok", "RETURN_BASIS_MISMATCH"),
        ("return_basis_ok", "CAUSALITY_INVALID"),
    ):
        status = ev.status_of(f"integrity.{gate_id}")
        reason_code = ev.get(f"integrity.{gate_id}.reason_code")
        if reason_code is None and isinstance(ev.get(f"integrity.{gate_id}"), Mapping):
            reason_code = ev.get(f"integrity.{gate_id}").get("reason_code")
        if status in BAD_EVIDENCE_STATUSES or (
            reason_code is not None and reason_code == reason
        ):
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="INTEGRITY_FAILURE",
                    rule_id=f"FI026_INTEGRITY_{gate_id}",
                    evidence_refs=(f"integrity.{gate_id}.evidence_status",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
                confidence_boost=0.0,
            )
            break  # one INTEGRITY_FAILURE record per card

    # Data quality failure: any data-quality metric with a bad status AND a
    # numeric-compute reason code implies a computation-level quality failure.
    dq_bad = False
    for dq_key in (
        "metrics.missing_ratio", "metrics.tie_ratio", "metrics.outlier_ratio",
        "metrics.staleness", "metrics.effective_n",
    ):
        if ev.status_of(dq_key) in BAD_EVIDENCE_STATUSES:
            dq_bad = True
            break
    if dq_bad:
        _triggered(
            out,
            DiagnosisRuleRef(
                tag="DATA_QUALITY_FAILURE",
                rule_id="FI026_DQ_BAD_STATUS",
                evidence_refs=(f"{dq_key}.evidence_status",),
            ),
            evidence=ev, policy=policy, health_ref=health_ref,
            factor_definition_id=factor_definition_id,
        )

    # ------------------------------------------------------------------ #
    # Data quality numeric rules (COMPUTED-only)                          #
    # ------------------------------------------------------------------ #
    cov_status = ev.status_of("metrics.coverage")
    if cov_status == EvidenceStatusVocabulary.COMPUTED:
        cov_v = ev.value_of("metrics.coverage")
        if cov_v is not None and cov_v < th.coverage_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="POOR_COVERAGE",
                    rule_id="FI026_COVERAGE_LT",
                    evidence_refs=("metrics.coverage.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    missing_status = ev.status_of("metrics.missing_ratio")
    if missing_status == EvidenceStatusVocabulary.COMPUTED:
        missing_v = ev.value_of("metrics.missing_ratio")
        if missing_v is not None and missing_v > th.missing_ratio_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="DATA_QUALITY_FAILURE",
                    rule_id="FI026_MISSING_GT",
                    evidence_refs=("metrics.missing_ratio.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    tie_status = ev.status_of("metrics.tie_ratio")
    if tie_status == EvidenceStatusVocabulary.COMPUTED:
        tie_v = ev.value_of("metrics.tie_ratio")
        if tie_v is not None and tie_v > th.tie_ratio_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="HIGH_TIE_RATIO",
                    rule_id="FI026_TIE_GT",
                    evidence_refs=("metrics.tie_ratio.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    eff_status = ev.status_of("metrics.effective_n")
    if eff_status == EvidenceStatusVocabulary.COMPUTED:
        eff_v = ev.value_of("metrics.effective_n")
        if eff_v is not None and eff_v < th.min_effective_n:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="SPARSE_FACTOR",
                    rule_id="FI026_EFFECTIVE_N_LT",
                    evidence_refs=("metrics.effective_n.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    nonfinite_status = ev.status_of("metrics.outlier_ratio")
    if nonfinite_status == EvidenceStatusVocabulary.COMPUTED:
        nonfinite_v = ev.value_of("metrics.outlier_ratio")
        if nonfinite_v is not None and nonfinite_v > th.outlier_ratio_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="NUMERICAL_INSTABILITY",
                    rule_id="FI026_OUTLIER_GT",
                    evidence_refs=("metrics.outlier_ratio.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    stale_status = ev.status_of("metrics.staleness")
    if stale_status == EvidenceStatusVocabulary.COMPUTED:
        stale_v = ev.value_of("metrics.staleness")
        if stale_v is not None and stale_v > th.stale_days_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="STALE_DATA",
                    rule_id="FI026_STALE_GT",
                    evidence_refs=("metrics.staleness.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    # ------------------------------------------------------------------ #
    # Predictive / stability / generalization (COMPUTED-only)             #
    # ------------------------------------------------------------------ #
    ric_status = ev.status_of("metrics.rank_ic")
    if ric_status == EvidenceStatusVocabulary.COMPUTED:
        ric = ev.value_of("metrics.rank_ic")
        if ric is not None and abs(ric) < th.rank_ic_low_abs:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="LOW_PREDICTIVE",
                    rule_id="FI026_RANK_IC_LOW",
                    evidence_refs=("metrics.rank_ic.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    icir_status = ev.status_of("metrics.rank_ic_ir")
    if (
        icir_status == EvidenceStatusVocabulary.COMPUTED
        and ric_status == EvidenceStatusVocabulary.COMPUTED
    ):
        ric = ev.value_of("metrics.rank_ic")
        icir = ev.value_of("metrics.rank_ic_ir")
        if (
            ric is not None and icir is not None
            and abs(ric) >= th.rank_ic_low_abs and icir < 0.3
        ):
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="LOW_STATISTICAL_CONFIDENCE",
                    rule_id="FI026_ICIR_LOW",
                    evidence_refs=("metrics.rank_ic_ir.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    ics_std_status = ev.status_of("metrics.rolling_ic_volatility")
    if ics_std_status == EvidenceStatusVocabulary.COMPUTED:
        ics_std = ev.value_of("metrics.rolling_ic_volatility")
        if ics_std is not None and ics_std > th.ic_vol_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="UNSTABLE_IC",
                    rule_id="FI026_IC_VOL_GT",
                    evidence_refs=("metrics.rolling_ic_volatility.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    recent_status = ev.status_of("metrics.ic_recent_vs_history_delta")
    if recent_status == EvidenceStatusVocabulary.COMPUTED:
        recent = ev.value_of("metrics.ic_recent_vs_history_delta")
        if recent is not None and recent < th.recent_degradation_delta_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="RECENT_DEGRADATION",
                    rule_id="FI026_RECENT_DELTA_LT",
                    evidence_refs=("metrics.ic_recent_vs_history_delta.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    retention_status = ev.status_of("generalization.validation_retention")
    if retention_status == EvidenceStatusVocabulary.COMPUTED:
        retention = ev.value_of("generalization.validation_retention")
        if retention is not None and retention < th.retention_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="OVERFIT_GENERALIZATION",
                    rule_id="FI026_RETENTION_LT",
                    evidence_refs=("generalization.validation_retention.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    # else: retention missing/bad — no numeric diagnosis fires; OVERFIT stays silent

    icir_delta_status = ev.status_of("generalization.train_validation_icir_delta")
    if icir_delta_status == EvidenceStatusVocabulary.COMPUTED:
        icir_delta = ev.value_of("generalization.train_validation_icir_delta")
        if icir_delta is not None and icir_delta > th.train_validation_icir_delta_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="OVERFIT_GENERALIZATION",
                    rule_id="FI026_ICIR_DELTA_GT",
                    evidence_refs=("generalization.train_validation_icir_delta.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    tstat_status = ev.status_of("metrics.hac_tstat")
    if tstat_status == EvidenceStatusVocabulary.COMPUTED:
        tstat = ev.value_of("metrics.hac_tstat")
        if tstat is not None and abs(tstat) < th.hac_tstat_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="LOW_STATISTICAL_CONFIDENCE",
                    rule_id="FI026_TSTAT_LT",
                    evidence_refs=("metrics.hac_tstat.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    # ------------------------------------------------------------------ #
    # Shape (COMPUTED-only)                                               #
    # ------------------------------------------------------------------ #
    u_status = ev.status_of("shape.u_shape_score")
    if u_status == EvidenceStatusVocabulary.COMPUTED:
        u_v = ev.value_of("shape.u_shape_score")
        if u_v is not None and u_v > th.u_shape_score_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="U_SHAPE",
                    rule_id="FI026_U_SHAPE_GT",
                    evidence_refs=("shape.u_shape_score.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    inv_status = ev.status_of("shape.inverted_u_score")
    if inv_status == EvidenceStatusVocabulary.COMPUTED:
        inv_v = ev.value_of("shape.inverted_u_score")
        if inv_v is not None and inv_v > th.inverted_u_shape_score_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="INVERTED_U",
                    rule_id="FI026_INVERTED_U_GT",
                    evidence_refs=("shape.inverted_u_score.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    top_cliff_status = ev.status_of("shape.top_tail_cliff")
    if top_cliff_status == EvidenceStatusVocabulary.COMPUTED:
        top_cliff = ev.value_of("shape.top_tail_cliff")
        if top_cliff is not None and top_cliff > th.top_tail_cliff_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="TOP_TAIL_COLLAPSE",
                    rule_id="FI026_TOP_CLIFF_GT",
                    evidence_refs=("shape.top_tail_cliff.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    bot_cliff_status = ev.status_of("shape.bottom_tail_cliff")
    if bot_cliff_status == EvidenceStatusVocabulary.COMPUTED:
        bot_cliff = ev.value_of("shape.bottom_tail_cliff")
        if bot_cliff is not None and bot_cliff > th.bottom_tail_cliff_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="BOTTOM_TAIL_COLLAPSE",
                    rule_id="FI026_BOTTOM_CLIFF_GT",
                    evidence_refs=("shape.bottom_tail_cliff.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    shape_stab_status = ev.status_of("shape.shape_stability")
    if shape_stab_status == EvidenceStatusVocabulary.COMPUTED:
        shape_stab = ev.value_of("shape.shape_stability")
        if shape_stab is not None and shape_stab < th.shape_stability_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="NONSTATIONARY_SHAPE",
                    rule_id="FI026_SHAPE_STAB_LT",
                    evidence_refs=("shape.shape_stability.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    # ------------------------------------------------------------------ #
    # Portfolio economics (COMPUTED-only)                                  #
    # ------------------------------------------------------------------ #
    turn_status = ev.status_of("metrics.factor_turnover_rate")
    if turn_status == EvidenceStatusVocabulary.COMPUTED:
        turn_v = ev.value_of("metrics.factor_turnover_rate")
        if turn_v is not None and turn_v > th.turnover_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="HIGH_TURNOVER",
                    rule_id="FI026_TURNOVER_GT",
                    evidence_refs=("metrics.factor_turnover_rate.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    cost_status = ev.status_of("metrics.cost_drag")
    if cost_status == EvidenceStatusVocabulary.COMPUTED:
        cost_v = ev.value_of("metrics.cost_drag")
        if cost_v is not None and cost_v > th.cost_drag_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="HIGH_COST_DRAG",
                    rule_id="FI026_COST_GT",
                    evidence_refs=("metrics.cost_drag.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    mdd_status = ev.status_of("metrics.max_drawdown")
    if mdd_status == EvidenceStatusVocabulary.COMPUTED:
        mdd = ev.value_of("metrics.max_drawdown")
        if mdd is not None and mdd < th.max_drawdown_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="HIGH_DRAWDOWN",
                    rule_id="FI026_MDD_LT",
                    evidence_refs=("metrics.max_drawdown.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    uw_status = ev.status_of("metrics.max_drawdown_duration")
    if uw_status == EvidenceStatusVocabulary.COMPUTED:
        uw = ev.value_of("metrics.max_drawdown_duration")
        if uw is not None and uw > th.underwater_days_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="LONG_UNDERWATER",
                    rule_id="FI026_UNDERWATER_GT",
                    evidence_refs=("metrics.max_drawdown_duration.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    cvar_status = ev.status_of("metrics.cvar_95")
    if cvar_status == EvidenceStatusVocabulary.COMPUTED:
        cvar = ev.value_of("metrics.cvar_95")
        if cvar is not None and cvar < th.cvar_95_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="HIGH_DRAWDOWN",
                    rule_id="FI026_CVAR_LT",
                    evidence_refs=("metrics.cvar_95.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    regime_disp_status = ev.status_of("metrics.regime_dispersion")
    if regime_disp_status == EvidenceStatusVocabulary.COMPUTED:
        regime_disp = ev.value_of("metrics.regime_dispersion")
        if regime_disp is not None and regime_disp > th.regime_dispersion_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="REGIME_DEPENDENT",
                    rule_id="FI026_REGIME_DISP",
                    evidence_refs=("metrics.regime_dispersion.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    else:
        regime_sig_status = ev.status_of("metrics.regime_sign_consistency")
        if regime_sig_status == EvidenceStatusVocabulary.COMPUTED:
            regime_sig = ev.value_of("metrics.regime_sign_consistency")
            if regime_sig is not None and regime_sig < th.regime_sign_consistency_min:
                _triggered(
                    out,
                    DiagnosisRuleRef(
                        tag="REGIME_DEPENDENT",
                        rule_id="FI026_REGIME_SIGN",
                        evidence_refs=("metrics.regime_sign_consistency.value",),
                    ),
                    evidence=ev, policy=policy, health_ref=health_ref,
                    factor_definition_id=factor_definition_id,
                )

    # ------------------------------------------------------------------ #
    # Exposure / purity (COMPUTED-only)                                    #
    # ------------------------------------------------------------------ #
    size_status = ev.status_of("exposure.size_exposure")
    if size_status == EvidenceStatusVocabulary.COMPUTED:
        size_v = ev.value_of("exposure.size_exposure")
        if size_v is not None and abs(size_v) > th.size_exposure_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="SIZE_EXPOSURE",
                    rule_id="FI026_SIZE_EXP",
                    evidence_refs=("exposure.size_exposure.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    ind_status = ev.status_of("exposure.industry_exposure")
    if ind_status == EvidenceStatusVocabulary.COMPUTED:
        ind_v = ev.value_of("exposure.industry_exposure")
        if ind_v is not None and abs(ind_v) > th.industry_exposure_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="INDUSTRY_EXPOSURE",
                    rule_id="FI026_IND_EXP",
                    evidence_refs=("exposure.industry_exposure.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    multi_status = ev.status_of("exposure.max_abs_style_exposure")
    if multi_status == EvidenceStatusVocabulary.COMPUTED:
        multi_v = ev.value_of("exposure.max_abs_style_exposure")
        n_style_status = ev.status_of("exposure.n_style_exposures")
        n_style = ev.value_of("exposure.n_style_exposures")
        if (
            multi_v is not None and multi_v > th.max_abs_style_exposure_max
            and (n_style is not None and n_style >= 2)
        ):
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="MULTI_STYLE_EXPOSURE",
                    rule_id="FI026_MULTI_STYLE",
                    evidence_refs=(
                        "exposure.max_abs_style_exposure.value",
                        "exposure.n_style_exposures.value",
                    ),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    # ------------------------------------------------------------------ #
    # Novelty / complexity (COMPUTED-only)                                 #
    # ------------------------------------------------------------------ #
    dup_status = ev.status_of("novelty.max_duplicate_corr")
    if dup_status == EvidenceStatusVocabulary.COMPUTED:
        dup_v = ev.value_of("novelty.max_duplicate_corr")
        if dup_v is not None:
            if dup_v >= th.duplicate_corr_min:
                _triggered(
                    out,
                    DiagnosisRuleRef(
                        tag="SEMANTIC_DUPLICATE",
                        rule_id="FI026_DUP_GE",
                        evidence_refs=("novelty.max_duplicate_corr.value",),
                    ),
                    evidence=ev, policy=policy, health_ref=health_ref,
                    factor_definition_id=factor_definition_id,
                )
            elif dup_v >= th.near_duplicate_corr_min:
                _triggered(
                    out,
                    DiagnosisRuleRef(
                        tag="VALUE_NEAR_DUPLICATE",
                        rule_id="FI026_NEAR_DUP_GE",
                        evidence_refs=("novelty.max_duplicate_corr.value",),
                    ),
                    evidence=ev, policy=policy, health_ref=health_ref,
                    factor_definition_id=factor_definition_id,
                )
    novelty_status = ev.status_of("novelty.novelty")
    if novelty_status == EvidenceStatusVocabulary.COMPUTED:
        novelty_v = ev.value_of("novelty.novelty")
        if novelty_v is not None and novelty_v < th.novelty_min:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="LOW_NOVELTY",
                    rule_id="FI026_NOVELTY_LT",
                    evidence_refs=("novelty.novelty.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )
    complex_status = ev.status_of("complexity.complexity_score")
    if complex_status == EvidenceStatusVocabulary.COMPUTED:
        complex_v = ev.value_of("complexity.complexity_score")
        if complex_v is not None and complex_v > th.complexity_max:
            _triggered(
                out,
                DiagnosisRuleRef(
                    tag="HIGH_COMPLEXITY",
                    rule_id="FI026_COMPLEX_GT",
                    evidence_refs=("complexity.complexity_score.value",),
                ),
                evidence=ev, policy=policy, health_ref=health_ref,
                factor_definition_id=factor_definition_id,
            )

    # Deduplicate (a tag may legitimately fire from multiple rules — e.g.
    # OVERFIT_GENERALIZATION — keep the highest-confidence record only).
    best: dict[str, DiagnosisTag] = {}
    for tag in out:
        prev = best.get(tag.tag)
        if prev is None or tag.confidence > prev.confidence:
            best[tag.tag] = tag
    ordered = tuple(best[t] for t in DIAGNOSIS_TAGS if t in best)
    return ordered


def diagnose_factor(
    *,
    factor_definition_id: str,
    health_ref: str = "",
    evidence: Mapping[str, object] | None = None,
    policy: DiagnosisPolicy | None = None,
    unknown_when_no_diagnosis: bool = False,
) -> tuple[DiagnosisTag, ...]:
    """Thin wrapper over :func:`detect_diagnoses`.

    ``unknown_when_no_diagnosis`` is accepted for call-site compatibility but
    deliberately **does not fabricate a sentinel diagnosis**: the FA diagnosis
    contract treats an empty sequence as a legal healthy factor (mirroring the
    FO ``DiagnosisView`` consumer boundary), and the ``UNKNOWN`` severity token
    only appears on genuine diagnosis records.  This wrapper therefore returns
    the empty tuple when no deterministic tag fired — it never invents a defect
    tag or a fake unknown record.
    """
    return detect_diagnoses(
        factor_definition_id=factor_definition_id,
        health_ref=health_ref,
        evidence=evidence,
        policy=policy,
    )
