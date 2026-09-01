"""export（任务书 §58 / §58.1 / §20 / §94.8）：Export Gate 重写。

Exporter 不再算指标：EvaluationRecord + RefinementResult + GlobalSeenIndex +
ExportRegistry → ExportDecision。不设 Top-N：0 合格 → 0，237 合格 → 237。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.contracts import (
    EvaluationRecord,
    ExportDecision,
    FactorCandidate,
    RejectionReason,
)
from alphaprobe.fitness import export_score as compute_export_score


def decide_export(
    candidate: FactorCandidate,
    evaluation: EvaluationRecord,
    *,
    hard_gate_failures: list[str],
    refinement_complete: bool,
    audit_pass: bool,
    novelty_ok: bool,
    seed_duplicate: bool,
    already_exported: bool,
    export_floor: float,
    tradability: float | None = None,
    stability: float | None = None,
) -> ExportDecision:
    """§58.1：ExportEligible = HardGatePass AND ExportScore >= floor AND AuditPass
    AND Novel AND NotSeed AND NotPreviouslyExported AND RefinementComplete。"""
    reasons: list[str] = []
    hard_pass = not hard_gate_failures
    reasons.extend(hard_gate_failures)
    if not refinement_complete:
        reasons.append("REFINEMENT_INCOMPLETE")
    if not audit_pass:
        reasons.append("AUDIT_FAIL")
    if not novelty_ok:
        reasons.append(RejectionReason.HIGH_CORRELATION.value)
    if seed_duplicate:
        reasons.append(RejectionReason.SEED_LIBRARY_DUPLICATE.value)
    if already_exported:
        reasons.append(RejectionReason.ALREADY_EXPORTED.value)

    mb = evaluation.metric_bundle
    sf = mb.get("search_fitness")
    score = None
    if sf is not None:
        score = compute_export_score(
            search_fitness=float(sf),
            stability=stability if stability is not None else float(mb.get("S", 0.0) or 0.0),
            novelty=float(mb.get("N", 0.0) or 0.0),
            tradability=tradability if tradability is not None else 1.0,
        )
        if score < export_floor:
            reasons.append("EXPORT_SCORE_BELOW_FLOOR")

    accepted = (
        hard_pass
        and refinement_complete
        and audit_pass
        and novelty_ok
        and not seed_duplicate
        and not already_exported
        and score is not None
        and score >= export_floor
    )
    return ExportDecision(
        factor_id=candidate.identity.factor_id,
        hard_gate_pass=hard_pass,
        export_score=score,
        audit_pass=audit_pass,
        novel=novelty_ok,
        seed_duplicate=seed_duplicate,
        already_exported=already_exported,
        refinement_complete=refinement_complete,
        accepted=accepted,
        reasons=reasons,
    )