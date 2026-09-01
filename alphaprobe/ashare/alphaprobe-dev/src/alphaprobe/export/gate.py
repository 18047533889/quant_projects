"""export.gate（任务书 §58 / §59）：ExportDecision 薄 wrapper。

decide_export 已在 export/__init__.py；此模块补：
build_export_decision(candidate, record, store, refinement_complete=False)
从 GlobalMemoryStore 查 already_exported / seed 判定并返回 ExportDecision。
refinement 未接外部引擎时 refinement_complete=False → 不 accepted（任务书 §59 正确行为）。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.contracts import (
    EvaluationRecord,
    ExportDecision,
    FactorCandidate,
    RejectionReason,
)
from alphaprobe.export import decide_export

__all__ = ["build_export_decision", "EXPORT_DEFAULT_FLOOR", "EXPORT_DEFAULT_TRADABILITY"]

EXPORT_DEFAULT_FLOOR = 0.0
EXPORT_DEFAULT_TRADABILITY = 1.0


def _factor_id_of(candidate: Any) -> str:
    identity = getattr(candidate, "identity", None)
    if identity is not None:
        return getattr(identity, "factor_id", "") or ""
    if isinstance(candidate, dict):
        return str(candidate.get("factor_id", ""))
    return ""


def _canonical_formula_of(candidate: Any) -> str:
    identity = getattr(candidate, "identity", None)
    if identity is not None:
        return getattr(identity, "canonical_formula", "") or ""
    if isinstance(candidate, dict):
        return str(candidate.get("canonical_formula", ""))
    return ""


def build_export_decision(
    candidate: Any,
    record: EvaluationRecord,
    store: Any,
    *,
    refinement_complete: bool = False,
    audit_pass: bool = True,
    novelty_ok: bool = True,
    export_floor: float = EXPORT_DEFAULT_FLOOR,
    tradability: float | None = None,
    stability: float | None = None,
    hard_gate_failures: list[str] | None = None,
) -> ExportDecision:
    """从 GlobalMemoryStore 查 already_exported / seed 判定并构建 ExportDecision。

    store: GlobalMemoryStore（需要 .already_exported()/.get_node()/.is_seed()）。

    行为（任务书 §59）：
    - refinement 未接外部引擎时 refinement_complete=False → 一律不 accepted；
    - store 中已导出的 factor → ALREADY_EXPORTED；
    - store 中标记 seed 的 factor → SEED_LIBRARY_DUPLICATE。
    """
    factor_id = _factor_id_of(candidate) or record.factor_id
    reasons: list[str] = list(hard_gate_failures or [])

    already_exported = False
    seed_duplicate = False
    if store is not None:
        try:
            already_exported = bool(store.already_exported(factor_id))
        except Exception:  # noqa: BLE001 - store 部分实现无该方法时视作未导出
            already_exported = False
        try:
            node = store.get_node(factor_id)
            seed_duplicate = bool(node and node.get("is_seed", 0))
        except Exception:  # noqa: BLE001
            seed_duplicate = False
    if already_exported:
        reasons.append(RejectionReason.ALREADY_EXPORTED.value)
    if seed_duplicate:
        reasons.append(RejectionReason.SEED_LIBRARY_DUPLICATE.value)

    decision = decide_export(
        candidate if isinstance(candidate, FactorCandidate) else _candidate_as_fc(candidate, factor_id),
        record,
        hard_gate_failures=reasons,
        refinement_complete=refinement_complete,
        audit_pass=audit_pass,
        novelty_ok=novelty_ok,
        seed_duplicate=seed_duplicate,
        already_exported=already_exported,
        export_floor=export_floor,
        tradability=tradability,
        stability=stability,
    )
    return decision


def _candidate_as_fc(candidate: Any, factor_id: str) -> FactorCandidate:
    """把裸 dict 升格为 FactorCandidate，供 decide_export 使用。"""
    from alphaprobe.contracts import FactorCandidate, FactorIdentity

    formula = _canonical_formula_of(candidate) or factor_id
    identity = FactorIdentity(
        factor_id=factor_id,
        canonical_formula=formula,
        canonical_ast_hash=formula,
        signal_equivalence_id=formula,
    )
    return FactorCandidate(identity=identity, source="mined")
