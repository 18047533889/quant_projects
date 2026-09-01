"""integration.refinement_client + consolidation_client（任务书 §59 / §67）。

AlphaPROBE 只做 client：真实 Refinement/Consolidation 引擎在外部公共包
（当前未落地 → 提供显式 NotImplemented stub + 可注入 adapter）。
"""

from __future__ import annotations

from typing import Any

from alphaprobe.contracts import EvaluationRecord, ExperimentContext, FactorCandidate


class ExternalRefinementNotAvailable(RuntimeError):
    pass


class ProtocolRefinementClient:
    """§59.1。refine_fn 注入外部实现；未注入时显式失败（不静默伪造）。"""

    def __init__(self, refine_fn: Any | None = None) -> None:
        self._refine_fn = refine_fn

    def refine(
        self,
        candidate: FactorCandidate,
        evaluation: EvaluationRecord,
        context: ExperimentContext,
        mode: str = "pre_export",
    ) -> Any:
        if self._refine_fn is None:
            raise ExternalRefinementNotAvailable(
                "公共 Factor Refinement Engine 未接入：AlphaPROBE 只持 client，"
                "禁止内置完整 refinement 实现（任务书 §1.4/§59）"
            )
        return self._refine_fn(candidate, evaluation, context, mode)


class ProtocolConsolidationClient:
    """§67：AlphaPROBE 不在内部聚类/PCA。未注入时返回空上下文（SearchValue 退化为无簇项）。"""

    def __init__(self, get_cluster_context_fn: Any | None = None) -> None:
        self._fn = get_cluster_context_fn

    def get_cluster_context(self, factor_ids: list[str]) -> dict[str, Any]:
        if self._fn is None:
            return {}
        return self._fn(factor_ids)