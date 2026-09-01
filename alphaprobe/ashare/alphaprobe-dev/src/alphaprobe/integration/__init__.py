"""Integration 层（任务书 §76）：FactorEngine / Evaluator / Refinement / Consolidation Client 协议。

AlphaPROBE 只定义 protocol + 临时 adapter（标记 legacy_compat），
不在内部复制 Evaluator 指标库 / Refinement Engine / Consolidation Engine。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from alphaprobe.contracts import (
    EvaluationRecord,
    ExperimentContext,
    FactorCandidate,
    FidelityLevel,
)


@runtime_checkable
class EvaluatorClient(Protocol):
    """§8 Unified Evaluator 接入协议。"""

    def evaluate(
        self,
        candidates: list[FactorCandidate],
        *,
        profile: str,
        fidelity: FidelityLevel,
        context: ExperimentContext,
    ) -> list[EvaluationRecord]: ...

    def get_artifact(self, artifact_id: str) -> Any: ...


@runtime_checkable
class RefinementClient(Protocol):
    """§59.1：AlphaPROBE 只实现 Client，不复制 refinement logic。"""

    def refine(
        self,
        candidate: FactorCandidate,
        evaluation: EvaluationRecord,
        context: ExperimentContext,
        mode: str = "pre_export",
    ) -> Any: ...


@runtime_checkable
class ConsolidationClient(Protocol):
    """§67：聚类/PCA/VIF/簇合成不进 AlphaPROBE。"""

    def get_cluster_context(
        self,
        factor_ids: list[str],
    ) -> dict[str, Any]: ...


@runtime_checkable
class FactorEngineAdapterProtocol(Protocol):
    """§3.1 新公式链唯一真值。"""

    def validate(self, formula: str, *, surface: str = "daily", mode: str = "research") -> Any: ...

    def canonicalize(self, formula: str) -> Any: ...

    def inspect(self, canonical: Any) -> Any: ...

    def run_many(self, formulas: list[Any], context: ExperimentContext) -> Any: ...