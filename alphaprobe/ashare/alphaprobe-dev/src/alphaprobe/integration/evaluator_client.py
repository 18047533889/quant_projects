"""integration.evaluator_client（任务书 §8 / §7 / §89.2）。

legacy_compat adapter：暂包现有 fe_bridge 指标（quant_evaluator 尚未接线），
上层代码只依赖 EvaluatorClient protocol；切换真 Evaluator 时搜索内核不改。
硬规则：Pool/Logger/Checkpoint/Exporter 全消费同一 EvaluationRecord（once-compute）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from alphaprobe.contracts import (
    EvaluationRecord,
    ExperimentContext,
    FactorCandidate,
    FidelityLevel,
)
from alphaprobe.evalcache import EvaluationCache, build_cache_key
from alphaprobe.research_protocol import LeakageGuard


class LegacyCompatEvaluator:
    """§1.3 临时 adapter，标记 legacy_compat，不得扩展成第二套评估库。

    evaluate_fn: (formulas, fidelity, context) -> list[dict[str, float|None]]
    由调用方注入当前真实指标计算（如 fe_bridge/metrics）。
    """

    def __init__(
        self,
        evaluate_fn: Any,
        *,
        context: ExperimentContext,
        cache: EvaluationCache | None = None,
        leakage_guard: LeakageGuard | None = None,
        evaluator_version: str = "legacy_compat_v0",
    ) -> None:
        self._evaluate_fn = evaluate_fn
        self.context = context
        self.cache = cache or EvaluationCache()
        self.guard = leakage_guard or LeakageGuard(context.split_spec)
        self.evaluator_version = evaluator_version

    def evaluate(
        self,
        candidates: list[FactorCandidate],
        *,
        profile: str = "search",
        fidelity: FidelityLevel = FidelityLevel.L2_FULL_TRAIN,
        context: ExperimentContext | None = None,
    ) -> list[EvaluationRecord]:
        ctx = context or self.context
        # §3.3：L0-L4 禁触 sealed test
        self.guard.assert_access_allowed(
            stage=fidelity,
            segment=ctx.split_spec.sealed_test if ctx.split_spec else "1970-01-01..2999-12-31",
            caller="LegacyCompatEvaluator.evaluate",
        )
        records: list[EvaluationRecord] = []
        # 批量去重：同 cache key 只算一次（§89.2）
        todo: list[tuple[FactorCandidate, str]] = []
        for c in candidates:
            key = self._key(c, fidelity, ctx)
            rec = self.cache.get(key)
            if rec is None:
                todo.append((c, key))
        if todo:
            metrics_list = self._evaluate_fn(
                [c.identity.canonical_formula for c, _ in todo],
                fidelity=fidelity,
                context=ctx,
            ) or [None] * len(todo)
            for (c, key), mb in zip(todo, metrics_list):
                rec = self._record(c, fidelity, ctx, mb)
                self.cache.put(key, rec)
        for c in candidates:
            rec = self.cache.get(self._key(c, fidelity, ctx))
            records.append(rec)
        return records

    def _key(self, c: FactorCandidate, fidelity: FidelityLevel, ctx: ExperimentContext) -> str:
        return build_cache_key(
            canonical_formula_hash=c.identity.canonical_ast_hash,
            orientation=c.identity.orientation,
            data_snapshot_id=ctx.data_snapshot_id,
            universe_snapshot_id=ctx.universe_snapshot_id,
            factor_engine_version=ctx.factor_engine_version,
            operator_semantics_version=ctx.operator_semantics_version,
            label_spec_hash=ctx.label_spec.hash_key,
            segment="train",
            fidelity=fidelity.value,
            evaluator_version=self.evaluator_version,
        )

    def _record(
        self,
        c: FactorCandidate,
        fidelity: FidelityLevel,
        ctx: ExperimentContext,
        metric_bundle: dict[str, float | None] | None,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            factor_id=c.identity.factor_id,
            segment="train",
            fidelity=fidelity.value,
            metric_bundle=dict(metric_bundle or {}),
            artifact_refs={},
            evaluator_version=self.evaluator_version,
            data_snapshot_id=ctx.data_snapshot_id,
            universe_snapshot_id=ctx.universe_snapshot_id,
            label_spec_hash=ctx.label_spec.hash_key,
            created_at=datetime.now(timezone.utc),
        )

    def get_artifact(self, artifact_id: str) -> Any:
        return self.cache.get(artifact_id)


def metric_of(record: EvaluationRecord, name: str) -> float | None:
    """统一取数入口：Pool/Logger/Exporter 只读 record，不重算。"""
    return record.metric_bundle.get(name)