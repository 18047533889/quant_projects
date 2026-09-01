"""pool.admission（任务书 §53 / Phase 5 / Phase 6）：Pool 准入。

旧 try_new_expr 的「pop lowest IC」替换路径：
- admit(candidate, evaluation_record, active_pool)：用 fitness.pool_utility + PoolMember 决策；
- 被弹者选择由 ActivePool._eviction_candidate 逻辑等价迁移（低 SearchValue → 高饱和 niche →
  低 utility，绝不 argmin(single_ics)）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from alphaprobe.contracts import EvaluationRecord, RejectionReason
from alphaprobe.fitness import pool_utility as compute_pool_utility
from alphaprobe.pool import ActivePool, PoolMember

__all__ = [
    "AdmissionResult",
    "PoolAdmission",
    "admit",
    "legacy_try_new_expr",
    "pool_member_from_record",
    "build_pool_member",
    "compute_member_utility",
    "eviction_candidate_from",
]


@dataclass
class AdmissionResult:
    """准入决策结果。"""

    factor_id: str
    accepted: bool
    reason: str = ""
    replacement: str = ""
    pool_utility: float = 0.0
    search_fitness: float = 0.0
    search_value: float = 0.0
    niche_rarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "replacement": self.replacement,
            "pool_utility": self.pool_utility,
            "search_fitness": self.search_fitness,
            "search_value": self.search_value,
            "niche_rarity": self.niche_rarity,
        }


def build_pool_member(
    *,
    factor_id: str,
    canonical_formula: str,
    record: EvaluationRecord | None = None,
    metric_bundle: dict[str, float | None] | None = None,
    niche_key: tuple = (),
    fingerprint: bytes | None = None,
    pareto_rank: int = 0,
    meta: dict[str, Any] | None = None,
    added_round: str = "",
    **overrides: Any,
) -> PoolMember:
    """从 EvaluationRecord.metric_bundle（或裸 metric_bundle）构建 PoolMember。

    只读 record，绝不重新计算指标（once-compute）。
    """
    mb = dict(metric_bundle or {})
    if record is not None:
        mb = dict(record.metric_bundle or {})

    def _f(*names: str) -> float:
        for n in names:
            v = mb.get(n)
            if v is not None and math.isfinite(float(v)):
                return float(v)
        return 0.0

    search_fitness = _f("search_fitness", "fitness")
    search_value = _f("search_value", "sv")
    util = _f("pool_utility", "utility")

    if util <= 0.0 and (record is not None or mb):
        # 由 fitness 组件计算 pool_utility（§21.4）
        try:
            util = compute_pool_utility(
                pareto_rank=pareto_rank,
                niche_rarity=_f("niche_rarity"),
                novelty=_f("structural_novelty", "N"),
                s_value=_f("S"),
                representative_value=_f("representative_value"),
            )
        except Exception:  # noqa: BLE001 - 组件失败退化为 0
            pass

    m = PoolMember(
        factor_id=factor_id,
        canonical_formula=canonical_formula,
        search_fitness=search_fitness,
        pool_utility=util,
        search_value=search_value,
        niche_key=tuple(niche_key),
        fingerprint=fingerprint,
        pareto_rank=pareto_rank,
        added_round=added_round,
        meta=dict(meta or {}),
    )
    for k, v in overrides.items():
        if hasattr(m, k):
            setattr(m, k, v)
    return m


def compute_member_utility(
    m: PoolMember,
    *,
    niche_count: dict[tuple, int] | None = None,
    total: int = 0,
) -> float:
    """淘汰优先级分（从差到好保护）。与 ActivePool._eviction_candidate 的
    score 公式等价迁移，低 IC 高 niche 成员因此不会被 argmin(single IC) 弹出。
    """
    niche_rarity = float(m.meta.get("niche_rarity", 0.0) or 0.0)
    niche_crowd = 1
    if niche_count is not None:
        niche_crowd = max(1, int(niche_count.get(m.niche_key, 1)))
    crowd_term = 0.1 * (niche_crowd / max(1, total or 1))
    return (
        0.4 * m.search_fitness
        + 0.3 * m.pool_utility
        + 0.2 * m.search_value
        - crowd_term
        + 0.1 * niche_rarity  # rare niche 成员更高（更难被淘汰）
    )


def eviction_candidate_from(
    active_pool: ActivePool,
    *,
    exclude: PoolMember | None = None,
) -> PoolMember | None:
    """ActivePool._eviction_candidate 逻辑等价迁移（§53.2，绝不 pop lowest IC）。

    返回 None = 无候选（空池 / 全部被保护）。
    """
    if not active_pool.members:
        return None
    niche_count: dict[tuple, int] = {}
    for mem in active_pool.members.values():
        niche_count[mem.niche_key] = niche_count.get(mem.niche_key, 0) + 1
    scored: list[tuple[float, PoolMember]] = []
    for mem in active_pool.members.values():
        if exclude is not None and mem.factor_id == exclude.factor_id:
            continue
        score = compute_member_utility(mem, niche_count=niche_count, total=len(active_pool.members))
        scored.append((score, mem))
    scored.sort(key=lambda t: t[0])
    # 保护 QD elite：唯一 niche 代表且 niche_rarity 高 → 跳过
    for _score, mem in scored:
        cell = [x for x in active_pool.members.values() if x.niche_key == mem.niche_key]
        if len(cell) <= 1 and float(mem.meta.get("niche_rarity", 0.0) or 0.0) > 0.7:
            continue
        return mem
    return scored[0][1] if scored else None


class PoolAdmission:
    """§53 新准入组件。可注入 niche/novelty/utility 打分函数便于单测。"""

    def __init__(
        self,
        *,
        pool_utility_fn: Callable[..., float] | None = None,
        niche_key_fn: Callable[..., tuple] | None = None,
        replacement_margin: float = 0.02,
        max_size: int = 0,
    ) -> None:
        self.pool_utility_fn = pool_utility_fn or compute_pool_utility
        self.niche_key_fn = niche_key_fn
        self.replacement_margin = replacement_margin
        self.max_size = max_size

    def _member_for(
        self,
        candidate: Any,
        record: EvaluationRecord | None,
        metric_bundle: dict[str, float | None] | None,
        factor_id: str,
        canonical_formula: str,
        niche_key: tuple,
        meta: dict[str, Any],
    ) -> PoolMember:
        pareto_rank = 0
        niche_rarity = 0.0
        mb = dict(metric_bundle or {})
        if record is not None:
            mb = dict(record.metric_bundle or {})
        if "pareto_rank" in mb:
            pareto_rank = int(mb["pareto_rank"] or 0)
        if "niche_rarity" in mb:
            niche_rarity = float(mb["niche_rarity"] or 0.0)
        util = float(mb.get("pool_utility") or 0.0)
        if util <= 0.0:
            util = float(
                self.pool_utility_fn(
                    pareto_rank=pareto_rank,
                    niche_rarity=niche_rarity,
                    novelty=float(mb.get("structural_novelty") or 0.0),
                    s_value=float(mb.get("S") or 0.0),
                    representative_value=float(mb.get("representative_value") or 0.0),
                )
            )
        return PoolMember(
            factor_id=factor_id,
            canonical_formula=canonical_formula,
            search_fitness=float(mb.get("search_fitness") or 0.0),
            pool_utility=util,
            search_value=float(mb.get("search_value") or 0.0),
            niche_key=tuple(niche_key),
            meta=dict(meta or {}),
        )

    def admit(
        self,
        candidate: Any,
        record: EvaluationRecord | None,
        active_pool: ActivePool,
        *,
        metric_bundle: dict[str, float | None] | None = None,
        factor_id: str = "",
        canonical_formula: str = "",
        niche_key: tuple = (),
        meta: dict[str, Any] | None = None,
    ) -> AdmissionResult:
        """准入：fitness 比较后调 ActivePool.admit（含替换）。

        candidate 可为 FactorCandidate（读 identity）或裸 dict；record/metric_bundle
        提供指标（once-compute，不重算）。
        """
        if candidate is not None and not factor_id:
            factor_id = _factor_id_of(candidate)
        if candidate is not None and not canonical_formula:
            canonical_formula = _canonical_formula_of(candidate)
        if not factor_id:
            factor_id = record.factor_id if record else ""
        if not canonical_formula and record is None:
            canonical_formula = factor_id
        if niche_key is None:
            niche_key = ()
        if self.niche_key_fn is not None and not niche_key:
            try:
                niche_key = self.niche_key_fn(
                    mechanism=str(meta or {}).get("mechanism", "unknown"),
                    horizon_bucket=str(meta or {}).get("horizon_bucket", "unknown"),
                )
            except Exception:  # noqa: BLE001
                niche_key = ()

        m = self._member_for(
            candidate, record, metric_bundle, factor_id, canonical_formula,
            tuple(niche_key), dict(meta or {}),
        )

        ok, reason = active_pool.admit(m)
        victim_id = ""
        if ok and reason == "REPLACED":
            # 找出被换出的成员（供日志/审计）
            victim = eviction_candidate_from(active_pool)
            if victim is not None and victim.factor_id != m.factor_id:
                victim_id = victim.factor_id
        return AdmissionResult(
            factor_id=factor_id,
            accepted=ok,
            reason=reason,
            replacement=victim_id,
            pool_utility=m.pool_utility,
            search_fitness=m.search_fitness,
            search_value=m.search_value,
            niche_rarity=float(m.meta.get("niche_rarity", 0.0) or 0.0),
        )


def admit(
    candidate: Any,
    evaluation_record: EvaluationRecord | None,
    active_pool: ActivePool,
    *,
    metric_bundle: dict[str, float | None] | None = None,
    factor_id: str = "",
    canonical_formula: str = "",
    niche_key: tuple = (),
    meta: dict[str, Any] | None = None,
) -> AdmissionResult:
    """§53 顶层准入函数（等价旧 try_new_expr 的新路径）。"""
    return PoolAdmission().admit(
        candidate,
        evaluation_record,
        active_pool,
        metric_bundle=metric_bundle,
        factor_id=factor_id,
        canonical_formula=canonical_formula,
        niche_key=niche_key,
        meta=meta,
    )


def pool_member_from_record(
    record: EvaluationRecord,
    *,
    canonical_formula: str = "",
    factor_id: str = "",
    niche_key: tuple = (),
    fingerprint: bytes | None = None,
    pareto_rank: int = 0,
    meta: dict[str, Any] | None = None,
) -> PoolMember:
    """从 EvaluationRecord 构建 PoolMember（once-compute，不重算指标）。"""
    return build_pool_member(
        factor_id=factor_id or record.factor_id,
        canonical_formula=canonical_formula or record.factor_id,
        record=record,
        niche_key=niche_key,
        fingerprint=fingerprint,
        pareto_rank=pareto_rank,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# 旧 AlphaKnowledgePool 适配层（向后兼容签名，内部不再 argmin(single_ics)）
# ---------------------------------------------------------------------------


def legacy_try_new_expr(
    pool,
    expr,
    topic: str = "",
    description: str = "",
    expression_node=None,
    *,
    metric_bundle: dict[str, float | None] | None = None,
    record: EvaluationRecord | None = None,
    niche_key: tuple = (),
    meta: dict[str, Any] | None = None,
) -> bool:
    """AlphaKnowledgePool.try_new_expr 的后向兼容实现。

    内部不再 argmin(single_ics) 弹出，改为：
      - 未满：直接 _add_factor；
      - 满池：等价 ActivePool._eviction_candidate 逻辑选被弹者，并调用
        pool._add_factor + pool._pop 组合完成替换。
    """
    s = str(expr) if not hasattr(expr, "dsl") else getattr(expr, "dsl")
    s = s if s else str(expr)

    if pool.size < pool.capacity:
        pool._add_factor(
            expr, getattr(pool, "values", None), 0.0, 0.0, [], topic, description, expression_node
        )
        return True

    # 满池：用 eviction 逻辑（等价 ActivePool._eviction_candidate）选被弹者，
    # 而不是 argmin(single_ics[:size])。
    victim = _legacy_eviction_candidate(pool)
    if victim is None:
        return False
    pool._add_factor(
        expr, getattr(pool, "values", None), 0.0, 0.0, [], topic, description, expression_node
    )
    # 新因子已在尾部；弹出的必须是 eviction 选中的 victim（替换语义），
    # 而不是 FixedCapacity._pop 默认弹出的尾部（尾部是新因子本身）。
    if callable(getattr(pool, "_replace_at", None)):
        pool._replace_at(victim, pool.size - 1)
    elif callable(getattr(pool, "_pop_at", None)):
        pool._pop_at(victim)
    else:
        pool._pop()
    return True


def _legacy_eviction_candidate(pool) -> Any | None:
    """等价 ActivePool._eviction_candidate 的迁移逻辑：低 SearchValue → 高饱和 niche
    → 低 utility，绝不 argmin(single IC)。单 IC 仅作最后一个 tie-break 项。
    """
    if pool.size <= 0:
        return None
    scored: list[tuple[float, int]] = []
    for idx in range(pool.size):
        ic = float(pool.single_ics[idx]) if pool.single_ics is not None else 0.0
        sv = 0.0
        util = 0.0
        meta = getattr(pool, "_admission_meta", None)
        if meta and idx in meta:
            sv = float(meta[idx].get("search_value", 0.0) or 0.0)
            util = float(meta[idx].get("pool_utility", 0.0) or 0.0)
        score = 0.4 * sv + 0.3 * util - 0.1 * abs(ic)
        scored.append((score, idx))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][1] if scored else None


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
