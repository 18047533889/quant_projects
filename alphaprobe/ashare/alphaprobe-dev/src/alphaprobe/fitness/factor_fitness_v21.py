"""FactorFitness V2.1 —— FactorFitness V2 扩展入口（plan Task 8）。

把 V2.1 的可选扩展（metric requirement missing policy / confidence shrinkage /
soft floors）全部作为 **带默认值的新参数** 挂在 V2 主函数之外的新函数
``factor_fitness_v2_1`` 上，避免破坏既有 V2 调用。

- ``factor_fitness_v2``：保持 V2 原公式不动（真零行为兼容，既有测试与调用方
  不受影响）。
- ``factor_fitness_v2_1``：在 V2 之上可选叠加 Task 8 三项扩展（默认全部关闭时
  结果与 V2 逐字一致）。

公式（F1）：

    Core    = 0.20P + 0.12Q + 0.35L + 0.15S + 0.10N + 0.08R   （P/Q/L/S/N/R 各自
              dimension 内先做 requirement-missing policy + confidence shrinkage）
    Fitness = clip(Core × gP × gL × gS × gR - penalties, 0, 1)

soft guard gX（X ∈ P/L/S/R）：平滑可配置，非隐藏硬门；X ≥ floor → 1.0，
X → 0 时 gX → 0，floor 越低惩罚越轻。很强 P 无法完全抵消灾难性 L（G #21）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from alphaprobe.fitness.confidence import DEFAULT_SHRINKAGE_K, shrink_dimension
from alphaprobe.fitness.contracts import (
    ComplexityInfo,
    EvaluationBundle,
    FactorFitnessResult,
    MetricRequirement,
    MetricRequirementPolicy,
    PenaltyReport,
)
from alphaprobe.fitness.penalties import (
    DEFAULT_COMPLEXITY_CONFIG,
    DEFAULT_COST_CONFIG,
    DEFAULT_FRAGILITY_CONFIG,
    ComplexityPenaltyConfig,
    CostPenaltyConfig,
    FragilityPenaltyConfig,
    compute_all_penalties,
)
from alphaprobe.fitness.components import (
    DEFAULT_LWEIGHTS,
    DEFAULT_NWEIGHTS,
    DEFAULT_PWEIGHTS,
    DEFAULT_QWEIGHTS,
    DEFAULT_RWEIGHTS,
    DEFAULT_SWEIGHTS,
    LWeights,
    NWeights,
    PWeights,
    QWeights,
    RWeights,
    SWeights,
)

EPS = 1e-9


# ---------------------------------------------------------------------------
# 默认配置（Part J3 中性默认值）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V21GuardFloors:
    """各 soft guard 的 floor（X ≥ floor → gX=1.0；X → 0 → gX → 0）。"""

    P: float = 0.0
    L: float = 0.0
    S: float = 0.0
    R: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"P": self.P, "L": self.L, "S": self.S, "R": self.R}


@dataclass(frozen=True)
class V21SoftFloorConfig:
    """soft floors 总配置。默认全 floor=0.0 → 各 guard 恒 1.0（与 V2 逐字一致）。

    仅当显式调用方（如 Part G #21 验收场景）传入 ``l_floor>0`` 时才激活灾难性 L
    压制。guard floor 全部可配置；默认中性，不宣称最优（Part J3）。
    """

    guard_floor: V21GuardFloors = V21GuardFloors(P=0.0, L=0.0, S=0.0, R=0.0)
    l_floor: float = 0.0

    def floor_for(self, dim: str) -> float:
        d = self.guard_floor.as_dict()
        if dim in d:
            return d[dim]
        if dim == "L":
            return self.l_floor
        return 0.0


DEFAULT_V21_SOFT_FLOORS = V21SoftFloorConfig()
DEFAULT_V21_GUARD_FLOORS = DEFAULT_V21_SOFT_FLOORS.guard_floor


def guard_g(utility: float, *, floor: float, steepness: float = 1.0) -> float:
    """平滑 soft guard（非硬门）。

    - utility >= floor → g = 1.0（无影响，V2 排序保持）。
    - 0 < utility < floor → g = (u/floor)^steepness 连续降到 0（平滑单调）。
    - utility -> 0 → g -> 0（灾难性维度把乘积压到 0）。
    - floor <= 0 → guard 关闭（恒 1.0，与 V2 逐字一致）。
    """
    u = max(0.0, min(1.0, float(utility)))
    fl = max(0.0, min(1.0, float(floor)))
    if fl <= EPS or u >= fl:
        return 1.0
    ratio = max(0.0, u / fl)
    return ratio ** max(steepness, EPS)


def product_guards(
    *,
    P: float, L: float, S: float, R: float,
    floors: V21GuardFloors = DEFAULT_V21_GUARD_FLOORS,
) -> tuple[float, float, float, float]:
    """返回 (gP, gL, gS, gR)。floors 全 0 → (1,1,1,1)。"""
    return (
        guard_g(P, floor=floors.P),
        guard_g(L, floor=floors.L),
        guard_g(S, floor=floors.S),
        guard_g(R, floor=floors.R),
    )


def _effective_n_for(
    dimension_n_eff: Mapping[str, Any] | None,
    *,
    key: str,
    metric_n_keys: tuple[str, ...],
    bundle: EvaluationBundle | None = None,
) -> float | None:
    """解析某维度的有效样本数。

    优先级：
      1. ``dimension_n_eff`` 显式 dict（{维度大写键: n_eff}）；
      2. bundle 里的 ``metric_n_keys``（n_eff 元数据键，如 n_ls_days）；
      3. 都没有 → None（不编造样本量；按 requirement 缺省语义处理）。
    """
    if dimension_n_eff:
        v = dimension_n_eff.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    if bundle is not None:
        for nk in metric_n_keys:
            v = bundle.get(nk)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


def factor_fitness_v2_1(
    bundle: EvaluationBundle | Mapping[str, Any],
    calibrator: Any,
    *,
    complexity: ComplexityInfo | None = None,
    weights: V2FactorWeights | None = None,
    cost_config: CostPenaltyConfig = DEFAULT_COST_CONFIG,
    complexity_config: ComplexityPenaltyConfig = DEFAULT_COMPLEXITY_CONFIG,
    fragility_config: FragilityPenaltyConfig = DEFAULT_FRAGILITY_CONFIG,
    d10_tau: float = 0.12,
    # --- V2.1 可选扩展（默认关闭/中性，行为与 V2 一致）---
    requirement_policy: MetricRequirementPolicy | None = None,
    dimension_requirement: Mapping[str, MetricRequirement] | None = None,
    dimension_n_eff: Mapping[str, Any] | None = None,
    metric_n_keys: Mapping[str, tuple[str, ...]] | None = None,
    k: float = DEFAULT_SHRINKAGE_K,
    apply_confidence_shrinkage: bool = False,
    soft_floor: V21SoftFloorConfig | None = None,
    apply_soft_floors: bool = False,
) -> FactorFitnessResult:
    """FactorFitness V2.1（plan Task 8）。

    六维分数各自先做 requirement missing policy（V2 原 utility 计算 + 缺失语义
    修正），再按需做 confidence shrinkage；随后 Core 加权，乘 soft guards
    （P/L/S/R），最后减三惩罚并 clip。

    兼容性：所有 V2.1 参数都有默认值；不传时本函数与 ``factor_fitness_v2``
    结果一致（P/S/R 维度 guard floor=0 → g=1；无 n_eff、无 requirement → 0 影响）。
    """
    policy = _resolve_policy(requirement_policy, dimension_requirement)
    from alphaprobe.fitness.factor_fitness import factor_fitness_v2, V2FactorWeights

    if weights is None:
        weights = V2FactorWeights()
    w = weights

    # 先算一次 V2（拿六维原始 utility；rejected/penalty 语义完全复用）
    base = factor_fitness_v2(
        bundle,
        calibrator,
        complexity=complexity,
        weights=w,
        cost_config=cost_config,
        complexity_config=complexity_config,
        fragility_config=fragility_config,
        d10_tau=d10_tau,
    )
    if base.rejected:
        return base

    dims = ("P", "Q", "L", "S", "N", "R")
    raw = {"P": base.P, "Q": base.Q, "L": base.L, "S": base.S,
           "N": base.N, "R": base.R}

    # 1) requirement missing policy + 2) confidence shrinkage
    adj: dict[str, float] = {}
    for dim in dims:
        u = raw[dim]
        req = policy.requirement_for(dim) if policy is not None else MetricRequirement.DIAGNOSTIC
        n_eff = _effective_n_for(
            dimension_n_eff,
            key=dim,
            metric_n_keys=dict(metric_n_keys or {}).get(dim, ()),
            bundle=bundle,
        )
        if not apply_confidence_shrinkage:
            # 无收缩：OPTIONAL 缺失仍要 0.5（missing policy 独立生效）
            if req == MetricRequirement.OPTIONAL and n_eff is None:
                u = 0.5
            adj[dim] = u
            continue
        adj[dim] = shrink_dimension(u, n_eff=n_eff, k=k, requirement=req)

    # Core = 加权六维（公式主结构不动）
    core = (
        w.P * adj["P"]
        + w.Q * adj["Q"]
        + w.L * adj["L"]
        + w.S * adj["S"]
        + w.N * adj["N"]
        + w.R * adj["R"]
    )

    # 3) soft guards（默认 P/S/R floor 0 → 恒 1；L floor 0.20 仅在 apply 时生效）
    sfl = soft_floor if soft_floor is not None else DEFAULT_V21_SOFT_FLOORS
    g = product_guards(
        P=adj["P"], L=adj["L"], S=adj["S"], R=adj["R"],
        floors=sfl.guard_floor if apply_soft_floors else V21GuardFloors(),
    )
    guarded = core * g[0] * g[1] * g[2] * g[3]

    total_pen = base.cost_penalty + base.complexity_penalty + base.fragility_penalty
    fitness = max(0.0, min(1.0, guarded - total_pen))
    return FactorFitnessResult(
        fitness=fitness,
        core=core,
        P=adj["P"], Q=adj["Q"], L=adj["L"], S=adj["S"], N=adj["N"], R=adj["R"],
        cost_penalty=base.cost_penalty,
        complexity_penalty=base.complexity_penalty,
        fragility_penalty=base.fragility_penalty,
        rejected=False,
        reject_reason="",
    )


def _effective_n_for(
    dimension_n_eff: Mapping[str, Any] | None,
    *,
    key: str,
    metric_n_keys: tuple[str, ...],
    bundle: EvaluationBundle | None = None,
) -> float | None:
    """见 factor_fitness._effective_n_for（re-export 便于 v21 模块独立使用）。"""
    from alphaprobe.fitness.factor_fitness import _effective_n_for as _impl

    return _impl(dimension_n_eff, key=key, metric_n_keys=metric_n_keys, bundle=bundle)


def _resolve_policy(
    requirement_policy: MetricRequirementPolicy | None,
    dimension_requirement: Mapping[str, MetricRequirement] | None,
) -> MetricRequirementPolicy | None:
    if requirement_policy is not None:
        return requirement_policy
    if dimension_requirement:
        return MetricRequirementPolicy(dimension_requirement)
    return None


def factor_fitness_v2_1_from_v2(
    base: FactorFitnessResult,
    *,
    calibrator: Any,
    bundle: EvaluationBundle,
    weights: V2FactorWeights | None = None,
    requirement_policy: MetricRequirementPolicy | None = None,
    dimension_requirement: Mapping[str, MetricRequirement] | None = None,
    dimension_n_eff: Mapping[str, Any] | None = None,
    metric_n_keys: Mapping[str, tuple[str, ...]] | None = None,
    k: float = DEFAULT_SHRINKAGE_K,
    apply_confidence_shrinkage: bool = False,
    soft_floor: V21SoftFloorConfig = DEFAULT_V21_SOFT_FLOORS,
    apply_soft_floors: bool = False,
) -> FactorFitnessResult:
    """从已算好的 V2 结果叠加 V2.1 扩展（rejected 语义直接透传）。"""
    if base.rejected:
        return base
    return factor_fitness_v2_1(
        bundle,
        calibrator,
        complexity=None,
        weights=weights,
        requirement_policy=requirement_policy,
        dimension_requirement=dimension_requirement,
        dimension_n_eff=dimension_n_eff,
        metric_n_keys=metric_n_keys,
        k=k,
        apply_confidence_shrinkage=apply_confidence_shrinkage,
        soft_floor=soft_floor,
        apply_soft_floors=apply_soft_floors,
    )
