"""FactorFitness V2 —— penalties（三惩罚）。

CostPenalty / ComplexityPenalty / FragilityPenalty 全部配置化，不写死进逻辑。

- CostPenalty ∈ [0, cost_cap]，cap 默认 0.03（L 已用 Net 口径，禁止双重重罚）。
- ComplexityPenalty 按 FactorEngine AST Analyzer 输出的节点数/深度分级：
  节点 ≤24 → 0；25~40 → 至多 0.005；41~64 → 至多 0.015；>64 → hard reject。
  depth soft ≤10（超过按比例计入 penalty），hard reject ~12。
  只吃 AST 计数（nodes/depth），不用字符串长度。
- FragilityPenalty 连续 severity（废除 flag count×0.01）：
  λ1·WinsorSensitivity + λ2·ContributionConcentration + λ3·CoverageInstability
  + λ4·DenominatorRisk + λ5·ExtremeValueDependence，cap ≤ 0.05。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from alphaprobe.fitness.calibration import single_utility
from alphaprobe.fitness.contracts import ComplexityInfo, EvaluationBundle, PenaltyReport

#: ComplexityPenalty 节点数分级（max penalty per bucket）
_COMPLEXITY_BUCKETS: tuple[tuple[int, int, float], ...] = (
    (0, 24, 0.0),     # ≤24 → 0
    (25, 40, 0.005),  # 25~40 → 至多 0.005
    (41, 64, 0.015),  # 41~64 → 至多 0.015
)


@dataclass(frozen=True)
class ComplexityPenaltyConfig:
    node_buckets: tuple[tuple[int, int, float], ...] = _COMPLEXITY_BUCKETS
    hard_reject_nodes: int = 64  # >64 → hard reject（不进 fitness）
    depth_soft: int = 10
    depth_hard: int = 12
    depth_max_penalty: float = 0.02


@dataclass(frozen=True)
class CostPenaltyConfig:
    cap: float = 0.03
    # base cost 口径（SearchFitness 用 gross 或 1x；Stress 只作 audit）
    cost_key: str = "cost_1x"


@dataclass(frozen=True)
class FragilityPenaltyConfig:
    lambdas: Mapping[str, float] = field(
        default_factory=lambda: {
            "winsor_sensitivity": 1.0,
            "contribution_concentration": 1.0,
            "coverage_instability": 1.0,
            "denominator_risk": 1.0,
            "extreme_value_dependence": 1.0,
        }
    )
    cap: float = 0.05


DEFAULT_COST_CONFIG = CostPenaltyConfig()
DEFAULT_COMPLEXITY_CONFIG = ComplexityPenaltyConfig()
DEFAULT_FRAGILITY_CONFIG = FragilityPenaltyConfig()


def cost_penalty(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    config: CostPenaltyConfig = DEFAULT_COST_CONFIG,
) -> float:
    """CostPenalty = clip(cost_cap · U(turnover-ish cost), 0, cost_cap)。

    L 维度已经用 Net 口径（net_sharpe / net_annualized_ls_return），这里只轻罚
    base cost（gross 或 1x），cap 默认 0.03 防双重重罚。
    """
    raw = bundle.get(config.cost_key)
    if raw is None:
        raw = bundle.get("turnover")  # 缺省退化为 turnover
    if raw is None:
        return 0.0
    u = single_utility(calibrator, config.cost_key if bundle.get(config.cost_key) is not None else "turnover",
                       raw, lower_is_better=True)
    return max(0.0, min(config.cap, float(config.cap) * u))


def complexity_penalty(
    complexity: ComplexityInfo,
    *,
    config: ComplexityPenaltyConfig = DEFAULT_COMPLEXITY_CONFIG,
) -> PenaltyReport:
    """按 AST 节点数/深度算复杂度惩罚。

    Returns PenaltyReport（complexity_rejected=True 表示 nodes > 64 或 depth
    超硬限，调用方应 hard reject，不进 fitness）。
    """
    nodes = max(0, int(complexity.nodes or 0))
    depth = max(0, int(complexity.depth or 0))

    # >64 → hard reject
    if nodes > config.hard_reject_nodes:
        return PenaltyReport(
            cost=0.0, complexity=0.0, fragility=0.0,
            complexity_rejected=True,
            reject_reason=f"AST nodes {nodes} > {config.hard_reject_nodes} (hard reject)",
        )
    # depth 硬限 ~12
    if depth > config.depth_hard:
        return PenaltyReport(
            cost=0.0, complexity=0.0, fragility=0.0,
            complexity_rejected=True,
            reject_reason=f"AST depth {depth} > {config.depth_hard} (hard reject)",
        )

    # 节点分级
    cx = 0.0
    for lo, hi, cap in config.node_buckets:
        if lo <= nodes <= hi:
            # 桶内线性：0~0.005 / 0~0.015（无惩罚区 0）
            if cap <= 0.0:
                cx = 0.0
            else:
                span = max(1, hi - lo)
                cx = cap * (nodes - lo) / span
            break
    else:
        cx = 0.0

    # depth soft：>10 按比例计入
    if depth > config.depth_soft:
        cx = min(config.depth_max_penalty, cx + config.depth_max_penalty * (depth - config.depth_soft))
    return PenaltyReport(
        cost=0.0, complexity=max(0.0, min(0.05, cx)), fragility=0.0,
        complexity_rejected=False,
    )


def fragility_penalty(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    config: FragilityPenaltyConfig = DEFAULT_FRAGILITY_CONFIG,
) -> float:
    """连续 severity：λ1·WinsorSens + λ2·ContributionConc + λ3·CoverageInstab
    + λ4·DenominatorRisk + λ5·ExtremeValueDependence，cap ≤ 0.05。

    全部走 calibrator.utility(lower_is_better=True)：各子项 ∈ [0,1] 连续，缺项
    为 0（不因缺测告警而惩罚）。不用 flag 计数。

    severity = Σ(λ_i·u_i) / (mean_λ·n_defined) —— 定义项的平均 severity ∈ [0,1]；
    penalty = severity × cap，保证连续且不因项数放大。
    """
    total = 0.0
    seen = 0
    for key, lam in config.lambdas.items():
        v = bundle.get(key)
        if v is None:
            continue
        u = single_utility(calibrator, key, v, lower_is_better=True)
        total += float(lam) * u
        seen += 1
    if seen == 0:
        return 0.0
    # 归一化到 λ 平均（避免「5 个可用项全超」让单一缺失项放大）
    mean_lam = sum(float(lam) for lam in config.lambdas.values()) / max(1, len(config.lambdas))
    severity = total / max(mean_lam, 1e-9) / seen
    severity = max(0.0, min(1.0, severity))
    return max(0.0, min(config.cap, severity * config.cap))


def compute_all_penalties(
    bundle: EvaluationBundle,
    calibrator: Any,
    complexity: ComplexityInfo,
    *,
    cost_config: CostPenaltyConfig = DEFAULT_COST_CONFIG,
    complexity_config: ComplexityPenaltyConfig = DEFAULT_COMPLEXITY_CONFIG,
    fragility_config: FragilityPenaltyConfig = DEFAULT_FRAGILITY_CONFIG,
) -> PenaltyReport:
    """一次算出三惩罚。complexity_rejected=True → 调用方应 hard reject。"""
    cx = complexity_penalty(complexity, config=complexity_config)
    if cx.complexity_rejected:
        return cx
    cost = cost_penalty(bundle, calibrator, config=cost_config)
    frag = fragility_penalty(bundle, calibrator, config=fragility_config)
    return PenaltyReport(
        cost=cost,
        complexity=cx.complexity,
        fragility=frag,
        complexity_rejected=False,
        fragility_parts={},
    )
