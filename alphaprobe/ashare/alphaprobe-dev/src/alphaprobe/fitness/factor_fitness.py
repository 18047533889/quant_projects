"""FactorFitness V2 —— factor_fitness（总公式与配置化权重）。

总公式（初版权重，配置化，不许硬编码进逻辑）：

    FactorFitnessCore = 0.20×P + 0.12×Q + 0.35×L + 0.15×S + 0.10×N + 0.08×R
    FactorFitness     = clip(Core − CostPenalty − ComplexityPenalty − FragilityPenalty, 0, 1)

六维核心分由 components.py 提供（只组合 QE 已有 metric 值）；三惩罚由 penalties.py
提供。复杂度 >64（或 depth 超硬限）→ hard reject，不进 fitness（fitness=0.0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
    long_short_score,
    novelty_score,
    predictive_score,
    quantile_score,
    robustness_score,
    stability_score,
)
from alphaprobe.fitness.contracts import (
    ComplexityInfo,
    EvaluationBundle,
    FactorFitnessResult,
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


@dataclass(frozen=True)
class V2FactorWeights:
    """FactorFitness 总权重（初版）。所有字段配置化，逻辑不许硬编码数值。"""

    P: float = 0.20
    Q: float = 0.12
    L: float = 0.35
    S: float = 0.15
    N: float = 0.10
    R: float = 0.08

    # 维度内部权重（透传给 components）
    p: PWeights = field(default_factory=lambda: DEFAULT_PWEIGHTS)
    q: QWeights = field(default_factory=lambda: DEFAULT_QWEIGHTS)
    l: LWeights = field(default_factory=lambda: DEFAULT_LWEIGHTS)
    s: SWeights = field(default_factory=lambda: DEFAULT_SWEIGHTS)
    n: NWeights = field(default_factory=lambda: DEFAULT_NWEIGHTS)
    r: RWeights = field(default_factory=lambda: DEFAULT_RWEIGHTS)

    @property
    def dimension_total(self) -> float:
        return self.P + self.Q + self.L + self.S + self.N + self.R


DEFAULT_V2_WEIGHTS = V2FactorWeights()


def factor_fitness_v2(
    bundle: EvaluationBundle | Mapping[str, Any],
    calibrator: Any,
    *,
    complexity: ComplexityInfo | None = None,
    weights: V2FactorWeights = DEFAULT_V2_WEIGHTS,
    cost_config: CostPenaltyConfig = DEFAULT_COST_CONFIG,
    complexity_config: ComplexityPenaltyConfig = DEFAULT_COMPLEXITY_CONFIG,
    fragility_config: FragilityPenaltyConfig = DEFAULT_FRAGILITY_CONFIG,
    d10_tau: float = 0.12,
) -> FactorFitnessResult:
    """计算 FactorFitness V2。

    Parameters
    ----------
    bundle : EvaluationBundle | dict
        评估 metric 集合（QE registry + probe_portfolio 20d cohort）。
    calibrator : fitness.MetricCalibrator
        现有 calibrator（warmup ordinal / freeze z-score）。
    complexity : ComplexityInfo | None
        FactorEngine AST Analyzer 输出；None 时视为 (0, 0)（无罚）。
    """
    eb = bundle if isinstance(bundle, EvaluationBundle) else EvaluationBundle(dict(bundle or {}))
    cx = complexity if complexity is not None else ComplexityInfo(nodes=0, depth=0)

    # 复杂度 hard-reject：不进 fitness
    cx_report = compute_all_penalties(
        eb, calibrator, cx,
        cost_config=cost_config,
        complexity_config=complexity_config,
        fragility_config=fragility_config,
    )
    if cx_report.complexity_rejected:
        return FactorFitnessResult(
            fitness=0.0,
            core=0.0,
            P=0.0, Q=0.0, L=0.0, S=0.0, N=0.0, R=0.0,
            cost_penalty=0.0,
            complexity_penalty=0.0,
            fragility_penalty=0.0,
            rejected=True,
            reject_reason=cx_report.reject_reason,
        )

    P = predictive_score(eb, calibrator, weights=weights.p)
    Q = quantile_score(eb, calibrator, weights=weights.q, d10_tau=d10_tau)
    L = long_short_score(eb, calibrator, weights=weights.l)
    S = stability_score(eb, calibrator, weights=weights.s)
    N = novelty_score(eb, calibrator, weights=weights.n)
    R = robustness_score(eb, calibrator, weights=weights.r)

    core = (
        weights.P * P
        + weights.Q * Q
        + weights.L * L
        + weights.S * S
        + weights.N * N
        + weights.R * R
    )
    total_pen = cx_report.cost + cx_report.complexity + cx_report.fragility
    fitness = max(0.0, min(1.0, core - total_pen))
    return FactorFitnessResult(
        fitness=fitness,
        core=core,
        P=P, Q=Q, L=L, S=S, N=N, R=R,
        cost_penalty=cx_report.cost,
        complexity_penalty=cx_report.complexity,
        fragility_penalty=cx_report.fragility,
        rejected=False,
        reject_reason="",
    )
