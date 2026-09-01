"""FactorFitness V2 —— contracts（共享数据类型）。

与 `alphaprobe.contracts`（FactorIdentity/EvaluationRecord/FactorCandidate 等全局契约）
分开：本文件只定义 fitness 组件内部的数据契约，供 components/penalties/calibration/
factor_fitness 之间传递。所有跨模块类型仍在 `alphaprobe.contracts` 唯一定义，
这里不重复定义任何全局已有类型。

核心原则（V2）：components 只组合 EvaluationBundle / 已有 metric 值，绝不自己算
基础金融指标。底层指标一律来自 quant_evaluator（QE registry STABLE 指标 +
metrics/probe_portfolio 20d cohort 层）。本文件的 EvaluationBundle 只是
`EvaluationRecord.metric_bundle` 的强类型视图，不新增指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 标准化 QE 指标键（与 quant_evaluator registry 一致；数值口径由数据源权威定义）。
# 本模块只读这些键，不重算。
QE_METRIC_KEYS = frozenset(
    {
        # P —— predictive（L2 full_train / L3 search_valid 段）
        "rankic_valid",
        "median_subperiod_rankic",
        "hac_tstat",
        "ic_lower_conf",
        "ic_lower_confidence_bound",
        # Q —— quantile shape
        "group_returns",
        "decile_monotonicity",
        "isotonic_fit_quality",
        "top10_excess",
        "d10_minus_d1",
        "top_tail_quality",
        "d10_cliff_penalty",  # funnel 层 gate 同源值（不做第二遍扣罚）
        "d10_drop_ratio",
        # L —— long-short / 组合层（metrics/probe_portfolio 20d cohort）
        "net_sharpe",
        "calmar_ratio",
        "sortino_ratio",
        "net_annualized_ls_return",
        "d10_long_only_active_return",
        "max_drawdown",
        "drawdown_persistence",
        "max_dd_duration",
        "tuw",
        "q20_rolling_sharpe",
        "positive_month_ratio",
        "long_short_return",
        "ls_net_return",
        "long_only_active_return",
        # S —— stability
        "rankicir",
        "q20_rolling_rankic",
        "positive_subperiod_ratio",
        "train_valid_retention",
        "worst_subperiod_rankic",
        "ic_decay",
        "ic_ir",
        # N —— novelty
        "rho_max",
        "mean_top5_abs_corr",
        "structural_novelty",
        "residual_rankic",
        "schema_novelty",
        # R —— robustness / data quality
        "coverage",
        "nan_inf_ratio",
        "untradeable_ratio",
        "winsor_sensitivity",
        "contribution_concentration",
        "coverage_instability",
        "denominator_risk",
        "extreme_value_dependence",
        # cost（SearchFitness 用 base cost 口径，Stress 只作 audit）
        "turnover",
        "cost_gross",
        "cost_1x",
        "cost_2x",
        "cost_3x",
    }
)

# 越小越好（lower_is_better=True）的 QE 指标。在 calibrator 里取负 z。
QE_LOWER_IS_BETTER = frozenset(
    {
        "max_drawdown",
        "max_dd_duration",
        "tuw",
        "turnover",
        "cost_gross",
        "cost_1x",
        "cost_2x",
        "cost_3x",
        "rho_max",
        "mean_top5_abs_corr",
        "nan_inf_ratio",
        "untradeable_ratio",
        "winsor_sensitivity",
        "contribution_concentration",
        "coverage_instability",
        "denominator_risk",
        "extreme_value_dependence",
        "ic_decay",  # 衰减越小越好
    }
)


@dataclass
class EvaluationBundle:
    """EvaluationRecord.metric_bundle 的强类型视图（只读映射，不新增指标）。

    metrics 键与 QE registry / probe_portfolio 对齐；unknown 键原样保留（透传）。
    """

    metrics: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.metrics

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]

    def raw(self) -> dict[str, Any]:
        return dict(self.metrics)


@dataclass(frozen=True)
class ComplexityInfo:
    """FactorEngine AST Analyzer 输出（V2 复杂度惩罚唯一权威）。

    - ``nodes``：AST 节点数（factor_engine IR 计数；本地降级用 adapter 的
      ``_local_ast_count`` 估算，禁止字符串长度当复杂度）。
    - ``depth``：AST 深度。
    - ``formula``：可选，仅审计用途。
    """

    nodes: int = 0
    depth: int = 0
    formula: str = ""


@dataclass(frozen=True)
class PenaltyReport:
    """三惩罚的分解结果（审计友好，不参与 score 计算之外的语义）。"""

    cost: float = 0.0
    complexity: float = 0.0
    fragility: float = 0.0
    complexity_rejected: bool = False  # nodes > 64 → 不进 fitness（hard reject）
    reject_reason: str = ""
    fragility_parts: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.cost + self.complexity + self.fragility


@dataclass(frozen=True)
class FactorFitnessResult:
    """FactorFitness V2 的完整输出。六维 score 均按各自权重合入 ``fitness``。

    ``fitness`` 已是 clip 后的 [0, 1]。``rejected`` 由复杂度 hard-reject
    （nodes > 64 或 depth 超限）触发；rejected 时 ``fitness=0.0``。
    """

    fitness: float
    core: float
    P: float
    Q: float
    L: float
    S: float
    N: float
    R: float
    cost_penalty: float
    complexity_penalty: float
    fragility_penalty: float
    rejected: bool = False
    reject_reason: str = ""

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = {
            "fitness": self.fitness,
            "factor_fitness_core": self.core,
            "P": self.P,
            "Q": self.Q,
            "L": self.L,
            "S": self.S,
            "N": self.N,
            "R": self.R,
            "cost_penalty": self.cost_penalty,
            "complexity_penalty": self.complexity_penalty,
            "fragility_penalty": self.fragility_penalty,
            "rejected": float(self.rejected),
        }
        if self.reject_reason:
            d["reject_reason"] = self.reject_reason
        return d
