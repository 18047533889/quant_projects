"""FactorFitness V2 —— components（六维核心分数）。

六个函数 predictive_score / quantile_score / long_short_score / stability_score /
novelty_score / robustness_score 只吃 metric dict / EvaluationBundle，并组合
quant_evaluator 已有 metric 值（QE registry STABLE + probe_portfolio 20d cohort）。
本文件绝不自己算基础金融指标。

权重约定：本文件所有函数默认权重在 V2FactorWeights（factor_fitness.py）里配置化，
这里只是初版数值的默认值；外部调用传入 weights 可整体替换，禁止硬编码进分支逻辑。

关于 Q 维的 D10 cliff：funnel 层（fitness/funnel.py `_l2_gate`）只做 gate（用
metric "d10_cliff_penalty" > 0 拒绝）；Q 层只做连续惩罚。两处同源读
"d10_cliff_penalty"（= max(0, d_top - tau)，tau 默认 0.12），语义一致且不叠加——
Q 里不再从 group_returns 重算 d10，避免扣两遍。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from alphaprobe.fitness import isotonic_fit_quality  # 现有 §15.2 工具（只做 shape）
from alphaprobe.fitness.calibration import single_utility, utility_for_bundle
from alphaprobe.fitness.contracts import EvaluationBundle

EPS = 1e-9


# ---------------------------------------------------------------------------
# 权重配置（初版；真实值在 V2FactorWeights 统一配置化，不许写死在分支里）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PWeights:
    u_rankic: float = 0.55
    u_subperiod: float = 0.25
    u_hac_tstat: float = 0.20
    # 预留升级位：IC_LowerConfidenceBound 就位后切 0.45/0.20/0.15/0.20
    # u_ic_lower: float = 0.45 / u_subperiod: 0.20 / u_hac: 0.15 / u_ic_lower bound 0.20
    use_ic_lower_conf: bool = False


@dataclass(frozen=True)
class QWeights:
    u_decile_mono: float = 0.25
    u_iso: float = 0.15
    u_top10: float = 0.20
    u_d10_minus_d1: float = 0.15
    u_top_tail: float = 0.25
    top_tail_lower_is_better: bool = False  # U_lower(TopTailQuality)


@dataclass(frozen=True)
class LWeights:
    u_net_sharpe: float = 0.22
    u_calmar: float = 0.15
    u_sortino: float = 0.08
    u_net_annualized_ls: float = 0.10
    u_d10_long_only_active: float = 0.10
    u_max_dd: float = 0.12
    u_drawdown_persistence: float = 0.08
    u_q20_rolling_sharpe: float = 0.10
    u_positive_month: float = 0.05
    # L 维度内部权重和（归一化 sanity 检查用）
    @property
    def total(self) -> float:
        return (
            self.u_net_sharpe
            + self.u_calmar
            + self.u_sortino
            + self.u_net_annualized_ls
            + self.u_d10_long_only_active
            + self.u_max_dd
            + self.u_drawdown_persistence
            + self.u_q20_rolling_sharpe
            + self.u_positive_month
        )


@dataclass(frozen=True)
class SWeights:
    u_icir: float = 0.30
    u_q20_rolling_rankic: float = 0.20
    u_positive_subperiod_ratio: float = 0.20
    u_train_valid_retention: float = 0.15
    u_worst_subperiod: float = 0.10
    u_ic_decay: float = 0.05

    @property
    def total(self) -> float:
        return (
            self.u_icir
            + self.u_q20_rolling_rankic
            + self.u_positive_subperiod_ratio
            + self.u_train_valid_retention
            + self.u_worst_subperiod
            + self.u_ic_decay
        )


@dataclass(frozen=True)
class NWeights:
    u_residual_rankic: float = 0.40
    u_structural_novelty: float = 0.30
    u_mean_top5_abs_corr: float = 0.30

    @property
    def total(self) -> float:
        return self.u_residual_rankic + self.u_structural_novelty + self.u_mean_top5_abs_corr


@dataclass(frozen=True)
class RWeights:
    u_coverage: float = 0.30
    u_nan_inf_ratio: float = 0.25
    u_untradeable_ratio: float = 0.25
    u_winsor_sensitivity: float = 0.10
    u_numerical_stability: float = 0.10

    @property
    def total(self) -> float:
        return (
            self.u_coverage
            + self.u_nan_inf_ratio
            + self.u_untradeable_ratio
            + self.u_winsor_sensitivity
            + self.u_numerical_stability
        )


#: 权重类 实例化 的默认集合（frozen dataclass 值对象）
DEFAULT_PWEIGHTS = PWeights()
DEFAULT_QWEIGHTS = QWeights()
DEFAULT_LWEIGHTS = LWeights()
DEFAULT_SWEIGHTS = SWeights()
DEFAULT_NWEIGHTS = NWeights()
DEFAULT_RWEIGHTS = RWeights()


# ---------------------------------------------------------------------------
# P —— Predictive
# ---------------------------------------------------------------------------


def predictive_score(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    weights: PWeights = DEFAULT_PWEIGHTS,
) -> float:
    """P = 0.55·U(RankIC_valid) + 0.25·U(MedianSubperiodRankIC) + 0.20·U(HAC_tstat)。

    IC_LowerConfidenceBound 升级位预留（0.45/0.20/0.15/0.20 分支）。
    """
    if weights.use_ic_lower_conf:
        return _weighted_sum(
            [
                (weights.u_rankic, single_utility(calibrator, "rankic_valid", bundle.get("rankic_valid"))),
                (0.20, single_utility(calibrator, "median_subperiod_rankic", bundle.get("median_subperiod_rankic"))),
                (0.15, single_utility(calibrator, "hac_tstat", bundle.get("hac_tstat"))),
                (0.20, single_utility(calibrator, "ic_lower_confidence_bound", bundle.get("ic_lower_confidence_bound"))),
            ]
        )
    return _weighted_sum(
        [
            (weights.u_rankic, single_utility(calibrator, "rankic_valid", bundle.get("rankic_valid"))),
            (weights.u_subperiod, single_utility(calibrator, "median_subperiod_rankic", bundle.get("median_subperiod_rankic"))),
            (weights.u_hac_tstat, single_utility(calibrator, "hac_tstat", bundle.get("hac_tstat"))),
        ]
    )


# ---------------------------------------------------------------------------
# Q —— Quantile Shape
# ---------------------------------------------------------------------------


def quantile_score(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    weights: QWeights = DEFAULT_QWEIGHTS,
    d10_tau: float = 0.12,
) -> float:
    """Q = 0.25·U(DecileMonotonicity) + 0.15·U(IsotonicFitQuality)
          + 0.20·U(Top10Excess) + 0.15·U(D10MinusD1) + 0.25·U_lower(TopTailQuality)

    D10 cliff（funnel gate 同源语义）只在 top-tail 子项里做连续惩罚，绝不叠加：
    Q 层读 bundle 里已有的 "d10_cliff_penalty"（= max(0, d_top - tau)，funnel 层
    gate 用同一值），只做 max(0, d_top - tau) 的单次扣减；如果 bundle 没有该值，
    才从 group_returns 用同一 d_top 公式算连续值（缺省值 fallback，非第二遍）。
    """
    u = utility_for_bundle(
        bundle, calibrator, keys=["decile_monotonicity", "isotonic_fit_quality",
                                  "top10_excess", "d10_minus_d1"]
    )
    # 单调性：优先已有 metric；缺失才从 group_returns 组合（shape 指标）
    decile_u = u.get("decile_monotonicity", 0.0)
    if decile_u == 0.0 and bundle.get("group_returns") is not None:
        decile_u = single_utility(calibrator, "decile_monotonicity",
                                  _decile_monotonicity(bundle.get("group_returns")))
    iso_u = u.get("isotonic_fit_quality", 0.0)
    if iso_u == 0.0 and bundle.get("group_returns") is not None:
        iso_u = single_utility(calibrator, "isotonic_fit_quality",
                               isotonic_fit_quality(bundle.get("group_returns")))

    # top-tail：U_lower(TopTailQuality)，TopTailQuality = 1 - CollapsePenalty。
    # 连续惩罚只扣一次：优先 bundle 的 d10_cliff_penalty（funnel 同源），
    # 没有才从 group_returns 算 max(0, d_top - tau)。
    top_tail_u = _top_tail_quality_utility(bundle, calibrator, d10_tau=d10_tau)
    q = (
        weights.u_decile_mono * decile_u
        + weights.u_iso * iso_u
        + weights.u_top10 * u.get("top10_excess", 0.0)
        + weights.u_d10_minus_d1 * u.get("d10_minus_d1", 0.0)
        + weights.u_top_tail * top_tail_u
    )
    return max(0.0, min(1.0, float(q)))


def _top_tail_quality_utility(
    bundle: EvaluationBundle, calibrator: Any, *, d10_tau: float = 0.12
) -> float:
    """U_lower(TopTailQuality)：TopTailQuality = 1 - CollapsePenalty。

    CollapsePenalty = max(0, d_top - tau)，d_top = median((G7-G10)+,(G8-G10)+,(G9-G10)+)
    / (|G10-G1| + eps)。取值优先级（保证 funnel gate 同源语义一致且不叠加）：
      1. bundle 已有 "top_tail_quality"（= 1 - CollapsePenalty，QE cohort 直接产出）
         → 直接用；
      2. bundle 已有 "d10_cliff_penalty"（= max(0, d_top - tau)，funnel gate 同源）
         → TopTailQuality = 1 - d10_cliff_penalty（单次扣减，不重算 d_top）；
      3. 都没有 → 从 group_returns 用同一 d_top 公式现算（fallback）。
    任何路径都不做第二遍扣罚。
    """
    tt = bundle.get("top_tail_quality")
    if tt is None:
        cp = bundle.get("d10_cliff_penalty")
        if cp is None:
            cp = _collapse_penalty_from_groups(bundle.get("group_returns"), d10_tau=d10_tau)
        if cp is not None:
            cp = max(0.0, min(1.0, float(cp)))
            tt = 1.0 - cp
        else:
            tt = 1.0  # 无任何来源信息 → 中性（不因缺测惩罚）
    tt = max(0.0, min(1.0, float(tt)))
    return single_utility(calibrator, "top_tail_quality", tt)


def _collapse_penalty_from_groups(group_returns: Any, *, d10_tau: float = 0.12) -> float | None:
    """从 G1..G10 算 CollapsePenalty（与 fitness.d10_cliff_penalty 同一公式）。"""
    from alphaprobe.fitness import d10_cliff_penalty

    return d10_cliff_penalty(group_returns, tolerance=d10_tau)


def _decile_monotonicity(group_returns: Any) -> float | None:
    """从 group_returns 组合单调性（现有 §15.1 工具；与 QE metric 同语义）。"""
    from alphaprobe.fitness import group_monotonicity

    return group_monotonicity(group_returns)


# ---------------------------------------------------------------------------
# L —— Long-Short（底层指标全部来自 QE registry / probe_portfolio cohort）
# ---------------------------------------------------------------------------


def long_short_score(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    weights: LWeights = DEFAULT_LWEIGHTS,
) -> float:
    """L = 0.22·U(NetSharpe) + 0.15·U(Calmar) + 0.08·U(Sortino)
          + 0.10·U(NetAnnualizedLSReturn) + 0.10·U(D10LongOnlyActiveReturn)
          + 0.12·U_lower(MaxDrawdown) + 0.08·U_lower(DrawdownPersistence)
          + 0.10·U(Q20RollingSharpe) + 0.05·U(PositiveMonthRatio)

    全部消费 QE metric 值，不重复实现。cost 口径：SearchFitness 用 base cost
    （gross 或 1x）；Stress（2x/3x）只作 audit，不进此分数。
    """
    # DrawdownPersistence = 0.5·norm(MaxDDDuration) + 0.5·norm(TimeUnderWater)
    dd_persist = _drawdown_persistence(bundle, calibrator)

    l = (
        weights.u_net_sharpe * single_utility(calibrator, "net_sharpe", bundle.get("net_sharpe"))
        + weights.u_calmar * single_utility(calibrator, "calmar_ratio", bundle.get("calmar_ratio", bundle.get("calmar")))
        + weights.u_sortino * single_utility(calibrator, "sortino_ratio", bundle.get("sortino_ratio", bundle.get("sortino")))
        + weights.u_net_annualized_ls * single_utility(calibrator, "net_annualized_ls_return", bundle.get("net_annualized_ls_return"))
        + weights.u_d10_long_only_active * single_utility(calibrator, "d10_long_only_active_return", bundle.get("d10_long_only_active_return"))
        + weights.u_max_dd * single_utility(calibrator, "max_drawdown", bundle.get("max_drawdown", bundle.get("mdd")), lower_is_better=True)
        + weights.u_drawdown_persistence * dd_persist
        + weights.u_q20_rolling_sharpe * single_utility(calibrator, "q20_rolling_sharpe", bundle.get("q20_rolling_sharpe"))
        + weights.u_positive_month * single_utility(calibrator, "positive_month_ratio", bundle.get("positive_month_ratio"))
    )
    return max(0.0, min(1.0, float(l)))


def _drawdown_persistence(bundle: EvaluationBundle, calibrator: Any) -> float:
    """DrawdownPersistence = 0.5·norm(MaxDDDuration) + 0.5·norm(TimeUnderWater)。

    直接消费 QE cohort 层已有值；缺失时返回 0.5（无信息中性，不奖励不惩罚）。
    """
    mdd_dur = bundle.get("max_dd_duration")
    tuw = bundle.get("tuw")
    if mdd_dur is None and tuw is None:
        return 0.5
    parts: list[float] = []
    if mdd_dur is not None:
        parts.append(single_utility(calibrator, "max_dd_duration", mdd_dur, lower_is_better=True))
    if tuw is not None:
        parts.append(single_utility(calibrator, "tuw", tuw, lower_is_better=True))
    if not parts:
        return 0.5
    return sum(parts) / len(parts)


# ---------------------------------------------------------------------------
# S —— Stability
# ---------------------------------------------------------------------------


def stability_score(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    weights: SWeights = DEFAULT_SWEIGHTS,
) -> float:
    """S：subperiod IC 稳定性 + IC decay，全部用现有 metric。

    ICIR / 20d 滚动 RankIC / 正子期占比 / train-valid retention / 最差子期
    / IC decay。
    """
    s = (
        weights.u_icir * single_utility(calibrator, "rankicir", bundle.get("rankicir", bundle.get("ic_ir")))
        + weights.u_q20_rolling_rankic * single_utility(calibrator, "q20_rolling_rankic", bundle.get("q20_rolling_rankic"))
        + weights.u_positive_subperiod_ratio * single_utility(calibrator, "positive_subperiod_ratio", bundle.get("positive_subperiod_ratio"))
        + weights.u_train_valid_retention * single_utility(calibrator, "train_valid_retention", bundle.get("train_valid_retention"))
        + weights.u_worst_subperiod * single_utility(calibrator, "worst_subperiod_rankic", bundle.get("worst_subperiod_rankic"))
        + weights.u_ic_decay * single_utility(calibrator, "ic_decay", bundle.get("ic_decay"), lower_is_better=True)
    )
    return max(0.0, min(1.0, float(s)))


# ---------------------------------------------------------------------------
# N —— Novelty（只占 10%，防奖励「低相关但垃圾」）
# ---------------------------------------------------------------------------


def novelty_score(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    weights: NWeights = DEFAULT_NWEIGHTS,
) -> float:
    """N：与最近邻 residual/incremental IC 或结构新颖度。

    权重刻意压低（V2FactorWeights.N=0.10），绝不奖励低相关垃圾。
    """
    n = (
        weights.u_residual_rankic * single_utility(calibrator, "residual_rankic", bundle.get("residual_rankic"))
        + weights.u_structural_novelty * single_utility(calibrator, "structural_novelty", bundle.get("structural_novelty"))
        + weights.u_mean_top5_abs_corr * single_utility(
            calibrator, "mean_top5_abs_corr", bundle.get("mean_top5_abs_corr"), lower_is_better=True
        )
    )
    return max(0.0, min(1.0, float(n)))


# ---------------------------------------------------------------------------
# R —— Robustness / Data Quality
# ---------------------------------------------------------------------------


def robustness_score(
    bundle: EvaluationBundle,
    calibrator: Any,
    *,
    weights: RWeights = DEFAULT_RWEIGHTS,
) -> float:
    """R：coverage、数值稳定、winsor 敏感等。全部 lower/upper 由默认表决定。

    注：WinsorSensitivity 既出现在 R（数据质量）又出现在 FragilityPenalty（连续
    severity λ1），两者语义不同：R 是质量回报，Fragility 是风险罚项，允许并存。
    """
    r = (
        weights.u_coverage * single_utility(calibrator, "coverage", bundle.get("coverage"))
        + weights.u_nan_inf_ratio * single_utility(calibrator, "nan_inf_ratio", bundle.get("nan_inf_ratio"), lower_is_better=True)
        + weights.u_untradeable_ratio * single_utility(calibrator, "untradeable_ratio", bundle.get("untradeable_ratio"), lower_is_better=True)
        + weights.u_winsor_sensitivity * single_utility(calibrator, "winsor_sensitivity", bundle.get("winsor_sensitivity"), lower_is_better=True)
        + weights.u_numerical_stability * single_utility(calibrator, "numerical_stability", bundle.get("numerical_stability"))
    )
    return max(0.0, min(1.0, float(r)))


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _weighted_sum(terms: list[tuple[float, float]]) -> float:
    total = 0.0
    for w, v in terms:
        if v is None or not math.isfinite(float(v)):
            continue
        total += float(w) * float(v)
    return max(0.0, min(1.0, total))
