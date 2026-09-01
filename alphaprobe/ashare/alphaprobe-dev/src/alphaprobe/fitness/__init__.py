"""fitness（任务书 §13-§21）：MetricCalibrator + 五维 SearchFitness + 生命周期评分。

所有原始指标先 utility 化（robust z-score），不能直接 RankIC+Sharpe-MDD。
Round Freeze：每 major round 冻结 calibrator，本轮同 raw metric → 同 utility。

FactorFitness V2：六维核心（P/Q/L/S/N/R）+ 三惩罚（Cost/Complexity/Fragility）。
- V2 实现在 ``fitness/factor_fitness.py``（总公式与配置化权重）、
  ``fitness/components.py``（六维 components）、``fitness/penalties.py``（三惩罚）、
  ``fitness/calibration.py``（U 与 MetricCalibrator 衔接）、``fitness/contracts.py``
  （fitness 内部数据契约）。
- 本文件保留 MetricCalibrator 与 ``compute_search_fitness``（legacy）公共签名，
  保证 trainer/pool.py 等既有调用方 0 改动；``compute_search_fitness`` 内部走 V2。
- D10 cliff 一致性：funnel 层（fitness/funnel.py）只做 gate（消费 metric
  ``d10_cliff_penalty`` > 0 拒绝），Q 层（components.quantile_score）只做连续
  惩罚（TopTailQuality = 1 - CollapsePenalty）。两处同源读同一 ``d10_cliff_penalty``
  值，语义一致且不叠加。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

EPS = 1e-9


class MetricCalibrator:
    """§13.1：U(m) = sigmoid(z_m / T_m)，z_m = (x-median)/(1.4826·MAD+ε)。

    越小越好项（MDD/DD Duration/TUW/Cost/Correlation/Turnover）→ U(-metric)。
    冻结后 z-score 语义：lower_is_better 分支对标准 z 整体取负
    （z = -( (x-median)/MAD )），保证更小 raw → 更高 utility。
    未冻结（warmup）时走 ordinal 百分位模式（样本内 rank，0~1，
    lower_is_better 反向），避免首轮 sigmoid(raw/T) 量纲不可比；
    可用 seed_prior() 用历史分布预填样本后直接 freeze。
    streaming quantile sketch：不把全历史 metric 放内存。
    """

    def __init__(
        self,
        temperature: float = 1.0,
        max_samples: int = 50_000,
        min_warmup_n: int = 500,
    ) -> None:
        self.T = temperature
        self.max_samples = max_samples
        self.min_warmup_n = min_warmup_n
        self._samples: dict[str, list[float]] = {}
        self._frozen: dict[str, tuple[float, float]] = {}  # metric -> (median, mad)
        self.frozen_round: str | None = None

    def observe(self, metric: str, value: float) -> None:
        if value is None or not math.isfinite(value):
            return
        buf = self._samples.setdefault(metric, [])
        buf.append(float(value))
        if len(buf) > self.max_samples:
            del buf[: len(buf) // 2]  # streaming 修剪，不存全历史

    def seed_prior(self, stats: dict[str, list[float]]) -> None:
        """用历史 evaluation 指标分布预填样本（如 seed library 10 万条）。"""
        for m, values in (stats or {}).items():
            for v in values:
                self.observe(m, v)

    def freeze(self, round_id: str) -> None:
        import statistics

        for m, buf in self._samples.items():
            if not buf:
                continue
            med = statistics.median(buf)
            mad = statistics.median([abs(x - med) for x in buf])
            self._frozen[m] = (med, 1.4826 * mad)
        self.frozen_round = round_id

    def utility(self, metric: str, value: float | None, *, lower_is_better: bool = False) -> float:
        if value is None or not math.isfinite(value):
            return 0.0
        v = float(value)
        if self.frozen_round is not None:
            frozen = self._frozen.get(metric)
            if frozen is not None:
                med, mad = frozen
                z = (v - med) / (mad + EPS)
                if lower_is_better:
                    z = -z
                return self._sigmoid(z / max(self.T, EPS))
            # freeze 前从未观测过该 metric：退化为 ordinal（该样本仅自己）
            return 0.5
        # 未冻结：warmup —— 样本内 ordinal 百分位（0~1，lower_is_better 反向）
        buf = self._samples.get(metric) or []
        if not buf:
            return 0.5
        if lower_is_better:
            return self._warmup_rank_ascending(v, buf)
        return self._warmup_rank_descending(v, buf)

    @staticmethod
    def _warmup_rank_descending(value: float, buf: list[float]) -> float:
        """值越大 utility 越高：u = count(x <= value)/n ∈ [1/n, 1]。"""
        if not buf:
            return 0.5
        n = len(buf)
        le = sum(1.0 for x in buf if x <= value)
        return le / n

    @staticmethod
    def _warmup_rank_ascending(value: float, buf: list[float]) -> float:
        """值越小 utility 越高：u = count(x >= value)/n ∈ [1/n, 1]。"""
        if not buf:
            return 0.5
        n = len(buf)
        ge = sum(1.0 for x in buf if x >= value)
        return ge / n

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def is_frozen(self) -> bool:
        return self.frozen_round is not None

    def warmup_active(self, metric: str) -> bool:
        """warmup 是否仍生效（未 freeze 或该 metric 样本不足 min_warmup_n）。"""
        if self.frozen_round is not None:
            return False
        return len(self._samples.get(metric) or []) < self.min_warmup_n


# ---------------------------------------------------------------------------
# §14-§18 五维 + §19 penalties
# ---------------------------------------------------------------------------


@dataclass
class FitnessInputs:
    """一个 candidate 的原始指标集合（全部 raw，绝不做 abs 混淆，§10.2）。"""

    rankic_valid: float | None = None
    median_subperiod_rankic: float | None = None
    # Q
    group_returns: list[float] | None = None  # G1..G10
    top10_excess: float | None = None
    d10_minus_d1: float | None = None
    # L
    net_sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    mdd: float | None = None
    max_dd_duration: float | None = None
    tuw: float | None = None
    q20_rolling_sharpe: float | None = None
    positive_month_ratio: float | None = None
    # S
    rankicir: float | None = None
    q20_rolling_rankic: float | None = None
    positive_subperiod_ratio: float | None = None
    train_valid_retention: float | None = None
    worst_subperiod_rankic: float | None = None
    # N
    rho_max: float | None = None
    mean_top5_abs_corr: float | None = None
    structural_novelty: float | None = None
    residual_rankic: float | None = None
    schema_novelty: float | None = None
    # penalties
    turnover: float | None = None
    ast_nodes: int = 0
    ast_depth: int = 0
    fragility_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FitnessWeights:
    # §13.2
    P: float = 0.26
    Q: float = 0.20
    L: float = 0.25
    S: float = 0.11
    N: float = 0.18
    # §14
    p_valid: float = 0.70
    p_subperiod: float = 0.30
    # §15.5
    q_mrank: float = 0.30
    q_miso: float = 0.25
    q_top10: float = 0.20
    q_tail: float = 0.25
    # §16.4
    l_sharpe: float = 0.30
    l_sortino: float = 0.10
    l_calmar: float = 0.15
    l_mdd: float = 0.10
    l_dddur: float = 0.08
    l_tuw: float = 0.07
    l_q20sharpe: float = 0.15
    l_posmonth: float = 0.05
    # §17
    s_icir: float = 0.35
    s_q20ric: float = 0.25
    s_possub: float = 0.15
    s_retention: float = 0.15
    s_worst: float = 0.10
    # §18
    n_corr: float = 0.45
    n_struct: float = 0.20
    n_residual: float = 0.20
    n_schema: float = 0.15
    # §19
    cost_max: float = 0.02
    fragility_max: float = 0.05


DEFAULT_WEIGHTS = FitnessWeights()


def group_monotonicity(group_returns: list[float]) -> float | None:
    """§15.1 Spearman([1..10], [G1..G10])。"""
    if not group_returns or len(group_returns) != 10:
        return None

    def _rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = _rank(list(range(10))), _rank(group_returns)
    n = 10
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    va = sum((a - ma) ** 2 for a in ra) ** 0.5
    vb = sum((b - mb) ** 2 for b in rb) ** 0.5
    if va * vb < EPS:
        return None
    return cov / (va * vb)


def isotonic_fit_quality(group_returns: list[float]) -> float | None:
    """§15.2 M_iso = 1 - SSE(G, G_iso)/SST。只评 shape，不自动改 factor。"""
    if not group_returns or len(group_returns) != 10:
        return None
    g = group_returns
    # PAVA（单调递增 isotonic）
    blocks: list[tuple[float, float]] = []  # (sum, count)
    for v in g:
        blocks.append((v, 1.0))
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append((s1 + s2, c1 + c2))
    fitted: list[float] = []
    for s, c in blocks:
        fitted.extend([s / c] * int(c))
    mean_g = sum(g) / len(g)
    sse = sum((a - b) ** 2 for a, b in zip(g, fitted))
    sst = sum((a - mean_g) ** 2 for a in g)
    return 1.0 - sse / (sst + EPS)


def d10_cliff_penalty(
    group_returns: list[float], tolerance: float = 0.12
) -> float | None:
    """§15.4：D10 允许比 D7/D8/D9 稍低（tol 0.10~0.15）；断崖才罚。

    CollapsePenalty = max(0, d_top - τ)，d_top = median[(Gi-G10)+]/|G10-G1|。
    """
    if not group_returns or len(group_returns) != 10:
        return None
    g = group_returns
    drops = [max(0.0, g[i] - g[9]) for i in (6, 7, 8)]
    drops.sort()
    median_drop = drops[1]
    denom = abs(g[9] - g[0]) + EPS
    d_top = median_drop / denom
    return max(0.0, d_top - tolerance)


def compute_search_fitness(
    x: FitnessInputs,
    calibrator: MetricCalibrator,
    *,
    weights: FitnessWeights = DEFAULT_WEIGHTS,
) -> dict[str, float]:
    """§13.2：F = 0.26P + 0.20Q + 0.25L + 0.11S + 0.18N - P_cost - P_cx - P_frag。

    内部走 FactorFitness V2（六维 + 三惩罚），保持既有调用方签名 0 改动。
    """
    from alphaprobe.fitness.contracts import ComplexityInfo, EvaluationBundle
    from alphaprobe.fitness.factor_fitness import factor_fitness_v2

    # 构造 EvaluationBundle（显式优先；缺省字段 fallback 到 FitnessInputs 的既有语义）
    _mb: dict[str, Any] = {}
    _mb["rankic_valid"] = x.rankic_valid
    _mb["median_subperiod_rankic"] = x.median_subperiod_rankic
    if x.group_returns is not None:
        _mb["group_returns"] = list(x.group_returns)
    _mb["top10_excess"] = x.top10_excess
    _mb["d10_minus_d1"] = x.d10_minus_d1
    _mb["net_sharpe"] = x.net_sharpe
    _mb["sortino"] = x.sortino
    _mb["calmar"] = x.calmar
    _mb["max_drawdown"] = x.mdd
    _mb["max_dd_duration"] = x.max_dd_duration
    _mb["tuw"] = x.tuw
    _mb["q20_rolling_sharpe"] = x.q20_rolling_sharpe
    _mb["positive_month_ratio"] = x.positive_month_ratio
    _mb["rankicir"] = x.rankicir
    _mb["q20_rolling_rankic"] = x.q20_rolling_rankic
    _mb["positive_subperiod_ratio"] = x.positive_subperiod_ratio
    _mb["train_valid_retention"] = x.train_valid_retention
    _mb["worst_subperiod_rankic"] = x.worst_subperiod_rankic
    _mb["rho_max"] = x.rho_max
    _mb["mean_top5_abs_corr"] = x.mean_top5_abs_corr
    _mb["structural_novelty"] = x.structural_novelty
    _mb["residual_rankic"] = x.residual_rankic
    _mb["schema_novelty"] = x.schema_novelty
    _mb["turnover"] = x.turnover
    # D10 同源：funnel gate 层只做 gate，Q 层只做连续惩罚（不叠加）
    if x.group_returns is not None:
        _mb["d10_cliff_penalty"] = d10_cliff_penalty(x.group_returns)
    _r = factor_fitness_v2(
        EvaluationBundle(_mb),
        calibrator,
        complexity=ComplexityInfo(nodes=int(x.ast_nodes or 0), depth=int(x.ast_depth or 0)),
    )
    return {
        "P": _r.P,
        "Q": _r.Q,
        "L": _r.L,
        "S": _r.S,
        "N": _r.N,
        "P_cost": _r.cost_penalty,
        "P_complexity": _r.complexity_penalty,
        "P_fragility": _r.fragility_penalty,
        "search_fitness": _r.fitness,
    }


# ---------------------------------------------------------------------------
# §20 Hard Gates（与 SearchFitness 分开，不能靠高 Fitness 救回）
# ---------------------------------------------------------------------------


class HardGateFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


HARD_GATE_REASONS = {
    "DSL_INVALID",
    "PIT_VIOLATION",
    "FORBIDDEN_FIELD",
    "INSUFFICIENT_COVERAGE",
    "NAN_INF_CATASTROPHIC",
    "EXACT_DUPLICATE",
    "SIGN_DUPLICATE",
    "SEED_LIBRARY_DUPLICATE",
    "ALREADY_EXPORTED",
    "SEALED_TEST_LEAKAGE",
    "UNTRADEABLE_CATASTROPHIC",
    "NUMERICAL_INVALID",
}


def check_hard_gates(
    *,
    dsl_valid: bool,
    pit_ok: bool,
    forbidden_field: bool,
    coverage: float | None,
    min_coverage: float = 0.5,
    nan_inf_ratio: float | None,
    exact_duplicate: bool,
    sign_duplicate: bool,
    seed_duplicate: bool,
    already_exported: bool,
    sealed_leak: bool,
    untradeable_ratio: float | None,
    max_untradeable: float = 0.5,
) -> list[str]:
    """返回触发的 gate 原因列表（空 = 通过）。"""
    fails: list[str] = []
    if not dsl_valid:
        fails.append("DSL_INVALID")
    if not pit_ok:
        fails.append("PIT_VIOLATION")
    if forbidden_field:
        fails.append("FORBIDDEN_FIELD")
    if coverage is not None and coverage < min_coverage:
        fails.append("INSUFFICIENT_COVERAGE")
    if nan_inf_ratio is not None and nan_inf_ratio > 0.5:
        fails.append("NAN_INF_CATASTROPHIC")
    if exact_duplicate:
        fails.append("EXACT_DUPLICATE")
    if sign_duplicate:
        fails.append("SIGN_DUPLICATE")
    if seed_duplicate:
        fails.append("SEED_LIBRARY_DUPLICATE")
    if already_exported:
        fails.append("ALREADY_EXPORTED")
    if sealed_leak:
        fails.append("SEALED_TEST_LEAKAGE")
    if untradeable_ratio is not None and untradeable_ratio > max_untradeable:
        fails.append("UNTRADEABLE_CATASTROPHIC")
    return fails


# ---------------------------------------------------------------------------
# §21 生命周期评分拆分
# ---------------------------------------------------------------------------


def asset_quality_score(x: FitnessInputs, calibrator: MetricCalibrator) -> float:
    """§21.2：偏 stability/survival/tradability/robustness/OOS。"""
    s = compute_search_fitness(x, calibrator)
    return 0.5 * s["S"] + 0.3 * calibrator.utility("net_sharpe", x.net_sharpe) + 0.2 * s["Q"]


def search_value(
    *,
    fertility: float,
    offspring_novelty: float,
    descendant_pool_gain: float,
    coverage_gap: float,
    uncertainty: float,
    frontierness: float,
    survival_opportunity: float,  # 已做置信收缩 + cap（§51）
    search_saturation: float,
    historical_compute_cost: float,
    w: dict[str, float] | None = None,
) -> float:
    """§21.3：SearchValue = w1·Fert + w2·OffNov + w3·PoolGain + w4·CovGap + w5·Unc
    + w6·Frontier + w7·Survival - w8·Saturation - w9·HistCost。"""
    d = {
        "w1": 0.20, "w2": 0.10, "w3": 0.15, "w4": 0.10,
        "w5": 0.10, "w6": 0.15, "w7": 0.10, "w8": 0.15, "w9": 0.10,
    }
    if w:
        d.update(w)
    return (
        d["w1"] * fertility
        + d["w2"] * offspring_novelty
        + d["w3"] * descendant_pool_gain
        + d["w4"] * coverage_gap
        + d["w5"] * uncertainty
        + d["w6"] * frontierness
        + d["w7"] * survival_opportunity
        - d["w8"] * search_saturation
        - d["w9"] * historical_compute_cost
    )


def survival_opportunity_shrunk(
    *,
    raw_survival_rate: float,
    support_count: int,
    prior: float = 0.5,
    shrinkage_strength: float = 20.0,
    cap: float = 0.15,
) -> float:
    """§51：置信度收缩。Direction A 90%/7个 不能大幅奖励；B 72%/3000 才可信。"""
    k = max(0.0, float(support_count))
    shrunk = (raw_survival_rate * k + prior * shrinkage_strength) / (k + shrinkage_strength)
    # cap 只约束「相对 prior 的增益」，小样本即使 raw 高也拿不到大幅奖励
    gain = max(0.0, shrunk - prior)
    return min(cap, gain)


def pool_utility(
    *,
    pareto_rank: int,
    niche_rarity: float,
    novelty: float,
    s_value: float,
    representative_value: float,
) -> float:
    """§21.4：值得占 Active Pool 一个 slot 吗。"""
    return (
        0.35 * (1.0 / (1.0 + pareto_rank))
        + 0.20 * niche_rarity
        + 0.20 * novelty
        + 0.15 * s_value
        + 0.10 * representative_value
    )


def export_score(
    *, search_fitness: float, stability: float, novelty: float, tradability: float
) -> float:
    """§21.5：比 SearchFitness 更严格的正式输出分。"""
    return 0.40 * search_fitness + 0.25 * stability + 0.20 * novelty + 0.15 * tradability


def complexity_efficiency(delta_fitness: float, delta_ast_nodes: int) -> float:
    """§19.2：进化时记录 ΔFitness/(1+ΔASTNodes)。"""
    return delta_fitness / (1.0 + max(0, delta_ast_nodes))