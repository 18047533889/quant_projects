"""Lineage Stagnation 判据（plan Task 11）——取代 depth hard exponential decay。

depth 惩罚的问题（plan Task 11 开头）：硬指数衰减 ``(1-gamma)^depth`` 会压制
「深但持续有增益」的 fertile branch。本模块把「该不该继续挖」从 depth 决定改为
**最近 K 代有没有真的在产出**（stagnation）决定；depth 只剩一个小复杂度先验。

判据（每条都配权重、可配置；无证据 = 中性，Part G #13）：

- ΔFactorFitness 趋势（last-K 的 gain 均值 / 是否连续下降）
- ΔPoolUtility 趋势
- novelty gain
- complexity growth（复杂度涨而 fitness 平 → 加速停）
- child nearest correlation（生成的孩子与既有池/近邻越来越像 → 停滞信号）
- failed generation count（连续失败次数）

``StagnationEvaluator`` 输入一个 branch 的 last-K 生成记录，输出停滞分
（越高越该停）；``observe`` 累积记录并给出最新判据。合成数据可测、纯函数、
零 LLM / 零模型 / 零网络。与 retriever 解耦：retriever 只消费
``stagnation_penalty``（分数）与 ``should_stop``（布尔），不依赖本模块表结构。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "GenerationRecord",
    "StagnationConfig",
    "StagnationResult",
    "StagnationEvaluator",
    "DEFAULT_STAGNATION_CONFIG",
    "evidence_neutral_gain",
    "rolling_mean",
]

EPS = 1e-9


# ---------------------------------------------------------------------------
# 输入契约：一条生成记录
# ---------------------------------------------------------------------------


@dataclass
class GenerationRecord:
    """一个 branch 的一代生成结果（delta 口径；None = 无证据 → 中性）。

    Parameters
    ----------
    factor_id : str
        该代产出的 child factor id（可为空字符串 = 生成失败/未落地）。
    delta_fitness : float | None
        相对 parent 的 ΔFactorFitness（None = 未评估 → 中性）。
    delta_pool : float | None
        ΔPoolUtility（None = 中性）。
    novelty_gain : float | None
        child 相对池内近邻的 novelty 增益（None = 中性）。
    complexity_delta : int | float | None
        ΔAST 节点数（正 = 变复杂；None = 中性）。
    nearest_correlation : float | None
        child 与结构最近邻的相关系数（∈[0,1]，越大越「没出新东西」；
        None = 中性 0.5）。
    generation : int
        代数（用于窗口切分与顺序）。
    outcome : str
        "ok" / "failed"（失败生成计数用）。
    """

    factor_id: str = ""
    delta_fitness: float | None = None
    delta_pool: float | None = None
    novelty_gain: float | None = None
    complexity_delta: int | float | None = None
    nearest_correlation: float | None = None
    generation: int = 0
    outcome: str = "ok"


# ---------------------------------------------------------------------------
# 可配置判据（Part G #30：所有新特性都有 ablation 开关）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StagnationConfig:
    """停滞判据权重与窗口（全部可配，逻辑不硬编码数值）。

    - ``window``：最近 K 代（缺省记录不足时按已有记录，>window 截断保留最新）。
    - ``min_records``：判据起效所需最少记录数；不足 → 中性（不误杀浅 branch）。
    - ``stale_generations``：连续多少代无 gain 判「停滞」（plan Task 11 测试的
      「浅但连续 6 代停滞被停」用默认 6）。
    - 各信号权重（相加归一化后乘到停滞分）。
    - ``complexity_accel``：复杂度涨而 fitness 平的加速系数。
    """

    window: int = 8
    min_records: int = 3
    stale_generations: int = 6
    w_fitness_trend: float = 0.35
    w_pool_trend: float = 0.20
    w_novelty: float = 0.15
    w_complexity: float = 0.15
    w_nearest_corr: float = 0.10
    w_failed: float = 0.05
    complexity_accel: float = 0.5
    enable_complexity_accel: bool = True
    #: 停滞分在 [0,1]，本阈值以上 ``should_stop`` 为 True
    stop_threshold: float = 0.5


DEFAULT_STAGNATION_CONFIG = StagnationConfig()


def _stale_streak_of(recs: Sequence[GenerationRecord], window: int) -> int:
    """窗口内最新连续「有产出但无 gain」代数（failed/无产出不计入）。

    从最新代往前数：遇到有 gain 或失败生成即停；全部无 gain 才累计满。
    无证据（delta 全 None）不算 gain → 会累计成 stale？不会：无评估证据的
    delta=None 记录按「无 gain」计，但与显式 df=0 不同——None 无证据应中性。
    因此：只有**至少一条显式评估证据**（任意 delta 非 None）的代数才可视为
    「确定无 gain」；全 None 代不累计 stale（避免无证据误停）。
    """
    streak = 0
    for r in reversed(recs):
        if r.outcome == "failed":
            break
        has_evidence = any(
            v is not None for v in (r.delta_fitness, r.delta_pool, r.novelty_gain)
        )
        if not has_evidence:
            break  # 无证据代：既不算 gain 也不算 stale（中性）
        if any(
            v is not None and float(v) > 1e-9
            for v in (r.delta_fitness, r.delta_pool, r.novelty_gain)
        ):
            break  # 该代有 gain → streak 清零
        streak += 1
    return streak


def rolling_mean(values: Sequence[float]) -> float:
    """非空序列均值（空 → 0.0）。"""
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def evidence_neutral_gain(value: float | None) -> float:
    """无证据 → 中性 0.0（不当作负 gain 惩罚，也不当作正 gain 奖励）。"""
    if value is None:
        return 0.0
    fv = float(value)
    if fv != fv:  # NaN
        return 0.0
    return max(-1.0, min(1.0, fv))


# ---------------------------------------------------------------------------
# 停滞结果
# ---------------------------------------------------------------------------


@dataclass
class StagnationResult:
    """一次判定的全部诊断。"""

    stagnation: float            # 综合停滞分 ∈ [0, 1]
    should_stop: bool
    gain_count: int
    stale_streak: int
    fitness_trend: float
    pool_trend: float
    novelty_mean: float
    complexity_trend: float
    nearest_corr_mean: float
    failed_count: int
    depth_prior: float           # depth 小先验（供 retriever 沿用，默认 0.5=中性）
    signals: dict[str, float]    # 各子信号（归一化后 0~1 分）

    @property
    def neutral(self) -> bool:
        """无足够证据 → 中性（不触发停，也不给惩罚）。"""
        return self.gain_count < 0 and self.stale_streak <= 0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class StagnationEvaluator:
    """last-K 生成记录的停滞判定。

    用法：对每个 branch 持有一个 evaluator（或按 branch key 分组喂给同一个），
    ``observe(record)`` 累积记录，``evaluate()`` 返回当前停滞分；判定后继续
    ``observe`` 新记录。判定是增量的：窗口内无增益代数累计成 stale_streak，
    一旦有 gain（任一正 delta）就清零。
    """

    def __init__(self, config: StagnationConfig | None = None) -> None:
        self.config = config if config is not None else DEFAULT_STAGNATION_CONFIG
        self._records: list[GenerationRecord] = []
        self._stale_streak = 0
        self._last_result: StagnationResult | None = None

    # ------------------------------------------------------------------
    # 累积
    # ------------------------------------------------------------------

    def observe(self, record: GenerationRecord) -> StagnationResult:
        """追加一条生成记录并返回最新停滞判定。"""
        self._records.append(record)
        return self.evaluate()

    def has_any_gain(self, r: GenerationRecord) -> bool:
        """任一正 delta（fitness/pool/novelty）即视为该代有 gain。"""
        return any(
            v is not None and float(v) > 1e-9
            for v in (r.delta_fitness, r.delta_pool, r.novelty_gain)
        )

    # ------------------------------------------------------------------
    # 主判定
    # ------------------------------------------------------------------

    def evaluate(self) -> StagnationResult:
        cfg = self.config
        # 窗口截断（保留最新 window 条）——stale streak 也按窗口语义：只统计
        # 窗口内最新的连续无 gain 代（有 gain 即清零；窗口外的不再参与）。
        recs = list(self._records[-max(cfg.window, 1):])
        self._stale_streak = _stale_streak_of(recs, cfg.window)
        n = len(recs)
        enough = n >= max(cfg.min_records, 1)

        # --- 子信号（全部归一化到 0~1 分，1 = 强停滞信号）---
        signals: dict[str, float] = {}
        gain_count = 0
        fitness_vals: list[float] = []
        pool_vals: list[float] = []
        novelty_vals: list[float] = []
        complexity_deltas: list[float] = []
        corr_vals: list[float] = []
        failed = 0

        for r in recs:
            if r.outcome == "failed":
                failed += 1
            if self.has_any_gain(r):
                gain_count += 1
            if r.delta_fitness is not None:
                fitness_vals.append(evidence_neutral_gain(r.delta_fitness))
            if r.delta_pool is not None:
                pool_vals.append(evidence_neutral_gain(r.delta_pool))
            if r.novelty_gain is not None:
                novelty_vals.append(evidence_neutral_gain(r.novelty_gain))
            if r.complexity_delta is not None:
                complexity_deltas.append(float(r.complexity_delta))
            if r.nearest_correlation is not None:
                fv = float(r.nearest_correlation)
                if fv == fv:  # NaN
                    corr_vals.append(max(0.0, min(1.0, fv)))

        # fitness trend：窗口内 ΔFitness 均值（正=好）；无证据 → 0 中性
        fitness_trend = rolling_mean(fitness_vals)
        pool_trend = rolling_mean(pool_vals)
        novelty_mean = rolling_mean(novelty_vals)

        # 归一化（越大越停滞）：
        #   Δ 均值 0 → 中性；负 → 停滞；正 → 非停滞
        signals["fitness_trend"] = _stall_from_delta(fitness_trend)
        signals["pool_trend"] = _stall_from_delta(pool_trend)
        signals["novelty"] = 1.0 - _squash(novelty_mean)          # 没 novelty → 停滞
        # 复杂度涨 = 停滞（复杂度上升却没换来 fitness/novelty）
        signals["complexity"] = _stall_from_complexity(complexity_deltas)
        # 最近邻相关度高 = 「孩子都在重复已有东西」→ 停滞
        signals["nearest_corr"] = 1.0 - (1.0 - rolling_mean(corr_vals)) if corr_vals else 0.5
        # 失败比例（窗口内）
        signals["failed"] = min(1.0, failed / max(n, 1)) if n else 0.0

        # --- 综合分（各权重相加）---
        w_sum = (
            cfg.w_fitness_trend + cfg.w_pool_trend + cfg.w_novelty
            + cfg.w_complexity + cfg.w_nearest_corr + cfg.w_failed
        ) or 1.0
        total = (
            cfg.w_fitness_trend * signals["fitness_trend"]
            + cfg.w_pool_trend * signals["pool_trend"]
            + cfg.w_novelty * signals["novelty"]
            + cfg.w_complexity * signals["complexity"]
            + cfg.w_nearest_corr * signals["nearest_corr"]
            + cfg.w_failed * signals["failed"]
        )
        if w_sum > 0:
            total /= w_sum

        # 复杂度加速：复杂度涨而 fitness 平 → 提高停滞（ablation 可关）
        if cfg.enable_complexity_accel and complexity_deltas:
            mean_cx = sum(complexity_deltas) / len(complexity_deltas)
            if mean_cx > 0.5 and fitness_trend <= 1e-6:
                total += cfg.complexity_accel * min(1.0, mean_cx / 5.0)

        total = max(0.0, min(1.0, total))

        # 深度小先验：0.5 中性（不在此惩罚 depth；retriever 若要保留极小先验
        # 可自行乘 depth_prior）
        depth_prior = 0.5
        stale_streak = self._stale_streak

        # 是否该停：
        #   a) 综合分超阈值，或
        #   b) stale_streak >= stale_generations（plan Task 11 硬验收：连续 6 代
        #      停滞停掉，浅 branch 也适用）
        # 证据不足（< min_records）→ 中性不判停。
        should_stop = enough and (
            total >= cfg.stop_threshold or stale_streak >= cfg.stale_generations
        )

        result = StagnationResult(
            stagnation=total,
            should_stop=should_stop,
            gain_count=gain_count,
            stale_streak=stale_streak,
            fitness_trend=fitness_trend,
            pool_trend=pool_trend,
            novelty_mean=novelty_mean,
            complexity_trend=sum(complexity_deltas) / len(complexity_deltas) if complexity_deltas else 0.0,
            nearest_corr_mean=rolling_mean(corr_vals),
            failed_count=failed,
            depth_prior=depth_prior,
            signals=dict(signals),
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> StagnationResult | None:
        return self._last_result

    def reset(self) -> None:
        self._records = []
        self._stale_streak = 0
        self._last_result = None


# ---------------------------------------------------------------------------
# 归一化 helpers
# ---------------------------------------------------------------------------


def _squash(x: float) -> float:
    """把任意实数值映射到 (0,1)：负 → 接近 0，0 → 0.5，正 → 接近 1。"""
    return _logistic(x)


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _stall_from_delta(mean_delta: float) -> float:
    """Δ 均值 → 停滞信号（0=非停滞，1=强停滞）。

    - mean >= 0.005 → 0（明确有增益，不停滞）。
    - mean <= -0.005 → 1（明确退化）。
    - 中间线性过渡；0 附近（无证据/微变）→ 0.5 中性偏不罚。
    """
    m = float(mean_delta)
    if m >= 0.005:
        return 0.0
    if m <= -0.005:
        return 1.0
    return 0.5 - m / 0.005 * 0.5  # m=0 → 0.5；m=0.0049 → ~0.01


def _stall_from_complexity(deltas: Sequence[float]) -> float:
    """复杂度总趋势：净增长显著 → 停滞信号强；净减 → 0。"""
    if not deltas:
        return 0.5  # 无证据中性
    net = sum(float(d) for d in deltas) / len(deltas)
    if net <= 0.0:
        return 0.0
    # net ∈ (0, +∞)：映射 net=1 → 0.5；net=5 → ~1
    return min(1.0, net / 5.0)
