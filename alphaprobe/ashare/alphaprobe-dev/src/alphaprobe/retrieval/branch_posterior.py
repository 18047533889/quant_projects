"""Beta 分支成功后验（任务书 §19-§20）。

Success 定义：child 过 L3（search_valid 段）AND（ΔFactorFitness > delta_fitness
OR ΔPoolUtility > delta_pool）。Beta(alpha0=1, beta0=1) 先验，后验
P_success = (alpha0 + s) / (alpha0 + beta0 + s + f)。

维护每 node 的 n_attempt / n_success / n_elite / mean_delta_fitness /
mean_novelty_gain / mean_pool_utility_gain。

A6（V3.1）：三个 running mean 各自带有效样本计数
（n_delta_fitness / n_novelty_gain / n_delta_pool）——对应 metric 为 None 的
attempt 只推进 n_attempt，不推进该 metric 的计数，也不稀释该 mean（均值只用
自己的有效观测数当分母）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "BetaBranchPosterior",
    "DEFAULT_ALPHA0",
    "DEFAULT_BETA0",
    "DEFAULT_DELTA_FITNESS",
    "DEFAULT_DELTA_POOL",
]

DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0
#: child 需过 L3 且 ΔFactorFitness 超过该值才记为 success（§19 阈值，可配）
DEFAULT_DELTA_FITNESS = 0.005
#: child 需 ΔPoolUtility 超过该值才记为 success（§19 阈值，可配）
DEFAULT_DELTA_POOL = 0.01


@dataclass
class BetaBranchPosterior:
    """node 维度（或 node×action）的分支后验统计。

    构造后未观测任何 attempt 时 posterior_success = 0.5（Beta(1,1) 先验）。
    ``observe_attempt`` 需要外部传入 is_l3_pass + delta_fitness + delta_pool，
    是否达标（success）由本类用阈值判定——保持确定性、可单测。
    """

    alpha0: float = DEFAULT_ALPHA0
    beta0: float = DEFAULT_BETA0
    delta_fitness: float = DEFAULT_DELTA_FITNESS
    delta_pool: float = DEFAULT_DELTA_POOL

    n_attempt: int = 0
    n_success: int = 0
    n_elite: int = 0
    mean_delta_fitness: float = 0.0
    mean_novelty_gain: float = 0.0
    mean_pool_utility_gain: float = 0.0
    # A6：per-metric 有效观测计数（分母与对应 mean 严格一致；None 观测不推进）
    n_delta_fitness: int = 0
    n_novelty_gain: int = 0
    n_delta_pool: int = 0

    _latest: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def observe_attempt(
        self,
        *,
        is_l3_pass: bool,
        delta_fitness: float | None = None,
        delta_pool: float | None = None,
        novelty_gain: float | None = None,
        elite: bool = False,
    ) -> bool:
        """记录一次 attempt，返回本次是否记为 success。

        Success = is_l3_pass AND (ΔFitness > delta_fitness OR ΔPoolUtility > delta_pool)。
        """
        gained = (
            (delta_fitness is not None and delta_fitness > self.delta_fitness)
            or (delta_pool is not None and delta_pool > self.delta_pool)
        )
        success = bool(is_l3_pass and gained)
        self.n_attempt += 1
        if success:
            self.n_success += 1
        if elite:
            self.n_elite += 1
        # A6：每个 mean 只用各自的**有效观测**推进（分母 = 该 metric 非 None 次数，
        # 不是 n_attempt）。metric 为 None → 不推进对应计数、不稀释对应均值。
        if delta_fitness is not None:
            self.n_delta_fitness += 1
            self.mean_delta_fitness = _running_mean(
                self.mean_delta_fitness, float(delta_fitness), self.n_delta_fitness
            )
        if novelty_gain is not None:
            self.n_novelty_gain += 1
            self.mean_novelty_gain = _running_mean(
                self.mean_novelty_gain, float(novelty_gain), self.n_novelty_gain
            )
        if delta_pool is not None:
            self.n_delta_pool += 1
            self.mean_pool_utility_gain = _running_mean(
                self.mean_pool_utility_gain, float(delta_pool), self.n_delta_pool
            )
        self._latest = {
            "is_l3_pass": bool(is_l3_pass),
            "delta_fitness": delta_fitness,
            "delta_pool": delta_pool,
            "novelty_gain": novelty_gain,
            "elite": bool(elite),
            "success": success,
        }
        return success

    def mark_elite(self) -> None:
        self.n_elite += 1

    # ------------------------------------------------------------------
    # 后验
    # ------------------------------------------------------------------

    @property
    def s(self) -> float:
        """Beta 后验成功计数（含先验 alpha0）。"""
        return self.alpha0 + self.n_success

    @property
    def f(self) -> float:
        """Beta 后验失败计数（含先验 beta0）。"""
        return self.beta0 + (self.n_attempt - self.n_success)

    @property
    def posterior_success(self) -> float:
        """P_success = (alpha0 + s) / (alpha0 + beta0 + s + f)。"""
        return self.s / (self.s + self.f)

    @property
    def saturation(self) -> float:
        """attempts ≥ 10 后才启用（与 memory store 的 exploration_state 一致）。"""
        if self.n_attempt < 10:
            return 0.0
        return 1.0 - self.n_success / max(self.n_attempt, 1)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "delta_fitness": self.delta_fitness,
            "delta_pool": self.delta_pool,
            "n_attempt": self.n_attempt,
            "n_success": self.n_success,
            "n_elite": self.n_elite,
            "mean_delta_fitness": self.mean_delta_fitness,
            "mean_novelty_gain": self.mean_novelty_gain,
            "mean_pool_utility_gain": self.mean_pool_utility_gain,
            # A6：per-metric 有效样本计数（序列化保真，往返后分母不丢）
            "n_delta_fitness": self.n_delta_fitness,
            "n_novelty_gain": self.n_novelty_gain,
            "n_delta_pool": self.n_delta_pool,
            "posterior_success": self.posterior_success,
            "saturation": self.saturation,
        }
        if self._latest:
            d["latest"] = dict(self._latest)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BetaBranchPosterior":
        d = dict(d or {})
        latest = d.pop("latest", None)
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # 历史 dict（V3.1 之前）没有 per-metric 计数 → 由 n_attempt 时代的
        # 均值反推有效样本数：只要 mean 非 0/有观测，令计数 = n_attempt。
        # （向后兼容：旧运行数据 reload 后均值语义不变。）
        obj = cls(**kwargs)
        if obj.n_delta_fitness <= 0 and obj.n_attempt > 0 and obj.mean_delta_fitness != 0.0:
            obj.n_delta_fitness = obj.n_attempt
        if obj.n_novelty_gain <= 0 and obj.n_attempt > 0 and obj.mean_novelty_gain != 0.0:
            obj.n_novelty_gain = obj.n_attempt
        if obj.n_delta_pool <= 0 and obj.n_attempt > 0 and obj.mean_pool_utility_gain != 0.0:
            obj.n_delta_pool = obj.n_attempt
        if isinstance(latest, dict):
            obj._latest = dict(latest)
        return obj


def _running_mean(current: float, new_value: float, n: int) -> float:
    if n <= 0:
        return current
    return current + (new_value - current) / n
