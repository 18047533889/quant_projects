"""Beta 分支成功后验（任务书 §19-§20）。

Success 定义：child 过 L3（search_valid 段）AND（ΔFactorFitness > delta_fitness
OR ΔPoolUtility > delta_pool）。Beta(alpha0=1, beta0=1) 先验，后验
P_success = (alpha0 + s) / (alpha0 + beta0 + s + f)。

维护每 node 的 n_attempt / n_success / n_elite / mean_delta_fitness /
mean_novelty_gain / mean_pool_utility_gain。
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
        if delta_fitness is not None:
            self.mean_delta_fitness = _running_mean(
                self.mean_delta_fitness, float(delta_fitness), self.n_attempt
            )
        if novelty_gain is not None:
            self.mean_novelty_gain = _running_mean(
                self.mean_novelty_gain, float(novelty_gain), self.n_attempt
            )
        if delta_pool is not None:
            self.mean_pool_utility_gain = _running_mean(
                self.mean_pool_utility_gain, float(delta_pool), self.n_attempt
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
        obj = cls(**kwargs)
        if isinstance(latest, dict):
            obj._latest = dict(latest)
        return obj


def _running_mean(current: float, new_value: float, n: int) -> float:
    if n <= 0:
        return current
    return current + (new_value - current) / n
