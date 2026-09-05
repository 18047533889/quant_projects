"""Macro search paradigm scheduler（plan Task 14 / F4 / AlphaBench CoE/ToT/EA 思想）。

把宏范式（往哪个方向搜）与微层 ActionScheduler（用哪个 arm）解耦：

    Context → {CoE, ToT, EA, Schema, LogicExplore, TrajectoryRepair}
    然后 paradigm 选择 micro action(s)。ActionScheduler 保持 micro 层不动。

``ParadigmScheduler.select`` 流程（全部可复现、零 LLM）：
1. 上下文证据（:class:`ParadigmContext`）→ 对每个 paradigm 判**资格**
   （``eligibility_of``；需求见 :data:`PARADIGM_REQUIREMENTS`）——不满足 =
   不可选（EA 无第二 parent 绝不可选，不合成假 parent；repair 无 critic
   标记绝不可选；COE 对高停滞 lineage 不可选……）。
2. eligible 范式按上下文强度打分（``paradigm_bias``，bandit 历史 reward/
   cost 偏置由注入统计面读，缺省中性）。
3. ``mode="sampled"`` → 按 softmax 分布采样（rng 可复现）；
   ``mode="argmax"`` → 取最高分。
4. 返回 :class:`ParadigmDecision`（范式 + 完整诊断：eligible/scores/dist/
   选中的轨道）。

config（ablation，#30）：``enabled`` / ``mode`` / ``temperature`` /
``repair_enabled`` / ``schema_enabled`` / ``ea_enabled``。``record_decision``
把一次选择记入内部每-paradigm 统计（attempts/reward 由调用方在评估后回传
``update_reward`` 结算——**reward 来自评估而非生成计数**，#26）；历史统计
也可整体注入（``history``），scheduler 不做持久化（持久化留给 ledger 层）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from alphaprobe.search.paradigms import (
    PARADIGM_ORDER,
    Paradigm,
    ParadigmBanditStats,
    ParadigmContext,
    ParadigmSelectionConfig,
    eligibility_of,
    paradigm_bias,
    softmax_distribution,
)

__all__ = [
    "ParadigmDecision",
    "ParadigmScheduler",
    "DEFAULT_PARADIGM_SCHEDULER_CONFIG",
]


@dataclass
class ParadigmDecision:
    """一次范式选择的结果（含完整诊断，测试/审计可断言行为区分）。"""

    paradigm: Paradigm
    eligible: tuple[Paradigm, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    distribution: dict[str, float] = field(default_factory=dict)
    mode: str = "sampled"
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paradigm": self.paradigm.value,
            "eligible": [p.value for p in self.eligible],
            "scores": dict(self.scores),
            "distribution": dict(self.distribution),
            "mode": self.mode,
            "seed": self.seed,
        }


DEFAULT_PARADIGM_SCHEDULER_CONFIG = ParadigmSelectionConfig()


@dataclass
class ParadigmScheduler:
    """宏范式调度器（上下文 → 范式分布 → 采样/argmax，可复现）。

    Parameters
    ----------
    config : ParadigmSelectionConfig | None
        全部开关/模式（ablation，#30）。
    rng : random.Random | None
        采样 rng（固定 seed → 同输入同输出，可复现）。
    history : Mapping[str, Any] | None
        注入的每-paradigm 历史统计面（``{paradigm: {"attempts": …,
        "reward_sum": …}}``）；缺省 None = 无历史（全中性，bandit 不参与）。
        语义与 ledger 的 paradigm 列对齐：由调用方从持久层读、在此注入，
        scheduler 不直接依赖 ledger 表结构。
    """

    config: ParadigmSelectionConfig = field(default_factory=ParadigmSelectionConfig)
    rng: random.Random | None = None
    history: Mapping[str, Any] | None = None
    _stats: dict[str, ParadigmBanditStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = DEFAULT_PARADIGM_SCHEDULER_CONFIG
        if self.rng is None:
            self.rng = random.Random(0)
        # 注入历史 → 内部统计面（缺省中性）
        if self.history:
            for k, v in self.history.items():
                p = _coerce_paradigm(k)
                if p is not None:
                    self._stats[p.value] = ParadigmBanditStats.from_mapping(v)

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------

    def select(self, ctx: ParadigmContext | None = None) -> ParadigmDecision:
        """按上下文选一个范式。

        - ``enabled=False`` → 返回缺省 COE 决策（资格/分布为空；
          ablation 回退语义——调用方应把该范式当普通 COE 微层路径走）。
        - 上下文/eligible 为空 → 均匀分布里也轮不到 ineligible；eligible 也
          空 → 返回 None-like 决策（paradigm=None）。调用方对 None 决策应
          回退微层 action（不额外发散）。
        - ``repair_enabled=False`` → TRAJECTORY_REPAIR 从 eligible 移除；
          ``schema_enabled=False`` → SCHEMA 移除；``ea_enabled=False`` →
          EA 移除（#30：全部范式/repair 有开关）。
        """
        if not self.config.enabled:
            return ParadigmDecision(
                paradigm=Paradigm.COE,
                eligible=(),
                scores={},
                distribution={Paradigm.COE.value: 1.0},
                mode=self.config.mode,
            )
        ctx = ctx if ctx is not None else ParadigmContext()
        eligible = [p for p in PARADIGM_ORDER if self._eligible(p, ctx)]
        if not eligible:
            return ParadigmDecision(
                paradigm=None, eligible=(), scores={}, distribution={},
                mode=self.config.mode,
            )
        # 上下文强度分 + bandit 历史偏置
        scores: dict[str, float] = {}
        for p in eligible:
            st = self._stats.get(p.value, ParadigmBanditStats())
            scores[p.value] = paradigm_bias(p, ctx, st)
        dist = softmax_distribution(
            scores, temperature=self.config.temperature, eligible=eligible
        )
        if self.config.mode == "argmax":
            chosen = max(dist, key=lambda k: (dist[k], k)) if dist else None
        else:
            chosen = self._sample(dist)
        return ParadigmDecision(
            paradigm=chosen,
            eligible=tuple(eligible),
            scores=scores,
            distribution={k: round(v, 6) for k, v in dist.items()},
            mode=self.config.mode,
            seed=getattr(self.rng, "seed_value", None),
        )

    def _eligible(self, paradigm: Paradigm, ctx: ParadigmContext) -> bool:
        if paradigm is Paradigm.TRAJECTORY_REPAIR and not self.config.repair_enabled:
            return False
        if paradigm is Paradigm.SCHEMA and not self.config.schema_enabled:
            return False
        if paradigm is Paradigm.EA and not self.config.ea_enabled:
            return False
        return eligibility_of(paradigm, ctx)

    def _sample(self, dist: Mapping[str, float]) -> Paradigm | None:
        """按分布采样（确定性 rng；边界：全 0/空 → None）。"""
        items = sorted(dist.items())  # 键序确定
        r = self.rng.random()
        acc = 0.0
        for k, prob in items:
            acc += max(0.0, prob)
            if r <= acc:
                p = _coerce_paradigm(k)
                return p
        return None

    # ------------------------------------------------------------------
    # 每-paradigm bandit 统计（attempts/mean_reward；reward 来自评估）
    # ------------------------------------------------------------------

    def record_decision(self, paradigm: Paradigm | str | None) -> None:
        """记录一次选择（attempt 计数 +1；reward 稍后由 update_reward 回传）。

        paradigm=None / 不可识别 → no-op（不污染统计）。
        """
        p = _coerce_paradigm(paradigm)
        if p is None:
            return
        st = self._stats.setdefault(p.value, ParadigmBanditStats())
        st = ParadigmBanditStats(
            attempts=st.attempts + 1,
            reward_sum=st.reward_sum,
            reward_sq_sum=st.reward_sq_sum,
            cost_sum=st.cost_sum,
        )
        self._stats[p.value] = st

    def update_reward(
        self,
        paradigm: Paradigm | str,
        reward: float,
        *,
        cost: float = 0.0,
    ) -> None:
        """评估后回传真实 reward（#26：来自评估闭环，不是生成数）。"""
        p = _coerce_paradigm(paradigm)
        if p is None:
            return
        st = self._stats.get(p.value, ParadigmBanditStats())
        r = float(reward)
        self._stats[p.value] = ParadigmBanditStats(
            attempts=max(0, st.attempts),
            reward_sum=st.reward_sum + r,
            reward_sq_sum=st.reward_sq_sum + r * r,
            cost_sum=st.cost_sum + max(0.0, float(cost)),
        )

    def stats(self) -> dict[str, dict[str, float]]:
        """当前每-paradigm 统计快照（attempts/mean_reward/cost）。"""
        out: dict[str, dict[str, float]] = {}
        for key, s in self._stats.items():
            out[str(key)] = {
                "attempts": float(s.attempts),
                "mean_reward": s.mean_reward,
                "mean_cost": s.mean_cost,
            }
        return out


def _coerce_paradigm(value: Any) -> Paradigm | None:
    if isinstance(value, Paradigm):
        return value
    try:
        return Paradigm(str(value).upper())
    except (TypeError, ValueError):
        return None
