"""search（任务书 §25-§32 / §76）：SearchAction 已在 contracts；此处 5 臂 + Scheduler。

GenerationRepairArm != Factor Refinement Engine（§25.5）：前者只把无效生成
改成合法公式；后者是对有效因子的经济/统计形状做后处理。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from alphaprobe.contracts import SearchAction, SearchActionType

ACTION_FAMILIES = [t.value for t in SearchActionType]


# ---------------------------------------------------------------------------
# §25.5 GenerationRepairArm：parser/type/field/DSL syntax/unsupported operator
# ---------------------------------------------------------------------------


class GenerationRepairArm:
    def __init__(self, repair_fn: Any | None = None) -> None:
        self._repair_fn = repair_fn

    def repair(self, raw: str) -> tuple[str | None, str]:
        """返回 (repaired_formula, note)。只修语法，不改统计形状。"""
        if self._repair_fn is not None:
            return self._repair_fn(raw)
        s = str(raw).strip().rstrip(";")
        # 常见括号不平衡修复
        if s.count("(") == s.count(")") + 1:
            s = s + ")"
        elif s.count(")") == s.count("(") + 1:
            s = "(" + s
        return (s if s.count("(") == s.count(")") else None), "syntax"


# ---------------------------------------------------------------------------
# §29 ActionScheduler：Thompson Sampling / UCB
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    attempts: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.attempts if self.attempts else 0.0

    @property
    def var(self) -> float:
        if self.attempts < 2:
            return 1.0
        m = self.mean
        return max(1e-6, self.reward_sq_sum / self.attempts - m * m)


class ActionScheduler:
    """Reward = ΔPoolUtility + ValidQualityGain + NoveltyGain + SchemaCoverageGain
    + SurvivalGain - EvaluationCost - LLMCost - FailurePenalty（§29）。"""

    def __init__(self, method: str = "thompson", *, rng: random.Random | None = None) -> None:
        self.method = method
        self.rng = rng or random.Random(0)
        self.stats: dict[str, ArmStats] = {}

    def select(self, action_families: list[str] | None = None) -> str:
        fams = action_families or ACTION_FAMILIES
        for f in fams:
            self.stats.setdefault(f, ArmStats())
        # 零尝试臂先轮询探索（UCB/Thompson 共同的 cold-start 规则）
        for f in fams:
            if self.stats[f].attempts == 0:
                return f
        if self.method == "thompson":
            best, best_v = None, -math.inf
            for f in fams:
                st = self.stats[f]
                # 正态 Thompson（Binghams 简化）
                try:
                    v = self.rng.gauss(st.mean, math.sqrt(st.var))
                except ValueError:
                    v = st.mean
                if v > best_v:
                    best, best_v = f, v
            return best
        # UCB1
        total = sum(self.stats[f].attempts for f in fams) or 1
        best, best_v = None, -math.inf
        for f in fams:
            st = self.stats[f]
            ucb = st.mean + math.sqrt(2 * math.log(total) / st.attempts)
            if ucb > best_v:
                best, best_v = f, ucb
        return best

    def update(self, action_family: str, reward: float) -> None:
        st = self.stats.setdefault(action_family, ArmStats())
        st.attempts += 1
        st.reward_sum += reward
        st.reward_sq_sum += reward * reward

    def state(self) -> dict[str, dict[str, float]]:
        return {
            f: {"attempts": s.attempts, "mean": s.mean}
            for f, s in self.stats.items()
        }


def make_action(
    *,
    action_type: str,
    parent_ids: list[str],
    round_id: str,
    generation: int,
    llm_model_class: str = "cheap",
    budget_class: str = "normal",
    target_role: str | None = None,
    target_schema: dict[str, str] | None = None,
) -> SearchAction:
    import uuid

    return SearchAction(
        action_id=f"act_{uuid.uuid4().hex[:12]}",
        action_type=SearchActionType(action_type),
        parent_ids=parent_ids,
        target_role=target_role,
        target_schema=target_schema,
        budget_class=budget_class,
        llm_model_class=llm_model_class,
        created_round=round_id,
        created_generation=generation,
    )


# ---------------------------------------------------------------------------
# §30 LLM 模型路由
# ---------------------------------------------------------------------------

CHEAP_ACTIONS = {
    "REFINE", "GENERATION_REPAIR", "FIELD_SUBSTITUTION", "OPERATOR_SUBSTITUTION",
    "WINDOW_SCALE",
}
STRONG_ACTIONS = {
    "CROSSOVER", "SCHEMA_EXPLORE", "STATE_CONDITION", "ROBUSTIFY",
}


def model_route(action_type: str, *, cheap_model: str = "cheap", strong_model: str = "strong") -> str:
    """不要所有 action 都用最强模型。"""
    return strong_model if action_type in STRONG_ACTIONS else cheap_model


# ---------------------------------------------------------------------------
# §56.3 Lineage Early Stop
# ---------------------------------------------------------------------------


class LineagePatienceTracker:
    """连续 patience 代无任何 gain（fitness/novelty/pool_utility/schema/survival）→ stop。"""

    def __init__(self, patience: int = 6) -> None:
        self.patience = patience
        self._no_gain_streak: dict[str, int] = {}

    def observe(self, lineage_key: str, gains: dict[str, float | None]) -> bool:
        """返回 True = 应停止该 branch。"""
        has_gain = any(
            (v is not None and v > 1e-9) for v in gains.values()
        )
        streak = self._no_gain_streak.get(lineage_key, 0)
        if has_gain:
            streak = 0
        else:
            streak += 1
        self._no_gain_streak[lineage_key] = streak
        return streak >= self.patience

    def strong_lineage_bonus(self, lineage_key: str) -> float:
        """强 lineage 获得更多预算（§56.3）。"""
        return 1.0 / (1.0 + self._no_gain_streak.get(lineage_key, 0))