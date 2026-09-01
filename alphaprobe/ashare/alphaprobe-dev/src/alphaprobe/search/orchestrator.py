"""SearchOrchestrator（任务书 §26-§32 / §76）。

单步流程：scheduler 选 arm（action）→ 构造 prompt（MemoryPacket.to_prompt_text，
§32 不塞完整 lineage）→ 调 llm_fn（stub/真实均可）→ parse → 返回候选列表 + 记录
AttemptRecord（cost/latency 首版 stub 0，§30 真实记账留给 LLM client 层）。

Phase 9 不调用真实 LLM：llm_fn 由调用方注入（测试给 stub；真实路径接
shared/utils/llm 的 OpenAIModel.chat_generate）。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from alphaprobe.contracts import AttemptRecord, SearchAction
from alphaprobe.search import ACTION_FAMILIES, ActionScheduler, make_action
from alphaprobe.search.arms import (
    BranchArm,
    EvolutionArm,
    RefineArm,
    SchemaExploreArm,
    _route,
)
from alphaprobe.search.structured_llm import (
    GeneratedCandidate,
    candidates_to_json,
    parse_generated,
)

LLMFn = Callable[[str, str, str], str]

# arm 单步默认期望候选数
ARM_EXPECTED: dict[str, int] = {
    "REFINE": 3,
    "WINDOW_SCALE": 3,
    "FIELD_SUBSTITUTION": 3,
    "OPERATOR_SUBSTITUTION": 3,
    "STATE_CONDITION": 3,
    "CROSSOVER": 3,
    "SCHEMA_EXPLORE": 3,
    "ROBUSTIFY": 3,
    "GENERATION_REPAIR": 1,
}

_ARM_BY_FAMILY: dict[str, Any] = {
    "REFINE": RefineArm,
    "STATE_CONDITION": BranchArm,
    "CROSSOVER": EvolutionArm,
    "SCHEMA_EXPLORE": SchemaExploreArm,
    # 其余 family 首版回落到 RefineArm 的轻量生成
}


@dataclass
class OrchestratorStep:
    """单步结果：候选 + 该步 AttemptRecord + action + 模型路由。"""

    action: SearchAction
    model_class: str
    candidates: list[GeneratedCandidate] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)


@dataclass
class SearchOrchestrator:
    """多臂搜索编排器。

    - scheduler: ActionScheduler（§29，已含 Thompson/UCB）
    - memory: 可选注入。提供 build_memory_packet(parent_node) / MemoryPacket，否则
      直接用 parent dict 拼最小 prompt（§32 降级）。
    - expected_num: 单步生成候选数
    """

    scheduler: ActionScheduler = field(default_factory=ActionScheduler)
    memory: Any | None = None
    expected_num: int = 3
    round_id: str = "r1"
    generation: int = 0

    # 允许测试注入计数/延迟
    latency_ms: int = 0
    llm_cost: float = 0.0
    eval_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.scheduler is None:
            self.scheduler = ActionScheduler()
        if self.generation == 0:
            self.generation = 1

    # ------------------------------------------------------------------
    # prompt 构造（§32：MemoryPacket.to_prompt_text 替代完整 lineage）
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        parents: Sequence[dict[str, Any]],
        action_type: str,
        num: int,
    ) -> tuple[str, str]:
        """构造 (system_prompt, user_prompt)。

        user_prompt 优先用 MemoryPacket.to_prompt_text（若 memory 提供 packet）；
        否则退化为 parent 摘要行（首版离线可用）。
        """
        from alphaprobe.search.arms import _build_prompt

        packet_text: str | None = None
        if self.memory is not None and parents:
            parent = parents[0]
            if hasattr(self.memory, "build_memory_packet"):
                packet = self.memory.build_memory_packet(parent_node=dict(parent))
                if hasattr(packet, "to_prompt_text"):
                    packet_text = packet.to_prompt_text()
        instruction = f"Generate up to {num} candidates under action {action_type}."
        sys_p, usr_p = _build_prompt(
            parents, action_type, num, instruction
        )
        if packet_text:
            usr_p = usr_p + "\n## MemoryPacket (retriever view, §32)\n" + packet_text
        return sys_p, usr_p

    # ------------------------------------------------------------------
    # 单步
    # ------------------------------------------------------------------

    def step(
        self,
        parent_node: dict[str, Any],
        llm_fn: LLMFn,
    ) -> OrchestratorStep:
        """选 arm → 构造 prompt → 调 llm_fn → parse → 记录 AttemptRecord。

        model_route 按 action 选 cheap/strong（§30）。
        """
        action_type = self.scheduler.select(list(ACTION_FAMILIES))
        model_class = _route(action_type)
        parents = [parent_node]
        num = ARM_EXPECTED.get(action_type, self.expected_num)
        sys_p, usr_p = self.build_prompt(parents, action_type, num)
        t0 = time.monotonic()
        text = llm_fn(sys_p, usr_p, model_class)
        self.latency_ms = int((time.monotonic() - t0) * 1000)

        # GenerationRepairArm 走 search/__init__.py 的修复逻辑（§25.5），其余走 arm.generate
        if action_type == "GENERATION_REPAIR":
            candidates = self._repair_candidates(parent_node, text, action_type)
        else:
            arm = self._arm_for(action_type)
            candidates = arm.generate(parents, None, llm_fn=None) if arm is not None else []

        action = make_action(
            action_type=action_type,
            parent_ids=[str(parent_node.get("factor_id") or parent_node.get("id") or "")]
            if parent_node
            else [],
            round_id=self.round_id,
            generation=self.generation,
            llm_model_class=model_class,
        )
        attempts = [
            AttemptRecord(
                attempt_id=f"at_{uuid.uuid4().hex[:12]}",
                action=action,
                candidate_factor_id=None,
                outcome="ok",
                llm_cost=self.llm_cost,
                eval_cost=self.eval_cost,
                latency_ms=self.latency_ms,
            )
            for _ in range(max(1, len(candidates)))
        ]
        return OrchestratorStep(
            action=action,
            model_class=model_class,
            candidates=candidates,
            attempts=attempts,
        )

    def step_with_llm(
        self,
        parent_node: dict[str, Any],
        llm_fn: LLMFn,
        *,
        reward: float = 0.0,
    ) -> OrchestratorStep:
        """step + 用 reward 更新 scheduler 统计（测试/闭环方便）。"""
        result = self.step(parent_node, llm_fn)
        self.scheduler.update(
            result.action.action_type.value, reward=reward or float(len(result.candidates))
        )
        return result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _arm_for(self, action_type: str) -> Any | None:
        cls = _ARM_BY_FAMILY.get(action_type)
        if cls is None:
            return None
        return cls()

    def _repair_candidates(
        self,
        parent_node: dict[str, Any],
        llm_text: str,
        action_type: str,
    ) -> list[GeneratedCandidate]:
        """GenerationRepairArm：先尝试 LLM 结构化输出，空/失败再本地 repair。

        本地 repair 复用 search/__init__.py 的 GenerationRepairArm（§25.5 括号修复）。
        """
        from alphaprobe.search import GenerationRepairArm

        # 1) 尝试结构化解析（若 llm 给了 candidates）
        cands = parse_generated(llm_text, expected_num=1, default_action_type=action_type)
        if cands:
            repaired = cands
        else:
            raw = str(parent_node.get("formula") or parent_node.get("canonical_formula") or "")
            note: str = "syntax"
            repaired_formula: str | None = raw
            try:
                repaired_formula, note = GenerationRepairArm().repair(raw)
            except Exception:
                repaired_formula = raw
            if repaired_formula is None:
                return []
            repaired = [
                GeneratedCandidate(
                    formula=repaired_formula,
                    explanation=f"repair ({note})",
                    hypothesis="修复无效生成，使其可被 factor_engine DSL 解析",
                    action_type=action_type,
                    parent_ids=[str(parent_node.get("factor_id") or "")],
                    schema_tags=dict(parent_node.get("schema_tags") or {}),
                )
            ]
        # 2) 修复后仍做一次括号自检（若仍不平衡则丢弃）
        final: list[GeneratedCandidate] = []
        for c in repaired:
            f = str(c.formula).strip()
            if f.count("(") == f.count(")"):
                final.append(c)
        return final


__all__ = ["SearchOrchestrator", "OrchestratorStep"]
