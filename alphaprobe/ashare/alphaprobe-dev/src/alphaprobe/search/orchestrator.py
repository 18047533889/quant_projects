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
    duplicates_filtered: int = 0


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

    # V2-D：可选 parent selector（Bayesian Retriever 排序）。None = 关闭，
    # 行为与旧版完全一致（向后兼容，全量测试不挂）。
    parent_selector: Any | None = None

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
        llm_fn: LLMFn | None,
        *,
        dedup_client: Any | None = None,
        extra_parents: Sequence[dict[str, Any]] | None = None,
        select_complement: bool = False,
        top_k: int = 5,
    ) -> OrchestratorStep:
        """选 arm → 构造 prompt → 调 llm_fn → parse → 记录 AttemptRecord。

        model_route 按 action 选 cheap/strong（§30）。

        P0-4：llm_fn 非 None 时，LLM 返回文本必须走 structured_llm.parse_generated
        真正产出 candidates（绝不丢弃/仅记 log）；parse 失败（空结果）进入 repair
        流程（GENERATION_REPAIR / GenerationRepairArm 括号修复），不静默丢。
        llm_fn=None 时走 arm.generate 本地确定性生成，不触发任何真实 LLM。

        dedup_client 可选注入（§41）：raw formula 列表生成后、进昂贵评估前，
        先 check_new_candidate 过滤硬重复（EXACT/SIGN）。默认 None = 行为不变。

        extra_parents：可选 multi-parent 补充池（§27 role-aware crossover 的第二
        parent 候选）。CROSSOVER 臂默认按结构远度从 extra_parents 选 1 个补充
        parent；显式传 select_complement=False 时禁用该挑选（原始 parents 语义）。

        兼容保证：不传 extra_parents / select_complement 时行为与旧版完全一致
        （parents=[parent_node]，无 multi-parent 挑选）。
        """
        action_type = self.scheduler.select(list(ACTION_FAMILIES))
        model_class = _route(action_type)
        parents = [parent_node]
        if extra_parents:
            chosen = list(extra_parents)
            # V2-D：parent_selector 注入时，用 Bayesian Retriever（RetrieverScore 降序）
            # 选补充 parent 池；默认 None → 走原 structural-distance 挑选。
            if self.parent_selector is not None:
                chosen = self.parent_selector.select_parents(
                    chosen, top_k=top_k, main=parent_node,
                    action_to_retrieve=action_type,
                )
            elif select_complement:
                chosen = self._complement_parents(parent_node, list(extra_parents), top_k=top_k)
            parents.extend(chosen)
        num = ARM_EXPECTED.get(action_type, self.expected_num)
        sys_p, usr_p = self.build_prompt(parents, action_type, num)
        t0 = time.monotonic()
        text = llm_fn(sys_p, usr_p, model_class) if llm_fn is not None else ""
        self.latency_ms = int((time.monotonic() - t0) * 1000)

        # P0-4：LLM 文本一律进 parse_generated；parse 失败进入 repair，不静默丢。
        if llm_fn is not None:
            candidates = self._parse_or_repair(text, action_type, num, parents, parent_node)
        elif action_type == "GENERATION_REPAIR":
            # 无 LLM 且是 repair arm：用 parent 的 raw formula 走本地 repair
            raw = str(parent_node.get("formula") or parent_node.get("canonical_formula") or "")
            candidates = self._repair_candidates(parent_node, raw, action_type) if raw else []
        else:
            arm = self._arm_for(action_type)
            candidates = arm.generate(parents, None, llm_fn=None) if arm is not None else []

        # §41：进昂贵评估前过滤硬重复（EXACT/SIGN）。挂 client 前行为完全不变。
        duplicates_filtered = 0
        if dedup_client is not None and candidates:
            kept: list[GeneratedCandidate] = []
            for cand in candidates:
                try:
                    verdict = dedup_client.check_new_candidate(cand.formula)
                except Exception:  # noqa: BLE001 - 去重失败不阻塞生成
                    verdict = None
                if verdict is not None and verdict.rejection_reason is not None:
                    duplicates_filtered += 1
                    continue
                kept.append(cand)
            candidates = kept

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
            duplicates_filtered=duplicates_filtered,
        )

    def step_with_llm(
        self,
        parent_node: dict[str, Any],
        llm_fn: LLMFn | None,
        *,
        reward: float = 0.0,
        dedup_client: Any | None = None,
    ) -> OrchestratorStep:
        """step + 用 reward 更新 scheduler 统计（测试/闭环方便）。"""
        result = self.step(parent_node, llm_fn, dedup_client=dedup_client)
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

    # ------------------------------------------------------------------
    # §27 multi-parent 补充选择：top-K 中按「结构远」挑 complement parent
    # ------------------------------------------------------------------

    @staticmethod
    def _complement_parents(
        main: dict[str, Any],
        candidates: Sequence[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """从候选池里挑 1 个与 main 结构最远的 parent 作 crossover 补充。

        距离 = 归一化字符串距离（Damerau-Levenshtein / 字符串编辑距离）。
        ``candidates`` 已视为 top-K 候选（调用方按 fitness 排好序）；
        这里只做简单确定性结构远度挑选，不跑模型、不做 embedding。
        挑不出（空/全等）→ 原样返回候选（不抛）。
        """
        import difflib

        cands = [c for c in candidates if c is not None]
        if not cands:
            return []
        main_f = str(main.get("formula") or main.get("canonical_formula") or "").strip()
        scored: list[tuple[float, dict[str, Any]]] = []
        for c in cands:
            cf = str(c.get("formula") or c.get("canonical_formula") or "").strip()
            if not cf:
                continue
            if main_f and cf == main_f:
                # 与 main 完全相同的公式：距离视为最远（不同 niche 的等价物），
                # 仍可作为补充（multi-parent crossover 允许同公式不同角色）。
                dist = 1.0
            else:
                dist = difflib.SequenceMatcher(None, main_f, cf).ratio()
                # ratio ∈ [0,1]，1=完全相同；取「结构远」= 1-ratio
                dist = 1.0 - dist
            scored.append((dist, c))
        if not scored:
            return list(candidates)
        scored.sort(key=lambda t: t[0], reverse=True)
        picked = [c for _, c in scored[:top_k]]
        return picked[:1]

    def _parse_or_repair(
        self,
        llm_text: str,
        action_type: str,
        num: int,
        parents: list[dict[str, Any]],
        parent_node: dict[str, Any],
    ) -> list[GeneratedCandidate]:
        """P0-4：LLM 文本一律走 parse_generated；parse 空/失败进 repair，不静默丢。

        GENERATION_REPAIR 直接走 _repair_candidates（其内部也先试结构化解析，
        空/失败再本地括号修复）；其余 arm 先 parse_generated，空结果走本地 repair
        兜底（GenerationRepairArm 括号修复）。
        """
        if action_type == "GENERATION_REPAIR":
            return self._repair_candidates(parent_node, llm_text, action_type)
        candidates = parse_generated(llm_text, expected_num=num, default_action_type=action_type)
        if candidates:
            return candidates
        # parse 失败（空结果）：进入 repair 流程，不静默丢
        raw = str(parent_node.get("formula") or parent_node.get("canonical_formula") or "")
        if not raw:
            return []
        return self._repair_candidates(parent_node, raw, action_type)

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
