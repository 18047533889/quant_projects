"""SearchOrchestrator（任务书 §26-§32 / §76）。

单步流程：scheduler 选 arm（action）→ 构造 prompt（MemoryPacket.to_prompt_text，
§32 不塞完整 lineage）→ 调 llm_fn（stub/真实均可）→ parse → 返回候选列表 + 记录
AttemptRecord（cost/latency 首版 stub 0，§30 真实记账留给 LLM client 层）。

Phase 9 不调用真实 LLM：llm_fn 由调用方注入（测试给 stub；真实路径接
shared/utils/llm 的 OpenAIModel.chat_generate）。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)

LLMFn = Callable[[str, str, str], str]


def _now_iso() -> str:
    """UTC ISO 时间戳（reward_settled_at 用；重放/audit 可比对）。"""
    return datetime.now(timezone.utc).isoformat()

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
    """单步结果：候选 + 该步 AttemptRecord + action + 模型路由。

    P0-B（delayed reward 闭环）：reward 来自评估闭环，不在生成时用
    candidate 数充数。``pending_reward=True`` 表示该步尚未回传真实 reward，
    待下游评估完成后用 ``SearchOrchestrator.settle_reward(step, reward)``
    回填并更新 scheduler 统计。
    """

    action: SearchAction
    model_class: str
    candidates: list[GeneratedCandidate] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    duplicates_filtered: int = 0
    #: 尚未回传真实 reward（delayed reward 闭环的标记；A8 幂等守卫配套）
    pending_reward: bool = True
    #: 该 step 关联的 attempt 主键（delayed reward 结算的账本锚点）。
    #: 默认 ``None`` = 未接入账本（settle_reward 退化为内存幂等守卫）。
    attempt_id: str | None = None
    #: 该 step 的 reward 是否已结算（幂等守卫：同一 step 只结算一次）。
    reward_settled: bool = False
    #: 结算时间戳（ISO，UTC）——重放/resume/audit 用。
    reward_settled_at: str | None = None
    #: 结算事件 id（子代评估事件 → reward 可溯源）。
    reward_event_id: str | None = None


@dataclass
class SearchOrchestrator:
    """多臂搜索编排器。

    - scheduler: ActionScheduler（§29，已含 Thompson/UCB）
    - memory: 可选注入。提供 build_memory_packet(parent_node) / MemoryPacket，否则
      直接用 parent dict 拼最小 prompt（§32 降级）。
    - ledger: 可选注入（alphaprobe.ledger.CandidateAttemptLedger，plan.md Task 4）。
      非 None 时 ``settle_reward`` 以账本 DB 仲裁为准（谁先写谁赢，重放/resume/
      worker retry 绝不重复计 reward）；None 时行为与旧版一致但不再重复计数
      （内存幂等守卫）。
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

    # V2-H：结构化 generation 开关。False（默认）= 行为与旧版完全一致（全量测试
    # 不挂）。True = 启用后，生成候选的 parents 来自 parent_selector 的 top-k DAG
    # 采样（而非单 parent），且每个候选记录 lineage（parent ids 列表）。
    structured_generation: bool = False
    #: 结构化 generator 注入（默认惰性构造 StructuredGenerator）。
    structured_generator: Any | None = None

    #: 允许测试注入计数/延迟
    latency_ms: int = 0
    llm_cost: float = 0.0
    eval_cost: float = 0.0

    # A8 / plan.md Task 4：CandidateAttemptLedger（可选注入）。None = 未接账本
    # （settle_reward 用内存幂等守卫，不重复计数）。
    ledger: Any | None = None

    def __post_init__(self) -> None:
        if self.scheduler is None:
            self.scheduler = ActionScheduler()
        if self.generation == 0:
            self.generation = 1
        # A8：ledger 缺省（``None``）时按环境变量/默认路径惰性构造真实账本。
        # 实例创建/连接在首次 settle 时才发生（``_settle_via_ledger``），因此
        # 默认构造 ``SearchOrchestrator()`` 不会在当前目录留空 DB 文件（磁盘
        # 纪律：不 settle 就不建库）。调用方可在构造后显式替换为 ``None``
        # 关闭账本（settle 退化为内存幂等守卫）。

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
                    chosen, k=top_k, main=parent_node,
                    action_to_retrieve=action_type,
                )
            elif select_complement:
                chosen = self._complement_parents(parent_node, list(extra_parents), top_k=top_k)
            parents.extend(chosen)

        # V2-H：结构化 generation 开关。启用后，parents 来自 parent_selector 的
        # top-k DAG 采样（而非单 parent），且每个候选记录 lineage（parent ids）。
        # 默认 False = 行为与旧版完全一致（全量测试不挂）。
        if self.structured_generation:
            return self._step_structured(
                parent_node, llm_fn, parents, action_type, model_class,
                dedup_client=dedup_client,
            )

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
        reward: float | None = None,
        dedup_client: Any | None = None,
        attempt_id: str | None = None,
    ) -> OrchestratorStep:
        """step + 可选更新 scheduler 统计（测试/闭环方便）。

        Parameters
        ----------
        reward : float | None
            - ``None``（默认）：**不更新 scheduler**，step 标记
              ``pending_reward=True``，等下游评估完成后用 ``settle_reward``
              回传真实 reward（delayed reward 闭环）。
            - 显式 float（含 ``0.0``）：照常 ``scheduler.update`` 并标记
              ``pending_reward=False``。
        attempt_id : str | None
            A8/plan.md Task 4：该 step 的账本锚点。None（默认）= 由本方法
            自动派生（``at_{uuid4}``），与旧版行为一致（不传即自动生成）。
            跨进程 resume 时调用方应复用账本里已持久化的 attempt_id，配合
            ledger 的 DB 仲裁保证只结算一次。

        P0-B：reward **绝不**退化为 candidate 数——生成 10 个垃圾的 reward
        不可能高于 2 个优质。reward 应来自评估闭环（公式参考）：
        ``0.45×ΔFactorFitness + 0.30×ΔPoolUtility + 0.15×NoveltyGain
        + 0.10×SchemaCoverageGain − 成本项``。
        """
        result = self.step(parent_node, llm_fn, dedup_client=dedup_client)
        result.attempt_id = attempt_id or f"at_{uuid.uuid4().hex[:12]}"
        if reward is None:
            # delayed reward：等 settle_reward 回传，暂不更新 scheduler
            result.pending_reward = True
            return result
        self.scheduler.update(
            result.action.action_type.value, reward=float(reward)
        )
        result.pending_reward = False
        result.reward_settled = True
        return result

    def settle_reward(self, step: OrchestratorStep, reward: float) -> None:
        """用真实 reward（评估闭环产出）回填一步并更新 scheduler 统计。

        A8 幂等语义（retry/resume/worker retry 不重复计 reward）：

        - 同一 ``OrchestratorStep`` 第二次 settle 为 no-op（``reward_settled``
          内存守卫——即使未接账本也绝不重复 ``scheduler.update``）；
        - 已用显式 reward 更新过的 step（``pending_reward=False`` /
          ``reward_settled=True``）再 settle 同样 no-op；
        - 接入 ``ledger`` 时以账本 DB 仲裁为准：同 ``attempt_id`` 谁先写谁赢，
          赢家才更新 scheduler；重放（resume 复用同一 ``attempt_id``）或两个
          worker 抢同一 attempt 结算都只有一个赢家（test #1/#2/#5 验收）。
        - 未接账本（ledger=None，测试/旧路径）：内存守卫保证同一 step 不重复
          计；跨进程语义需调用方复用持久化 attempt_id（TODO 见 ledger.py 头）。
        """
        # 内存幂等守卫：同一 step 只结算一次（A8 兜底，即使无账本也不重复计）
        if getattr(step, "reward_settled", False):
            return
        family = step.action.action_type.value
        attempt_id = getattr(step, "attempt_id", None) or f"stp_{uuid.uuid4().hex[:12]}"
        event_id = getattr(step, "reward_event_id", None)
        if event_id is None:
            event_id = f"stt_{uuid.uuid4().hex[:12]}"
        step.reward_settled_at = _now_iso()
        won = True
        reason = None
        if self.ledger is not None:
            won, reason = self._settle_via_ledger(attempt_id, float(reward), event_id)
        if not won:
            step.reward_event_id = event_id
            if reason and reason.startswith("ledger error"):
                # 瞬时账本故障：结算状态未知，step 保持 pending 供调用方重试
                logger.warning(
                    "settle_reward ledger error for %s (event %s): %s",
                    attempt_id, event_id, reason,
                )
                return
            # 输 = 账本仲裁定局（同 attempt 已被他人/重放结算）：reward 归属已
            # 尘埃落定，step 关闭结算（绝不再尝试、绝不更新 scheduler）。
            if reason:
                logger.warning(
                    "settle_reward lost arbitration for %s (event %s): %s",
                    attempt_id, event_id, reason,
                )
            step.pending_reward = False
            step.reward_settled = True
            return
        self.scheduler.update(family, reward=float(reward))
        step.pending_reward = False
        step.reward_settled = True
        step.reward_event_id = event_id

    # ------------------------------------------------------------------
    # A8：settle_reward 幂等（同一 step / 同一 attempt 绝不重复计 reward）
    # ------------------------------------------------------------------

    def _settle_via_ledger(
        self,
        attempt_id: str,
        reward: float,
        reward_event_id: str,
    ) -> tuple[bool, str | None]:
        """账本仲裁路径：返回 (won, reason)。ledger 为 None 时恒 (True, None)。

        ledger 实例惰性构造：第一次真实 settle 时才建库（``ALPHAPROBE_LEDGER_DB``
        环境变量可覆盖默认路径；测试注入 tmp_path 账本后此处不再自建）。
        """
        from alphaprobe.ledger import CandidateAttemptLedger

        if not isinstance(self.ledger, CandidateAttemptLedger):
            if self.ledger is None:
                self.ledger = CandidateAttemptLedger()
            elif not (
                hasattr(self.ledger, "settle_reward")
                and hasattr(self.ledger, "db_path")
            ):
                # 无账本语义的对象（鸭子类型不合格）：回退 None → 不仲裁
                self.ledger = None
        if self.ledger is None:
            return True, None
        try:
            won, reason = self.ledger.settle_reward(
                attempt_id=attempt_id,
                reward=float(reward),
                reward_event_id=reward_event_id,
            )
            return bool(won), reason
        except Exception as exc:  # noqa: BLE001 - 账本故障不阻塞闭环
            logger.warning("ledger settle failed for %s: %s", attempt_id, exc)
            return False, f"ledger error: {exc}"

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _arm_for(self, action_type: str) -> Any | None:
        cls = _ARM_BY_FAMILY.get(action_type)
        if cls is None:
            return None
        return cls()

    def _step_structured(
        self,
        parent_node: dict[str, Any],
        llm_fn: LLMFn | None,
        parents: list[dict[str, Any]],
        action_type: str,
        model_class: str,
        *,
        dedup_client: Any | None = None,
    ) -> OrchestratorStep:
        """结构化 generation：parents 来自 parent_selector 的 top-k DAG 采样。

        - llm_fn 输出经 ``normalize_llm_output`` 归一化为 action dict 列表
          （旧文本输出 → identity action fallback，向后兼容）；
        - ``StructuredGenerator.generate`` 把 action 应用到 parents → FE validate
          → 候选池（非法/非白名单 op/FE 校验失败一律拒绝并记录原因）；
        - 每个候选记录 lineage（parent ids 列表）。
        """
        from alphaprobe.generation.structured import (
            StructuredGenerator,
            normalize_llm_output,
        )

        gen = self.structured_generator
        if gen is None:
            gen = StructuredGenerator()
            self.structured_generator = gen

        num = ARM_EXPECTED.get(action_type, self.expected_num)
        sys_p, usr_p = self.build_prompt(parents, action_type, num)
        t0 = time.monotonic()
        output = llm_fn(sys_p, usr_p, model_class) if llm_fn is not None else None
        self.latency_ms = int((time.monotonic() - t0) * 1000)

        actions = normalize_llm_output(output, parents=parents, default_action_type=action_type)
        result = gen.generate(actions, parents, default_action_type=action_type)

        candidates: list[GeneratedCandidate] = []
        for cand in result.candidates:
            candidates.append(
                GeneratedCandidate(
                    formula=cand.formula,
                    explanation=cand.explanation,
                    hypothesis=cand.hypothesis,
                    action_type=cand.action_type,
                    parent_ids=list(cand.parent_ids),
                    schema_tags=dict(cand.schema_tags),
                )
            )

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
