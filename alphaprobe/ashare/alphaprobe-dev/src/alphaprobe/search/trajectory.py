"""Search Trajectory Credit Assignment（plan Task 15 / QuantaAlpha 思想）。

把一条 lineage 上的生成序列看成**轨迹**：

    F1 --a1--> F2 --a2--> F3 --a3--> F4

每一步记录：ΔFitness / ΔPoolUtility / novelty / complexity / cost /
failure_reason（plan Task 15 字段全清单）。``TrajectoryCritic`` 在最近 K 步里
找「低价值边」（崩塌边/纯复杂化边/失败边）——修复目标是**那条坏边**，而不是
丢弃整个成功前缀（#28）。``repair_target_of`` 返回坏边位置；``compose_fragments``
把两个兼容轨迹片段在 action/schema 层组合。

语义约束（plan + Non-negotiable）：
- **repair 只在停滞或显式 critic 条件后触发**（``should_repair`` 需要
  stagnation 证据 ≥ 阈值或显式 ``critic_condition``；不是每个 candidate 都触发）。
- **action 级 vs parent 级分离**：重复失败 suffix 提高局部 action 的饱和度
  （在 ``SearchTrajectory.local_action_saturation`` 反映），但**不**判 parent
  全局死亡——parent 的 fertility 归 stagnation/parent 级判据管。
- **fragment 组合**：action/schema 层级兼容才可组合；schema/domain 不兼容 →
  拒绝组合（不抛、返回 None，绝不产生假组合）。组合产物的**最终表达式仍由
  FE AST 编译**——本模块只记录**组合 plan**（parent ids、每个片段的
  provenance、组合模式），编译验证留给生成链（契约在本模块 docstring 写清）。
- 每一步的评估事实（ΔFitness 等）由调用方在评估后回传；无证据字段缺省
  None → 中性（#13），critic 不会把「没测过」当坏边。

零 LLM / 零模型 / 零网络，纯合成数据可测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

__all__ = [
    "TrajectoryStep",
    "SearchTrajectory",
    "TrajectoryCritic",
    "CriticConfig",
    "CriticVerdict",
    "repair_target_of",
    "ComposePlan",
    "compose_fragments",
    "default_local_saturation",
]

#: 有效字段集（trajectory 持久化/诊断白名单；plan Task 15 全清单）
STEP_FIELDS = (
    "factor_id", "parent_id", "action_type", "action_id",
    "delta_fitness", "delta_pool", "novelty", "complexity",
    "cost", "failure_reason",
)


@dataclass
class TrajectoryStep:
    """轨迹的一步：F_k --a_k--> F_{k+1} 的边（child 视角记录）。

    Parameters
    ----------
    factor_id : str
        本步产出的 child factor id（可为 "" = 失败/未落地）。
    parent_id : str
        该步的 parent factor id。
    action_type : str
        微层 action（REFINE / CROSSOVER / STATE_CONDITION …）。
    action_id : str
        可选的 action 唯一 id（对齐 ledger / action_edges）。
    delta_fitness : float | None
        相对 parent 的 ΔFitness（None = 未评估 → 中性）。
    delta_pool : float | None
        ΔPoolUtility（None = 中性）。
    novelty : float | None
        该步 novelty 增益（None = 中性）。
    complexity : int
        该步 AST 节点数增量（0 = 无变化）。
    cost : float
        该步成本（LLM+评估；缺省 0）。
    failure_reason : str
        失败原因（空 = 成功）。与 contracts.RejectionReason 值对齐但本模块
        不 import 契约层（保持独立可测）；critic 只关心「是否失败」。
    """

    factor_id: str = ""
    parent_id: str = ""
    action_type: str = ""
    action_id: str = ""
    delta_fitness: float | None = None
    delta_pool: float | None = None
    novelty: float | None = None
    complexity: int = 0
    cost: float = 0.0
    failure_reason: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.failure_reason)

    @property
    def has_gain(self) -> bool:
        """任一正向评估证据（ΔFitness/ΔPool/novelty > 0）即视为有增益。

        无证据（全 None）不算失败也不算 gain（#13）。
        """
        if self.failed:
            return False
        for v in (self.delta_fitness, self.delta_pool, self.novelty):
            if v is not None and float(v) > 1e-9:
                return True
        return False

    @property
    def has_evidence(self) -> bool:
        return any(
            v is not None for v in (self.delta_fitness, self.delta_pool, self.novelty)
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in STEP_FIELDS}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrajectoryStep":
        return cls(
            factor_id=str(d.get("factor_id") or ""),
            parent_id=str(d.get("parent_id") or ""),
            action_type=str(d.get("action_type") or ""),
            action_id=str(d.get("action_id") or ""),
            delta_fitness=_opt_float(d.get("delta_fitness")),
            delta_pool=_opt_float(d.get("delta_pool")),
            novelty=_opt_float(d.get("novelty")),
            complexity=int(d.get("complexity") or 0),
            cost=float(d.get("cost") or 0.0),
            failure_reason=str(d.get("failure_reason") or ""),
        )


@dataclass
class SearchTrajectory:
    """一条 lineage 的轨迹：有序步骤 + 局部 action 失败饱和度（action 级）。

    - ``lineage_key``：branch 根 factor_id / lineage 路径 key（与 retriever 的
      stagnation 按同一 key 聚合）。
    - ``steps``：按生成顺序追加的 TrajectoryStep。
    - ``local_action_saturation``：``action_type -> 连续失败/重复失败计数``
      （**局部**：只表示该 action 在该轨迹上下文里失败多，不判 parent 全局死）。
    - ``repaired_at``：最近一次 repair 发生的步骤下标（无 = -1）。

    设计：本模块**只做 credit 判定与 plan 记录**；不写 memory/ledger（持久化
    由 lineage/__init__ 的接线 hook 消费本类字段）。
    """

    lineage_key: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    local_action_saturation: dict[str, int] = field(default_factory=dict)
    repaired_at: int = -1

    def append(self, step: TrajectoryStep) -> "SearchTrajectory":
        """追加一步并更新局部 action 失败饱和度。

        重复失败 suffix 语义（plan Task 15 test 2）：**同 action 连续/累积失败**
        只抬升该 action 的局部饱和度；成功一步（任意正向 gain）清零该 action
        计数——action 级与 parent 级分离，这里永不判 parent 死亡。
        """
        self.steps.append(step)
        if step.failed:
            key = step.action_type or "_"
            self.local_action_saturation[key] = (
                self.local_action_saturation.get(key, 0) + 1
            )
        elif step.has_gain:
            key = step.action_type or "_"
            self.local_action_saturation[key] = 0
        # 无证据 step：不动饱和度（中性，#13）
        return self

    def append_record(
        self,
        *,
        factor_id: str = "",
        parent_id: str = "",
        action_type: str = "",
        action_id: str = "",
        delta_fitness: float | None = None,
        delta_pool: float | None = None,
        novelty: float | None = None,
        complexity: int = 0,
        cost: float = 0.0,
        failure_reason: str = "",
    ) -> "SearchTrajectory":
        return self.append(
            TrajectoryStep(
                factor_id=factor_id,
                parent_id=parent_id,
                action_type=action_type,
                action_id=action_id,
                delta_fitness=delta_fitness,
                delta_pool=delta_pool,
                novelty=novelty,
                complexity=complexity,
                cost=cost,
                failure_reason=failure_reason,
            )
        )

    def mark_repaired(self, index: int) -> None:
        self.repaired_at = int(index)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_key": self.lineage_key,
            "steps": [s.to_dict() for s in self.steps],
            "local_action_saturation": dict(self.local_action_saturation),
            "repaired_at": self.repaired_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SearchTrajectory":
        return cls(
            lineage_key=str(d.get("lineage_key") or ""),
            steps=[TrajectoryStep.from_dict(s) for s in (d.get("steps") or [])],
            local_action_saturation={
                str(k): int(v) for k, v in (d.get("local_action_saturation") or {}).items()
            },
            repaired_at=int(d.get("repaired_at") or -1),
        )


def default_local_saturation(step: TrajectoryStep, current: int) -> int:
    """默认局部饱和度更新规则（独立纯函数，测试可注入自定义规则）。

    - 失败 → +1；
    - 有 gain 的成功 → 0（该 action 已证明能出活）；
    - 无证据 → 保持（中性）。
    """
    if step.failed:
        return current + 1
    if step.has_gain:
        return 0
    return current


# ---------------------------------------------------------------------------
# Critic：隔离坏边（#28）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticConfig:
    """critic 判据（全部可配，#30）。

    - ``window``：只看最近 K 步。
    - ``collapse_delta``：ΔFitness 低于该值判「崩塌」。
    - ``complexity_fitness_ratio``：复杂度涨而 fitness 平的坏边判据权重。
    - ``max_failed_ratio``：窗口内失败比例阈值（超 → 判该窗口有坏边）。
    """

    window: int = 8
    collapse_delta: float = -0.01
    max_complexity_growth: int = 3
    w_fitness: float = 0.6
    w_complexity: float = 0.25
    w_failed: float = 0.15
    fail_threshold: float = 0.5

    #: repair 触发：需要显式停滞证据 ≥ 该值，或 critic 显式标记（#13/#30）。
    min_stagnation_for_repair: float = 0.4


@dataclass(frozen=True)
class CriticVerdict:
    """critic 判定结果。

    - ``bad_edge_index``：坏边下标（**相对轨迹 steps 的绝对下标**）或 -1。
    - ``low_value_indexes``：窗口内所有低价值边下标。
    - ``score``：坏边分数（0~1；越接近 1 越该修）。
    - ``reasons``：机器可读原因码（fitness_collapse / complexity_without_gain /
      failure_streak / no_evidence 等）。
    - ``suggests_repair``：critic 是否显式建议 repair（只在停滞或显式条件
      下才 True——repair 不因单个 candidate 触发）。
    """

    bad_edge_index: int = -1
    low_value_indexes: tuple[int, ...] = ()
    score: float = 0.0
    reasons: tuple[str, ...] = ()
    suggests_repair: bool = False

    @property
    def has_bad_edge(self) -> bool:
        return self.bad_edge_index >= 0


def _edge_value(step: TrajectoryStep) -> float:
    """单条边的价值（0~1，1 = 高价值好边；0.5 = 无证据中性）。"""
    if step.failed:
        return 0.0
    if not step.has_evidence:
        return 0.5  # 无证据 → 中性（不判好也不判坏，#13）
    v = 0.5
    n_ev = 0
    for key in ("delta_fitness", "delta_pool", "novelty"):
        raw = getattr(step, key)
        if raw is None:
            continue
        n_ev += 1
        # 增量信号（如 +0.02）映射到 [0,1]：+x → 0.5 + 25x；-x → 0.5 - 25x
        v += 0.5 * max(-1.0, min(1.0, 25.0 * float(raw))) / n_ev
    # 复杂度惩罚：净涨而价值中性/负 → 拉低
    if step.complexity and step.complexity > 0:
        if v <= 0.5:
            v -= min(0.3, 0.05 * int(step.complexity))
    return max(0.0, min(1.0, v))


def _bounded(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _edge_value_rank(steps: Sequence[TrajectoryStep]) -> list[float]:
    return [_edge_value(s) for s in steps]


class TrajectoryCritic:
    """最近 K 步的低价值边检测（QuantaAlpha-style credit assignment）。

    ``judge(trajectory, stagnation=…)`` 返回 :class:`CriticVerdict`。坏边判定
    是**增量**的：只看窗口内，成功前缀保留（repair target = 坏边，不是丢弃
    整个前缀，#28）。
    """

    def __init__(self, config: CriticConfig | None = None) -> None:
        self.config = config if config is not None else CriticConfig()

    # ------------------------------------------------------------------
    # 主判定
    # ------------------------------------------------------------------

    def judge(
        self,
        trajectory: SearchTrajectory,
        *,
        stagnation: float | None = None,
        critic_condition: bool = False,
    ) -> CriticVerdict:
        """判定最近窗口的低价值边，并给出 repair 建议。

        Parameters
        ----------
        trajectory : SearchTrajectory
            已累积 steps 的轨迹。
        stagnation : float | None
            lineage 层停滞分（0~1；None/0 = 无停滞证据）。
        critic_condition : bool
            显式 critic 条件（trajectory critic 之外的外部信号；
            与停滞 ≥ 阈值同权触发 repair 建议）。

        Returns
        -------
        CriticVerdict
        """
        cfg = self.config
        recs = list(trajectory.steps)
        if not recs:
            return CriticVerdict()
        recs = recs[-max(cfg.window, 1):]
        base = len(trajectory.steps) - len(recs)  # 窗口在整条轨迹里的起始下标

        values = _edge_value_rank(recs)
        reasons: list[str] = []
        # 1) fitness 崩塌：窗口内最差价值边（失败/负 ΔFitness）
        min_v = min(values)
        min_i = values.index(min_v)
        abs_min_i = base + min_i
        bad = -1
        if min_v < 0.45:  # 低于中性明显（崩塌/失败/复杂化）
            bad = abs_min_i
            step = recs[min_i]
            if step.failed:
                reasons.append("failed_step")
            elif step.delta_fitness is not None and float(step.delta_fitness) <= cfg.collapse_delta:
                reasons.append("fitness_collapse")
            elif step.complexity and step.complexity > cfg.max_complexity_growth:
                reasons.append("complexity_without_gain")
            else:
                reasons.append("low_value_edge")
        # 2) 失败比例（窗口内）
        failed = [i for i, s in enumerate(recs) if s.failed]
        failed_ratio = len(failed) / len(recs) if recs else 0.0
        if failed_ratio >= cfg.fail_threshold:
            if bad < 0 and failed:
                bad = base + failed[-1]
            reasons.append("failure_streak")

        # 低价值边全集（供 #28 隔离测试断言）
        low_idx = tuple(
            base + i for i, v in enumerate(values) if v < 0.45
        )

        # 3) 分数：取最差边相对中性的落差（越大越该修）
        score = 0.0 if bad < 0 else (0.5 - min_v) / 0.5
        score = max(0.0, min(1.0, score))
        if not reasons and bad >= 0:
            reasons.append("low_value_edge")

        # 4) repair 建议 = 停滞证据足够（≥ min_stagnation_for_repair）或显式
        #    critic 条件。**不是每个 candidate 都触发**（plan Task 15）。
        stag = float(stagnation or 0.0)
        suggests = bool(
            (bad >= 0)
            and (critic_condition or stag >= cfg.min_stagnation_for_repair)
        )
        if suggests and "stagnation" not in reasons:
            reasons.append("stagnation")

        return CriticVerdict(
            bad_edge_index=bad,
            low_value_indexes=low_idx,
            score=score,
            reasons=tuple(dict.fromkeys(reasons)),
            suggests_repair=suggests,
        )


#: 兼容别名：自由函数形式（plan Task 15 契约点）
def repair_target_of(
    trajectory: SearchTrajectory,
    critic: TrajectoryCritic | None = None,
    *,
    stagnation: float | None = None,
    critic_condition: bool = False,
) -> CriticVerdict:
    """返回坏边判定（critic.judge 的薄封装）。"""
    c = critic if critic is not None else TrajectoryCritic()
    return c.judge(
        trajectory, stagnation=stagnation, critic_condition=critic_condition
    )


# ---------------------------------------------------------------------------
# Fragment 组合（action/schema 层；最终表达式由 FE AST 编译）
# ---------------------------------------------------------------------------


@dataclass
class ComposePlan:
    """一个轨迹组合的**计划**（不是表达式本身）。

    plan 契约（最终表达式仍由 FE AST 编译，本模块不产公式）：
    - ``prefix_fragment`` / ``suffix_fragment``：被组合的两个轨迹片段
      （:class:`TrajectoryFragment`）；
    - ``parent_factor_ids``：组合产物的 DAG parents（两个 fragment 的锚点因子）；
    - ``mode``：``"prefix_concat"``（前缀动作接后缀动作）或
      ``"schema_cross"``（A 的 schema × B 的动作，schema 层组合）；
    - ``schema_tags``：组合后的目标 schema 九维 tag（调用方传给生成链做
      alignment / DSL 编译用）。

    生成链拿到 ``ComposePlan`` 后必须走 FE AST 编译验证（本模块不编译、不
    拼接公式字符串）。
    """

    prefix_fragment: "TrajectoryFragment"
    suffix_fragment: "TrajectoryFragment"
    parent_factor_ids: tuple[str, ...]
    mode: str = "prefix_concat"
    schema_tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_fragment": self.prefix_fragment.to_dict(),
            "suffix_fragment": self.suffix_fragment.to_dict(),
            "parent_factor_ids": list(self.parent_factor_ids),
            "mode": self.mode,
            "schema_tags": dict(self.schema_tags),
        }


@dataclass(frozen=True)
class TrajectoryFragment:
    """轨迹的一段可复用片段（前缀/后缀）。

    - ``fragment_id``：稳定 id（调用方给；组合/去重用）。
    - ``steps``：有序步骤（**已成功**——组合只消费成功片段）。
    - ``anchor_factor_id``：片段末端的锚点因子（作为后续动作的 parent）。
    - ``schema_id``：片段所属 schema（组合兼容性判定用）。
    - ``domain_tags``：片段的数据域（DataDomain / 机制 tag；组合兼容性用）。
    - ``end_fitness``：片段末端 fitness（组合打分用，可选）。
    """

    fragment_id: str
    steps: tuple[TrajectoryStep, ...] = ()
    anchor_factor_id: str = ""
    schema_id: str = ""
    domain_tags: tuple[str, ...] = ()
    end_fitness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "steps": [s.to_dict() for s in self.steps],
            "anchor_factor_id": self.anchor_factor_id,
            "schema_id": self.schema_id,
            "domain_tags": list(self.domain_tags),
            "end_fitness": self.end_fitness,
        }


def compose_fragments(
    a: TrajectoryFragment,
    b: TrajectoryFragment,
    *,
    mode: str = "prefix_concat",
    compatible_schemas: bool = False,
) -> ComposePlan | None:
    """把两个兼容轨迹片段组合成 :class:`ComposePlan`（不产公式）。

    兼容判定（plan Task 15 test 3）：
    - ``mode="prefix_concat"``：A 的 action 序列接 B 的 action 序列——
      要求 schema/domain 兼容（``compatible_schemas`` 或 schema_id/domain
      tags 无冲突）；不兼容 → 拒绝（返回 None）。
    - ``mode="schema_cross"``：A 的 schema × B 的动作——仍要求 B 的动作
      domain 与 A 的 schema domain 不冲突；冲突 → None。

    拒绝 = 返回 None（不抛、不合成假组合）。schema_id 相同（或一侧空）视为
    兼容；两侧都非空且不同 → 需 ``compatible_schemas=True`` 显式放行
    （调用方保证已做语义对齐）。**缺省兼容判定从严**：``compatible_schemas``
    默认 False（不同的 schema_id 默认拒绝；调用方显式证明语义兼容才放行）。
    """
    if a is None or b is None:
        return None
    if not a.steps or not b.steps:
        return None
    if not _domains_compatible(a, b):
        return None
    if mode == "schema_cross":
        if a.schema_id and b.schema_id and a.schema_id != b.schema_id:
            if not compatible_schemas:
                return None
    elif mode != "prefix_concat":
        return None  # 未知组合模式 → fail-closed
    else:
        if a.schema_id and b.schema_id and a.schema_id != b.schema_id:
            if not compatible_schemas:
                return None
    parents = []
    for f in (a, b):
        if f.anchor_factor_id and f.anchor_factor_id not in parents:
            parents.append(f.anchor_factor_id)
    return ComposePlan(
        prefix_fragment=a,
        suffix_fragment=b,
        parent_factor_ids=tuple(parents),
        mode=mode,
        schema_tags={"schema_id": a.schema_id or b.schema_id},
    )


def _domains_compatible(a: TrajectoryFragment, b: TrajectoryFragment) -> bool:
    """domain tags 冲突才不兼容；任一侧空 = 无约束 → 兼容。

    冲突 = 两侧都有 domain tag 且集合不相交 → 组合会跨语义域（如
    PRICE 域动作接 FUNDAMENTAL 域条件）——本层只做**表面冲突拒绝**；精细
    语义对齐留给生成链的 deterministic alignment。
    """
    ta = {str(t).upper() for t in a.domain_tags if str(t).strip()}
    tb = {str(t).upper() for t in b.domain_tags if str(t).strip()}
    if not ta or not tb:
        return True
    return bool(ta & tb)


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = [
    "TrajectoryStep",
    "SearchTrajectory",
    "TrajectoryCritic",
    "CriticConfig",
    "CriticVerdict",
    "repair_target_of",
    "TrajectoryFragment",
    "ComposePlan",
    "compose_fragments",
    "default_local_saturation",
]
