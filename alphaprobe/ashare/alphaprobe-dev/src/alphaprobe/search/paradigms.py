"""Macro search paradigms（plan Task 14 / Part F4）——六范式定义与需求声明。

宏层范式把「**往哪个方向搜**」从微层 ActionScheduler 的「**用哪个 arm**」中
分离出来。F4 语义：上下文 → {CoE, ToT, EA, Schema, LogicExplore,
TrajectoryRepair}，然后范式选择微 action(s)。ActionScheduler 保持 micro 层不动
（本模块不接管 action 选择）。

范式枚举（plan 全集）：

- ``COE`` — Chain of Experience exploitation：fertile 低饱和 lineage 上继续
  **利用**——沿同一 parent 的经验链做小步 refine，把已验证的增益延续下去。
- ``TOT`` — Tree of Thought divergent branching：lineage **停滞**但 parent 本身
  仍好 → 不再沿原链打磨，而是从 parent 发散出多个**机制假设分支**，找新出路。
- ``EA`` — mutation + complementary crossover：存在**互补 cluster**（结构/语义
  远的第二个 parent 可用）→ 用变异 + 互补交叉组合两个成熟信号。**绝无第二个
  parent 时不可选**（不合成假 parent，#27 语义）。
- ``SCHEMA`` — underexplored semantic region：当前 schema 被挖得少（n_impl 低）
  或整个语义区存在大 gap → 主动探索 schema 空间。
- ``TRAJECTORY_REPAIR`` — 轨迹修补：trajectory critic 标记坏边/连续停滞 →
  放弃坏后缀、从好前缀继续（credit assignment 的消费方，见 trajectory.py）。
- ``LOGIC_EXPLORE`` — market logic library 探索：存在低 support / 高 gap 的
  可复用市场逻辑 → 绕开 formula 层的局部寻优，跳到机制层找新信号。

设计约束：

- 每个 paradigm 声明自己的**上下文需求**（``ParadigmContextRequirements``）——
  scheduler 只有在上下文证据满足需求时才让该 paradigm 参与选择（缺证据 =
  不满足 ≠ 中性，见 Part G #27 的区分度要求）。需求是**白名单**：不在此声明
  的上下文键，scheduler 不得用来为该 paradigm 抬分。
- 全模块纯函数/纯 dataclass，零 LLM、零模型、零网络。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    "Paradigm",
    "ParadigmFamily",
    "ParadigmContext",
    "ParadigmContextRequirements",
    "PARADIGM_ORDER",
    "PARADIGM_CONTEXT_KEYS",
    "eligibility_of",
    "ParadigmBanditStats",
    "paradigm_bias",
    "softmax_distribution",
    "ParadigmSelectionConfig",
]


# ---------------------------------------------------------------------------
# 范式枚举
# ---------------------------------------------------------------------------


class Paradigm(str, Enum):
    """宏搜索范式全集（plan Task 14 / F4）。值即规范名。"""

    COE = "COE"
    TOT = "TOT"
    EA = "EA"
    SCHEMA = "SCHEMA"
    TRAJECTORY_REPAIR = "TRAJECTORY_REPAIR"
    LOGIC_EXPLORE = "LOGIC_EXPLORE"


#: 规范顺序（诊断/分布序列化用；COE/ToT/EA 排在 plan 测试前三位）
PARADIGM_ORDER: tuple[Paradigm, ...] = (
    Paradigm.COE,
    Paradigm.TOT,
    Paradigm.EA,
    Paradigm.SCHEMA,
    Paradigm.TRAJECTORY_REPAIR,
    Paradigm.LOGIC_EXPLORE,
)


class ParadigmFamily(str, Enum):
    """范式的搜索行为族（Part G #27 行为区分 / TOT/COE/EA 的语义锚）。"""

    EXPLOIT = "EXPLOIT"    # 沿已有经验链利用（COE）
    DIVERGE = "DIVERGE"    # 从 parent 发散分支（TOT）
    RECOMBINE = "RECOMBINE"  # 跨互补簇变异/交叉（EA）
    EXPLORE = "EXPLORE"    # 语义区/机制层探索（SCHEMA / LOGIC_EXPLORE）


#: 范式 → 行为族（诊断与区分测试消费；每个范式恰属一族）
PARADIGM_FAMILY: dict[Paradigm, ParadigmFamily] = {
    Paradigm.COE: ParadigmFamily.EXPLOIT,
    Paradigm.TOT: ParadigmFamily.DIVERGE,
    Paradigm.EA: ParadigmFamily.RECOMBINE,
    Paradigm.SCHEMA: ParadigmFamily.EXPLORE,
    Paradigm.LOGIC_EXPLORE: ParadigmFamily.EXPLORE,
    Paradigm.TRAJECTORY_REPAIR: ParadigmFamily.DIVERGE,
}


# ---------------------------------------------------------------------------
# 范式选择上下文（plan Task 14：parent fitness / lineage stagnation /
# cluster saturation / schema rarity / complementary parent availability /
# historical paradigm reward/cost；#13：缺证据 = 中性）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParadigmContext:
    """一次范式选择所需的全部上下文（缺省全部中性/无证据）。

    每个键的语义与取值（全部可空 = 无证据）：

    - ``parent_fitness`` : float | None — 主 parent 的归一化 fitness
      （None = 未评估 → 中性 0.5）。
    - ``lineage_stagnation`` : float | None — 该 lineage 的停滞分
      （0 = 不停滞 / 中性，1 = 全停；None = 无停滞证据 → 中性 0.0）。
    - ``parent_fitness_trend`` : float | None — 主 lineage 最近代数 Δfitness
      均值（None = 中性 0.0；正 = fertile）。
    - ``cluster_saturation`` : float | None — 主 parent 所在 cluster 的饱和
      （0~1，越大越被挖烂；None = 无 cluster 证据 → 中性 0.0）。
    - ``has_complementary_parent`` : bool — 是否存在结构/语义互补的第二 parent
      （可用作 EA 交叉；**缺省 False**——EA 需要的证据必须显式给）。
    - ``n_second_parents`` : int — 显式给出的第二 parent 数（EA 资格真值；
      缺省 0）。
    - ``schema_gap`` : float | None — 当前语义区/目标 schema 的 gap 分
      （越大越没被挖；None = 中性 0.0）。
    - ``schema_rarity`` : float | None — 目标 schema 的稀有度
      （0~1，越大越稀有/少实现；None = 中性 0.0）。
    - ``schema_n_impl`` : int | None — 目标 schema 已实现数（None = 未知）。
    - ``logic_gap`` : float | None — market logic 层 gap
      （低 support / 高机制缺口；None = 中性 0.0）。
    - ``trajectory_repair_requested`` : bool — trajectory critic 是否显式标记
      需要 repair（坏边 / 停滞）。**不是每个 candidate 都触发**（plan Task 15：
      repair 只在停滞或显式 critic 条件后触发）。
    - ``trajectory_stagnation`` : float | None — trajectory 层停滞证据
      （None = 中性 0.0）。
    - ``lineage_key`` : str — 主 lineage 的 key（bandit 历史按 lineage 聚合时
      用；缺省 "" = 无）。
    - ``cluster_id`` : str — 主 parent 所在 cluster id（诊断用）。
    - ``meta`` : Mapping[str, Any] — 附加上下文（EA 的第二 parent fitness 等；
      只承载**展示/强度**信息，不允许用未声明键改 eligibility）。

    Part G #13：未出现的键一律中性（绝不把「没测过」当「没价值」）。
    """

    parent_fitness: float | None = None
    lineage_stagnation: float | None = None
    parent_fitness_trend: float | None = None
    cluster_saturation: float | None = None
    has_complementary_parent: bool = False
    n_second_parents: int = 0
    schema_gap: float | None = None
    schema_rarity: float | None = None
    schema_n_impl: int | None = None
    logic_gap: float | None = None
    trajectory_repair_requested: bool = False
    trajectory_stagnation: float | None = None
    lineage_key: str = ""
    cluster_id: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    #: 全部有效上下文键（需求声明 / 诊断共用）
    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.__dataclass_fields__)

    def get(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(f"unknown paradigm context key: {key!r}")
        return getattr(self, key)

    def neutral_value(self, key: str) -> Any:
        """该键的「无证据中性」取值（#13；scheduler 缺省用它填分布）。"""
        if key == "meta":
            return {}
        fv = self.__dataclass_fields__[key].default
        if key in ("has_complementary_parent", "trajectory_repair_requested"):
            return False
        if isinstance(fv, bool):
            return False
        if isinstance(fv, int):
            return 0
        if isinstance(fv, float) or fv is None:
            return 0.0
        return fv


@dataclass(frozen=True)
class ParadigmContextRequirements:
    """一个范式参与选择所需的**最小上下文证据**（资格判定）。

    语义（plan Task 14）：paradigm 只有在需求满足时才 eligible；不满足 =
    **不可选**（不是中性参与）。本类同时充当需求**声明**（Part G #27 的
    「每个 paradigm 的上下文需求声明」）：

    - ``needs_second_parent`` : bool — EA 需要显式第二 parent
      （无第二 parent → EA 不可选，绝不合成假 parent）。
    - ``min_stagnation`` : float — 需要停滞分达到的下限
      （0.0 = 不要求停滞）。
    - ``max_stagnation`` : float | None — 停滞分上限
      （None = 不设上限；COE 要求低停滞 = 需显式低值证据）。
    - ``min_schema_gap`` : float — schema gap 下限（SCHEMA/LOGIC_EXPLORE 用；
      缺省 0 = 不要求）。
    - ``min_schema_rarity`` : float — schema rarity 下限。
    - ``repair_flag`` : bool — 需要显式 repair 标记
      （TRAJECTORY_REPAIR 的触发闸门，plan Task 15）。
    """

    needs_second_parent: bool = False
    min_stagnation: float = 0.0
    max_stagnation: float | None = None
    min_schema_gap: float = 0.0
    min_schema_rarity: float = 0.0
    repair_flag: bool = False
    require_fertile_trend: bool = False  # 保留：后续如需「须有正趋势证据」

    def satisfied_by(self, ctx: ParadigmContext) -> bool:
        """判定上下文是否满足需求（缺证据 = 不满足；布尔键要求显式 True）。"""
        # EA：必须显式存在互补 parent；绝不因「没测过」就当 eligible
        if self.needs_second_parent:
            if not ctx.has_complementary_parent or ctx.n_second_parents < 1:
                return False
        # 停滞下限：需要「停滞 ≥ min」——缺证据（0.0）自然不满足 >0 的下限
        stag = float(ctx.lineage_stagnation or 0.0)
        if stag < self.min_stagnation - 1e-9:
            return False
        # 停滞上限：要求「停滞 ≤ max」——缺证据 0 满足 0 上限，
        # 但 COE 的「low-saturation fertile」另有 fertility/趋势门槛，见下。
        if self.max_stagnation is not None and stag > self.max_stagnation + 1e-9:
            return False
        # schema gap / rarity
        gap = float(ctx.schema_gap or 0.0)
        if gap < self.min_schema_gap - 1e-9:
            return False
        rarity = float(ctx.schema_rarity or 0.0)
        if rarity < self.min_schema_rarity - 1e-9:
            return False
        # repair 标记：critic 显式请求才算
        if self.repair_flag and not ctx.trajectory_repair_requested:
            return False
        return True


#: 每个 paradigm 的资格需求（范式级静态声明）。
#:
#: - COE：**低停滞**（利用 fertile lineage）。需求上限 ``max_stagnation`` 要求
#:   停滞分显式很低（≤0.3 才 eligible）；同时 cluster 饱和低的判断在权重层做
#:   （saturation 不是资格而是强度信号）。fertility 用 ``lineage_stagnation``
#:   低 + ``parent_fitness_trend`` 高表达。
#: - TOT：停滞但 parent 好。停滞下限 0.35（够停滞才发散）；无上限（发散本就为
#:   停滞解围）。parent 好坏的强度信号在权重层。
#: - EA：资格 = 有互补第二 parent（真值闸门）；其余信号只调强度不调资格。
#: - SCHEMA：schema gap / rarity 门槛（缺证据不可选）。
#: - LOGIC_EXPLORE：logic gap 门槛（缺证据不可选）。
#: - TRAJECTORY_REPAIR：critic 显式 repair 标记（硬闸门）。
PARADIGM_REQUIREMENTS: dict[Paradigm, ParadigmContextRequirements] = {
    Paradigm.COE: ParadigmContextRequirements(max_stagnation=0.3),
    Paradigm.TOT: ParadigmContextRequirements(min_stagnation=0.35),
    Paradigm.EA: ParadigmContextRequirements(needs_second_parent=True),
    Paradigm.SCHEMA: ParadigmContextRequirements(
        min_schema_gap=0.25, min_schema_rarity=0.25
    ),
    Paradigm.LOGIC_EXPLORE: ParadigmContextRequirements(min_schema_gap=0.2),
    Paradigm.TRAJECTORY_REPAIR: ParadigmContextRequirements(repair_flag=True),
}


#: 上下文键全集（eligibility 白名单校验用）
PARADIGM_CONTEXT_KEYS: tuple[str, ...] = tuple(ParadigmContext.__dataclass_fields__)


def eligibility_of(
    paradigm: Paradigm | str,
    ctx: ParadigmContext,
    *,
    requirements: Mapping[Paradigm, ParadigmContextRequirements] | None = None,
) -> bool:
    """单个范式在当前上下文是否 eligible（资格白名单 + 需求判定）。

    - 未知 paradigm → False（fail-closed，不静默放行）；
    - 需求未声明（requirements 缺该 paradigm）→ False（无需求声明的范式
      不许参与，防止「没想清楚就放行」）；
    - 其余按 ``ParadigmContextRequirements.satisfied_by`` 判定（缺证据 =
      不满足，#13 / #27）。
    """
    p = _coerce_paradigm(paradigm)
    if p is None:
        return False
    reqs = requirements if requirements is not None else PARADIGM_REQUIREMENTS
    req = reqs.get(p)
    if req is None:
        return False
    return req.satisfied_by(ctx)


def _coerce_paradigm(value: Any) -> Paradigm | None:
    if isinstance(value, Paradigm):
        return value
    try:
        return Paradigm(str(value).upper())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 历史 paradigm reward（bandit 语义：每 paradigm 有 attempts / mean_reward）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParadigmBanditStats:
    """一个范式在注入统计面里的历史表现。

    字段：
    - ``attempts`` : int — 该范式历史尝试次数（无尝试 → 0，不参与统计）。
    - ``reward_sum`` / ``reward_sq_sum`` : float — 累计 reward（均值 =
      reward_sum / attempts；无尝试 → 中性）。
    - ``cost_sum`` : float — 累计成本（与 reward 一起做 cost-adjusted 偏置；
      无尝试 → 0）。
    - ``reliability`` : float — 尝试置信（0~1；shrink 向中性用）。
    """

    attempts: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    cost_sum: float = 0.0
    reliability: float = 0.0

    @property
    def mean_reward(self) -> float:
        """尝试数 >0 才给均值；否则中性 0.5（#13，绝不把「没试过」当 0/1）。"""
        if self.attempts <= 0:
            return 0.5
        return max(0.0, min(1.0, self.reward_sum / self.attempts))

    @property
    def mean_cost(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return max(0.0, self.cost_sum / self.attempts)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any] | None) -> "ParadigmBanditStats":
        if not m:
            return cls()
        return cls(
            attempts=max(0, int(m.get("attempts") or 0)),
            reward_sum=float(m.get("reward_sum") or 0.0),
            reward_sq_sum=float(m.get("reward_sq_sum") or 0.0),
            cost_sum=float(m.get("cost_sum") or 0.0),
            reliability=float(m.get("reliability") or 0.0),
        )


# ---------------------------------------------------------------------------
# bandit 偏置 / softmax（合成纯函数，可复现）
# ---------------------------------------------------------------------------


def _shrunk(value: float, stats: ParadigmBanditStats, prior: float = 0.5) -> float:
    """尝试少的 paradigm 的 reward 向中性先验收缩（与 #29 同款置信语义）。"""
    n = float(stats.attempts)
    k = 8.0  # paradigm 级收缩强度（经验常数：尝试 ~8 次才到 ~50% 置信）
    w = n / (n + k) if n > 0 else 0.0
    return float(prior + w * (value - prior))


def paradigm_bias(
    paradigm: Paradigm | str,
    ctx: ParadigmContext,
    stats: ParadigmBanditStats | None = None,
) -> float:
    """一个 eligible paradigm 的上下文强度分（0~1，不掺入 eligibility 逻辑）。

    这是**权重/强度**层：上下文证据越支持该范式，分越高。缺失的上下文
    （None / False / 0）一律取中性（#13），绝不因缺证据加分。

    各范式语义：
    - COE：fertile = 低停滞（1 - stagnation）+ 高 fitness 趋势。两个信号
      独立缺省中性 → 相乘不塌（无趋势证据时只按停滞低给分）。
    - TOT：停滞高（stagnation）+ parent 本身好（fitness 高）。
    - EA：互补 parent 存在（已经过 eligibility 闸门，这里取 1）+
      第二 parent fitness 高 + cluster 饱和高（两簇都成熟才值得交叉）。
    - SCHEMA：gap / rarity 高（未探索区）。
    - TRAJECTORY_REPAIR：显式 repair 请求（eligibility 已保证）+
      停滞高（越停滞越该修）。
    - LOGIC_EXPLORE：logic gap 高（机制层缺口）。
    - 所有范式乘一个 cost-adjusted bandit 偏置：历史 mean_reward 高的范式
      略占优；cost 高则压一点；无历史 → 中性 1.0（#13 / #26：reward 来自
      评估而非生成计数）。
    """
    p = _coerce_paradigm(paradigm)
    if p is None:
        return 0.0
    stag = float(ctx.lineage_stagnation or 0.0)
    trend = float(ctx.parent_fitness_trend or 0.0)
    fitness = float(ctx.parent_fitness) if ctx.parent_fitness is not None else 0.5
    sat = float(ctx.cluster_saturation or 0.0)
    gap = float(ctx.schema_gap or 0.0)
    rarity = float(ctx.schema_rarity or 0.0)
    lgap = float(ctx.logic_gap or 0.0)
    s2 = float(ctx.n_second_parents or 0)
    s2fit = _last_second_fitness(ctx)

    if p is Paradigm.COE:
        # 低停滞 = 高利用价值；trend 有证据时加成（无证据中性 0）
        score = (1.0 - stag) * 0.7 + min(1.0, max(0.0, trend) / 0.05) * 0.3
    elif p is Paradigm.TOT:
        score = stag * 0.6 + max(0.0, fitness - 0.4) * 2.0 * 0.4
        score = max(0.0, min(1.0, score))
    elif p is Paradigm.EA:
        score = (1.0 if s2 >= 1 else 0.0) * 0.4 + sat * 0.3 + s2fit * 0.3
    elif p is Paradigm.SCHEMA:
        score = gap * 0.5 + rarity * 0.5
    elif p is Paradigm.TRAJECTORY_REPAIR:
        score = 0.5 + 0.5 * stag
    elif p is Paradigm.LOGIC_EXPLORE:
        score = min(1.0, lgap)
    else:  # pragma: no cover - 枚举已穷尽
        score = 0.0
    score = max(0.0, min(1.0, float(score)))

    # bandit 偏置（成本调整后的 reward 记忆；无历史中性 1.0，#13 / #26）
    if stats is not None and stats.attempts > 0:
        mr = _shrunk(stats.mean_reward, stats)
        cost_pen = min(0.3, stats.mean_cost)  # cost ∈ [0,1] 归一化
        bias = 0.5 + 1.0 * (mr - 0.5) - 0.5 * cost_pen
        bias = max(0.3, min(1.6, bias))
        score = max(0.0, min(1.0, score * bias))
    return score


def _last_second_fitness(ctx: ParadigmContext) -> float:
    """第二 parent 的 fitness（EA 用）；ctx 无 meta → 中性 0.5。"""
    meta = getattr(ctx, "meta", None)
    if isinstance(meta, Mapping):
        v = meta.get("second_parent_fitness")
        if v is not None:
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return 0.5
    return 0.5


def softmax_distribution(
    scores: Mapping[Paradigm | str, float],
    *,
    temperature: float = 1.0,
    eligible: Sequence[Paradigm | str] | None = None,
) -> dict[Paradigm, float]:
    """把范式分转成 softmax 概率分布（可复现：同输入同输出）。

    - ``scores`` 只含 eligible 范式；未提供 scores 的 eligible 范式 → 0 分。
    - 空 eligible 或全 0 → 均匀分布在 eligible 上（仍不选 ineligible）。
    - temperature <= 0 → 退化为 argmax（最高分 1.0，其余 0）。
    """
    ele = [_coerce_paradigm(x) for x in (eligible or list(scores))]
    ele = [p for p in ele if p is not None]
    if not ele:
        return {}
    vals: list[tuple[Paradigm, float]] = []
    for p in ele:
        raw = 0.0
        for k, v in scores.items():
            kp = _coerce_paradigm(k)
            if kp is p:
                raw = float(v or 0.0)
                break
        vals.append((p, max(0.0, raw)))
    if temperature is not None and temperature <= 0:
        best = max(v for _, v in vals) if vals else 0.0
        return {p: (1.0 if v >= best else 0.0) for p, v in vals}
    temp = float(temperature) if temperature else 1.0
    mx = max((v for _, v in vals), default=0.0)
    exps = [(p, math.exp((v - mx) / max(temp, 1e-9))) for p, v in vals]
    total = sum(e for _, e in exps) or 1.0
    return {p: e / total for p, e in exps}


# ---------------------------------------------------------------------------
# scheduler 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParadigmSelectionConfig:
    """范式选择的全部可配置项（Part G #30：ablation 开关都在这）。

    - ``enabled`` : 总开关（False = 关闭全部范式调度）。
    - ``mode`` : "sampled"（按分布采样，rng 可复现）| "argmax"（取最高分）。
    - ``temperature`` : softmax 温度（<=0 退化为 argmax）。
    - ``repair_enabled`` : TRAJECTORY_REPAIR 单独开关。
    - ``schema_enabled`` / ``ea_enabled`` : SCHEMA/EA 单独开关（可配）——
      语义与 plan「范式/repair 全部有开关」对齐；需求/权重逻辑本身不硬编码。
    """

    enabled: bool = True
    mode: str = "sampled"
    temperature: float = 1.0
    repair_enabled: bool = True
    schema_enabled: bool = True
    ea_enabled: bool = True


__all__ = [
    "Paradigm",
    "ParadigmFamily",
    "ParadigmContext",
    "ParadigmContextRequirements",
    "ParadigmBanditStats",
    "PARADIGM_ORDER",
    "PARADIGM_FAMILY",
    "PARADIGM_CONTEXT_KEYS",
    "PARADIGM_REQUIREMENTS",
    "eligibility_of",
    "paradigm_bias",
    "softmax_distribution",
    "ParadigmSelectionConfig",
]
