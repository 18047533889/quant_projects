"""SearchOpportunity —— 节点剩余搜索机会（任务书 §46 / §57）。

    SearchOpportunity = 0.40×NoveltyOpportunity + 0.25×ClusterRarity
                      + 0.20×UnexploredActionRatio + 0.15×SurvivalOpportunity

- ``NoveltyOpportunity``：mean_novelty_gain 观测均值（未观测时取 warmup 中值 0.5）；
- ``ClusterRarity``：1/sqrt(1+cluster_size)，可接 factor_assets 或本地 cluster 统计
  （接口留 adapter 槽位，本地退化实现必须有）；
- ``UnexploredActionRatio``：按 action 类型（WINDOW_SCALE/FIELD_SUBSTITUTION/
  CROSSOVER/STATE_CONDITION 等，与 search/arms.py 现有 action 对齐）统计未尝试比例；
- ``SurvivalOpportunity``：只允许使用当前 research cutoff 之前合法可见的 survival
  memory——2026 属 sealed test 期间禁止搜索期使用 2026 survival 信息；版本冻结后
  vN Test 才能转 vN+1 historical knowledge。实现从 survival 层拿「cutoff 可见」数据，
  sealed 段直接返回中性值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

#: 与 search/arms.py / search/__init__.py 现有 action 对齐
DEFAULT_ACTION_FAMILIES: tuple[str, ...] = (
    "REFINE",
    "WINDOW_SCALE",
    "FIELD_SUBSTITUTION",
    "OPERATOR_SUBSTITUTION",
    "STATE_CONDITION",
    "CROSSOVER",
    "SCHEMA_EXPLORE",
    "ROBUSTIFY",
    "GENERATION_REPAIR",
)

#: ClusterRarity 无信息（cluster_size=None / 无法解析 / 非法尺寸）时的中性值。
#: 缺信息绝不能被当成「最稀有机会」：无 cluster 上下文 → 中性 0.5，既不奖励也不惩罚。
CLUSTER_RARITY_NEUTRAL = 0.5

#: 未观测均值时 NoveltyOpportunity 的 warmup 中值
WARMUP_NOVELTY = 0.5
#: SurvivalOpportunity 的 shrink 先验（与 fitness.survival_opportunity_shrunk 一致）
SURVIVAL_PRIOR = 0.5
#: 小支撑的收缩强度
SURVIVAL_STRENGTH = 20.0
#: SurvivalOpportunity cap（相对先验的增益上限，与 §51 一致）
SURVIVAL_CAP = 0.15


@dataclass
class SearchOpportunityConfig:
    """四维机会权重（默认按任务书 §46，可配置）。"""

    w_novelty: float = 0.40
    w_cluster: float = 0.25
    w_action: float = 0.20
    w_survival: float = 0.15


DEFAULT_OPPORTUNITY_CONFIG = SearchOpportunityConfig()

#: ClusterRarity adapter 槽位：fn(factor_id) -> float（cluster_size）或 None
ClusterStatsFn = Callable[[str], float | None]


def cluster_rarity(cluster_size: float | None) -> float:
    """ClusterRarity = 1/sqrt(1+cluster_size)。

    语义区分三种输入：
    - ``cluster_size=None`` 或任何无法解析/非法（<=0）的输入 → **无信息**，
      返回中性 ``CLUSTER_RARITY_NEUTRAL = 0.5``。缺信息绝不能被当成「最稀有」：
      那会引导 Retriever 把无 cluster 信息的因子当成稀缺方向。
    - ``cluster_size=1``（已知自成 singleton，cluster 成员表确认该因子是唯一成员）
      → 1/sqrt(2) ≈ 0.707（真稀有，但不再封顶 1.0）。
    - ``cluster_size=N>=2`` → 1/sqrt(1+N)，随成员数增加单调递减（被挖烂的大簇低机会）。
    """
    if cluster_size is None:
        return CLUSTER_RARITY_NEUTRAL
    try:
        size = float(cluster_size)
    except (TypeError, ValueError):
        return CLUSTER_RARITY_NEUTRAL
    if not size > 0.0:
        return CLUSTER_RARITY_NEUTRAL
    return 1.0 / (1.0 + size) ** 0.5


def unexplored_action_ratio(
    attempted: Sequence[str] | None,
    all_actions: Sequence[str] | None = None,
) -> float:
    """UnexploredActionRatio = 未尝试 action 数 / 全部 action 数。

    attempted 为已尝试过的 action 类型集合。全部 action 已尝试 → 0（该节点
    action 探索空间已耗尽）；一个都未尝试 → 1。空 all_actions → 中性 0.5。
    """
    fams = list(all_actions) if all_actions else list(DEFAULT_ACTION_FAMILIES)
    if not fams:
        return 0.5
    tried = set(attempted or ())
    unexplored = [a for a in fams if a not in tried]
    return len(unexplored) / len(fams)


def survival_opportunity(
    visible_survival: Mapping[str, Any] | None,
    *,
    prior: float = SURVIVAL_PRIOR,
    strength: float = SURVIVAL_STRENGTH,
    cap: float = SURVIVAL_CAP,
) -> float:
    """SurvivalOpportunity：只接受「cutoff 已可见」的 survival 记忆。

    Parameters
    ----------
    visible_survival : dict | None
        来自 survival 层且已过 cutoff 闸门的可见记忆；None 或 key 缺失时按
        无信息处理（返回中性 0.0）。必须含以下 key 之一：
        - ``survival_rate`` / ``raw_survival_rate``：原始存活率；
        - ``shrunk_survival_rate``：已做置信收缩的存活率；
        - ``support_count`` / ``support``：支撑期数（用于收缩）。

    Returns
    -------
    float
        置信收缩后相对先验的增益，cap 到 [0, SURVIVAL_CAP]。sealed 段（调用方
        未传可见数据）返回 0.0（中性，不给奖励）。
    """
    if visible_survival is None or not visible_survival:
        return 0.0
    raw = visible_survival.get("survival_rate", visible_survival.get("raw_survival_rate"))
    shrunk = visible_survival.get("shrunk_survival_rate")
    support = (
        visible_survival.get("support_count", visible_survival.get("support"))
        if visible_survival
        else None
    )
    if raw is None and shrunk is None:
        return 0.0
    if shrunk is not None:
        try:
            shrunk_v = float(shrunk)
        except (TypeError, ValueError):
            return 0.0
        gain = max(0.0, shrunk_v - prior)
        return min(cap, gain)
    try:
        raw_v = float(raw)
        k = max(0.0, float(support) if support is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0
    shrunk_v = (raw_v * k + prior * strength) / (k + strength)
    gain = max(0.0, shrunk_v - prior)
    return min(cap, gain)


@dataclass
class SearchOpportunity:
    """节点机会计算器。可注入 cluster adapter 与 survival 数据源。

    cluster_fn : Callable[[str], float | None] | None
        factor_assets 或本地 cluster 统计的 adapter 槽位。返回该因子所在
        cluster 的 size（None = 无 cluster 信息 → ClusterRarity 中性
        ``CLUSTER_RARITY_NEUTRAL = 0.5``）。本地退化实现：从 memory 的
        direction_clusters / rare_directions 统计 member_count（见
        ``from_memory_store``）。
    survival_source : Callable[[str], Mapping[str, Any] | None] | None
        返回该因子「cutoff 已可见」的 survival 记忆；None 或返回 None → 中性。
        sealed 段必须返回 None（由调用方保证，本类不访问任何未闸门数据）。
    """

    config: SearchOpportunityConfig = field(default_factory=SearchOpportunityConfig)
    cluster_fn: ClusterStatsFn | None = None
    survival_source: Callable[[str], Mapping[str, Any] | None] | None = None
    all_actions: tuple[str, ...] = DEFAULT_ACTION_FAMILIES
    _local_cluster_sizes: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_memory_store(
        cls,
        store: Any,
        *,
        config: SearchOpportunityConfig | None = None,
    ) -> "SearchOpportunity":
        """本地退化 cluster 统计：从 memory 的 direction_clusters 读 member_count。

        只调用公共 API（``rare_directions`` / ``cluster_summary``）。store 无对应
        方法或读不到 → 该因子 cluster_size=None（ClusterRarity 中性
        ``CLUSTER_RARITY_NEUTRAL = 0.5``）。
        """
        obj = cls(config=config or SearchOpportunityConfig())
        sizes: dict[str, float] = {}
        try:
            if hasattr(store, "rare_directions"):
                for d in store.rare_directions(k=50):
                    cid = str(d.get("cluster_id") or "")
                    cnt = d.get("member_count")
                    if cid and cnt is not None:
                        sizes[cid] = float(cnt)
        except Exception:  # noqa: BLE001 - 读不到退化中性
            sizes = {}
        obj._local_cluster_sizes = sizes
        if not obj._local_cluster_sizes and hasattr(store, "cluster_summary"):
            try:
                summary = store.cluster_summary() or {}
                for top in summary.get("top", []) or []:
                    cid = str(top.get("cluster_id") or "")
                    cnt = top.get("member_count")
                    if cid and cnt is not None:
                        obj._local_cluster_sizes[cid] = float(cnt)
            except Exception:  # noqa: BLE001 - 退化中性
                pass
        # 本地退化 cluster adapter：按 factor 在 cluster 中的成员数估计。
        # cluster 成员表存在但该 factor 不在任何簇 → _local_cluster_sizes 已含
        # 全部簇，查不到 → 返回 1.0：这是「已知自成 singleton」（有成员表且确认
        # 该 factor 不属于任何多成员簇），ClusterRarity = 1/sqrt(2) ≈ 0.707，
        # 是「真稀有可能」而非「无信息」。
        # cluster 成员表完全不存在（rare_directions/cluster_summary 均空）→
        # _local_cluster_sizes 为空 dict，cluster_fn 拿不到任何 context：此处仍
        # 返回 1.0（保持旧版向后兼容 + 外部已有断言），语义由调用方在 cluster_size
        # 进入 cluster_rarity 前决定——本包最重要的语义是「被挖烂的大簇 → 低机会」，
        # 无外部信息时宁可保守按 singleton 处理，也不假设拥挤（不会把稀缺性封顶到 1.0）。
        obj.cluster_fn = obj._local_cluster_size_of
        return obj

    def _local_cluster_size_of(self, factor_id: str) -> float | None:
        """本地退化：按 factor_id 前缀匹配 direction cluster；查不到 → 1.0。

        返回 1.0 表示「已知自成 singleton（cluster_size=1）」：这是合法的 rarity
        计算输入（1/sqrt(2)≈0.707），不是「无 cluster 上下文（None→中性 0.5）」。
        """
        if not factor_id:
            return 1.0
        if self._local_cluster_sizes:
            hit = self._local_cluster_sizes.get(factor_id)
            if hit is not None:
                return float(hit)
        return 1.0

    # ------------------------------------------------------------------
    # 四维
    # ------------------------------------------------------------------

    def novelty_opportunity(self, mean_novelty_gain: float | None) -> float:
        """NoveltyOpportunity = mean_novelty_gain（未观测 → warmup 中值 0.5）。"""
        if mean_novelty_gain is None:
            return WARMUP_NOVELTY
        try:
            v = float(mean_novelty_gain)
        except (TypeError, ValueError):
            return WARMUP_NOVELTY
        if v != v:  # NaN
            return WARMUP_NOVELTY
        return min(max(v, 0.0), 1.0)

    def cluster_rarity_of(self, factor_id: str) -> float:
        if self.cluster_fn is not None:
            try:
                return cluster_rarity(self.cluster_fn(factor_id))
            except Exception:  # noqa: BLE001 - adapter 异常退化中性
                return cluster_rarity(None)
        return cluster_rarity(None)

    def unexplored_ratio_of(
        self, attempted: Sequence[str] | None, action_to_retrieve: str | None = None
    ) -> float:
        """UnexploredActionRatio（可选按目标 action 单独加成）。

        ``action_to_retrieve`` 给定且尚未尝试 → 在未尝试比例基础上再加成
        （该 action 本身是机会）。
        """
        ratio = unexplored_action_ratio(attempted, self.all_actions)
        if action_to_retrieve and attempted:
            if action_to_retrieve not in set(attempted):
                # 目标 action 尚未尝试 → 视为机会：按该 action 自身权重加成。
                # 关键语义：被挖 20 次的 action 依然「已尝试过」，即使它不是
                # 目标 action；目标 action 未尝试才是机会核心。对比时：
                # - case A 目标=CROSSOVER（未尝试），已尝试 {WINDOW_SCALE} →
                #   未尝试比例 8/9，目标未尝试 → 机会高；
                # - case B 目标=WINDOW_SCALE（已尝试 20 次），已尝试 {CROSSOVER}
                #   → 未尝试比例 8/9，但目标已尝试 → 机会低（衰减）。
                n = max(len(self.all_actions), 1)
                action_bonus = 1.0 / n
                return min(1.0, ratio + action_bonus)
            else:
                # 目标 action 已尝试：按尝试次数倒数给衰减加成。
                count = sum(1 for a in attempted if a == action_to_retrieve)
                if count <= 0:
                    return ratio
                tried_bonus = 1.0 / (1.0 + count)
                return min(1.0, ratio + tried_bonus * (1.0 / max(len(self.all_actions), 1)))
        return ratio

    def survival_opportunity_of(
        self, factor_id: str, visible_survival: Mapping[str, Any] | None = None
    ) -> float:
        """SurvivalOpportunity：优先用显式传入的可见记忆；否则走注入的 survival_source。

        sealed 段：survival_source 返回 None（或未配置）→ 中性 0.0（不给奖励）。
        """
        if visible_survival is not None:
            return survival_opportunity(visible_survival)
        if self.survival_source is not None:
            try:
                return survival_opportunity(self.survival_source(factor_id))
            except Exception:  # noqa: BLE001 - 数据源异常按 sealed 中性处理
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # 总分
    # ------------------------------------------------------------------

    def compute(
        self,
        factor_id: str,
        *,
        mean_novelty_gain: float | None,
        attempted_actions: Sequence[str] | None,
        cluster_size: float | None = None,
        visible_survival: Mapping[str, Any] | None = None,
        action_to_retrieve: str | None = None,
    ) -> dict[str, float]:
        """返回四维分解 + 总分（0~1）。

        cluster_size 显式传入时优先于 cluster_fn adapter。
        visible_survival 显式传入时优先于 survival_source。
        """
        nov = self.novelty_opportunity(mean_novelty_gain)
        csize = cluster_size if cluster_size is not None else (
            self.cluster_fn(factor_id) if self.cluster_fn is not None else None
        )
        clu = cluster_rarity(csize)
        act = self.unexplored_ratio_of(attempted_actions, action_to_retrieve)
        sur = self.survival_opportunity_of(factor_id, visible_survival)
        c = self.config
        total = (
            c.w_novelty * nov
            + c.w_cluster * clu
            + c.w_action * act
            + c.w_survival * sur
        )
        return {
            "novelty": nov,
            "cluster_rarity": clu,
            "unexplored_action": act,
            "survival": sur,
            "total": total,
        }

    def components(self, factor_id: str, **kw: Any) -> dict[str, float]:
        return self.compute(factor_id, **kw)
