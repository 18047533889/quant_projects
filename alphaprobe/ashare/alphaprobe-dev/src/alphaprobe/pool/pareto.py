"""ActivePool 真 Pareto（任务书 §42-§43）。

把 ActivePool 的淘汰从「单维排序 / 加权综合分」升级为 multi-objective
non-dominated sorting（NSGA-II 风格）：

- 目标维度（§42）：[FactorFitness, Novelty(或 cluster rarity), 低复杂度,
  低换手]。fitness / novelty 越高越好；复杂度 / 换手越低越好。
- ``non_dominated_rank``：返回每个成员的 Pareto rank（0 = 最前 front）。
- 同 rank 内用 crowding distance 或确定性 tie-break 排序（§43）。
- 淘汰策略：入池/出池都按 Pareto rank（rank 越大越差 → 越该淘汰），
  禁止按单一 IC pop。

本模块只做 Pareto 排序与淘汰选择，不持有池状态；ActivePool 持有成员并调用
本模块的纯函数。合成数据可测，零 LLM / 零模型 / 零网络。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Objective",
    "ParetoPoint",
    "non_dominated_rank",
    "crowding_distance",
    "pareto_eviction_candidate",
    "pool_snapshot",
    "pareto_rank_of",
]


@dataclass(frozen=True)
class Objective:
    """一个目标维度。

    ``maximize=True`` 表示值越大越好（fitness / novelty）；
    ``maximize=False`` 表示值越小越好（复杂度 / 换手）。
    """

    name: str
    maximize: bool = True


#: 默认目标维度（§42）：fitness / novelty 最大化，复杂度 / 换手最小化。
DEFAULT_OBJECTIVES: tuple[Objective, ...] = (
    Objective("fitness", maximize=True),
    Objective("novelty", maximize=True),
    Objective("complexity", maximize=False),
    Objective("turnover", maximize=False),
)


@dataclass(frozen=True)
class ParetoPoint:
    """一个待排序成员的目标向量。

    ``values`` 与 ``objectives`` 一一对应（原始值，方向由 objective 决定）。
    ``factor_id`` 用于淘汰选择与审计。
    """

    factor_id: str
    values: tuple[float, ...]
    objectives: tuple[Objective, ...] = DEFAULT_OBJECTIVES

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if len(self.values) != len(self.objectives):
            raise ValueError("values must match objectives length")
        for v in self.values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError("values must be non-boolean numbers")
            fv = float(v)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError("values must be finite")
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))

    def _maximized(self) -> tuple[float, ...]:
        """把每个目标统一为「越大越好」的向量（minimize 目标取负）。"""
        return tuple(
            float(v) if obj.maximize else -float(v)
            for v, obj in zip(self.values, self.objectives)
        )


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """a 支配 b：a 在所有目标 >= b，且至少一个目标严格 > b。"""
    ge = all(x >= y for x, y in zip(a, b))
    strict = any(x > y for x, y in zip(a, b))
    return ge and strict


def non_dominated_rank(
    points: Sequence[ParetoPoint],
) -> dict[str, int]:
    """NSGA-II 非支配排序，返回 factor_id -> rank（0 = 最前 front）。

    复杂度 O(n²·m)（n=点数，m=目标数），ActivePool 有界（max_size）可接受。
    """
    if not points:
        return {}
    maximized = [p._maximized() for p in points]
    n = len(points)
    # S[i] = 点 i 支配的点集合；n_dominated[i] = 支配点 i 的点数
    S: list[set[int]] = [set() for _ in range(n)]
    n_dominated: list[int] = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(maximized[i], maximized[j]):
                S[i].add(j)
                n_dominated[j] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if n_dominated[i] == 0]
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for i in current:
            for j in S[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
        current = next_front

    rank: dict[str, int] = {}
    for r, front in enumerate(fronts):
        for i in front:
            rank[points[i].factor_id] = r
    return rank


def crowding_distance(
    points: Sequence[ParetoPoint],
    rank: dict[str, int],
) -> dict[str, float]:
    """同 rank 内的 crowding distance（越大越稀疏 → 越该保留）。

    边界点（某目标极值）crowding = inf。用于同 rank 内 tie-break：
    淘汰时优先淘汰 crowding 最小（最拥挤）的成员。
    """
    if not points:
        return {}
    maximized = [p._maximized() for p in points]
    n = len(points)
    m = len(points[0].objectives)
    dist: dict[str, float] = {p.factor_id: 0.0 for p in points}

    # 按 rank 分组
    by_rank: dict[int, list[int]] = {}
    for i, p in enumerate(points):
        by_rank.setdefault(rank[p.factor_id], []).append(i)

    for r, idxs in by_rank.items():
        # 同 rank 全等价（任意两点全维都无差异）→ 无法从目标空间区分谁更差，
        # 整组按边界 inf 处理，淘汰侧退化为纯 tie-break（factor_id 字典序）——
        # 与任务书「完全等价点按字典序确定性淘汰」一致；且绝不误伤真边界。
        vals = {tuple(maximized[i]) for i in idxs}
        if len(vals) == 1:
            for i in idxs:
                dist[points[i].factor_id] = float("inf")
            continue
        if len(idxs) <= 2:
            # 同 rank 少于 3 个：无内部点可区分，NSGA-II 惯例一律视作边界 inf
            # （极值保护）→ 全排最不该淘汰，由确定性 tie-break 兜底。
            for i in idxs:
                dist[points[i].factor_id] = float("inf")
            continue
        # 每维排序后，若出现重复极值（span == 0）不能把整组当 inf：
        # 那样会让「完全相同点」与真边界点混淆，甚至漏掉应淘汰的内部重复者。
        # 正确做法：只把当前维严格两端标 inf，剩余内部点按间距累加；
        # 同 rank 存在部分等价（某维重复极值）→ 内部重复者 crowding 保持 0
        # （最拥挤），由淘汰侧 tie-break 确定性选择，杜绝淘汰边界。
        for obj in range(m):
            idxs_sorted = sorted(idxs, key=lambda i: maximized[i][obj])
            dist[points[idxs_sorted[0]].factor_id] = float("inf")
            dist[points[idxs_sorted[-1]].factor_id] = float("inf")
            fmin = maximized[idxs_sorted[0]][obj]
            fmax = maximized[idxs_sorted[-1]][obj]
            span = fmax - fmin
            if span <= 0.0:
                continue
            for k in range(1, len(idxs_sorted) - 1):
                i = idxs_sorted[k]
                prev = maximized[idxs_sorted[k - 1]][obj]
                nxt = maximized[idxs_sorted[k + 1]][obj]
                dist[points[i].factor_id] += (nxt - prev) / span
    return dist


def pareto_eviction_candidate(
    points: Sequence[ParetoPoint],
    *,
    exclude_factor_id: str | None = None,
    tie_break: Callable[[str, str], int] | None = None,
) -> ParetoPoint | None:
    """按 Pareto rank 选淘汰候选（rank 越大越差 → 越该淘汰）。

    同 rank 内：crowding distance 越小（越拥挤）越该淘汰；crowding 相同
    （含边界 inf）用确定性 tie-break（factor_id 字典序，稳定可复现）。

    Parameters
    ----------
    points : Sequence[ParetoPoint]
        候选成员（不含被排除者）。
    exclude_factor_id : str | None
        排除的成员（如入池者自身）。
    tie_break : Callable[[str, str], int] | None
        同 rank 且 crowding 相同时的确定性比较；缺省用 factor_id 字典序
        （返回负 = 前者优先淘汰）。

    Returns
    -------
    ParetoPoint | None
        最该淘汰的成员；空池返回 None。
    """
    if not points:
        return None
    cands = [p for p in points if p.factor_id != exclude_factor_id]
    if not cands:
        return None
    rank = non_dominated_rank(cands)
    crowding = crowding_distance(cands, rank)

    def _tie(a: str, b: str) -> int:
        if tie_break is not None:
            return tie_break(a, b)
        return -1 if a < b else (1 if a > b else 0)

    # 淘汰优先级（「更该淘汰」= 比较序更小，min 取最该淘汰者）：
    #   1) rank 越大越差 → 更该淘汰（rank 0 = 最前 front，绝不能因 min 反成淘汰它）；
    #   2) 同 rank：crowding 越小（越拥挤）越该淘汰；crowding=inf（边界点，
    #      最稀疏）→ 视为 +inf，最不该淘汰（排最后）；
    #   3) 完全并列：tie_break / factor_id 字典序确定性（返回负 = 前者优先淘汰）。
    # 不可用单一 max：max 对 crowding 取最大值会把最稀疏者误选为淘汰对象（方向反）。
    # 不可用 min(rank, ...)：rank 0 是最优 front，min 会把最优者误选为淘汰对象。
    def _cmp(a: ParetoPoint, b: ParetoPoint) -> int:
        ra = rank[a.factor_id]
        rb = rank[b.factor_id]
        if ra != rb:
            return -1 if ra > rb else 1  # rank 大者（更差 front）优先淘汰
        ca = crowding.get(a.factor_id, 0.0)
        cb = crowding.get(b.factor_id, 0.0)
        # 有限值越小越拥挤 → 越该淘汰；inf（边界）视为 +inf → 最不该淘汰。
        ca_key = float("inf") if ca == float("inf") else ca
        cb_key = float("inf") if cb == float("inf") else cb
        if ca_key != cb_key:
            return -1 if ca_key < cb_key else 1
        return _tie(a.factor_id, b.factor_id)

    return min(cands, key=cmp_to_key(_cmp))


def pareto_rank_of(
    points: Sequence[ParetoPoint],
    factor_id: str,
) -> int:
    """查询某成员的 Pareto rank（不在池中返回 -1）。"""
    rank = non_dominated_rank(points)
    return rank.get(factor_id, -1)


def pool_snapshot(
    members: Iterable[Any],
    *,
    cluster_key_fn: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """ActivePool 快照（供 retrieval/search_opportunity 消费 cluster size 分布）。

    Parameters
    ----------
    members : Iterable[Any]
        池成员（PoolMember 或任意带 factor_id 的对象）。
    cluster_key_fn : Callable[[Any], str] | None
        取成员 cluster 键的函数；缺省用 ``meta['cluster_id']``（无则
        ``meta['niche_key']`` 字符串化，再退化为 factor_id 自身）。

    Returns
    -------
    dict
        - ``total``：成员总数；
        - ``cluster_sizes``：cluster_key -> 成员数（供 cluster_rarity）；
        - ``by_cluster``：cluster_key -> [factor_id, ...]；
        - ``pareto_ranks``：factor_id -> rank（若成员带目标向量）。
    """
    members = list(members)
    total = len(members)
    cluster_sizes: dict[str, int] = {}
    by_cluster: dict[str, list[str]] = {}
    for m in members:
        ck = _cluster_key_of(m, cluster_key_fn)
        cluster_sizes[ck] = cluster_sizes.get(ck, 0) + 1
        by_cluster.setdefault(ck, []).append(m.factor_id)

    # Pareto rank（若成员可构造 ParetoPoint）
    pareto_ranks: dict[str, int] = {}
    points = [_to_pareto_point(m) for m in members]
    points = [p for p in points if p is not None]
    if points:
        pareto_ranks = non_dominated_rank(points)

    return {
        "total": total,
        "cluster_sizes": cluster_sizes,
        "by_cluster": by_cluster,
        "pareto_ranks": pareto_ranks,
    }


def _cluster_key_of(m: Any, fn: Callable[[Any], str] | None) -> Any:
    if fn is not None:
        return fn(m)
    meta = getattr(m, "meta", None) or {}
    cid = meta.get("cluster_id")
    if cid:
        return str(cid)
    niche = getattr(m, "niche_key", None)
    if niche:
        return tuple(niche)
    return str(getattr(m, "factor_id", "unknown"))


def _to_pareto_point(m: Any) -> ParetoPoint | None:
    """从 PoolMember 构造 ParetoPoint（缺目标维度时返回 None）。

    目标向量：fitness / novelty / complexity / turnover。
    - fitness = search_fitness
    - novelty = meta['novelty'] 或 meta['structural_novelty']（缺省 0）
    - complexity = meta['complexity'] 或 meta['ast_nodes']（缺省 0，越低越好）
    - turnover = meta['turnover']（缺省 0，越低越好）
    """
    meta = getattr(m, "meta", None) or {}
    fitness = float(getattr(m, "search_fitness", 0.0) or 0.0)
    novelty = float(meta.get("novelty", meta.get("structural_novelty", 0.0)) or 0.0)
    complexity = float(meta.get("complexity", meta.get("ast_nodes", 0.0)) or 0.0)
    turnover = float(meta.get("turnover", 0.0) or 0.0)
    return ParetoPoint(
        factor_id=str(getattr(m, "factor_id", "")),
        values=(fitness, novelty, complexity, turnover),
        objectives=DEFAULT_OBJECTIVES,
    )
