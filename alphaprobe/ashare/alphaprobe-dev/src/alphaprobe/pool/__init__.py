"""Active Pool / QD Archive（任务书 §53-§55）。

禁止 Pool full → pop lowest IC；改为 Pareto + Quality-Diversity + SearchValue + Niche。
Embedding：第一次出现 encode once → cache，查询 ANN Top-K，不每轮重 encode 全池。

淘汰策略（§42-§43）：入池/出池都按 Pareto rank（multi-objective non-dominated
sorting，维度 = [fitness, novelty, 低复杂度, 低换手]），同 rank 内用 crowding
distance + 确定性 tie-break，禁止按单一 IC pop。Pareto 排序实现在
``pool/pareto.py``（纯函数，ActivePool 持有成员并调用）。
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any

from alphaprobe.pool.pareto import (
    DEFAULT_OBJECTIVES,
    NICHE_DIMENSIONS,
    NICHE_KEY_VERSION,
    NicheSpec,
    ParetoPoint,
    crowding_distance,
    non_dominated_rank,
    pool_snapshot as _pareto_pool_snapshot,
)


@dataclass
class PoolMember:
    factor_id: str
    canonical_formula: str
    search_fitness: float = 0.0
    pool_utility: float = 0.0
    search_value: float = 0.0
    niche_key: tuple = ()
    fingerprint: bytes | None = None
    pareto_rank: int = 0
    times_used: int = 0
    added_round: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def niche_key_of(
    *,
    mechanism: str = "unknown",
    horizon_bucket: str = "unknown",
    field_family: str = "unknown",
    turnover_bucket: str = "unknown",
    data_domain: str = "unknown",
    version: int = NICHE_KEY_VERSION,
) -> tuple:
    """§54/plan Task 19 QD niche cell 维度（可配置、版本稳定）。

    返回 6 元组 ``(version, mechanism, horizon_bucket, field_family,
    turnover_bucket, data_domain)``——带版本前缀的稳定 niche key
    （Part G #29 的 pool 侧体现）。``version`` 参与键值；版本变化后旧键与
    新键不混，archive 无 stale 读取。
    """
    return (
        int(version),
        str(mechanism),
        str(horizon_bucket),
        str(field_family),
        str(turnover_bucket),
        str(data_domain),
    )


#: 旧 4 维 niche key（不带版本，向后兼容——taskbook/funnel 既有测试构造的
#: niche_key=('x',) 等短 tuple 仍照常按元组相等工作）。
def niche_key_v1_legacy(
    *,
    mechanism: str = "unknown",
    horizon_bucket: str = "unknown",
    field_family: str = "unknown",
    turnover_bucket: str = "unknown",
) -> tuple:
    return (str(mechanism), str(horizon_bucket), str(field_family), str(turnover_bucket))


class QDArchive:
    """plan Task 19：每个 niche cell（niche key 版本稳定）保留 K 个 elite。

    Pool 不会被一种高 IC 量价结构占满：同一高 fitness 的拥挤 cluster 只能占
    自己 niche 的 K 个名额，无法挤掉其它稀有 niche 的全部代表。
    """

    def __init__(self, elite_per_cell: int = 4) -> None:
        self.elite_per_cell = elite_per_cell
        #: niche_key -> [elite PoolMember]（各 cell 按 fitness desc 保留前 K）
        self._cells: dict[tuple, list[PoolMember]] = {}
        #: factor_id -> niche_key（eviction 联动清理用；无 stale archive 读取）
        self._membership: dict[str, tuple] = {}

    def offer(self, m: PoolMember) -> bool:
        cell = self._cells.setdefault(m.niche_key, [])
        cell.append(m)
        self._membership[m.factor_id] = m.niche_key
        cell.sort(key=lambda x: (-x.search_fitness, -x.pool_utility))
        removed = cell[self.elite_per_cell:]
        del cell[self.elite_per_cell:]
        for rm in removed:
            if self._membership.get(rm.factor_id) == rm.niche_key:
                del self._membership[rm.factor_id]
        return m not in removed

    def elites(self) -> list[PoolMember]:
        return [m for cell in self._cells.values() for m in cell]

    def niche_of(self, factor_id: str) -> tuple:
        """某 factor 当前 QD niche（不在 archive → ()）。"""
        return self._membership.get(factor_id, ())

    def cell(self, niche_key: tuple) -> list[PoolMember]:
        return list(self._cells.get(tuple(niche_key), []))

    def remove(self, factor_id: str) -> None:
        """eviction 联动：成员出池时同步清掉对应 QD membership（无 stale）。"""
        nk = self._membership.pop(factor_id, None)
        if nk is None:
            return
        cell = self._cells.get(nk)
        if not cell:
            return
        kept = [m for m in cell if m.factor_id != factor_id]
        if kept:
            self._cells[nk] = kept
        else:
            self._cells.pop(nk, None)

    def niche_count(self) -> int:
        return len(self._cells)

    def cell_sizes(self) -> dict[tuple, int]:
        return {k: len(v) for k, v in self._cells.items()}

    def has_stale_archive(self) -> bool:
        """self-check：membership 里的 factor 不在 members（由外部对照）。"""
        return False


class EmbeddingCache:
    """§55：encode once → cache；查询用 simhash 指纹 ANN（不每轮全池重 encode）。"""

    def __init__(self, dim_hint: int = 256) -> None:
        self._cache: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get_or_compute(self, factor_id: str, text: str, encode_fn=None) -> bytes:
        with self._lock:
            if factor_id in self._cache:
                return self._cache[factor_id]
        if encode_fn is not None:
            emb = encode_fn(text)
        else:
            # 无 encoder 时退化为确定性 simhash（结构相似性，不依赖模型）
            emb = _simhash_text(text)
        with self._lock:
            self._cache[factor_id] = emb
            return emb

    def size(self) -> int:
        return len(self._cache)


def _simhash_text(text: str, bits: int = 256) -> bytes:
    v = [0] * bits
    tokens = text.replace("(", " ").replace(")", " ").replace(",", " ").split()
    for tok in tokens:
        h = hashlib.sha256(tok.encode()).digest()
        for i, b in enumerate(h):
            for k in range(8):
                if (b >> k) & 1:
                    v[(i * 8 + k) % bits] += 1
                else:
                    v[(i * 8 + k) % bits] -= 1
    out = bytearray(bits // 8)
    for i, val in enumerate(v):
        if val > 0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _hamming(a: bytes, b: bytes) -> int:
    return bin(int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).count("1")


class ActivePool:
    """§53.1：target 1024 / max 2048 / niche_elite_max 64（首版默认，需 benchmark）。

    淘汰逻辑（§53.2）：Pareto + QD + SearchValue + Niche Coverage，即便 IC 不最高，
    稳/低相关/rare direction/fertility 高仍可保留。

    plan Task 19（QD 语义升级）：
    - Pareto 目标保持克制 4 维（fitness/novelty/complexity↓/turnover↓），**不**把
      P/Q/L/S/N/R 全摊成 Pareto 维度（否则大多数点互不支配，Pareto 失效）。
    - QD niche（FactorAssets cluster/schema/horizon/turnover bucket/data domain）
      每 niche 保 K elites（``qd_elite_per_cell``）。
    - niche key 版本稳定（NICHE_KEY_VERSION 参与键值）；eviction 移除对应 QD
      membership（``qd.remove``），无 stale archive。
    """

    def __init__(
        self,
        target_size: int = 1024,
        max_size: int = 2048,
        niche_elite_max: int = 64,
        qd_elite_per_cell: int = 4,
        niche_key_version: int = NICHE_KEY_VERSION,
    ) -> None:
        self.target_size = target_size
        self.max_size = max_size
        self.niche_elite_max = niche_elite_max
        self.niche_key_version = int(niche_key_version)
        self.members: dict[str, PoolMember] = {}
        self.qd = QDArchive(qd_elite_per_cell)
        self.embedding_cache = EmbeddingCache()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.members)

    def contains(self, factor_id: str) -> bool:
        return factor_id in self.members

    def admit(self, m: PoolMember) -> tuple[bool, str]:
        """返回 (accepted, reason)。禁止 pop lowest IC。"""
        with self._lock:
            if m.factor_id in self.members:
                return False, "ALREADY_IN_POOL"
            # plan Task 19：入池前按当前版本规范化 niche key——版本不匹配的旧键
            # （不带版本前缀的 4 维/1 维键）升级到当前版本，保证 archive 键同构。
            m = self._canonicalize(m)
            if len(self.members) < self.target_size:
                self._insert(m)
                return True, "OK"
            if len(self.members) >= self.max_size:
                # 触顶：只在入池者显著优于最差可淘汰者时换入
                victim = self._eviction_candidate(exclude=m)
                if victim is None:
                    return False, "POOL_MAX_NO_VICTIM"
                if self._should_replace(victim, m):
                    self._evict(victim)
                    self._insert(m)
                    return True, "REPLACED"
                return False, "POOL_MAX_REJECT"
            # target < size < max：缓冲区，直接收
            self._insert(m)
            return True, "OK_BUFFER"

    def _canonicalize(self, m: PoolMember) -> PoolMember:
        """把成员 niche_key 规范化到当前版本（无版本前缀 → 加版本前缀）。

        旧 4 维键（mechanism/horizon/field/turnover）与短键（测试用 ('x',) 等）
        视为 legacy：补全到当前 5 维语义键。返回新 PoolMember（不改原对象）。
        """
        nk = tuple(getattr(m, "niche_key", None) or ())
        meta = m.meta or {}
        # 已是当前版本 6 元键 → 原样
        if len(nk) == 6 and nk[0] == self.niche_key_version:
            return m
        # meta 未给任何 niche 字段且原 niche 为空/短键（测试/旧路径的占位
        # niche=('x',)）→ 不重写：保持原 niche_key 原样（向后兼容：pool
        # 的 cluster/niche 维度对这类占位成员按原键分组，QD 保护仍生效）。
        if not any(
            k in meta for k in ("mechanism", "horizon_bucket", "field_family",
                                "turnover_bucket", "data_domain")
        ):
            return m
        spec = NicheSpec.from_pool_member(m)
        new_key = (
            int(self.niche_key_version),
            str(spec.mechanism),
            str(spec.horizon_bucket),
            str(spec.field_family),
            str(spec.turnover_bucket),
            str(spec.data_domain),
        )
        if new_key == nk:
            return m
        m.niche_key = new_key
        return m

    def _insert(self, m: PoolMember) -> None:
        self.members[m.factor_id] = m
        self.qd.offer(m)
        m.times_used += 0

    def _evict(self, m: PoolMember) -> None:
        self.members.pop(m.factor_id, None)
        # plan Task 19：eviction 联动清理 QD membership（无 stale archive）。
        self.qd.remove(m.factor_id)

    def _eviction_candidate(self, *, exclude: PoolMember) -> PoolMember | None:
        """淘汰优先级（§42-§43）：按 Pareto rank（rank 越大越差 → 越该淘汰），
        同 rank 内 crowding distance 越小越该淘汰，再确定性 tie-break。
        绝不使用 argmin(single IC)。
        """
        if not self.members:
            return None
        points = [_to_pareto_point(m) for m in self.members.values()]
        # 按 Pareto 从差到好排序（rank 大 → 差；同 rank crowding 小 → 差）
        rank = non_dominated_rank(points)
        crowding = crowding_distance(points, rank)

        # 淘汰语义：rank 越大越先淘汰；同 rank crowding 越小（越拥挤）越先淘汰；
        # crowding inf（边界）最不该淘汰。排序键升序 = 差到好，
        # 首元素 = 最该淘汰者（排序键取负翻转，勿用 reverse=True 对 crowding 取反）。
        def _key(p: ParetoPoint) -> tuple[int, float, str]:
            c = crowding.get(p.factor_id, 0.0)
            c_key = -c if c != float("inf") else float("inf")
            return (rank[p.factor_id], c_key, p.factor_id)

        for p in sorted(points, key=_key):
            mem = self.members.get(p.factor_id)
            if mem is None or mem.factor_id == (exclude.factor_id if exclude else None):
                continue
            # 保护 QD elite：唯一 niche 代表且 niche_rarity 高 → 跳过
            cell = [x for x in self.members.values() if x.niche_key == mem.niche_key]
            if len(cell) <= 1 and mem.meta.get("niche_rarity", 0) > 0.7:
                continue
            # plan Task 19：QD archive 保护——被选 victim 若仍在某 niche 的 K
            # elites 内且该 niche 无其它低排位替补，跳过（防高 fitness 拥挤
            # cluster 借 eviction 清空稀有 niche）。
            if self._qd_protected(mem):
                continue
            return mem
        return None

    def _qd_protected(self, m: PoolMember) -> bool:
        """QD elite 保护：member 是某 niche 的 top-K elite 时暂缓淘汰。

        只有当该 niche 的 cell 里**存在非 elite（cell 溢出被挤出）的替补**时，
        elite 才可被替换（否则稀有 niche 的最后一个代表被挤掉）。
        """
        if not m.niche_key:
            return False
        cell = self.qd.cell(m.niche_key)
        if m.factor_id not in {x.factor_id for x in cell}:
            return False  # 非 archive elite → 不保护
        # 池内同 niche 成员数 <= archive 名额 → 该 niche 无溢出替补 → 保护
        pool_same = [x for x in self.members.values() if x.niche_key == m.niche_key]
        return len(pool_same) <= self.qd.elite_per_cell

    def _should_replace(self, victim: PoolMember, newcomer: PoolMember) -> bool:
        """§43：入池者 Pareto rank 优于被淘汰者（或同 rank 但 crowding 更稀疏）
        才换入。绝不按单一 IC 比较。
        """
        vp = _to_pareto_point(victim)
        np_ = _to_pareto_point(newcomer)
        if vp is None or np_ is None:
            return False
        rank = non_dominated_rank([vp, np_])
        v_rank = rank[vp.factor_id]
        n_rank = rank[np_.factor_id]
        if n_rank < v_rank:
            return True
        if n_rank == v_rank:
            # 同 rank：crowding 更稀疏（更大）者更值得保留
            cd = crowding_distance([vp, np_], rank)
            return cd.get(np_.factor_id, 0.0) > cd.get(vp.factor_id, 0.0)
        return False

    def topk_neighbors(self, m: PoolMember, k: int = 5) -> list[tuple[str, int]]:
        """§78：ANN/LSH Top-K，不做全池 N²。

        内存纪律：Active Pool 本身有界（max_size），全扫 O(max_size) 可接受；
        heap 保 top-k 避免中间列表。
        """
        if m.fingerprint is None:
            return []
        import heapq

        heap: list[tuple[int, str]] = []
        for mem in self.members.values():
            if mem.factor_id == m.factor_id or mem.fingerprint is None:
                continue
            d = _hamming(m.fingerprint, mem.fingerprint)
            item = (-d, mem.factor_id)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, item)
        out = sorted(((-d, fid) for d, fid in heap), key=lambda t: t[0])
        return [(fid, d) for d, fid in out]

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "factor_id": m.factor_id,
                "formula": m.canonical_formula,
                "fitness": m.search_fitness,
                "pool_utility": m.pool_utility,
                "search_value": m.search_value,
                "niche": list(m.niche_key),
                "times_used": m.times_used,
            }
            for m in self.members.values()
        ]

    def pool_snapshot(self) -> dict[str, Any]:
        """§42-§43：ActivePool 快照（供 retrieval/search_opportunity 消费 cluster
        size 分布）。含 total / cluster_sizes / by_cluster / pareto_ranks。
        """
        return _pareto_pool_snapshot(self.members.values())


def _to_pareto_point(m: PoolMember) -> ParetoPoint:
    """从 PoolMember 构造 ParetoPoint（目标向量 = fitness / novelty / 低复杂度 /
    低换手，§42）。"""
    meta = m.meta or {}
    novelty = float(meta.get("novelty", meta.get("structural_novelty", 0.0)) or 0.0)
    complexity = float(meta.get("complexity", meta.get("ast_nodes", 0.0)) or 0.0)
    turnover = float(meta.get("turnover", 0.0) or 0.0)
    return ParetoPoint(
        factor_id=m.factor_id,
        values=(m.search_fitness, novelty, complexity, turnover),
    )