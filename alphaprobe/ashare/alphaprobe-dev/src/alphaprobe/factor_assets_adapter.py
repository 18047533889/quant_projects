"""factor_assets adapter（任务书 §33 / §42-§43）。

把 factor_assets 的 identity + 聚类 + ANN 能力包装给 AlphaPROBE 消费，同时
明确与 AlphaPROBE 自带 ``seen/`` 索引的读写分工，避免双写冲突。

读写分工（docstring 权威，代码不重复实现）：
----------------------------------------------------------------------------
| 能力            | 写入方（authority）        | 读取方（consumer）          |
|-----------------|----------------------------|-----------------------------|
| 去重 / seen     | AlphaPROBE ``seen/``       | 本 adapter 只读（不写）      |
|                 | (DedupService / SeenStore) |                              |
| identity hash   | FE ``factor_engine.identity`` | 本 adapter 只读（不写）   |
|                 | (authority.build_identity_view) |                          |
| cluster/family  | factor_assets              | 本 adapter 只读（不写）      |
|                 | (IncrementalPolicy /       |                              |
|                 |  IncrementalAssignResult)  |                              |
| ANN 近邻        | factor_assets FaissANNIndex| 本 adapter 只读（不写）      |
|                 | (或 AlphaPROBE seen/       |                              |
|                 |  NearestIndex 只读)        |                              |
----------------------------------------------------------------------------

- AlphaPROBE ``seen/`` 是去重/seen 的 primary（Phase B 已接线 identity 三级
  降级链），本 adapter 绝不向 ``seen/`` 写入。
- factor_assets 是 cluster/family/ANN 的权威：``cluster_assign`` 把因子交给
  factor_assets 的 identity + 聚类，``nearest_neighbors`` 用 FaissANNIndex
  查近邻。两者各写各的，互不覆盖。
- 因子身份（identity）的唯一权威是 FE ``factor_engine.identity``：本 adapter
  的 ``identity_of`` 一律经 ``alphaprobe.authority.build_identity_view`` 取 FE
  的 ``canonical_ast_hash``（canonical_hash）与 ``signal_equivalence_id``
  （fe_identity_ref）。FE 不可用 → fail-closed 抛异常，绝不静默 sha256 兜底。

降级链（fail-closed，绝不静默吞掉）：
- factor_assets 不可导入 → 抛 :class:`FactorAssetsUnavailable`（显式异常，
  调用方决定是否降级到本地 seen/ 索引）。本 adapter 不自动静默降级。
- FE 身份权威不可用（``FactorIdentityAuthorityError``）→ ``identity_of`` /
  ``cluster_assign`` 抛 :class:`FactorAssetsUnavailable`（fail-closed，不静默
  回落文本 sha256）。
- cluster context 缺失（``cluster_versions`` 为空且未显式允许 singleton
  fallback）→ ``cluster_assign`` 抛 :class:`ClusterContextUnavailable`
  （fail-closed；OFFLINE_TEST / 测试可显式 ``allow_singleton_fallback=True``）。
- faiss 不可用 → ``nearest_neighbors`` 抛 :class:`FactorAssetsUnavailable`
  （FaissANNIndex 构造本身会抛 ImportError，这里包成统一异常）。ANN 索引默认
  惰性 faiss：不 ``add`` 不 ``build`` 就不触碰 faiss，测试可用注入的假 backend。
- 显式降级入口：``nearest_neighbors_local`` 用 AlphaPROBE ``seen/`` 的
  NearestIndex（只读）作为显式降级路径，并记录日志。

零 LLM、零模型、零网络。合成数据可测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "FactorAssetsUnavailable",
    "ClusterContextUnavailable",
    "ClusterAssignResult",
    "NearestNeighborResult",
    "PersistentSimilarityIndex",
    "ANNBackend",
    "FactorAssetsAdapter",
    "build_factor_assets_adapter",
    "GlobalFactorContext",
    "GlobalFactorGate",
    "GateVerdict",
]


class FactorAssetsUnavailable(RuntimeError):
    """factor_assets 不可导入或能力缺失（fail-closed，不静默降级）。"""


class ClusterContextUnavailable(FactorAssetsUnavailable):
    """cluster context 缺失：未提供 cluster versions 且未显式允许 singleton
    fallback（production fail-closed；测试可显式开启 fallback）。"""


# ---------------------------------------------------------------------------
# ANN 后端协议（faiss 惰性：不 add / 不 build 不触碰 faiss；测试可注入假后端）
# ---------------------------------------------------------------------------


@runtime_checkable
class ANNBackend(Protocol):
    """ANN 后端最小协议（FaissANNIndex 是默认实现）。

    仅声明 build / search 两个动作；``add`` 为可选（无增量能力的后端由
    :class:`PersistentSimilarityIndex` 在 ``search`` 前统一重建）。
    """

    def build(
        self, factor_ids: Sequence[str], embeddings: Any
    ) -> None: ...  # noqa: E704

    def search(
        self, query_embedding: Any, k: int = 10, min_similarity: Optional[float] = None
    ) -> list[Any]: ...  # noqa: E704


# ---------------------------------------------------------------------------
# 结果类型（不依赖 factor_assets 具体类，便于 AlphaPROBE 消费与测试）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterAssignResult:
    """一次 cluster_assign 的返回结构。

    ``assignments`` 每项含 factor_id / logical_cluster_id / kind / affinity /
    cluster_set_version_ref；``cluster_sizes`` 为 logical_cluster_id -> 成员数
    （供 retrieval/search_opportunity 消费 cluster size 分布）。每项 assignment
    带 ``cluster_context`` 标记（``"PROVIDED"`` | ``"SINGLETON_FALLBACK"``），
    让下游可甄别结果来自真实 cluster 版本还是 singleton fallback。
    """

    batch_id: str
    assignments: tuple[dict[str, Any], ...]
    cluster_sizes: dict[str, int]
    representative_by_cluster: dict[str, str] = field(default_factory=dict)
    cluster_context: str = "PROVIDED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "assignments": list(self.assignments),
            "cluster_sizes": dict(self.cluster_sizes),
            "representative_by_cluster": dict(self.representative_by_cluster),
            "cluster_context": self.cluster_context,
        }


@dataclass(frozen=True)
class NearestNeighborResult:
    """一次 nearest_neighbors 的返回结构（相似度检索）。"""

    factor_id: str
    distance: float
    similarity_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "distance": self.distance,
            "similarity_score": self.similarity_score,
        }


# ---------------------------------------------------------------------------
# 惰性导入 + 可用性探测
# ---------------------------------------------------------------------------


def _factor_assets_importable() -> bool:
    """探测 factor_assets 是否可导入（不抛异常）。"""
    try:
        import factor_assets  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - 探测失败即不可用
        return False


def _faiss_importable() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 持久相似度索引（版本化 key；同一版本复用，避免每次重 build 失去 ANN 意义）
# ---------------------------------------------------------------------------


class PersistentSimilarityIndex:
    """进程内持久 ANN 近邻索引。

    构造参数拼成**版本化 key**（embedding_dim / fingerprint_version /
    data_snapshot / universe_snapshot / cluster_version）。key 一致 → 复用
    backend；key 变化 → 自动失效并在下一次 ``search`` 前整体重建。

    - 惰性 faiss：不 ``add`` 不 ``build`` 就不触碰 faiss。默认 backend 是
      :class:`FaissANNIndex`（venv 无 faiss 时可用注入的假 backend 测持久化
      逻辑本身）。
    - ``add`` 增量注册因子（同 id 覆盖，幂等）；``add_many`` 批量注册。
    - backend 支持增量 ``add`` 则注册后直接落库；否则标记 dirty，下次
      ``search`` 时整库重建。``rebuild()`` 显式强制重建。
    - ``index_rebuild_count`` 计数器暴露重建次数（测试断言用）。
    """

    def __init__(
        self,
        embedding_dim: int,
        fingerprint_version: str,
        data_snapshot: str = "default",
        universe_snapshot: str = "default",
        cluster_version: str = "default",
        backend: Optional[ANNBackend] = None,
    ) -> None:
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an int")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not fingerprint_version:
            raise ValueError("fingerprint_version is required")
        self.embedding_dim = int(embedding_dim)
        self.fingerprint_version = fingerprint_version
        self.data_snapshot = data_snapshot
        self.universe_snapshot = universe_snapshot
        self.cluster_version = cluster_version
        #: 版本化 key：任一构造维度变化即整体失效。
        self.version_key = (
            self.embedding_dim,
            self.fingerprint_version,
            self.data_snapshot,
            self.universe_snapshot,
            self.cluster_version,
        )
        #: backend 可注入（默认 FaissANNIndex，惰性：需要时再构造）。
        self._backend: Optional[ANNBackend] = backend
        self._factor_ids: dict[str, Sequence[float]] = {}
        self._dirty = False
        self.index_rebuild_count = 0

    # -- backend 管理 ------------------------------------------------------

    def _ensure_backend(self) -> ANNBackend:
        """返回可用的 backend；默认 backend 惰性构造（此时才触碰 faiss）。"""
        if self._backend is None:
            self._backend = self._build_default_backend()
        return self._backend

    def _build_default_backend(self) -> ANNBackend:
        """默认 backend：factor_assets FaissANNIndex（faiss 不可用即抛，fail-closed）。"""
        try:
            from factor_assets.similarity.ann import FaissANNIndex
        except Exception as exc:  # noqa: BLE001
            raise FactorAssetsUnavailable(
                "factor_assets unavailable for default ANN backend: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not _faiss_importable():
            raise FactorAssetsUnavailable(
                "faiss is not importable; FaissANNIndex unavailable. "
                "Inject an ANNBackend (e.g. a fake backend) or pass explicit "
                "factor_ids/embeddings to use the persistent index."
            )
        return FaissANNIndex(embedding_dim=self.embedding_dim)  # type: ignore[return-value]

    # -- 注册 --------------------------------------------------------------

    def add(self, factor_id: str, embedding: Sequence[float]) -> None:
        """增量注册一个因子（同 id 覆盖，幂等）。"""
        if not factor_id:
            raise ValueError("factor_id is required")
        emb = list(embedding)
        if len(emb) != self.embedding_dim:
            raise ValueError(
                f"embedding dim {len(emb)} != index dim {self.embedding_dim}"
            )
        if factor_id in self._factor_ids and self._factor_ids[factor_id] == tuple(emb):
            return  # 幂等：同 id 同 embedding 不标记 dirty
        self._factor_ids[factor_id] = tuple(emb)
        self._dirty = True

    def add_many(
        self,
        factor_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """批量注册（长度须一致）。"""
        if len(factor_ids) != len(embeddings):
            raise ValueError("factor_ids must match embeddings length")
        for fid, emb in zip(factor_ids, embeddings):
            self.add(fid, emb)

    # -- 检索 --------------------------------------------------------------

    def search(
        self,
        query_embedding: Sequence[float],
        k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> list[Any]:
        """查近邻。注册内容自上次重建后变化 → 先整库重建（保持 backend 一致）。"""
        backend = self._ensure_backend()
        if self._factor_ids and self._dirty:
            self.rebuild()
        if not self._factor_ids:
            return []
        q = _as_1d(query_embedding)
        return list(backend.search(q, k=k, min_similarity=min_similarity))

    # -- 重建 --------------------------------------------------------------

    def rebuild(self) -> None:
        """显式重建：把全部已注册因子交给 backend 重新 build。"""
        backend = self._ensure_backend()
        if self._factor_ids:
            ids = list(self._factor_ids)
            mat = _as_2d([self._factor_ids[i] for i in ids], self.embedding_dim)
            backend.build(ids, mat)
        self._dirty = False
        self.index_rebuild_count += 1

    @property
    def num_factors(self) -> int:
        return len(self._factor_ids)

    @property
    def dirty(self) -> bool:
        return self._dirty


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class FactorAssetsAdapter:
    """包装 factor_assets 能力供 AlphaPROBE 消费。

    - ``identity_of``：走 FE 身份权威（authority.build_identity_view），返回
      canonical_repr / canonical_hash（FE canonical_ast_hash）/ fe_identity_ref
      （FE signal_equivalence_id）。FE 不可用 fail-closed 抛异常。``canonical_hash``
      由 ``canonical_ast_hash`` 承担（AST 级）。
    - ``cluster_assign``：把因子（formula/AST）交给 factor_assets 的 identity
      + 聚类（IncrementalPolicy / IncrementalAssignResult），返回
      cluster_id / family representative；cluster context 缺失且未显式允许
      singleton fallback → fail-closed 抛 :class:`ClusterContextUnavailable`。
    - ``nearest_neighbors``：用持久 ANN 近邻索引（FaissANNIndex / 注入后端）
      查近邻（相似度检索）；同版本 key 直接 search，不重复重建。
    - ``nearest_neighbors_local``：显式降级路径，用 AlphaPROBE ``seen/`` 的
      NearestIndex（只读）查近邻，并记录日志。

    读写分工见模块 docstring：本 adapter 只读 ``seen/``，绝不写入；cluster/
    family/ANN 权威在 factor_assets，identity 权威在 FE。
    """

    def __init__(
        self,
        *,
        embedding_spec: str = "simhash256",
        snapshot: str = "default",
        universe: str = "default",
        window: str = "default",
        cluster_set_version_ref: str = "csv_default",
        affinity_threshold: float = 0.5,
        ambiguity_gap: float = 0.05,
        max_candidates: int = 32,
        min_measure_floor: float = 0.25,
        allow_singleton_fallback: bool = False,
    ) -> None:
        if not _factor_assets_importable():
            raise FactorAssetsUnavailable(
                "factor_assets is not importable; cannot build FactorAssetsAdapter. "
                "Caller must decide whether to degrade to the local seen/ index."
            )
        self.embedding_spec = embedding_spec
        self.snapshot = snapshot
        self.universe = universe
        self.window = window
        self.cluster_set_version_ref = cluster_set_version_ref
        self.affinity_threshold = affinity_threshold
        self.ambiguity_gap = ambiguity_gap
        self.max_candidates = max_candidates
        self.min_measure_floor = min_measure_floor
        #: singleton fallback 是显式开关：OFFLINE_TEST / 测试场景才开。
        self.allow_singleton_fallback = allow_singleton_fallback
        #: 进程内持久 ANN 索引（版本化 key；可用 get_or_build_index 或直接注入）。
        self._sim_index: Optional[PersistentSimilarityIndex] = None

    # -- identity ----------------------------------------------------------

    def identity_of(self, formula: str) -> dict[str, Any]:
        """用 FE 身份权威取 identity（不重复实现 canonical hash）。

        经 ``alphaprobe.authority.build_identity_view`` 委托
        ``factor_engine.identity.get_factor_identity``：
        - canonical_repr ← FE canonical_formula；
        - canonical_hash ← FE canonical_ast_hash（AST 级，非文本 sha256）；
        - fe_identity_ref ← FE signal_equivalence_id（含全局取负等价）。

        FE 权威不可用（FactorIdentityAuthorityError）→ 抛
        :class:`FactorAssetsUnavailable`（fail-closed，不静默回落 sha256）。
        """
        from alphaprobe import authority
        from alphaprobe.authority import (
            FactorIdentityAuthorityError,
            build_identity_view,
        )

        if not formula:
            raise ValueError("formula is required")
        try:
            view = build_identity_view(formula)
        except FactorIdentityAuthorityError as exc:
            raise FactorAssetsUnavailable(
                "FE identity authority unavailable for identity_of; refusing "
                f"sha256 fallback (fail-closed): {exc}"
            ) from exc
        return {
            "canonical_repr": view.get("canonical_formula") or formula,
            "canonical_hash": view["canonical_ast_hash"],
            "fe_identity_ref": view["signal_equivalence_id"],
        }

    # -- cluster -----------------------------------------------------------

    def cluster_assign(
        self,
        formulas: Sequence[str],
        *,
        factor_ids: Optional[Sequence[str]] = None,
        cluster_versions: Optional[Mapping[str, Any]] = None,
        fingerprints_by_id: Optional[Mapping[str, Any]] = None,
        policy: Optional[Any] = None,
        batch_id: Optional[str] = None,
        allow_singleton_fallback: Optional[bool] = None,
    ) -> ClusterAssignResult:
        """把因子（formula）交给 factor_assets 的 identity + 聚类。

        Parameters
        ----------
        formulas : Sequence[str]
            因子公式列表（每个公式一个因子）。
        factor_ids : Sequence[str] | None
            因子 id 列表；缺省时用 ``identity_of`` 的 canonical_hash 前缀生成。
        cluster_versions : Mapping[str, Any] | None
            logical_cluster_id -> ClusterVersionArtifact（factor_assets 的
            生产 cluster 版本）。
        fingerprints_by_id : Mapping[str, Any] | None
            factor_id -> SimilarityFingerprintArtifact（覆盖既有成员 + 新因子）。
            缺省时用本地 simhash 指纹构造。
        policy : IncrementalPolicy | None
            聚类策略；缺省用本 adapter 的阈值构造。
        allow_singleton_fallback : bool | None
            单次调用覆盖构造参数。None → 用 adapter 的构造参数。cluster
            context 缺失（``cluster_versions`` 为空/缺省）且不允许 fallback →
            抛 :class:`ClusterContextUnavailable`（production fail-closed，
            不静默让每个因子自成一簇误导 Retriever）。

        Returns
        -------
        ClusterAssignResult
            含 assignments（factor_id / logical_cluster_id / kind / affinity /
            cluster_set_version_ref / cluster_context）、cluster_sizes 分布与
            顶层 ``cluster_context`` 标记（``"PROVIDED"`` | ``"SINGLETON_FALLBACK"``）。
        """
        from factor_assets.clustering.incremental import (
            IncrementalPolicy,
            incremental_assign,
        )
        from factor_assets.contracts.cluster_governance import (
            ClusterScale,
            ClusterVersionArtifact,
        )
        from factor_assets.contracts.fingerprint import (
            SimilarityFingerprintArtifact,
        )

        if not formulas:
            raise ValueError("formulas must be non-empty")
        # identity authority 先行：缺省 factor_ids 时这里就 fail-closed
        # （FE 不可用绝不静默 sha256）。
        if factor_ids is None:
            factor_ids = [self._default_factor_id(f) for f in formulas]
        if len(factor_ids) != len(formulas):
            raise ValueError("factor_ids must match formulas length")
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor_ids must be unique")

        # 构造 fingerprint（本地 simhash 确定性 embedding，与身份无关）
        if fingerprints_by_id is None:
            fingerprints_by_id = {
                fid: self._fingerprint(fid, formula)
                for fid, formula in zip(factor_ids, formulas)
            }

        # cluster context：缺省空 → 未显式允许 singleton fallback 即 fail-closed
        if cluster_versions is None:
            cluster_versions = {}
        if not cluster_versions:
            fallback = (
                self.allow_singleton_fallback
                if allow_singleton_fallback is None
                else bool(allow_singleton_fallback)
            )
            if not fallback:
                raise ClusterContextUnavailable(
                    "cluster context unavailable: cluster_versions is empty and "
                    "allow_singleton_fallback=False. Production must supply "
                    "cluster versions (or explicitly enable singleton fallback "
                    "for OFFLINE_TEST scenarios); refusing to mislead the "
                    "Retriever with all-SINGLETON assignments."
                )
            # 显式 singleton fallback：每个新因子自成一簇（SINGLETON），
            # 不触发 incremental_assign 的「cluster_versions 必须非空」约束。
            assignments = []
            for fid in factor_ids:
                assignments.append(
                    {
                        "factor_id": fid,
                        "logical_cluster_id": f"CL_{fid}",
                        "kind": "SINGLETON",
                        "cluster_set_version_ref": self.cluster_set_version_ref,
                        "affinity": None,
                        "cluster_context": "SINGLETON_FALLBACK",
                    }
                )
            cluster_sizes = {f"CL_{fid}": 1 for fid in factor_ids}
            representatives = {f"CL_{fid}": fid for fid in factor_ids}
            return ClusterAssignResult(
                batch_id=batch_id or "singleton",
                assignments=tuple(assignments),
                cluster_sizes=cluster_sizes,
                representative_by_cluster=representatives,
                cluster_context="SINGLETON_FALLBACK",
            )

        if policy is None:
            policy = IncrementalPolicy(
                affinity_threshold=self.affinity_threshold,
                ambiguity_gap=self.ambiguity_gap,
                max_candidates=self.max_candidates,
                min_measure_floor=self.min_measure_floor,
            )

        result = incremental_assign(
            [fingerprints_by_id[fid] for fid in factor_ids],
            cluster_versions,
            fingerprints_by_id,
            policy,
            batch_id=batch_id,
        )

        assignments = []
        cluster_sizes: dict[str, int] = {}
        representatives: dict[str, str] = {}
        for a in result.assignments:
            ad = a.to_dict()
            ad["cluster_context"] = "PROVIDED"
            assignments.append(ad)
            cid = a.logical_cluster_id
            if cid is not None:
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
        # family representative：取每个 cluster 的既有 representative（若在
        # cluster_versions 中），否则用该 cluster 首个成员。
        for cid, cv in cluster_versions.items():
            if cv.representative_factor_id:
                representatives[cid] = cv.representative_factor_id
            elif cv.member_factor_ids:
                representatives[cid] = cv.member_factor_ids[0]
        return ClusterAssignResult(
            batch_id=result.batch_id,
            assignments=tuple(assignments),
            cluster_sizes=cluster_sizes,
            representative_by_cluster=representatives,
            cluster_context="PROVIDED",
        )

    # -- ANN 近邻 ----------------------------------------------------------

    def get_or_build_index(
        self,
        factor_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        *,
        version_key: Mapping[str, Any] | Sequence[Any] | str,
        _backend: Optional[ANNBackend] = None,
    ) -> PersistentSimilarityIndex:
        """按版本化 key 取/建持久 ANN 索引。

        首次（或 version_key 变化）→ 构造并注册；key 相同 → 复用缓存索引，
        ``search`` 不再重复 build。调用方也可直接 ``adapter.persistent_index =
        idx`` 注入管理生命周期。

        ``_backend``：测试注入假 ANN backend；缺省走 FaissANNIndex（faiss
        不可用 fail-closed）。get_or_build_index 本身不吞缺省后端不可用——
        测试可在拿到索引后 ``idx._backend = fake`` 再 ``idx.rebuild()``。
        """
        from factor_assets.similarity.ann import ANNSearchResult

        keys = _coerce_version_key(version_key)
        # 复用判定：keys 首位是 embedding_dim 占位 0，而索引.version_key 首位是真实
        # embedding_dim —— 只比较后 4 位（fp/data/universe/cluster），否则每次
        # coerce 出的 keys 与缓存索引的 version_key 恒不等，缓存永不命中。
        if (
            self._sim_index is not None
            and tuple(self._sim_index.version_key[1:]) == tuple(keys[1:])
        ):
            # 复用：补充注册新因子（幂等 add_many），不重建已建好的 backend。
            self._sim_index.add_many(factor_ids, embeddings)
            if _backend is not None:
                self._sim_index._backend = _backend  # 测试注入
            return self._sim_index
        # 构造新索引（维度从首个 embedding 推断；无 embedding 时用已有索引维度）
        if embeddings:
            dim = len(list(embeddings[0]))
        elif self._sim_index is not None:
            dim = self._sim_index.embedding_dim
        else:
            raise ValueError("embeddings must be non-empty on first index build")
        idx = PersistentSimilarityIndex(
            embedding_dim=dim,
            fingerprint_version=keys[1],
            data_snapshot=keys[2],
            universe_snapshot=keys[3],
            cluster_version=keys[4],
            backend=_backend,  # 注入假 backend（测试用）
        )
        idx.add_many(factor_ids, embeddings)
        # 首次构建即显式 rebuild：注入 backend 则成功；缺省后端（faiss）不可用
        # 时 fail-closed 抛异常（不吞）。
        if idx.num_factors:
            try:
                idx.rebuild()
            except FactorAssetsUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - 统一 fail-closed
                raise FactorAssetsUnavailable(
                    f"ANN index build failed: {type(exc).__name__}: {exc}"
                ) from exc
        self._sim_index = idx
        return idx

    @property
    def persistent_index(self) -> Optional[PersistentSimilarityIndex]:
        """已缓存持久索引（无则 None）。"""
        return self._sim_index

    @persistent_index.setter
    def persistent_index(self, idx: Optional[PersistentSimilarityIndex]) -> None:
        """注入 / 替换 / 清空持久索引（生命周期由调用方管理）。"""
        self._sim_index = idx

    def nearest_neighbors(
        self,
        query_embedding: Sequence[float],
        k: int = 10,
        *,
        factor_ids: Optional[Sequence[str]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        min_similarity: Optional[float] = None,
        version_key: Optional[Mapping[str, Any] | Sequence[Any] | str] = None,
        _backend: Optional[ANNBackend] = None,
    ) -> list[NearestNeighborResult]:
        """用持久 ANN 近邻索引查近邻（相似度检索）。

        - 未传 ``factor_ids/embeddings``：直接用缓存索引（版本 key 一致不重建）。
        - 传入 ``factor_ids + embeddings``：走持久索引路径
          （``get_or_build_index``，同 version_key 复用缓存）。
        - 无缓存且无 factor_ids → 空索引返回空列表（零 faiss 依赖，纯查询路径
          不触碰 faiss）。
        - faiss / factor_assets 不可用且无注入后端 → 抛
          :class:`FactorAssetsUnavailable`（fail-closed）。

        ``version_key``：可传 dict / tuple / str，参与索引版本化 key
        （fingerprint_version / data_snapshot / universe_snapshot / cluster_version）。
        ``_backend``：测试注入假 ANN backend（避免依赖 faiss）。
        """
        if not query_embedding:
            raise ValueError("query_embedding must be non-empty")
        dim = len(list(query_embedding))

        if factor_ids is not None or embeddings is not None:
            if factor_ids is None or embeddings is None:
                raise ValueError("factor_ids and embeddings must be provided together")
            idx = self.get_or_build_index(
                factor_ids,
                embeddings,
                version_key=(
                    version_key
                    if version_key is not None
                    else (dim, self.embedding_spec, self.snapshot, self.universe, self.window)
                ),
                _backend=_backend,
            )
        elif self._sim_index is None:
            # 纯查询且无缓存：空库直接返回（零 faiss 依赖）。
            return []
        else:
            idx = self._sim_index

        if _backend is not None and idx._backend is None:
            idx._backend = _backend
        if idx._backend is None:
            # 惰性：需要搜才触碰 faiss/factor_assets（默认 backend）。
            try:
                idx._ensure_backend()
            except FactorAssetsUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                raise FactorAssetsUnavailable(
                    f"ANN backend unavailable: {type(exc).__name__}: {exc}"
                ) from exc
        results = idx.search(query_embedding, k=k, min_similarity=min_similarity)
        return [
            NearestNeighborResult(
                factor_id=r.factor_id,
                distance=float(r.distance),
                similarity_score=(
                    float(r.similarity_score) if r.similarity_score is not None else None
                ),
            )
            for r in results
        ]

    def nearest_neighbors_local(
        self,
        seen_store: Any,
        fingerprint: Optional[bytes],
        k: int = 20,
        *,
        fingerprint_version: str = "v1",
        max_hamming: Optional[int] = None,
    ) -> list[NearestNeighborResult]:
        """显式降级路径：用 AlphaPROBE ``seen/`` 的 NearestIndex（只读）查近邻。

        记录日志说明走的是本地降级（factor_assets/faiss 不可用时的显式降级，
        不是静默吞掉）。
        """
        from alphaprobe.seen.nearest import NearestIndex

        logger.info(
            "FactorAssetsAdapter.nearest_neighbors_local: degrading to AlphaPROBE "
            "seen/ NearestIndex (read-only) for nearest-neighbor search."
        )
        idx = NearestIndex(seen_store)
        results = idx.nearest(
            fingerprint,
            k=k,
            fingerprint_version=fingerprint_version,
            max_hamming=max_hamming,
        )
        return [
            NearestNeighborResult(
                factor_id=r.factor_id,
                distance=float(r.hamming),
                similarity_score=(
                    float(r.correlation_estimate)
                    if r.correlation_estimate is not None
                    else None
                ),
            )
            for r in results
        ]

    # -- helpers -----------------------------------------------------------

    def _default_factor_id(self, formula: str) -> str:
        ident = self.identity_of(formula)
        return f"F{ident['canonical_hash'][:16]}"

    def _fingerprint(self, factor_id: str, formula: str) -> Any:
        from factor_assets.contracts.fingerprint import (
            SimilarityFingerprintArtifact,
        )

        emb = _simhash_embedding(formula)
        return SimilarityFingerprintArtifact(
            factor_id=factor_id,
            embedding=tuple(emb),
            embedding_spec=self.embedding_spec,
            snapshot=self.snapshot,
            universe=self.universe,
            window=self.window,
        )


def build_factor_assets_adapter(**kwargs: Any) -> FactorAssetsAdapter:
    """构造 FactorAssetsAdapter；factor_assets 不可导入时抛 FactorAssetsUnavailable。"""
    return FactorAssetsAdapter(**kwargs)


# ---------------------------------------------------------------------------
# 全局因子上下文（plan Task 5）——FactorAssets 进入搜索闭环的消费面
# ---------------------------------------------------------------------------
# 模块级实现：GlobalFactorContext / GlobalFactorGate 都不 import factor_assets
# / factor_engine 具体类（只依赖注入的 registry / identity_provider），保证
# factor_assets 缺失 / FE 缺失的降级路径下本模块仍可导入。


@dataclass(frozen=True)
class GlobalFactorContext:
    """一个候选因子的全局搜索上下文（plan Task 5 提议的 dataclass）。

    字段（plan 原文 + 可推导口径）：
    - ``factor_id``：候选在 FactorAssets 全局库的 factor_id；
    - ``seen_status``：``"NEW"``（从未见）| ``"SEEN_EXACT"``（exact 重复）|
      ``"SEEN_SIGN"``（sign 等价重复）| ``"RESERVED"``（本轮已预留）；
    - ``nearest``：ANN 近邻（:class:`NearestNeighborResult` 元组）；
    - ``cluster_id`` / ``cluster_size`` / ``cluster_elite_rate``：候选所在
      cluster 的上下文；**无 cluster 证据 → 三者全 None**（#13：缺信息 = 中性，
      绝不当成最稀有的 singleton）；
    - ``family_id``：FE ``parameter_family_id``（无 → None）。

    ``cluster_rarity`` 与 :mod:`alphaprobe.retrieval.search_opportunity` 的
    ``cluster_rarity`` 同一口径（1/sqrt(1+size)；无证据 → 中性 0.5）。
    """

    factor_id: str
    seen_status: str = "NEW"
    nearest: tuple[NearestNeighborResult, ...] = ()
    cluster_id: str | None = None
    cluster_size: int | None = None
    cluster_elite_rate: float | None = None
    family_id: str | None = None

    @property
    def cluster_rarity(self) -> float:
        """cluster rarity（与 search_opportunity.cluster_rarity 同口径）。"""
        return cluster_rarity_of(self.cluster_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "seen_status": self.seen_status,
            "nearest": [n.to_dict() for n in self.nearest],
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "cluster_elite_rate": self.cluster_elite_rate,
            "family_id": self.family_id,
            "cluster_rarity": self.cluster_rarity,
        }


@dataclass(frozen=True)
class GateVerdict:
    """一次 GlobalFactorGate.check_candidate 的判定结果。

    - ``blocked=True`` → candidate 是 exact/sign duplicate（**在 FE 完整执行
      / QE 之前**停止）；``reason`` 为 ``"EXACT"`` / ``"SIGN"``；
    - ``blocked=False`` → 放行；``factor_id`` 为候选的全局 factor_id；
    - ``existing_factor_id``：重复时指向全局库已存在的因子（alias /
      rediscovery，不是新全局节点）；放行时为 ""。
    - ``context``：候选的 :class:`GlobalFactorContext`。
    """

    blocked: bool
    reason: str = ""
    factor_id: str = ""
    existing_factor_id: str = ""
    identity: Mapping[str, Any] = field(default_factory=dict)
    context: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "factor_id": self.factor_id,
            "existing_factor_id": self.existing_factor_id,
            "context": self.context.to_dict() if self.context is not None else None,
        }


def cluster_rarity_of(cluster_size: int | float | None) -> float:
    """cluster rarity = 1/sqrt(1+size)；无证据/非法 → 中性 0.5（#13）。"""
    if cluster_size is None:
        return 0.5
    try:
        size = float(cluster_size)
    except (TypeError, ValueError):
        return 0.5
    if not size > 0.0:
        return 0.5
    return 1.0 / (1.0 + size) ** 0.5


class GlobalFactorGate:
    """候选因子进搜索前查 FactorAssets 全局库的门卫（L0 gate 前置）。

    Task 5 行为：
    1. **先拒**：candidate identity（FE canonical_hash / signal_id）命中全局库
       seen → exact/sign duplicate，在 FE 完整执行 / QE 之前即停；
    2. **ANN 同版本不重建**：``context_for`` 走 adapter 持久索引，同版本 key
       复用缓存（``index_rebuild_count`` 不增，#16）；
    3. **cluster context**：``cluster_read`` 从注入的 cluster_size_fn 读
       cluster 上下文；无 cluster 证据 → None → 中性 0.5（#13）；
    4. **cold-start 只进全局库**：``register_many`` 只写入 registry，不产生
       DAG 节点；选中的 seed 才经 ``as_dag_roots`` 物化（#24）；
    5. **alias/rediscovery**：重复 register 返回 existing_factor_id，不建
       新全局节点。

    依赖全部可注入（``registry`` / ``identity_provider`` / ``cluster_size_fn``），
    factor_assets 缺失时由调用方决定降级级别（本类不 import factor_assets
    具体类）。零 LLM / 零模型 / 零网络。
    """

    def __init__(
        self,
        *,
        adapter: FactorAssetsAdapter,
        registry: Any,
        identity_provider: Callable[[str], Mapping[str, Any]] | None = None,
        cluster_size_fn: Callable[[str], int | float | None] | None = None,
        version_key: Mapping[str, Any] | Sequence[Any] | str | None = None,
        allow_reserve_default: bool = False,
    ) -> None:
        if adapter is None:
            raise ValueError("adapter is required")
        if registry is None:
            raise ValueError("registry is required")
        self.adapter = adapter
        self.registry = registry
        self.identity_provider = identity_provider
        self.cluster_size_fn = cluster_size_fn
        self.version_key = version_key
        self.allow_reserve_default = allow_reserve_default
        self.dag_node_created_count = 0
        self._candidate_serial = 0

    # -- identity（FE authority 经 adapter.identity_of；可注入覆盖） ----------

    def _identity_of(self, formula: str) -> dict[str, Any]:
        """返回 identity 视图（canonical_repr / canonical_hash / fe_identity_ref /
        parameter_family_id）。identity_provider 注入时优先（测试不依赖 FE）。"""
        if self.identity_provider is not None:
            view = dict(self.identity_provider(formula) or {})
            if not view.get("canonical_hash") or not view.get("fe_identity_ref"):
                raise FactorAssetsUnavailable(
                    "identity_provider returned incomplete identity for "
                    f"{formula!r} (fail-closed)"
                )
            return view
        return dict(self.adapter.identity_of(formula))

    # -- 门卫判定（先拒 / 放行） --------------------------------------------

    def check_candidate(
        self,
        formula: str,
        *,
        allow_reserve: bool | None = None,
        candidate_id: str | None = None,
    ) -> GateVerdict:
        """L0 gate：全局库 seen 命中即拒（不触发 FE 全执行/QE）。

        - 已注册（exact：canonical_hash 命中；sign：signal_id 命中）→
          ``blocked=True``，``reason`` = EXACT / SIGN，existing_factor_id 指向
          已有全局节点；
        - 未见 → 放行；``allow_reserve``（缺省用构造默认）时把候选注册进
          全局库（本轮预留，同 formula 下次判 EXACT）。
        """
        if not str(formula or "").strip():
            raise ValueError("formula is required")
        ident = self._identity_of(formula)
        chash = str(ident["canonical_hash"])
        iref = str(ident["fe_identity_ref"])
        existing = self._find_existing(chash, iref)
        reserve = self.allow_reserve_default if allow_reserve is None else bool(allow_reserve)
        if existing is not None:
            reason = "EXACT" if str(existing.get("canonical_hash") or "") == chash else "SIGN"
            ctx = GlobalFactorContext(
                factor_id=str(existing.get("factor_id") or ""),
                seen_status="SEEN_EXACT" if reason == "EXACT" else "SEEN_SIGN",
                family_id=_as_str(ident.get("parameter_family_id")),
            )
            return GateVerdict(
                blocked=True,
                reason=reason,
                existing_factor_id=str(existing.get("factor_id") or ""),
                identity=ident,
                context=ctx,
            )
        fid = candidate_id or self._new_factor_id(chash)
        if reserve:
            created, existing_id = self._register(fid, ident)
            if not created:
                # 竞态：另一 miner 已注册 → alias/rediscovery，不是新节点
                return GateVerdict(
                    blocked=True,
                    reason="SIGN" if existing_id else "EXACT",
                    factor_id=fid,
                    existing_factor_id=existing_id,
                    identity=ident,
                    context=GlobalFactorContext(
                        factor_id=existing_id or fid, seen_status="SEEN_EXACT"
                    ),
                )
            return GateVerdict(
                blocked=False,
                factor_id=fid,
                identity=ident,
                context=GlobalFactorContext(
                    factor_id=fid, seen_status="RESERVED",
                    family_id=_as_str(ident.get("parameter_family_id")),
                ),
            )
        return GateVerdict(
            blocked=False,
            factor_id=fid,
            identity=ident,
            context=GlobalFactorContext(
                factor_id=fid, seen_status="NEW",
                family_id=_as_str(ident.get("parameter_family_id")),
            ),
        )

    def reserve(self, formula: str, *, candidate_id: str | None = None) -> GateVerdict:
        """显式预留（放行后调用；NEW→RESERVED）。冲突返回 blocked。"""
        return self.check_candidate(formula, allow_reserve=True, candidate_id=candidate_id)

    # -- 全局上下文（ANN + cluster + family） --------------------------------

    def context_for(
        self,
        formula: str,
        *,
        candidate_id: str | None = None,
        universe: Sequence[str] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        k: int = 10,
        _backend: Any = None,
    ) -> GlobalFactorContext:
        """构造候选的完整全局上下文（seen + ANN nearest + cluster + family）。

        ANN 走 adapter 持久索引：同版本 key 复用缓存（不重建）。universe 为空
        或缺省 → nearest 空元组（零 faiss 依赖的纯查询路径不触碰 faiss）。
        """
        ident = self._identity_of(formula)
        chash = str(ident["canonical_hash"])
        iref = str(ident["fe_identity_ref"])
        existing = self._find_existing(chash, iref)
        seen_status = (
            "SEEN_EXACT" if existing is not None else "NEW"
        )
        if existing is not None:
            factor_id = str(existing.get("factor_id") or "")
            nearest = self._nearest(formula, universe, embeddings, k=k, _backend=_backend)
            cluster_id, csize, elite = self._cluster_of(factor_id)
            return GlobalFactorContext(
                factor_id=factor_id,
                seen_status=seen_status,
                nearest=nearest,
                cluster_id=cluster_id,
                cluster_size=csize,
                cluster_elite_rate=elite,
                family_id=_as_str(existing.get("parameter_family_id")),
            )
        fid = candidate_id or self._new_factor_id(chash)
        nearest = self._nearest(formula, universe, embeddings, k=k, _backend=_backend)
        cluster_id, csize, elite = self._cluster_of(fid)
        return GlobalFactorContext(
            factor_id=fid,
            seen_status=seen_status,
            nearest=nearest,
            cluster_id=cluster_id,
            cluster_size=csize,
            cluster_elite_rate=elite,
            family_id=_as_str(ident.get("parameter_family_id")),
        )

    def _nearest(
        self,
        formula: str,
        universe: Sequence[str] | None,
        embeddings: Sequence[Sequence[float]] | None,
        *,
        k: int,
        _backend: Any,
    ) -> tuple[NearestNeighborResult, ...]:
        """ANN nearest（持久索引同版本不重建；空 universe 不触碰 faiss）。"""
        if not universe and not embeddings:
            return ()
        if embeddings is None:
            # 无显式 embedding 时用 adapter 的本地 simhash embedding（确定性）
            emb = _simhash_embedding(formula)
            embs = [_simhash_embedding(u) for u in universe]
            ids = [u if u.startswith(("F", "fa_", "seed_")) else f"fa_{u}" for u in universe]
        else:
            embs = list(embeddings)
            ids = [u if u.startswith(("F", "fa_", "seed_")) else f"fa_{u}" for u in universe] if universe else [
                f"fa_{i}" for i in range(len(embs))
            ]
            emb = list(embs[0]) if embs else [0.0] * len(list(embeddings[0]) if embeddings else [])
        vk = self.version_key if self.version_key is not None else {
            "fingerprint_version": self.adapter.embedding_spec,
            "data_snapshot": self.adapter.snapshot,
            "universe_snapshot": self.adapter.universe,
            "cluster_version": self.adapter.window,
        }
        try:
            results = self.adapter.nearest_neighbors(
                emb, k=k, factor_ids=ids, embeddings=embs,
                version_key=vk, _backend=_backend,
            )
        except FactorAssetsUnavailable:
            # 显式降级：faiss / factor_assets 不可用 → 返回空近邻（不静默吞，
            # 记日志说明走了降级路径）。
            logger.info(
                "GlobalFactorGate._nearest: ANN unavailable, returning empty "
                "nearest (explicit degrade): factor=%r", formula,
            )
            return ()
        return tuple(results)

    def _cluster_of(
        self, factor_id: str,
    ) -> tuple[str | None, int | None, float | None]:
        """读 cluster 上下文：cluster_size_fn 注入时用；否则无证据（#13）。

        无 cluster 证据 → cluster_id/size/elite_rate 全 None（中性，绝不把
        「不知道」当成 singleton 误导 Retriever）。
        """
        if self.cluster_size_fn is None:
            return None, None, None
        try:
            size = self.cluster_size_fn(factor_id)
        except Exception:  # noqa: BLE001 - cluster_fn 异常按无证据中性
            return None, None, None
        if size is None:
            return None, None, None
        cid = getattr(self.cluster_size_fn, "cluster_id_for", None)
        cid_val = cid(factor_id) if callable(cid) else None
        return (
            _as_str(cid_val) or f"CL_{factor_id}",
            int(size) if size > 0 else None,
            None,  # elite_rate 由调用方 cluster 表提供；无则 None
        )

    # -- 批量注册（cold-start 只进全局库，#24） ------------------------------

    def register_many(
        self,
        metas: Sequence[Mapping[str, Any]],
    ) -> int:
        """把一批 metadata（cold-start 库 / 算法历史）注册进全局库。

        只写 registry，不产生 DAG 节点（``dag_node_created_count`` 不增）。
        返回实际新建数（重复 register 会计入 alias 但不新建）。
        """
        created = 0
        for m in metas:
            chash = str(m.get("canonical_hash") or m.get("identity_ref") or "")
            iref = str(m.get("fe_identity_ref") or m.get("signal_equivalence_id") or chash or "")
            if not chash or not iref:
                continue
            fid = str(m.get("factor_id") or "")
            ok, _ = self._register(
                fid,
                {
                    "canonical_repr": str(m.get("canonical_repr") or m.get("formula") or ""),
                    "canonical_hash": chash,
                    "fe_identity_ref": iref,
                    "parameter_family_id": m.get("parameter_family_id"),
                },
            )
            if ok:
                created += 1
        return created

    def as_dag_roots(self, factor_ids: Sequence[str]) -> list[dict[str, Any]]:
        """把**选中的** seed 物化为 DAG root（parents 空 + lineage_root）。

        只物化传入的 factor_ids（已从全局库选中的 seed）；未选中的永远留在
        全局库。#24：10 万 seed 不整体塞进 DAG。
        """
        roots: list[dict[str, Any]] = []
        for fid in factor_ids:
            meta = self._registry_get(fid)
            if meta is None:
                continue
            self.dag_node_created_count += 1
            roots.append(
                {
                    "factor_id": fid,
                    "formula": str(meta.get("canonical_repr") or meta.get("formula") or ""),
                    "parents": (),
                    "lineage_root": True,
                    "identity_ref": str(meta.get("fe_identity_ref") or ""),
                    "canonical_hash": str(meta.get("canonical_hash") or ""),
                }
            )
        return roots

    # -- registry helpers（可注入 duck-typing） -------------------------------

    def _find_existing(
        self, canonical_hash: str, fe_identity_ref: str,
    ) -> dict[str, Any] | None:
        """按 canonical_hash 优先、fe_identity_ref 次之查全局库。"""
        reg = self.registry
        for method, arg in (
            ("find_by_hash", canonical_hash),
            ("find_by_identity", fe_identity_ref),
        ):
            fn = getattr(reg, method, None)
            if fn is None:
                continue
            try:
                out = fn(arg)
            except Exception:  # noqa: BLE001 - 查询失败按未命中
                out = None
            if out is not None:
                if isinstance(out, Mapping):
                    return dict(out)
                # FactorAsset 对象 → 转 dict
                meta = getattr(out, "metadata", None) or out
                return {
                    "factor_id": str(
                        getattr(out, "factor_id", "") or getattr(meta, "factor_id", "") or ""
                    ),
                    "canonical_hash": str(
                        getattr(out, "canonical_hash", "")
                        or getattr(meta, "canonical_hash", "") or ""
                    ),
                    "canonical_repr": str(
                        getattr(out, "canonical_repr", "")
                        or getattr(meta, "canonical_repr", "") or ""
                    ),
                    "fe_identity_ref": str(
                        getattr(out, "fe_identity_ref", "")
                        or getattr(meta, "fe_identity_ref", "") or ""
                    ),
                    "parameter_family_id": getattr(meta, "parameter_family_id", None),
                }
        # 兼容：exists_by_hash / exists_by_identity + get 组合
        if hasattr(reg, "exists_by_hash") and reg.exists_by_hash(canonical_hash):
            return self._registry_get_by_hash(canonical_hash)
        if hasattr(reg, "exists_by_identity") and reg.exists_by_identity(fe_identity_ref):
            return self._registry_get_by_identity(fe_identity_ref)
        return None

    def _register(
        self, factor_id: str, ident: Mapping[str, Any],
    ) -> tuple[bool, str]:
        """注册到全局库；返回 (created, existing_factor_id)。

        先探测存在性（find 接口或 exists 接口），存在 → (False, existing)；
        否则调 register/create。create 返回 (created, existing) 语义时用之。
        """
        chash = str(ident.get("canonical_hash") or "")
        iref = str(ident.get("fe_identity_ref") or "")
        existing = self._find_existing(chash, iref)
        if existing is not None:
            return False, str(existing.get("factor_id") or "")
        reg = self.registry
        create = getattr(reg, "register", None)
        if create is not None:
            try:
                out = create(
                    factor_id=factor_id,
                    canonical_hash=chash,
                    canonical_repr=str(ident.get("canonical_repr") or ""),
                    fe_identity_ref=iref,
                    parameter_family_id=ident.get("parameter_family_id"),
                )
            except TypeError:
                try:
                    out = create(factor_id, ident)
                except Exception:  # noqa: BLE001
                    out = None
            if isinstance(out, tuple):
                created, existing_id = out[0], (out[1] if len(out) > 1 else "")
                return bool(created), str(existing_id or "")
            return True, ""
        # 兜底接口：exists_by_hash / exists_by_identity + insert
        if hasattr(reg, "insert"):
            try:
                reg.insert(
                    factor_id=factor_id, canonical_hash=chash,
                    canonical_repr=str(ident.get("canonical_repr") or ""),
                    fe_identity_ref=iref,
                    parameter_family_id=ident.get("parameter_family_id"),
                )
                return True, ""
            except Exception:  # noqa: BLE001 - 已存在 → alias
                existing = self._find_existing(chash, iref)
                return False, str((existing or {}).get("factor_id") or "")
        return True, ""

    def _registry_get(self, factor_id: str) -> dict[str, Any] | None:
        reg = self.registry
        fn = getattr(reg, "get", None)
        if fn is None:
            return None
        try:
            out = fn(factor_id)
        except Exception:  # noqa: BLE001
            return None
        if out is None:
            return None
        if isinstance(out, Mapping):
            return dict(out)
        meta = getattr(out, "metadata", None) or out
        return {
            "factor_id": str(getattr(out, "factor_id", "") or getattr(meta, "factor_id", "") or ""),
            "canonical_hash": str(getattr(out, "canonical_hash", "") or getattr(meta, "canonical_hash", "") or ""),
            "canonical_repr": str(getattr(out, "canonical_repr", "") or getattr(meta, "canonical_repr", "") or ""),
            "fe_identity_ref": str(getattr(out, "fe_identity_ref", "") or getattr(meta, "fe_identity_ref", "") or ""),
            "parameter_family_id": getattr(meta, "parameter_family_id", None),
        }

    def _registry_get_by_hash(self, canonical_hash: str) -> dict[str, Any] | None:
        reg = self.registry
        fn = getattr(reg, "find_by_hash", None)
        if fn is None:
            return None
        try:
            return self._coerce_found(fn(canonical_hash))
        except Exception:  # noqa: BLE001
            return None

    def _registry_get_by_identity(self, fe_identity_ref: str) -> dict[str, Any] | None:
        reg = self.registry
        fn = getattr(reg, "find_by_identity", None)
        if fn is None:
            return None
        try:
            return self._coerce_found(fn(fe_identity_ref))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _coerce_found(out: Any) -> dict[str, Any] | None:
        if out is None:
            return None
        if isinstance(out, Mapping):
            return dict(out)
        meta = getattr(out, "metadata", None) or out
        return {
            "factor_id": str(getattr(out, "factor_id", "") or getattr(meta, "factor_id", "") or ""),
            "canonical_hash": str(getattr(out, "canonical_hash", "") or getattr(meta, "canonical_hash", "") or ""),
            "canonical_repr": str(getattr(out, "canonical_repr", "") or getattr(meta, "canonical_repr", "") or ""),
            "fe_identity_ref": str(getattr(out, "fe_identity_ref", "") or getattr(meta, "fe_identity_ref", "") or ""),
            "parameter_family_id": getattr(meta, "parameter_family_id", None),
        }

    def _new_factor_id(self, canonical_hash: str) -> str:
        self._candidate_serial += 1
        return f"fa_{canonical_hash[:16]}_{self._candidate_serial}"


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# ---------------------------------------------------------------------------
# 版本化 key 归一化 / 本地确定性 embedding（simhash 256-bit → 256 维 ±1 向量）
# ---------------------------------------------------------------------------


def _coerce_version_key(key: Mapping[str, Any] | Sequence[Any] | str) -> tuple[Any, ...]:
    """把调用方 version_key 归一化为 5 元组（embedding_dim 占位，由索引持有）。

    - dict → 按字段名取（fingerprint_version / data_snapshot / universe_snapshot /
      cluster_version）；缺省字段回落 "default"。
    - tuple/list → 补足 5 元；长度 >5 截断（容忍额外前缀维度）。
    - str → (str, "default", "default", "default", "default")。
    """
    if isinstance(key, str):
        return (0, key, "default", "default", "default")
    if isinstance(key, Mapping):
        return (
            0,
            str(key.get("fingerprint_version") or "default"),
            str(key.get("data_snapshot") or "default"),
            str(key.get("universe_snapshot") or "default"),
            str(key.get("cluster_version") or "default"),
        )
    seq = tuple(key)
    if len(seq) == 1:
        return (0, str(seq[0]), "default", "default", "default")
    # 容忍：(dim 占位 0, fp, data, universe, cluster)[:5]
    padded = tuple(str(v) for v in seq)
    if len(padded) >= 5:
        return (0,) + padded[:4]
    return (0,) + padded + ("default",) * (4 - len(padded))


def _simhash_embedding(text: str, bits: int = 256) -> list[float]:
    import hashlib

    v = [0.0] * bits
    tokens = text.replace("(", " ").replace(")", " ").replace(",", " ").split()
    for tok in tokens:
        h = hashlib.sha256(tok.encode()).digest()
        for i, b in enumerate(h):
            for k in range(8):
                if (b >> k) & 1:
                    v[(i * 8 + k) % bits] += 1.0
                else:
                    v[(i * 8 + k) % bits] -= 1.0
    return v


def _as_1d(x: Sequence[float]) -> Any:
    import numpy as np

    return np.asarray(list(x), dtype=np.float64)


def _as_2d(x: Sequence[Sequence[float]], dim: int) -> Any:
    import numpy as np

    arr = np.asarray([list(row) for row in x], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"embeddings must be 2D with dim={dim}")
    return arr
