"""factor_assets adapter（任务书 §33 / §42-§43）。

把 factor_assets 的 identity + 聚类 + ANN 能力包装给 AlphaPROBE 消费，同时
明确与 AlphaPROBE 自带 ``seen/`` 索引的读写分工，避免双写冲突。

读写分工（docstring 权威，代码不重复实现）：
----------------------------------------------------------------------------
| 能力            | 写入方（authority）        | 读取方（consumer）          |
|-----------------|----------------------------|-----------------------------|
| 去重 / seen     | AlphaPROBE ``seen/``       | 本 adapter 只读（不写）      |
|                 | (DedupService / SeenStore) |                              |
| identity hash   | AlphaPROBE ``dedup_client``| 本 adapter 只读（不写）      |
|                 | (三级降级链)               |                              |
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

降级链（fail-closed，绝不静默吞掉）：
- factor_assets 不可导入 → 抛 :class:`FactorAssetsUnavailable`（显式异常，
  调用方决定是否降级到本地 seen/ 索引）。本 adapter 不自动静默降级。
- faiss 不可用 → ``nearest_neighbors`` 抛 :class:`FactorAssetsUnavailable`
  （FaissANNIndex 构造本身会抛 ImportError，这里包成统一异常）。
- 显式降级入口：``nearest_neighbors_local`` 用 AlphaPROBE ``seen/`` 的
  NearestIndex（只读）作为显式降级路径，并记录日志。

零 LLM、零模型、零网络。合成数据可测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "FactorAssetsUnavailable",
    "ClusterAssignResult",
    "NearestNeighborResult",
    "FactorAssetsAdapter",
    "build_factor_assets_adapter",
]


class FactorAssetsUnavailable(RuntimeError):
    """factor_assets 不可导入或能力缺失（fail-closed，不静默降级）。"""


# ---------------------------------------------------------------------------
# 结果类型（不依赖 factor_assets 具体类，便于 AlphaPROBE 消费与测试）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterAssignResult:
    """一次 cluster_assign 的返回结构。

    ``assignments`` 每项含 factor_id / logical_cluster_id / kind / affinity /
    cluster_set_version_ref；``cluster_sizes`` 为 logical_cluster_id -> 成员数
    （供 retrieval/search_opportunity 消费 cluster size 分布）。
    """

    batch_id: str
    assignments: tuple[dict[str, Any], ...]
    cluster_sizes: dict[str, int]
    representative_by_cluster: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "assignments": list(self.assignments),
            "cluster_sizes": dict(self.cluster_sizes),
            "representative_by_cluster": dict(self.representative_by_cluster),
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
# Adapter
# ---------------------------------------------------------------------------


class FactorAssetsAdapter:
    """包装 factor_assets 能力供 AlphaPROBE 消费。

    - ``cluster_assign``：把因子（formula/AST）交给 factor_assets 的 identity
      + 聚类（IncrementalPolicy / IncrementalAssignResult），返回
      cluster_id / family representative。
    - ``nearest_neighbors``：用 FaissANNIndex 查近邻（相似度检索）。
    - ``nearest_neighbors_local``：显式降级路径，用 AlphaPROBE ``seen/`` 的
      NearestIndex（只读）查近邻，并记录日志。

    读写分工见模块 docstring：本 adapter 只读 ``seen/``，绝不写入；cluster/
    family/ANN 权威在 factor_assets。
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

    # -- identity ----------------------------------------------------------

    def identity_of(self, formula: str) -> dict[str, Any]:
        """用 factor_assets 的 FactorIdentityProvider 语义取 identity。

        本 adapter 不重复实现 canonical hash（factor_assets 也不实现，它通过
        FE adapter 协议获取）。这里提供确定性本地降级：sha256(formula) 作为
        canonical_hash，供 cluster_assign 的 fingerprint 使用。
        """
        import hashlib

        if not formula:
            raise ValueError("formula is required")
        canonical_hash = hashlib.sha256(formula.encode("utf-8")).hexdigest()
        return {
            "canonical_repr": formula,
            "canonical_hash": canonical_hash,
            "fe_identity_ref": None,
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
            生产 cluster 版本）。缺省时构造一个空 singleton 版本集（每个新
            因子自成一簇，kind=SINGLETON）。
        fingerprints_by_id : Mapping[str, Any] | None
            factor_id -> SimilarityFingerprintArtifact（覆盖既有成员 + 新因子）。
            缺省时用本地 simhash 指纹构造。
        policy : IncrementalPolicy | None
            聚类策略；缺省用本 adapter 的阈值构造。

        Returns
        -------
        ClusterAssignResult
            含 assignments（factor_id / logical_cluster_id / kind / affinity /
            cluster_set_version_ref）与 cluster_sizes 分布。
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
        if factor_ids is None:
            factor_ids = [self._default_factor_id(f) for f in formulas]
        if len(factor_ids) != len(formulas):
            raise ValueError("factor_ids must match formulas length")
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor_ids must be unique")

        # 构造 fingerprint（本地 simhash 确定性 embedding）
        if fingerprints_by_id is None:
            fingerprints_by_id = {
                fid: self._fingerprint(fid, formula)
                for fid, formula in zip(factor_ids, formulas)
            }

        # 构造 cluster_versions（缺省：空 singleton 版本集）
        if cluster_versions is None:
            cluster_versions = {}
        if not cluster_versions:
            # 空版本集：每个新因子自成一簇（SINGLETON），不触发 incremental_assign
            # 的「cluster_versions 必须非空」约束——这里直接返回 singleton 结果。
            assignments = []
            for fid in factor_ids:
                assignments.append(
                    {
                        "factor_id": fid,
                        "logical_cluster_id": f"CL_{fid}",
                        "kind": "SINGLETON",
                        "cluster_set_version_ref": self.cluster_set_version_ref,
                        "affinity": None,
                    }
                )
            cluster_sizes = {f"CL_{fid}": 1 for fid in factor_ids}
            representatives = {f"CL_{fid}": fid for fid in factor_ids}
            return ClusterAssignResult(
                batch_id=batch_id or "singleton",
                assignments=tuple(assignments),
                cluster_sizes=cluster_sizes,
                representative_by_cluster=representatives,
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
            assignments.append(a.to_dict())
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
        )

    # -- ANN 近邻 ----------------------------------------------------------

    def nearest_neighbors(
        self,
        query_embedding: Sequence[float],
        k: int = 10,
        *,
        factor_ids: Optional[Sequence[str]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        min_similarity: Optional[float] = None,
    ) -> list[NearestNeighborResult]:
        """用 FaissANNIndex 查近邻（相似度检索）。

        factor_ids / embeddings 提供索引库（既有成员）。缺省时用空库（返回空）。
        faiss 不可用 → 抛 :class:`FactorAssetsUnavailable`（fail-closed）。
        """
        from factor_assets.similarity.ann import FaissANNIndex

        if not _faiss_importable():
            raise FactorAssetsUnavailable(
                "faiss is not importable; FaissANNIndex unavailable. "
                "Use nearest_neighbors_local() for the explicit seen/ degrade path."
            )
        if not query_embedding:
            raise ValueError("query_embedding must be non-empty")
        dim = len(query_embedding)
        idx = FaissANNIndex(embedding_dim=dim)
        if factor_ids and embeddings:
            idx.build(list(factor_ids), _as_2d(embeddings, dim))
        results = idx.search(
            _as_1d(query_embedding), k=k, min_similarity=min_similarity
        )
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
# 本地确定性 embedding（simhash 256-bit → 256 维 ±1 向量）
# ---------------------------------------------------------------------------


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
