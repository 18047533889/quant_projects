"""V3 P0-D factor_assets_adapter 修复测试（P0_PROGRESS_LOG P0-D）。

覆盖三项修复：
- identity_of 删 sha256(raw formula) → 走 FE authority（canonical_hash ==
  canonical_ast_hash，fe_identity_ref == signal_equivalence_id）；FE 不可用
  fail-closed 抛 FactorAssetsUnavailable；``_default_factor_id`` 链路依赖仍在。
- ANN 持久索引：PersistentSimilarityIndex 同 version_key 二次 search 不重建；
  add 增量 / add_many；版本变化自动失效重建（index_rebuild_count 计数）。
- cluster fallback 不全 singleton：缺省空版本抛 ClusterContextUnavailable；
  allow_singleton_fallback=True → SINGLETON + SINGLETON_FALLBACK 标记。

全合成数据、假 backend 注入（venv 无 torch/faiss），零 LLM / 零模型 / 零网络。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from alphaprobe.factor_assets_adapter import (
    ClusterContextUnavailable,
    ClusterAssignResult,
    FactorAssetsAdapter,
    FactorAssetsUnavailable,
    NearestNeighborResult,
    PersistentSimilarityIndex,
    build_factor_assets_adapter,
)

# ---------------------------------------------------------------------------
# factor_assets 可用性（照抄 test_factor_assets_adapter 的 sys.path 引导模式）
# ---------------------------------------------------------------------------


def _fa_importable() -> bool:
    try:
        import factor_assets  # noqa: F401

        return True
    except Exception:
        return False


def _ensure_fa_importable() -> bool:
    if _fa_importable():
        return True
    for parent in Path(__file__).resolve().parents:
        if (parent / "factor_assets").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return _fa_importable()
    return False


FA_AVAILABLE = _ensure_fa_importable()
FAISS_SKIP = pytest.mark.skipif(not FA_AVAILABLE, reason="factor_assets not importable")


# ---------------------------------------------------------------------------
# 假 ANN backend（无 faiss；记录 build 次数 + 保序搜索）
# ---------------------------------------------------------------------------


class FakeANNBackend:
    """最小 ANNBackend 实现：记录 build/search 调用，搜索用余弦近似。"""

    def __init__(self) -> None:
        self.build_count = 0
        self.search_count = 0
        self.factor_ids: list[str] = []
        self.mat: np.ndarray | None = None

    def build(self, factor_ids: Sequence[str], embeddings: Any) -> None:
        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim != 2 or len(factor_ids) != arr.shape[0]:
            raise ValueError("fake backend: bad build inputs")
        self.factor_ids = list(factor_ids)
        self.mat = arr
        self.build_count += 1

    def search(
        self,
        query_embedding: Any,
        k: int = 10,
        min_similarity: float | None = None,
    ) -> list[object]:
        self.search_count += 1
        q = np.asarray(query_embedding, dtype=np.float64)
        if self.mat is None or len(self.factor_ids) == 0:
            return []
        qn = q / (np.linalg.norm(q) + 1e-12)
        rows = self.mat / (
            np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12
        )
        sims = rows @ qn
        order = np.argsort(-sims)
        out = []
        for i in order:
            s = float(sims[i])
            if min_similarity is not None and s < min_similarity:
                continue
            out.append(
                _FakeResult(
                    factor_id=self.factor_ids[i],
                    distance=float(1.0 - s),
                    similarity_score=s,
                )
            )
            if len(out) >= k:
                break
        return out


class _FakeResult:
    def __init__(self, factor_id: str, distance: float, similarity_score: float):
        self.factor_id = factor_id
        self.distance = distance
        self.similarity_score = similarity_score


# ---------------------------------------------------------------------------
# 1. identity_of → FE authority（不再 sha256）
# ---------------------------------------------------------------------------


class TestIdentityOfFeAuthority:
    def test_delegates_to_fe_authority(self, monkeypatch):
        """monkeypatch build_identity_view → canonical_hash==FE ast hash，
        fe_identity_ref==signal_id，确定性保持。"""
        import alphaprobe.authority as authority_mod

        calls: list[str] = []

        def _fake_view(formula: str):
            calls.append(formula)
            return {
                "canonical_formula": f"C({formula})",
                "canonical_ast_hash": "a" * 64,
                "signal_equivalence_id": "b" * 64,
            }

        monkeypatch.setattr(authority_mod, "build_identity_view", _fake_view)
        ad = FactorAssetsAdapter()
        a = ad.identity_of("rank(close)")
        b = ad.identity_of("rank(close)")
        assert calls == ["rank(close)", "rank(close)"]
        assert a["canonical_hash"] == b["canonical_hash"] == "a" * 64
        assert a["fe_identity_ref"] == b["fe_identity_ref"] == "b" * 64
        assert a["canonical_repr"] == "C(rank(close))"
        # 三个兼容 key 全保留
        assert {"canonical_repr", "canonical_hash", "fe_identity_ref"} <= set(a)

    def test_no_sha256_fallback(self, monkeypatch):
        """FE 抛 FactorIdentityAuthorityError → FactorAssetsUnavailable，
        不静默回落 sha256。"""
        import alphaprobe.authority as authority_mod
        from alphaprobe.authority import FactorIdentityAuthorityError

        def _boom(formula: str):
            raise FactorIdentityAuthorityError("FE down")

        monkeypatch.setattr(authority_mod, "build_identity_view", _boom)
        ad = FactorAssetsAdapter()
        with pytest.raises(FactorAssetsUnavailable) as ei:
            ad.identity_of("rank(close)")
        assert "sha256" in str(ei.value) or "fail-closed" in str(ei.value)

    def test_empty_raises_value_error(self):
        ad = FactorAssetsAdapter()
        with pytest.raises(ValueError):
            ad.identity_of("")

    def test_default_factor_id_uses_fe_hash(self, monkeypatch):
        """_default_factor_id → identity_of（canonical_hash 前缀）链路仍工作。"""
        import alphaprobe.authority as authority_mod

        monkeypatch.setattr(
            authority_mod,
            "build_identity_view",
            lambda f: {
                "canonical_formula": f,
                "canonical_ast_hash": "f" * 64,
                "signal_equivalence_id": "s" * 64,
            },
        )
        ad = FactorAssetsAdapter()
        fid = ad._default_factor_id("rank(close)")
        assert fid.startswith("F")
        assert fid == "F" + "f" * 16


# ---------------------------------------------------------------------------
# 2. PersistentSimilarityIndex（假 backend；版本化 key 自动重建）
# ---------------------------------------------------------------------------


class TestPersistentIndex:
    def test_add_and_search_same_version_no_rebuild(self):
        """同 version_key 二次 search 不触发 rebuild（index_rebuild_count 不增）。"""
        fake = FakeANNBackend()
        idx = PersistentSimilarityIndex(
            embedding_dim=4,
            fingerprint_version="v1",
            backend=fake,
        )
        idx.add_many(["a", "b"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # 首次 search 前自动 rebuild（dirty 注册内容）
        r1 = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert fake.build_count == 1
        assert [x.factor_id for x in r1] == ["a", "b"]
        # 幂等 add 同 id 同 embedding → 不 dirty、不重建
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        r2 = idx.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert fake.build_count == 1
        assert r2[0].factor_id == "a"
        assert not idx.dirty
        assert idx.index_rebuild_count == 1

    def test_add_incremental_triggers_rebuild(self):
        """新增因子 → dirty → 下次 search 前整库重建（build_count 增加）。"""
        fake = FakeANNBackend()
        idx = PersistentSimilarityIndex(embedding_dim=4, fingerprint_version="v1", backend=fake)
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert fake.build_count == 1
        idx.add("b", [0.0, 1.0, 0.0, 0.0])  # 增量
        assert idx.dirty
        out = idx.search([0.0, 1.0, 0.0, 0.0], k=1)
        assert fake.build_count == 2
        assert out[0].factor_id == "b"
        assert idx.index_rebuild_count == 2

    def test_version_key_change_auto_rebuild(self):
        """版本 key 变化 → 新索引实例（自动失效），不沿用旧 backend。"""
        idx1 = PersistentSimilarityIndex(embedding_dim=4, fingerprint_version="v1")
        idx2 = PersistentSimilarityIndex(embedding_dim=4, fingerprint_version="v2")
        assert idx1.version_key != idx2.version_key
        # 不同构造参数 → 不同 version key
        idx3 = PersistentSimilarityIndex(
            embedding_dim=4,
            fingerprint_version="v1",
            cluster_version="c2",
        )
        assert idx1.version_key != idx3.version_key

    def test_version_change_in_adapter_rebuilds(self):
        """adapter.get_or_build_index：version_key 变化 → 重建；相同 → 复用缓存。"""
        fake = FakeANNBackend()
        ad = FactorAssetsAdapter()
        idx_a = ad.get_or_build_index(
            ["a", "b"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            version_key={
                "fingerprint_version": "v1",
                "data_snapshot": "d1",
                "universe_snapshot": "u1",
                "cluster_version": "c1",
            },
            _backend=fake,
        )
        assert idx_a._backend is fake
        assert idx_a.index_rebuild_count == 1
        assert ad.persistent_index is idx_a
        # 同 key 再取 → 同一实例（不重建）
        idx_b = ad.get_or_build_index(
            ["c"],
            [[1.0, 0.0, 0.0, 0.0]],
            version_key={
                "fingerprint_version": "v1",
                "data_snapshot": "d1",
                "universe_snapshot": "u1",
                "cluster_version": "c1",
            },
            _backend=fake,
        )
        assert idx_b is idx_a
        # version_key 变化 → 新实例（重建路径）
        idx_c = ad.get_or_build_index(
            ["d"],
            [[1.0, 0.0, 0.0, 0.0]],
            version_key={
                "fingerprint_version": "v2",
                "data_snapshot": "d1",
                "universe_snapshot": "u1",
                "cluster_version": "c1",
            },
            _backend=fake,
        )
        assert idx_c is not idx_a
        assert ad.persistent_index is idx_c

    def test_nearest_neighbors_reuses_persistent_index(self):
        """nearest_neighbors 同 version_key：首次建索引，二次不再重建（假 backend）。"""
        fake = FakeANNBackend()
        ad = FactorAssetsAdapter()
        vk = {
            "fingerprint_version": "simhash256",
            "data_snapshot": "default",
            "universe_snapshot": "default",
            "cluster_version": "default",
        }
        r1 = ad.nearest_neighbors(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            factor_ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            version_key=vk,
            _backend=fake,
        )
        assert fake.build_count == 1
        assert len(r1) == 2
        idx = ad.persistent_index
        assert idx is not None
        assert idx._backend is fake
        # 二次调用（不同 embedding 也复用缓存，不重建）
        r2 = ad.nearest_neighbors(
            [0.0, 1.0, 0.0, 0.0],
            k=1,
            factor_ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            version_key=vk,
            _backend=fake,
        )
        assert ad.persistent_index is idx
        assert fake.build_count == 1
        assert r2[0].factor_id == "b"

    def test_lazy_faiss_no_faiss_touched(self):
        """惰性 faiss：不 add/build/search 不触碰 faiss；纯查询无缓存返回空。"""
        ad = FactorAssetsAdapter()
        # 无 factor_ids 纯查询（无缓存）→ 空结果（零 faiss 依赖）
        assert ad.nearest_neighbors([1.0, 0.0, 0.0, 0.0], k=3) == []
        # 有 factor_ids 但 faiss/factor_assets 后端不可用 → fail-closed
        idx = PersistentSimilarityIndex(embedding_dim=4, fingerprint_version="v1")
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(FactorAssetsUnavailable):
            idx.search([1.0, 0.0, 0.0, 0.0], k=1)

    def test_adapter_persistent_index_injectable(self):
        """调用方可直接注入 persistent_index 管理生命周期。"""
        fake = FakeANNBackend()
        ad = FactorAssetsAdapter()
        idx = PersistentSimilarityIndex(embedding_dim=4, fingerprint_version="v1", backend=fake)
        idx.add_many(["a"], [[1.0, 0.0, 0.0, 0.0]])
        idx.rebuild()
        ad.persistent_index = idx
        res = ad.nearest_neighbors([1.0, 0.0, 0.0, 0.0], k=1)
        assert res[0].factor_id == "a"
        assert fake.build_count == 1
        assert res[0].distance == pytest.approx(0.0, abs=1e-6)
        assert res[0].similarity_score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. cluster fallback 不全 singleton
# ---------------------------------------------------------------------------


class TestClusterContext:
    def test_default_empty_versions_raises(self):
        """缺省 cluster_versions 空 + fallback=False → ClusterContextUnavailable。"""
        import alphaprobe.authority as authority_mod

        monkeypatch_default = pytest.MonkeyPatch()
        monkeypatch_default.setattr(
            authority_mod,
            "build_identity_view",
            lambda f: {
                "canonical_formula": f,
                "canonical_ast_hash": "c" * 64,
                "signal_equivalence_id": "s" * 64,
            },
        )
        try:
            ad = FactorAssetsAdapter()
            # 显式 factor_ids：不触发 identity_of，只验证 cluster context 守卫
            with pytest.raises(ClusterContextUnavailable):
                ad.cluster_assign(["rank(close)"], factor_ids=["f1"])
            assert issubclass(ClusterContextUnavailable, FactorAssetsUnavailable)
        finally:
            monkeypatch_default.undo()

    def test_singleton_fallback_explicit(self, monkeypatch):
        """allow_singleton_fallback=True → SINGLETON 且标记 SINGLETON_FALLBACK。"""
        import alphaprobe.authority as authority_mod

        monkeypatch.setattr(
            authority_mod,
            "build_identity_view",
            lambda f: {
                "canonical_formula": f,
                "canonical_ast_hash": "d" * 64,
                "signal_equivalence_id": "s" * 64,
            },
        )
        ad = FactorAssetsAdapter(allow_singleton_fallback=True)
        res = ad.cluster_assign(
            ["rank(close)", "rank(open)"], factor_ids=["f1", "f2"]
        )
        assert isinstance(res, ClusterAssignResult)
        assert res.cluster_context == "SINGLETON_FALLBACK"
        assert len(res.assignments) == 2
        for a in res.assignments:
            assert a["kind"] == "SINGLETON"
            assert a["cluster_context"] == "SINGLETON_FALLBACK"
            assert a["logical_cluster_id"].startswith("CL_")
        assert all(v == 1 for v in res.cluster_sizes.values())

    def test_provided_versions_marked_provided(self, monkeypatch):
        """有 cluster_versions → 走 incremental_assign，标记 PROVIDED。"""
        if not FA_AVAILABLE:
            pytest.skip("factor_assets not importable")
        import alphaprobe.authority as authority_mod
        from factor_assets.contracts.cluster_governance import (
            ClusterScale,
            ClusterVersionArtifact,
        )
        from factor_assets.contracts.fingerprint import (
            SimilarityFingerprintArtifact,
        )

        monkeypatch.setattr(
            authority_mod,
            "build_identity_view",
            lambda f: {
                "canonical_formula": f,
                "canonical_ast_hash": "e" * 64,
                "signal_equivalence_id": "s" * 64,
            },
        )
        ad = FactorAssetsAdapter()
        cv = ClusterVersionArtifact(
            logical_cluster_id="CL_A",
            cluster_set_version_ref="csv1",
            member_factor_ids=("f0",),
            representative_factor_id="f0",
            scale=ClusterScale.MICRO_CLUSTER,
        )
        fps = {
            "f0": SimilarityFingerprintArtifact(
                factor_id="f0", embedding=(1.0, 0.0, 0.0, 0.0),
                embedding_spec="s", snapshot="s", universe="u", window="w",
            ),
            "f1": SimilarityFingerprintArtifact(
                factor_id="f1", embedding=(0.9, 0.1, 0.0, 0.0),
                embedding_spec="s", snapshot="s", universe="u", window="w",
            ),
        }
        res = ad.cluster_assign(
            ["rank(close)"],
            factor_ids=["f1"],
            cluster_versions={"CL_A": cv},
            fingerprints_by_id=fps,
        )
        assert res.cluster_context == "PROVIDED"
        assert res.assignments[0]["cluster_context"] == "PROVIDED"
        assert res.representative_by_cluster["CL_A"] == "f0"
