"""factor_assets_adapter 测试（任务书 §33 / §42-§43）。

覆盖：
- adapter 降级链（factor_assets 不可导入 → FactorAssetsUnavailable，fail-closed）
- cluster_assign 返回结构（singleton 路径 + incremental_assign 路径）
- nearest_neighbors 返回结构（faiss 不可用 → FactorAssetsUnavailable）
- nearest_neighbors_local 显式降级（seen/ NearestIndex 只读）

合成数据，零 LLM / 零模型 / 零网络。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from alphaprobe.factor_assets_adapter import (
    ClusterAssignResult,
    FactorAssetsAdapter,
    FactorAssetsUnavailable,
    NearestNeighborResult,
    build_factor_assets_adapter,
)

# ---------------------------------------------------------------------------
# factor_assets 可用性探测（不抛异常；失败只 skip 相关用例）
# ---------------------------------------------------------------------------


def _fa_importable() -> bool:
    try:
        import factor_assets  # noqa: F401

        return True
    except Exception:
        return False


def _ensure_fa_importable() -> bool:
    """把 quant_projects 加进 sys.path 使 factor_assets 可导入。"""
    if _fa_importable():
        return True
    # 从当前文件向上找 quant_projects（factor_assets 的父目录）
    for parent in Path(__file__).resolve().parents:
        if (parent / "factor_assets").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return _fa_importable()
    return False


FA_AVAILABLE = _ensure_fa_importable()


# ---------------------------------------------------------------------------
# 1. 降级链
# ---------------------------------------------------------------------------


class TestDegradationChain:
    def test_build_raises_when_fa_unavailable(self, monkeypatch):
        """factor_assets 不可导入 → build 抛 FactorAssetsUnavailable（fail-closed）。"""
        import alphaprobe.factor_assets_adapter as mod

        monkeypatch.setattr(mod, "_factor_assets_importable", lambda: False)
        with pytest.raises(FactorAssetsUnavailable):
            build_factor_assets_adapter()

    def test_nearest_neighbors_raises_when_faiss_unavailable(self, monkeypatch):
        """faiss 不可用 → nearest_neighbors 抛 FactorAssetsUnavailable。"""
        import alphaprobe.factor_assets_adapter as mod

        if not FA_AVAILABLE:
            pytest.skip("factor_assets not importable")
        ad = build_factor_assets_adapter()
        monkeypatch.setattr(mod, "_faiss_importable", lambda: False)
        with pytest.raises(FactorAssetsUnavailable):
            ad.nearest_neighbors([1.0, 0.0, 0.0, 0.0], k=3)

    def test_nearest_neighbors_local_is_explicit_degrade(self):
        """nearest_neighbors_local 走 seen/ NearestIndex（只读），不抛异常。"""
        from alphaprobe.seen.store import SeenStore

        store = SeenStore(":memory:")
        ad = build_factor_assets_adapter()
        # 空库 → 空结果（不抛）
        res = ad.nearest_neighbors_local(store, b"\x00" * 32, k=3)
        assert res == []


# ---------------------------------------------------------------------------
# 2. cluster_assign 返回结构
# ---------------------------------------------------------------------------


class TestClusterAssign:
    def test_singleton_path_structure(self):
        """空 cluster_versions → 每个因子自成一簇（SINGLETON），返回结构完整。"""
        if not FA_AVAILABLE:
            pytest.skip("factor_assets not importable")
        ad = build_factor_assets_adapter()
        res = ad.cluster_assign(["rank(close)", "rank(open)"])
        assert isinstance(res, ClusterAssignResult)
        assert len(res.assignments) == 2
        for a in res.assignments:
            assert a["kind"] == "SINGLETON"
            assert a["logical_cluster_id"].startswith("CL_")
            assert a["cluster_set_version_ref"] == "csv_default"
        assert len(res.cluster_sizes) == 2
        assert all(v == 1 for v in res.cluster_sizes.values())
        assert len(res.representative_by_cluster) == 2

    def test_incremental_assign_path(self):
        """有既有 cluster → 走 incremental_assign，返回 ASSIGNED / PENDING。"""
        if not FA_AVAILABLE:
            pytest.skip("factor_assets not importable")
        from factor_assets.contracts.cluster_governance import (
            ClusterScale,
            ClusterVersionArtifact,
        )
        from factor_assets.contracts.fingerprint import (
            SimilarityFingerprintArtifact,
        )

        ad = build_factor_assets_adapter()
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
            "f2": SimilarityFingerprintArtifact(
                factor_id="f2", embedding=(0.0, 0.0, 1.0, 0.0),
                embedding_spec="s", snapshot="s", universe="u", window="w",
            ),
        }
        res = ad.cluster_assign(
            ["rank(close)", "rank(open)"],
            factor_ids=["f1", "f2"],
            cluster_versions={"CL_A": cv},
            fingerprints_by_id=fps,
        )
        kinds = {a["factor_id"]: a["kind"] for a in res.assignments}
        assert kinds["f1"] == "ASSIGNED"
        assert kinds["f2"] == "PENDING_GLOBAL_REFRESH"
        assert res.representative_by_cluster["CL_A"] == "f0"

    def test_identity_of_deterministic(self):
        """identity_of 确定性：同 formula → 同 hash。"""
        if not FA_AVAILABLE:
            pytest.skip("factor_assets not importable")
        ad = build_factor_assets_adapter()
        a = ad.identity_of("rank(close)")
        b = ad.identity_of("rank(close)")
        assert a["canonical_hash"] == b["canonical_hash"]
        assert len(a["canonical_hash"]) == 64


# ---------------------------------------------------------------------------
# 3. nearest_neighbors 返回结构
# ---------------------------------------------------------------------------


class TestNearestNeighbors:
    def test_nearest_neighbors_structure(self):
        """faiss 可用时返回 NearestNeighborResult 列表。"""
        if not FA_AVAILABLE:
            pytest.skip("factor_assets not importable")
        import alphaprobe.factor_assets_adapter as mod

        if not mod._faiss_importable():
            pytest.skip("faiss not importable")
        ad = build_factor_assets_adapter()
        res = ad.nearest_neighbors(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            factor_ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        assert len(res) == 2
        assert all(isinstance(r, NearestNeighborResult) for r in res)
        assert res[0].factor_id == "a"  # 与 query 最相似
        assert res[0].similarity_score is not None
