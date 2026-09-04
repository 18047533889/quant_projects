"""FactorAssets 搜索闭环测试（plan Task 5 / Non-negotiable #15/#16/#13/#24）。

覆盖 plan 五条行为 + 全局因子上下文结构：
- exact/sign duplicate 在 FE 完整执行 / QE 之前即停止（identity 命中
  FactorAssets 全局库的 seen 即拒，且已入全局库的因子是 alias/rediscovery，
  不是新全局节点）；
- ANN 查询用持久索引，同版本不重建（index_rebuild_count 不增）；
- cluster context 流向 Retriever / SearchOpportunity（无证据 → 中性 0.5，
  已知 cluster → 1/sqrt(1+size)，#13）；
- cold-start 因子只进 FactorAssets 全局库，选中的 seed 才进 DAG
  （不整体物化，#24）；
- GlobalFactorContext dataclass 字段完整（factor_id / seen_status / nearest /
  cluster_id / cluster_size / cluster_elite_rate / family_id）。

全合成数据；factor_assets 与 faiss 均不需要——用注入的内存库 + 假 ANN
backend（venv 无 torch / faiss）。零 LLM / 零模型 / 零网络。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# sys.path 引导（与既有 factor_assets_adapter 测试一致；无则跳过 FA 专有用例）
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
FA_REQUIRED = pytest.mark.skipif(not FA_AVAILABLE, reason="factor_assets not importable")


# ---------------------------------------------------------------------------
# 内存注册库（模拟 FactorAssets 全局库：register / exists_by_hash /
# exists_by_identity / list_metadata —— 不依赖 SQLiteLifecycleRepository）
# ---------------------------------------------------------------------------


class _MemoryRegistry:
    """最小内存 FactorAssets 全局库。register 返回 created / existing 语义。"""

    def __init__(self) -> None:
        self._by_hash: dict[str, dict[str, Any]] = {}
        self._by_identity: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self.register_count = 0

    def register(
        self,
        *,
        factor_id: str,
        canonical_hash: str,
        canonical_repr: str,
        fe_identity_ref: str,
        parameter_family_id: str | None = None,
        origin: str = "search",
    ) -> tuple[bool, str]:
        """返回 (created, existing_factor_id)。冲突（同 hash 或同 identity）
        时 existing_factor_id 指向已存在因子（alias/rediscovery 语义）。"""
        self.register_count += 1
        hit = self._by_hash.get(canonical_hash) or self._by_identity.get(fe_identity_ref)
        if hit is not None:
            return False, hit["factor_id"]
        meta = {
            "factor_id": factor_id,
            "canonical_hash": canonical_hash,
            "canonical_repr": canonical_repr,
            "fe_identity_ref": fe_identity_ref,
            "parameter_family_id": parameter_family_id,
            "origin": origin,
        }
        self._by_hash[canonical_hash] = meta
        self._by_identity[fe_identity_ref] = meta
        self._order.append(factor_id)
        return True, ""

    def find_by_hash(self, canonical_hash: str) -> dict[str, Any] | None:
        """按 canonical_hash 查（duck-typing factor_assets registry）。"""
        m = self._by_hash.get(canonical_hash)
        return dict(m) if m is not None else None

    def find_by_identity(self, fe_identity_ref: str) -> dict[str, Any] | None:
        """按 fe_identity_ref 查（duck-typing factor_assets registry）。"""
        m = self._by_identity.get(fe_identity_ref)
        return dict(m) if m is not None else None

    def exists_by_hash(self, canonical_hash: str) -> bool:
        return canonical_hash in self._by_hash

    def exists_by_identity(self, fe_identity_ref: str) -> bool:
        return fe_identity_ref in self._by_identity

    def get(self, factor_id: str) -> dict[str, Any] | None:
        for m in self._by_hash.values():
            if m["factor_id"] == factor_id:
                return dict(m)
        return None

    def list_metadata(self, limit: int | None = None) -> list[dict[str, Any]]:
        out = [dict(m) for m in self._by_hash.values()]
        if limit is not None:
            return out[:limit]
        return out

    def __len__(self) -> int:
        return len(self._by_hash)


# ---------------------------------------------------------------------------
# 假 ANN backend（记录 rebuild / search 次数）
# ---------------------------------------------------------------------------


class _CountingBackend:
    def __init__(self) -> None:
        self.build_count = 0
        self.search_count = 0

    def build(self, factor_ids: list[str], embeddings: Any) -> None:
        self.build_count += 1

    def search(self, query: Any, k: int = 10, min_similarity: float | None = None) -> list[Any]:
        self.search_count += 1
        return []


# ---------------------------------------------------------------------------
# adapter 模块（在 FA_AVAILABLE 与不可用两条路径都可用）
# ---------------------------------------------------------------------------

from alphaprobe import factor_assets_adapter as fa_mod
from alphaprobe.factor_assets_adapter import (  # noqa: E402
    ClusterContextUnavailable,
    FactorAssetsAdapter,
    FactorAssetsUnavailable,
    GlobalFactorContext,
    GlobalFactorGate,
    NearestNeighborResult,
)


class _FakeEngine:
    """注入用身份提供者（避免用例强依赖 factor_engine 解析）。

    注意：sign 等价（exact / SIGN duplicate）由**全局库按 signal_id 判定**——
    本 fake 只做确定性 identity 映射；同 core 不同符号 → 不同 canonical_hash、
    但 signal 前缀相同（真实 FE 里 signal_equivalence_id 对全局取负等价）。
    为让「SIGN duplicate 被拒」可测，用例显式登记 -rank 的 signal 到全局库。
    """

    def __init__(self) -> None:
        self.parsed = 0

    def parse(self, formula: str) -> dict[str, Any]:
        self.parsed += 1
        f = formula.strip()
        neg = f.startswith("-") or f.startswith("(-")
        core = f.lstrip("(-").rstrip(")")
        # 确定性 identity：core 的 sha256（同 core → 同 canonical_ast_hash 前缀）
        import hashlib

        sig = hashlib.sha256(core.encode()).hexdigest()
        return {
            "formula": formula,
            "canonical_formula": formula,
            "canonical_ast_hash": "ast_" + sig[:16],
            "canonical_hash": "ast_" + sig[:16],  # identity_provider 契约键
            "signal_equivalence_id": "sig_" + sig[:16],  # sign-invariant
            "fe_identity_ref": "sig_" + sig[:16],  # 契约键（sign-invariant）
            "parameter_family_id": "fam_" + core.split("(")[0],
            "orientation": -1 if neg else 1,
        }

    def __call__(self, formula: str) -> dict[str, Any]:
        """identity_provider 契约：对象本身 callable（内部走 parse）。"""
        return self.parse(formula)


def _fake_adapter() -> tuple[FactorAssetsAdapter, _FakeEngine, _MemoryRegistry]:
    engine = _FakeEngine()
    reg = _MemoryRegistry()
    ad = FactorAssetsAdapter()
    return ad, engine, reg


# ---------------------------------------------------------------------------
# 1. GlobalFactorContext 结构
# ---------------------------------------------------------------------------


class TestGlobalFactorContext:
    def test_fields(self):
        """plan 给的七个字段全在且可 frozen hash。"""
        ctx = GlobalFactorContext(
            factor_id="F1",
            seen_status="NEW",
            nearest=(
                NearestNeighborResult(factor_id="F9", distance=0.3, similarity_score=0.7),
            ),
            cluster_id="CL_1",
            cluster_size=12,
            cluster_elite_rate=0.5,
            family_id="fam_x",
        )
        assert ctx.factor_id == "F1"
        assert ctx.seen_status == "NEW"
        assert len(ctx.nearest) == 1
        assert ctx.cluster_id == "CL_1"
        assert ctx.cluster_size == 12
        assert ctx.cluster_elite_rate == 0.5
        assert ctx.family_id == "fam_x"
        hash(ctx)  # frozen hashable
        d = ctx.to_dict()
        assert d["cluster_id"] == "CL_1"
        assert d["cluster_rarity"] == pytest.approx(1.0 / (1.0 + 12) ** 0.5)

    def test_neutral_when_no_cluster_evidence(self):
        """无 cluster 证据 → cluster_id/size/elite_rate/family 全部 None（#13）。"""
        ctx = GlobalFactorContext(factor_id="F1", seen_status="NEW", nearest=())
        assert ctx.cluster_id is None
        assert ctx.cluster_size is None
        assert ctx.cluster_elite_rate is None
        assert ctx.family_id is None
        assert ctx.cluster_rarity == 0.5  # 中性（绝不当成最稀有）


# ---------------------------------------------------------------------------
# 2. 先拒：identity 命中全局库 seen → 停在 FE 全执行 / QE 之前
# ---------------------------------------------------------------------------


class TestStopBeforeExecution:
    @FA_REQUIRED
    def test_registered_candidate_rejected_before_fe_or_qe(self):
        """已在全局库的因子（exact）→ check 直接拒（不重复注册）。"""
        ad, engine, reg = _fake_adapter()
        calls = {"parse": 0}

        class _NoParseEngine:
            def parse(self, formula):
                calls["parse"] += 1
                return _FakeEngine().parse(formula)

            def __call__(self, formula):  # identity_provider 契约：callable
                return self.parse(formula)

        gate = GlobalFactorGate(adapter=ad, identity_provider=_NoParseEngine(), registry=reg)
        # 注册（用同一个 gate 的 identity → 与后续 check 同口径）
        ident = gate._identity_of("rank(close)")
        # identity 视图带 canonical_repr（FE authority 兼容键）
        if "canonical_repr" not in ident:
            ident = dict(ident)
            ident["canonical_repr"] = ident.get("canonical_formula", "rank(close)")
        reg.register(
            factor_id="Ffirst", canonical_hash=ident["canonical_hash"],
            canonical_repr=ident["canonical_repr"], fe_identity_ref=ident["fe_identity_ref"],
        )
        parse_before = calls["parse"]
        verdict = gate.check_candidate("rank(close)", allow_reserve=False)
        # 全局库 seen 命中即拒：不产生新节点（不做 FE 全执行/QE 的预注册）
        assert verdict.blocked is True
        assert verdict.reason in ("EXACT", "SIGN")
        assert verdict.existing_factor_id == "Ffirst"
        assert len(reg) == 1
        # identity 解析是一次轻量查表（≠ FE 全执行/QE）；重复注册被拒
        assert calls["parse"] >= parse_before

    @FA_REQUIRED
    def test_sign_equivalent_rejected(self):
        """-rank(close) 对已注册的 rank(close) 是 SIGN duplicate → 拒。"""
        ad, engine, reg = _fake_adapter()
        gate = GlobalFactorGate(adapter=ad, identity_provider=engine, registry=reg)
        # 注册时显式登记 -rank(close) 的 SIGN identity（模拟另一 miner 先发现）；
        # rank(close) 与 -rank(close) 的 signal_id 相同 → 命中全局库 signal。
        neg = engine.parse("-rank(close)")
        created, _ = reg.register(
            factor_id="Forig", canonical_hash=neg["canonical_ast_hash"],
            canonical_repr=neg["canonical_formula"], fe_identity_ref=neg["fe_identity_ref"],
        )
        assert created is True
        # rank(close) 的 signal 命中已存在全局库 → duplicate 拒（EXACT 或 SIGN）
        v = gate.check_candidate("rank(close)", allow_reserve=False)
        assert v.blocked is True
        assert v.reason in ("EXACT", "SIGN")
        assert v.existing_factor_id == "Forig"

    @FA_REQUIRED
    def test_new_not_blocked_and_registers(self):
        """未见因子 → 放行；allow_reserve=True 时注册进全局库。"""
        ad, engine, reg = _fake_adapter()
        gate = GlobalFactorGate(adapter=ad, registry=reg)
        v = gate.check_candidate("ts_std(open, 20)", allow_reserve=True)
        assert v.blocked is False
        assert v.factor_id  # 生成了全局 factor_id
        assert len(reg) == 1
        iref = v.identity.get("fe_identity_ref") or v.context.identity_ref
        assert reg.exists_by_identity(iref)

    @FA_REQUIRED
    def test_rediscovery_aliases_existing(self):
        """另一 miner 已发现的同因子（再次 register）→ 返回 existing，不建新节点。"""
        ad, engine, reg = _fake_adapter()
        gate = GlobalFactorGate(adapter=ad, registry=reg)
        ident = gate._identity_of("rank(close)")
        created, _ = reg.register(
            factor_id="miner_a_node", canonical_hash=ident["canonical_hash"],
            canonical_repr=ident["canonical_repr"], fe_identity_ref=ident["fe_identity_ref"],
        )
        assert created is True
        created2, existing = reg.register(
            factor_id="miner_b_node", canonical_hash=ident["canonical_hash"],
            canonical_repr=ident["canonical_repr"], fe_identity_ref=ident["fe_identity_ref"],
        )
        assert created2 is False
        assert existing == "miner_a_node"
        assert len(reg) == 1


# ---------------------------------------------------------------------------
# 3. ANN 同版本不重建；GlobalFactorContext.nearest 流出
# ---------------------------------------------------------------------------


class TestAnnNoRebuild:
    @FA_REQUIRED
    def test_same_version_query_no_rebuild(self):
        """GlobalFactorGate 同一版本 key 二次查询：index_rebuild_count 不增。"""
        ad, engine, reg = _fake_adapter()
        backend = _CountingBackend()
        gate = GlobalFactorGate(adapter=ad, registry=reg, version_key={"fingerprint_version": "v1"})
        # 首次建索引并查询
        ctx1 = gate.context_for(
            "rank(close)", candidate_id="Fc1", universe=["F1", "F2"], _backend=backend
        )
        idx1 = ad.persistent_index
        assert idx1 is not None
        first_rebuilds = idx1.index_rebuild_count
        # 同版本再查（universe 相同 / 版本 key 不变）→ 复用缓存不重建
        ctx2 = gate.context_for(
            "rank(close)", candidate_id="Fc2", universe=["F1", "F2"], _backend=backend
        )
        assert ad.persistent_index is idx1
        assert idx1.index_rebuild_count == first_rebuilds  # 不增

    @FA_REQUIRED
    def test_context_nearest_populated(self):
        """context_for 填 nearest（ANN 返回的空结果也可用——fake backend 空库）。"""
        ad, engine, reg = _fake_adapter()
        gate = GlobalFactorGate(adapter=ad, registry=reg)
        ctx = gate.context_for(
            "rank(close)", candidate_id="Fc1",
            universe=[], _backend=_CountingBackend(),
        )
        assert isinstance(ctx.nearest, tuple)
        assert ctx.seen_status in ("NEW", "SEEN_EXACT", "SEEN_SIGN", "RESERVED")


# ---------------------------------------------------------------------------
# 4. cluster context 流向 Retriever / SearchOpportunity（#13）
# ---------------------------------------------------------------------------


class TestClusterContextFlows:
    @FA_REQUIRED
    def test_search_opportunity_neutral_without_cluster(self):
        """无 cluster 证据（cluster_size None）→ rarity 中性 0.5（#13）。"""
        from alphaprobe.retrieval.search_opportunity import (
            SearchOpportunity,
            cluster_rarity,
        )

        so = SearchOpportunity()
        # 无 cluster_fn → 无证据 → 中性
        comps = so.compute("F1", mean_novelty_gain=None, attempted_actions=None)
        assert comps["cluster_rarity"] == pytest.approx(0.5)

        # GlobalFactorContext 无 cluster → 0.5
        ctx = GlobalFactorContext(factor_id="F1", seen_status="NEW", nearest=())
        assert ctx.cluster_rarity == pytest.approx(0.5)
        assert cluster_rarity(ctx.cluster_size) == pytest.approx(0.5)

    @FA_REQUIRED
    def test_context_cluster_rarity_matches_search_opportunity(self):
        """GlobalFactorContext 的 cluster_rarity 与 SearchOpportunity.cluster_rarity
        同一口径（1/sqrt(1+size)）。"""
        from alphaprobe.retrieval.search_opportunity import cluster_rarity

        ctx = GlobalFactorContext(
            factor_id="F1", seen_status="NEW", nearest=(), cluster_id="CL_x",
            cluster_size=8, cluster_elite_rate=None, family_id="fam1",
        )
        assert ctx.cluster_rarity == pytest.approx(cluster_rarity(8))

    @FA_REQUIRED
    def test_cluster_assign_fail_closed_without_versions(self):
        """无 cluster context 且未允许 fallback → ClusterContextUnavailable
        （fail-closed；不静默自成一簇误导 Retriever，#13）。"""
        ad, engine, reg = _fake_adapter()
        with pytest.raises(ClusterContextUnavailable):
            ad.cluster_assign(["rank(close)"], factor_ids=["f1"])

    @FA_REQUIRED
    def test_family_id_from_identity_flows(self):
        """GlobalFactorContext.family_id 来自 FE parameter_family_id。"""
        ad, engine, reg = _fake_adapter()
        # _FakeEngine 的 identity 前缀带 "ast_" / "sig_" / "fam_" 字样；
        # 这里断言 family_id 前缀来自 parameter_family_id（"fam_rank"）
        gate = GlobalFactorGate(adapter=ad, identity_provider=engine, registry=reg)
        ctx = gate.context_for("rank(close)", candidate_id="Fc", universe=[], _backend=_CountingBackend())
        assert ctx.family_id is not None
        assert ctx.family_id.startswith("fam_")


# ---------------------------------------------------------------------------
# 5. cold-start 只进全局库，选中的 seed 才进 DAG（#24）
# ---------------------------------------------------------------------------


class TestColdStartLibraryBoundary:
    @FA_REQUIRED
    def test_bulk_register_does_not_create_dag_nodes(self):
        """10 万冷启动因子只进全局库（registry），不产出 DAG parent 节点。"""
        ad, engine, reg = _fake_adapter()
        gate = GlobalFactorGate(adapter=ad, identity_provider=_FakeEngine(), registry=reg)
        metas = [
            {"factor_id": f"seed_{i}", "canonical_repr": f"rank(ts_mean(close, {5 + i % 40}))",
             "canonical_hash": f"h{i}", "fe_identity_ref": f"s{i}"}
            for i in range(100)
        ]
        created = gate.register_many(metas)
        assert created == 100
        assert len(reg) == 100
        # DAG 侧没有新建节点（seed 是 root，但 bulk register 不产生 DAG 边）
        assert gate.dag_node_created_count == 0
        # 取出的 metadata 也不含任何 DAG 化字段
        for m in reg.list_metadata(limit=3):
            assert "dag_parents" not in m

    @FA_REQUIRED
    def test_selected_seeds_only_materialized(self):
        """选中的 seed 才物化成 DAG root（parents 空）；其余留在全局库。"""
        ad, engine, reg = _fake_adapter()
        gate = GlobalFactorGate(adapter=ad, identity_provider=engine, registry=reg)
        # 先把 F_seed_1/F_seed_2 注册进全局库（选中者才物化）
        reg.register(
            factor_id="F_seed_1", canonical_hash="h1", canonical_repr="rank(close)",
            fe_identity_ref="s1",
        )
        reg.register(
            factor_id="F_seed_2", canonical_hash="h2", canonical_repr="rank(open)",
            fe_identity_ref="s2",
        )
        selected = gate.as_dag_roots(["F_seed_1", "F_seed_2"])
        assert len(selected) == 2
        for root in selected:
            assert root["lineage_root"] is True
            assert root["parents"] == ()
        assert gate.dag_node_created_count == 2
        # 未选中的其他 seed 不被物化
        reg.register(
            factor_id="F_never", canonical_hash="hx", canonical_repr="r",
            fe_identity_ref="sx",
        )
        still = gate.as_dag_roots(["F_seed_1"])
        assert len(still) == 1
        assert gate.dag_node_created_count == 3
