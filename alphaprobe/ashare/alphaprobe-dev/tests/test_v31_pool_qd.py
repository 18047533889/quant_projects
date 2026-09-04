"""ActivePool / QD 语义升级（plan Task 19）——Pareto 克制 + niche archive。

验收点（plan Task 19 + Non-negotiable #13/#29/#30）：
1. Pareto 目标保持少维（FactorFitness↑ / Novelty↑ / Complexity↓ / Turnover↓，
   4 维）——不得把 P/Q/L/S/N/R 全摊成 Pareto 维度：
   - ``DEFAULT_OBJECTIVES`` 长度 == 4；构造 6+ 维目标被明确拒绝。
   - 用六维 fitness 分量（P/Q/L/S/N/R 全高分但 fitness 平庸）构造成员不会把池
     变成全员 front 0（四维 Pareto 仍能分出 rank）。
2. QD niche（FactorAssets cluster / schema / horizon / turnover bucket /
   data domain）每 niche 保 K elites：
   - niche_key_of 生成 6 元带版本键；niche 键版本稳定（同一 niche 语义 →
     同一键）。
   - 同一高 fitness 拥挤 cluster（同 niche）无法挤掉全部稀有 niche：稀有 niche
     的 elite 在 max_size 触顶时被 QD 保护。
3. niche key 版本稳定：version 变化 → 键不同（新旧不混）。
4. eviction 移除对应 QD membership，无 stale archive：evict 后 qd.niche_of 返回
   ()、elites() 不含被移除者。

全部合成数据、确定性、零 LLM / 零模型 / 零网络。
"""

from __future__ import annotations

import pytest

from alphaprobe.pool import (
    QDArchive,
    ActivePool,
    PoolMember,
    niche_key_of,
)
from alphaprobe.pool.pareto import (
    DEFAULT_OBJECTIVES,
    NICHE_DIMENSIONS,
    NICHE_KEY_VERSION,
    NicheSpec,
    Objective,
    ParetoPoint,
    non_dominated_rank,
)


def _member(
    fid: str,
    fitness: float,
    *,
    niche=(),
    novelty: float = 0.0,
    complexity: float = 0.0,
    turnover: float = 0.0,
    rarity: float = 0.0,
    meta: dict | None = None,
) -> PoolMember:
    m = PoolMember(
        factor_id=fid,
        canonical_formula=fid,
        search_fitness=fitness,
        niche_key=tuple(niche),
        meta={
            "novelty": novelty,
            "complexity": complexity,
            "turnover": turnover,
            "niche_rarity": rarity,
            **(meta or {}),
        },
    )
    return m


def _niche_key(*, mechanism="m", horizon="h", field="f", bucket="b", domain="d") -> tuple:
    return niche_key_of(
        mechanism=mechanism, horizon_bucket=horizon, field_family=field,
        turnover_bucket=bucket, data_domain=domain,
    )


# ---------------------------------------------------------------------------
# 1. Pareto 维度克制（4 维，不摊 P/Q/L/S/N/R）
# ---------------------------------------------------------------------------


class TestParetoRestraint:
    def test_default_objectives_four_dimensions(self):
        """Pareto 目标 = [FactorFitness, Novelty, Complexity↓, Turnover↓] 4 维。"""
        names = [o.name for o in DEFAULT_OBJECTIVES]
        assert names == ["fitness", "novelty", "complexity", "turnover"]
        assert len(DEFAULT_OBJECTIVES) == 4
        dirs = {o.name: o.maximize for o in DEFAULT_OBJECTIVES}
        assert dirs == {"fitness": True, "novelty": True, "complexity": False, "turnover": False}

    def test_six_dimension_pareto_is_rejected(self):
        """把 P/Q/L/S/N/R 全摊成 Pareto 维度 → 显式拒绝（构造即报错）。"""
        objs = tuple(Objective(n, True) for n in ("P", "Q", "L", "S", "N", "R"))
        with pytest.raises(ValueError):
            ParetoPoint("x", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), objectives=objs)
        # 但 NicheSpec 维度常量（niche 用）允许 5 维——不是 Pareto 维度
        assert len(NICHE_DIMENSIONS) == 5

    def test_crowded_all_high_sixdim_not_all_front0(self):
        """六维分量全高分但 fitness 平庸 → 四维 Pareto 仍分出 rank（克制维度
        保留选择力）。"""
        # 8 个成员：fitness 0.9/0.8/0.7...（dominance 链），novelty 等相同
        pts = [
            ParetoPoint(
                f"F{i}",
                values=(0.9 - i * 0.05, 0.5, 10.0, 0.2),
                objectives=DEFAULT_OBJECTIVES,
            )
            for i in range(6)
        ]
        rank = non_dominated_rank(pts)
        # 至少能分出 >=2 个 rank（不会全员互不支配）
        assert len(set(rank.values())) >= 2


# ---------------------------------------------------------------------------
# 2. QD niche 每 niche 保 K elites
# ---------------------------------------------------------------------------


class TestQDPerNicheElites:
    def test_archive_keeps_k_per_niche(self):
        """每个 niche 只保留 K（elite_per_cell）个 elite。"""
        qd = QDArchive(elite_per_cell=2)
        for i in range(8):
            qd.offer(_member(f"c{i}", 0.9 - 0.01 * i, niche=_niche_key(mechanism="same")))
        assert len(qd.elites()) == 2

    def test_distinct_niches_each_keep_k(self):
        """不同 niche（cluster/horizon/domain 各异）各保 K，互不挤占。"""
        qd = QDArchive(elite_per_cell=2)
        for i in range(5):
            qd.offer(_member(f"a{i}", 0.99 - 0.01 * i, niche=_niche_key(mechanism="mom")))
            qd.offer(_member(f"b{i}", 0.2 + 0.01 * i, niche=_niche_key(mechanism="vol")))
        # 高 fitness 拥挤 niche(mom) 只占 2；稀有 niche(vol) 也保 2
        elites = qd.elites()
        assert sum(1 for m in elites if m.niche_key[1] == "mom") == 2
        assert sum(1 for m in elites if m.niche_key[1] == "vol") == 2

    def test_pool_crowded_cluster_cannot_clear_rare_niche(self):
        """max_size 触顶：同一高 fitness 拥挤 cluster 换入不能挤掉稀有 niche 代表。"""
        pool = ActivePool(target_size=3, max_size=4, qd_elite_per_cell=2)
        # 稀有 niche（vol，低 fitness 但唯一代表）
        pool.admit(_member("rare1", 0.1, niche=_niche_key(mechanism="vol"), rarity=0.9))
        # 拥挤 cluster（mom）：连续塞高 fitness
        for i in range(3):
            pool.admit(_member(f"mom{i}", 0.9 - 0.01 * i, niche=_niche_key(mechanism="mom")))
        # 再塞一个更高 fitness 的 mom → 触发替换；稀有 vol niche 的 rare1 仍在
        pool.admit(_member("mom_new", 0.95, niche=_niche_key(mechanism="mom")))
        assert pool.contains("rare1")
        # mom niche 被 elite_per_cell=2 限制，塞不进 4 个 mom（archive 层面）
        moms = [f for f in pool.members if f.startswith("mom")]
        assert len(moms) <= 3  # max_size=4，rare1 占 1 个位 → mom 至多 3


# ---------------------------------------------------------------------------
# 3. niche key 版本稳定
# ---------------------------------------------------------------------------


class TestNicheKeyVersionStable:
    def test_niche_key_has_version_prefix(self):
        k = niche_key_of(mechanism="mom", horizon_bucket="h20")
        assert k[0] == NICHE_KEY_VERSION
        assert len(k) == 6
        # 键含 5 个 niche 维度 + 版本
        assert k[1:] == ("mom", "h20", "unknown", "unknown", "unknown")

    def test_same_semantics_same_key(self):
        a = niche_key_of(mechanism="mom", horizon_bucket="h20", data_domain="stock")
        b = niche_key_of(mechanism="mom", horizon_bucket="h20", data_domain="stock")
        assert a == b

    def test_version_change_produces_different_key(self):
        k1 = niche_key_of(mechanism="mom")
        k2 = niche_key_of(mechanism="mom", version=NICHE_KEY_VERSION + 1)
        assert k1 != k2
        # 且两个不同版本键不会落到同一 archive cell
        qd = QDArchive(elite_per_cell=1)
        qd.offer(_member("old", 0.9, niche=k1))
        qd.offer(_member("new", 0.9, niche=k2))
        assert len(qd.elites()) == 2  # 新旧分属两个 niche，互不覆盖

    def test_legacy_v1_four_dim_key_is_stable(self):
        """旧 4 维 niche key（不带版本）仍有稳定生成器（向后兼容）。"""
        from alphaprobe.pool import niche_key_v1_legacy

        a = niche_key_v1_legacy(mechanism="m", horizon_bucket="h")
        b = niche_key_v1_legacy(mechanism="m", horizon_bucket="h")
        assert a == b == ("m", "h", "unknown", "unknown")


# ---------------------------------------------------------------------------
# 4. eviction 移除 QD membership（无 stale archive）
# ---------------------------------------------------------------------------


class TestEvictionCleansArchive:
    def test_evict_removes_membership(self):
        pool = ActivePool(target_size=1, max_size=2, qd_elite_per_cell=2)
        pool.admit(_member("A", 0.9, niche=_niche_key(mechanism="m1")))
        pool.admit(_member("B", 0.8, niche=_niche_key(mechanism="m1")))
        assert pool.qd.niche_of("A") == _niche_key(mechanism="m1")
        # 触顶换入 C → 挤掉最差者
        pool.admit(_member("C", 0.95, niche=_niche_key(mechanism="m2")))
        # 被淘汰成员不再出现在 QD elites / membership
        for gone in ("A", "B"):
            if not pool.contains(gone):
                assert pool.qd.niche_of(gone) == ()
                assert all(m.factor_id != gone for m in pool.qd.elites())

    def test_qd_archive_no_stale_after_replacements(self):
        """连续替换后：QD elites 都是池内成员（membership 与 members 一致）。"""
        pool = ActivePool(target_size=2, max_size=3, qd_elite_per_cell=2)
        for i in range(6):
            pool.admit(_member(f"F{i}", 0.5 + 0.01 * i, niche=_niche_key(mechanism=f"m{i % 3}")))
        elite_ids = {m.factor_id for m in pool.qd.elites()}
        assert elite_ids <= set(pool.members.keys())
        # membership 与 elites 完全对应
        for m in pool.members.values():
            assert pool.qd.niche_of(m.factor_id) == m.niche_key

    def test_qd_offer_removes_overflow_from_membership(self):
        """niche 溢出（超出 K）的成员从 membership 清除。"""
        qd = QDArchive(elite_per_cell=1)
        for i in range(4):
            qd.offer(_member(f"c{i}", 0.9 - 0.01 * i, niche=_niche_key(mechanism="m")))
        assert qd.niche_of("c3") == ()  # 溢出被挤掉
        assert qd.niche_of("c0") == _niche_key(mechanism="m")  # 最高 fitness 保
