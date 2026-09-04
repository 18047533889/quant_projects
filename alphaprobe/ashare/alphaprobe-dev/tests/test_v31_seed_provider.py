"""SeedProvider 测试（plan Task 20 / Non-negotiable #24）。

规则覆盖：
- 混合比例采样（默认 40% underexplored good / 20% fertile / 15% rare schema /
  10% survival / 10% cluster rep / 5% random）。比例可配置；候选不足时按可用
  桶比例重归一并在 diagnostics 标注（degraded_buckets）。
- seed 无迭代历史是合法 DAG root（parents 为空 / lineage_root 语义）。
- 采样前去重（按 FE signal 等价 identity）。
- 合成 10 万 metadata 规模下采样 O(k) 级：不整体物化、不全量打分——用
  确定性 index-space reservoir（按桶预洗牌顺序取 k），输入走惰性可重放的
  metadata 源（列表或索引函数），不把 100K 物化成 dict 后再过滤。

全合成、确定性、零 LLM / 零模型 / 零网络；factor_assets 与 FE identity
均不需要（identity 函数由调用方注入）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import pytest

from alphaprobe.seed_provider import (
    DEFAULT_SEED_MIX,
    SeedBucket,
    SeedCandidate,
    SeedProvider,
    SeedProviderConfig,
)


# ---------------------------------------------------------------------------
# 合成 metadata 源（惰性：不把全量物化成 dict 列表）
# ---------------------------------------------------------------------------


@dataclass
class _LazyCatalog:
    """可重放的惰性 metadata 源：按索引产出 dict，长度已知。"""

    n: int
    _seed: int = 0
    base_formula: str = "rank(ts_mean(close, {w}))"
    #: 索引 → bucket 的确定性模式（每 100 个索引按 40/20/15/10/10/5 分布，
    #: 与默认 mix 一致；索引本身永远唯一 identity）
    _bucket_pattern: tuple[SeedBucket, ...] = field(default_factory=lambda: (
        (SeedBucket.UNDEREXPLORED_GOOD,) * 40
        + (SeedBucket.FERTILE,) * 20
        + (SeedBucket.RARE_SCHEMA,) * 15
        + (SeedBucket.SURVIVAL,) * 10
        + (SeedBucket.CLUSTER_REPRESENTATIVE,) * 10
        + (SeedBucket.RANDOM,) * 5
    ))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Any]:
        if i < 0 or i >= self.n:
            raise IndexError(i)
        w = 5 + (i % 40)
        return {
            "factor_id": f"F{i:08d}",
            "canonical_repr": self.base_formula.format(w=w),
            "canonical_hash": f"hash_{i % 97:02x}" + f"{i:012d}",
            "fe_identity_ref": f"sig_{i % 53:02x}" + f"{i:012d}",
            "parameter_family_id": f"fam_{(i // 7) % 31:03d}",
            "domains": ("PRICE",),
            "bucket": self._bucket_pattern[i % 100],
            "fitness": 0.0,
        }


class _IndexIdentity:
    """确定性 identity：候选与索引 i 用同公式模式，i 相同 identity 相同。

    模拟 FE signal_equivalence 的等价关系：f1(i=5) 与 f1(i=5) 等价；
    i 不同（w 不同）identity 不同。供「采样前去重」与「10 万规模」用例使用。
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, formula: str) -> dict[str, str]:
        self.calls += 1
        # 从公式里把 w 数字取出来当 identity 后缀（与 _LazyCatalog 同模式）
        m = formula.rstrip(")").rsplit(", ", 1)
        w = m[-1] if len(m) > 1 else formula
        return {
            "canonical_repr": formula,
            "canonical_ast_hash": f"ast_{w}",
            "signal_equivalence_id": f"sig_{w}",
            "parameter_family_id": f"fam_{w}",
        }


def _default_provider(**kw: Any) -> SeedProvider:
    cfg = SeedProviderConfig(**kw)
    return SeedProvider(config=cfg)


def _synthetic_catalog(n: int) -> _LazyCatalog:
    return _LazyCatalog(n)


# ---------------------------------------------------------------------------
# 1. 混合比例采样 + 可配置 + 降级标注
# ---------------------------------------------------------------------------


class TestMixtureSampling:
    def test_default_buckets_defined(self):
        """默认六桶 + 比例（40/20/15/10/10/5，合计 100）。"""
        names = {b.name for b in SeedBucket}
        assert names == {
            "UNDEREXPLORED_GOOD",
            "FERTILE",
            "RARE_SCHEMA",
            "SURVIVAL",
            "CLUSTER_REPRESENTATIVE",
            "RANDOM",
        }
        assert len(DEFAULT_SEED_MIX) == len(SeedBucket)
        total = sum(v for v in DEFAULT_SEED_MIX.values())
        assert total == pytest.approx(1.0)
        assert DEFAULT_SEED_MIX[SeedBucket.UNDEREXPLORED_GOOD] == pytest.approx(0.40)
        assert DEFAULT_SEED_MIX[SeedBucket.FERTILE] == pytest.approx(0.20)
        assert DEFAULT_SEED_MIX[SeedBucket.RARE_SCHEMA] == pytest.approx(0.15)
        assert DEFAULT_SEED_MIX[SeedBucket.SURVIVAL] == pytest.approx(0.10)
        assert DEFAULT_SEED_MIX[SeedBucket.CLUSTER_REPRESENTATIVE] == pytest.approx(0.10)
        assert DEFAULT_SEED_MIX[SeedBucket.RANDOM] == pytest.approx(0.05)

    def test_sample_returns_requested_k_with_bucket_tags(self):
        """10 万 catalog 上采样 100 个 seed：数量正确且都带 bucket 标签。"""
        cat = _synthetic_catalog(100_000)
        sp = _default_provider(rng_seed=7)
        out = sp.sample(cat, k=100)
        assert len(out) == 100
        assert all(isinstance(s, SeedCandidate) for s in out)
        assert all(s.bucket is not None for s in out)

    def test_no_duplicates_within_sample(self):
        """同一 k 内不重复（identity 去重）。"""
        cat = _synthetic_catalog(100_000)
        sp = _default_provider(rng_seed=3)
        out = sp.sample(cat, k=200)
        ids = [s.factor_id for s in out]
        assert len(ids) == len(set(ids))

    def test_bucket_ratios_approximately_respected(self):
        """大 k 下各桶比例接近配置（宽松容差，随机采样）。"""
        cat = _synthetic_catalog(200_000)
        # 全桶均匀分配，便于统计
        sp = _default_provider(rng_seed=11)
        out = sp.sample(cat, k=2000)
        counts: dict[str, int] = {}
        for s in out:
            counts[s.bucket.name] = counts.get(s.bucket.name, 0) + 1
        n = len(out)
        assert n == 2000
        for b in SeedBucket:
            actual = counts.get(b.name, 0) / n
            expected = DEFAULT_SEED_MIX[b]
            assert actual == pytest.approx(expected, abs=0.06), (
                f"{b.name}: {actual:.3f} vs {expected:.3f}"
            )

    def test_configurable_mix_used(self):
        """自定义 mix：FERTILE=1.0 → 全部来自 fertile 桶（当候选充足时）。"""
        cat = _LazyCatalog(10_000)
        # 全 fertile（索引公式模式固定，identity 全同 → 去重后只剩 1 个；
        # 换成按索引变 w 才够候选。此处专门造多变候选）
        class _FertileCat:
            n = 10_000

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                w = 5 + (i % 100)
                return {
                    "factor_id": f"F{i:05d}",
                    "canonical_repr": f"rank(ts_mean(close, {w}))",
                    "canonical_hash": f"h{i:06d}",
                    "fe_identity_ref": f"s{i:06d}",
                    "parameter_family_id": None,
                    "domains": ("PRICE",),
                    "bucket": SeedBucket.FERTILE,
                    "fitness": 0.01,
                }

        sp = SeedProvider(
            config=SeedProviderConfig(
                mix={SeedBucket.FERTILE: 1.0}, rng_seed=5
            )
        )
        out = sp.sample(_FertileCat(), k=50)
        assert len(out) == 50
        assert all(s.bucket == SeedBucket.FERTILE for s in out)
        assert all(s.formula.startswith("rank(ts_mean(close,") for s in out)

    def test_mix_weights_must_sum_to_one(self):
        """mix 权重合计必须为 1.0（其余 raise ValueError）。"""
        with pytest.raises(ValueError):
            SeedProviderConfig(mix={SeedBucket.RANDOM: 0.5})

    def test_unknown_bucket_in_mix_rejected(self):
        """mix 里出现未知 key → TypeError/ValueError。"""
        with pytest.raises((TypeError, ValueError)):
            SeedProviderConfig(mix={SeedBucket.RANDOM: 1.0, "BOGUS": 0.0})  # type: ignore[dict-item]

    def test_diagnostics_record_degraded_when_short(self):
        """候选不足：按可用桶重归一并标注 degraded_buckets；数量 min(k, 候选)。"""
        cat = _LazyCatalog(10)  # 只有 10 个（全部索引 0..9）
        sp = _default_provider(rng_seed=1)
        out = sp.sample(cat, k=500)
        assert len(out) == 10
        diag = sp.diagnostics.to_dict()
        assert diag["requested"] == 500
        assert diag["returned"] == 10
        assert diag["total_available"] == 10
        assert diag["degraded_buckets"]  # 标注过降级
        # 不重复
        assert len({s.factor_id for s in out}) == len(out)

    def test_no_history_seed_is_valid_dag_root(self):
        """seed 无迭代历史是合法 DAG root：parents 空 + lineage_root 语义。"""
        cat = _synthetic_catalog(50)
        sp = _default_provider(rng_seed=2)
        out = sp.sample(cat, k=5)
        for s in out:
            assert s.parents == ()
            assert s.lineage_root is True
            assert s.iteration_history == ()


# ---------------------------------------------------------------------------
# 2. 采样前 identity 去重
# ---------------------------------------------------------------------------


class TestDedupBeforeSample:
    def test_signal_duplicates_deduped(self):
        """同 signal identity（不同表面公式）只保留一个 seed。"""
        entries = [
            {"factor_id": "F_a", "canonical_repr": "rank(close)",
             "canonical_hash": "ast_1", "fe_identity_ref": "sig_x", "domains": ("PRICE",),
             "bucket": SeedBucket.RANDOM, "fitness": 0.0},
            {"factor_id": "F_b", "canonical_repr": "-rank(close)",  # SIGN 等价
             "canonical_hash": "ast_2", "fe_identity_ref": "sig_x", "domains": ("PRICE",),
             "bucket": SeedBucket.RANDOM, "fitness": 0.0},
            {"factor_id": "F_c", "canonical_repr": "rank(open)",
             "canonical_hash": "ast_3", "fe_identity_ref": "sig_y", "domains": ("PRICE",),
             "bucket": SeedBucket.RANDOM, "fitness": 0.0},
        ]
        sp = _default_provider(rng_seed=0)
        out = sp.sample(entries, k=5)
        # sig_x 只出现一次 → 结果 2 个不同 identity
        assert len(out) == 2
        sigs = {s.identity_ref for s in out}
        assert sigs == {"sig_x", "sig_y"}

    def test_exact_duplicates_deduped(self):
        """同 canonical_hash 精确重复只保留一个。"""
        entries = [
            {"factor_id": f"F_{i}", "canonical_repr": f"formula_{i % 3}",
             "canonical_hash": f"h_{i % 2}", "fe_identity_ref": f"s_{i % 2}",
             "domains": ("PRICE",), "bucket": SeedBucket.RANDOM, "fitness": 0.0}
            for i in range(6)
        ]
        sp = _default_provider(rng_seed=0)
        out = sp.sample(entries, k=6)
        assert len(out) == 2  # h_0 / h_1
        hashes = {s.canonical_hash for s in out}
        assert hashes == {"h_0", "h_1"}

    def test_seed_provider_accepts_identity_fn(self):
        """显式 identity_fn 控制去重口径（调用方注入 FE 等价语义）。

        条目不含 hash/identity → identity_fn 被调用作为去重口径。"""
        entries = [
            {"factor_id": "F1", "canonical_repr": "a", "domains": ("PRICE",),
             "bucket": SeedBucket.RANDOM, "fitness": 0.0},
            {"factor_id": "F2", "canonical_repr": "b", "domains": ("PRICE",),
             "bucket": SeedBucket.RANDOM, "fitness": 0.0},
        ]
        seen: list[str] = []

        def _idf(formula: str):
            seen.append(formula)
            return {"canonical_repr": formula, "canonical_ast_hash": formula,
                    "signal_equivalence_id": f"id_{formula}",
                    "parameter_family_id": None}

        sp = SeedProvider(config=SeedProviderConfig(), identity_fn=_idf)
        out = sp.sample(entries, k=2)
        assert len(out) == 2
        assert set(seen) == {"a", "b"}  # identity_fn 确实被调用（作为去重口径）


# ---------------------------------------------------------------------------
# 3. 10 万规模：不物化 / 不全量打分（O(k) 级）
# ---------------------------------------------------------------------------


class TestScale100k:
    def test_no_full_materialization(self):
        """10 万 catalog 采样只触碰 O(k) 个条目（不全量物化/打分）。

        用会抛异常的自定义 getitem 验证：只访问被选中索引附近的条目，
        绝不遍历全 10 万。"""
        n = 100_000
        accesses: list[int] = []
        pattern = ((SeedBucket.UNDEREXPLORED_GOOD,) * 40 + (SeedBucket.FERTILE,) * 20
                   + (SeedBucket.RARE_SCHEMA,) * 15 + (SeedBucket.SURVIVAL,) * 10
                   + (SeedBucket.CLUSTER_REPRESENTATIVE,) * 10 + (SeedBucket.RANDOM,) * 5)

        class _TrackingCat:
            def __len__(self):
                return n

            def __getitem__(self, i):
                accesses.append(i)
                w = 5 + (i % 40)
                return {
                    "factor_id": f"F{i:08d}",
                    "canonical_repr": f"rank(ts_mean(close, {w}))",
                    "canonical_hash": f"hash_{i % 97:02x}{i:012d}",
                    "fe_identity_ref": f"sig_{i % 53:02x}{i:012d}",
                    "parameter_family_id": None,
                    "domains": ("PRICE",),
                    "bucket": pattern[i % 100],
                    "fitness": 0.0,
                }

        sp = _default_provider(rng_seed=9)
        out = sp.sample(_TrackingCat(), k=100)
        assert len(out) == 100
        # reservoir 只触碰少量条目（去重回退时最多 O(k) 次重抽）
        assert len(accesses) <= max(2000, n // 10), (
            f"sampling touched {len(accesses)} items — expected O(k)"
        )

    def test_many_duplicates_still_dedup_before_return(self):
        """identity 高度重复（大 catalog 只有 7 个唯一 identity）：
        返回全部唯一 seed，不足 k 时用唯一数（仍不遍历全表打分）。"""
        n = 100_000
        pattern = ((SeedBucket.UNDEREXPLORED_GOOD,) * 40 + (SeedBucket.FERTILE,) * 20
                   + (SeedBucket.RARE_SCHEMA,) * 15 + (SeedBucket.SURVIVAL,) * 10
                   + (SeedBucket.CLUSTER_REPRESENTATIVE,) * 10 + (SeedBucket.RANDOM,) * 5)

        class _DupCat:
            def __len__(self):
                return n

            def __getitem__(self, i):
                u = i % 7
                return {
                    "factor_id": f"F{i:08d}",
                    "canonical_repr": f"formula_{u}",
                    "canonical_hash": f"h_{u}",
                    "fe_identity_ref": f"s_{u}",
                    "parameter_family_id": None,
                    "domains": ("PRICE",),
                    "bucket": pattern[i % 100],
                    "fitness": 0.0,
                }

        sp = _default_provider(rng_seed=4)
        out = sp.sample(_DupCat(), k=1000)
        # 7 个唯一 identity 分布在 6 个桶里 → 每个桶取不到独特身份时
        # 由 RANDOM 兜底 → 最终返回全部 7 个
        assert len(out) == 7
        assert len({s.identity_ref for s in out}) == 7

    def test_sample_is_reasonably_fast(self):
        """100k catalog 采样 1000 seed 在宽松时限内完成（无全量相关矩阵）。"""
        cat = _synthetic_catalog(100_000)
        sp = _default_provider(rng_seed=6)
        t0 = time.time()
        out = sp.sample(cat, k=1000)
        elapsed = time.time() - t0
        assert len(out) == 1000
        # 宽松：O(k) 级采样在合成源上应远小于 5s
        assert elapsed < 5.0, f"sample took {elapsed:.2f}s — not O(k)"

    def test_deterministic_repeat(self):
        """同 catalog + 同 rng_seed → 两次采样结果逐位一致。"""
        cat = _synthetic_catalog(50_000)
        sp1 = _default_provider(rng_seed=42)
        sp2 = _default_provider(rng_seed=42)
        out1 = sp1.sample(cat, k=50)
        out2 = sp2.sample(cat, k=50)
        assert [s.factor_id for s in out1] == [s.factor_id for s in out2]
        assert [s.formula for s in out1] == [s.formula for s in out2]
        assert [s.bucket for s in out1] == [s.bucket for s in out2]


# ---------------------------------------------------------------------------
# 4. runner 接线（Task 20：parents 来自 seed 采样而非 YAML 位置序；
#    OFFLINE_TEST 兼容开关保持既有路径）
# ---------------------------------------------------------------------------


class TestRunnerWiring:
    def test_seed_provider_catalog_helper_dir_catalog(self, tmp_path):
        """runner._open_factor_assets_catalog：目录 seed_catalog.json → list。"""
        import json

        from alphaprobe.runner import _open_factor_assets_catalog

        catalog = [
            {"factor_id": "F1", "canonical_repr": "rank(close)",
             "canonical_hash": "h1", "fe_identity_ref": "s1"},
            {"factor_id": "F2", "canonical_repr": "ts_mean(close, 20)",
             "canonical_hash": "h2", "fe_identity_ref": "s2"},
        ]
        d = tmp_path / "fa_lib"
        d.mkdir()
        (d / "seed_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        out = _open_factor_assets_catalog(str(d))
        assert out is not None
        assert len(out) == 2
        assert out[0]["factor_id"] == "F1"

    def test_seed_provider_catalog_helper_sqlite(self, tmp_path):
        """runner._open_factor_assets_catalog：SQLite assets 表 → metadata list。"""
        import json
        import sqlite3

        from alphaprobe.runner import _open_factor_assets_catalog

        db = tmp_path / "fa.sqlite3"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE assets(factor_id TEXT, canonical_hash TEXT, payload TEXT)"
        )
        payload = json.dumps(
            {"metadata": {"factor_id": "F1", "canonical_repr": "rank(close)",
                          "canonical_hash": "h1", "fe_identity_ref": "s1"}}
        )
        conn.execute("INSERT INTO assets VALUES (?,?,?)", ("F1", "h1", payload))
        conn.commit()
        conn.close()
        out = _open_factor_assets_catalog(str(db))
        assert out is not None
        assert len(out) == 1
        assert out[0]["canonical_repr"] == "rank(close)"

    def test_missing_path_returns_none(self):
        """不存在的路径 → None（调用方回落 resolve_cold_start，不抛）。"""
        from alphaprobe.runner import _open_factor_assets_catalog

        assert _open_factor_assets_catalog("/nonexistent/fa_lib_xyz") is None

    def test_resolve_initial_parents_default_offline_compat(self, monkeypatch):
        """缺省开关 → OFFLINE_TEST 兼容路径（resolve_cold_start 位置序）。"""
        from alphaprobe import runner as runner_mod

        captured: dict[str, Any] = {}
        initial = []
        for i in range(3):
            initial.append(type("E", (), {"dsl": f"rank(ts_mean(close, {i + 5}))"})())
        monkeypatch.setattr(
            runner_mod, "resolve_cold_start", lambda *a, **k: initial
        )

        class _Args:
            seed_provider = None
            seed_provider_fa_lib = None
            seed_sample_size = None
            generate_num = 5
            seed = 0

        parents = runner_mod._resolve_initial_parents(
            None, _Args(), exec_mode=None
        )
        assert len(parents) == 3
        assert parents[0]["factor_id"] == "seed_0"
        assert parents[0]["formula"].startswith("rank(ts_mean(close, 5))")
        captured["ok"] = True
        assert captured["ok"]

    def test_resolve_initial_parents_on_path_samples(self, monkeypatch, tmp_path):
        """--seed-provider on + fa_lib → parents 来自 SeedProvider 采样。"""
        import json

        from alphaprobe import runner as runner_mod

        catalog = [
            {"factor_id": f"F{i}", "canonical_repr": f"rank(ts_mean(close, {5 + i}))",
             "canonical_hash": f"h{i}", "fe_identity_ref": f"s{i}"}
            for i in range(10)
        ]
        d = tmp_path / "fa"
        d.mkdir()
        (d / "seed_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

        class _Args:
            seed_provider = "on"
            seed_provider_fa_lib = str(d)
            seed_sample_size = 5
            generate_num = 5
            seed = 3

        parents = runner_mod._resolve_initial_parents(
            None, _Args(), exec_mode=None
        )
        # SeedProvider 路径：parents 是采样出的 seed（带 bucket / identity）
        assert len(parents) == 5
        assert parents[0]["factor_id"].startswith("F")
        assert "bucket" in parents[0]
        assert "identity_ref" in parents[0]

    def test_resolve_initial_parents_on_no_fa_lib_falls_back(self, monkeypatch):
        """--seed-provider on 但无 fa_lib → 回落 resolve_cold_start（绝不停摆）。"""
        from alphaprobe import runner as runner_mod

        initial = []
        for i in range(2):
            initial.append(type("E", (), {"dsl": f"rank(close_{i})"})())
        monkeypatch.setattr(
            runner_mod, "resolve_cold_start", lambda *a, **k: initial
        )

        class _Args:
            seed_provider = "on"
            seed_provider_fa_lib = None
            seed_sample_size = None
            generate_num = 5
            seed = 0

        parents = runner_mod._resolve_initial_parents(
            None, _Args(), exec_mode=None
        )
        assert len(parents) == 2
        assert parents[0]["factor_id"] == "seed_0"
