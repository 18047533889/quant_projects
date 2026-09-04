"""V3.1 Schema Space 验收（plan Task 12 / Non-negotiable #29 / Part E）。

覆盖 plan Task 12 三条：
① 同一 schema 不同公式实现共享 schema_id（SchemaPlan 稳定版本化 ID）：
   - 同九维语义 → 同 schema_id（与实现公式无关）；
   - 仅 name/description 不同 → 同 id；
   - 任一语义字段改变 → id 不同（版本化，新旧不混）。
② schema reward 聚合实现层结果且带置信计数（n_impl/n_success/n_elite 等）：
   - 同一 schema 多个实现的 reward/elite 独立聚合到 schema 层；
   - 计数可查询（n_impl / n_eval / n_success / n_elite / n_failed_impl）；
   - aggregate_reward 带置信收缩：实现少 → 向中性 0.5 收缩（绝不当最好）；
   - schema 统计独立于公式层（registry 不复制公式层/身份层数据）。
③ 语义字段（DataDomain / Horizon 等）对照 DA taxonomy 校验，非法即拒：
   - DataDomain 不在 DA 词表 → SchemaValidationError；
   - DataDomain 大小写不敏感归一（PRICE/price）；
   - 显式非法 Horizon / Direction / Normalization 值 → 拒；
   - 空/unknown 维放行。

外加行为契约：
- failed implementation 记录（status=failed + failure_reason）但**不自动**把
  schema 标记 failed（is_failed 仅显式 mark_schema_failed 或全部实现失败时
  为 True）；schema saturation / elite_rate 可供 Retriever 消费。
- SchemaRegistry 内存(:memory:) + 文件持久化（SQLite）都可用；重开连接后
  统计仍在。

全部合成数据、确定性、零 LLM / 零模型 / 零网络 / 零 FE。DataDomain 词表用
模块内 fake（= DA VALID_FIELD_DOMAINS 的显式子集——真 DA 词表通过注入面在
集成处使用；本测试不 import data_access）。
"""

from __future__ import annotations

import json

import pytest

from alphaprobe.research_space.contracts import (
    DEFAULT_SHRINKAGE_K,
    ImplementationOutcome,
    ImplementationRecord,
    SchemaPlan,
    SchemaStats,
)
from alphaprobe.research_space.registry import SchemaRegistry
from alphaprobe.research_space.schema import (
    DEFAULT_DOMAIN_VOCABULARY,
    SchemaValidationError,
    validate_domain_token,
    validate_schema_plan,
)


# ---------------------------------------------------------------------------
# 词表 / 构造辅助
# ---------------------------------------------------------------------------


def _plan(**kw) -> SchemaPlan:
    base = dict(
        event="earnings_surprise",
        context="high_vol",
        qualities="reversal",
        direction="long",
        output="rank",
        data_domain="FUNDAMENTAL.QUALITY",
        horizon="medium",
        normalization="cross_section",
        tradability="liquid",
    )
    base.update(kw)
    return SchemaPlan(**base)


# ---------------------------------------------------------------------------
# ① 同一 schema 不同公式实现共享 schema_id
# ---------------------------------------------------------------------------


class TestStableVersionedSchemaId:
    def test_same_semantics_share_schema_id_across_implementations(self):
        """不同公式实现同一机制 → 同一 schema_id（plan Task 12 核心）。"""
        s1 = _plan(event="earnings_surprise", data_domain="FUNDAMENTAL.QUALITY")
        s2 = _plan(
            event="earnings_surprise",
            data_domain="FUNDAMENTAL.QUALITY",
            context="low_vol",  # 仅不同 name 描述 → 仍同语义？No——context 是语义维。
        )
        # 构造真正同语义：只改展示字段（所有语义维与 s1b 一致，仅大小写/展示不同）
        s1b = _plan(event="earnings_surprise", data_domain="FUNDAMENTAL.QUALITY")
        s2b = _plan(
            event="earnings_surprise",
            data_domain="fundamental.quality",  # 大小写不同 → 语义同
            name="另一个名字",
            description="另一段描述",
        )
        assert s1b.derive_schema_id() == s2b.derive_schema_id()
        assert s1.derive_schema_id() == s1b.derive_schema_id()
        # context 是语义维：改 context 应得到不同 schema
        assert s1.derive_schema_id() != s2.derive_schema_id()

    def test_semantic_field_change_changes_id(self):
        a = _plan(data_domain="FUNDAMENTAL.QUALITY")
        b = _plan(data_domain="FUNDAMENTAL.VALUE")
        assert a.derive_schema_id() != b.derive_schema_id()

    def test_id_deterministic_across_processes(self):
        a = _plan(data_domain="FUNDAMENTAL.QUALITY", horizon="medium")
        b = _plan(data_domain="FUNDAMENTAL.QUALITY", horizon="medium")
        assert a.derive_schema_id() == b.derive_schema_id()
        assert a.derive_schema_id().startswith("SCH_")

    def test_display_name_and_owner_not_part_of_id(self):
        a = SchemaPlan(
            event="x", data_domain="price", name="A", owner="alice"
        )
        b = SchemaPlan(event="x", data_domain="PRICE")
        assert a.derive_schema_id() == b.derive_schema_id()


class TestSchemaIdVersioning:
    def test_version_prefix_in_payload(self):
        p = _plan()
        payload = p.semantic_payload()
        assert payload["version"] == "1"

    def test_canonical_tags_case_normalize(self):
        p = _plan(data_domain="price", horizon="MEDIUM")
        tags = p.canonical_tags()
        assert tags["DataDomain"] == "PRICE"
        assert tags["Horizon"] == "MEDIUM"  # 非 DA 维度保留原文（仅 DataDomain 大写）

    def test_as_tags_round_trip_plan(self):
        p = _plan()
        q = SchemaPlan.from_tags(p.as_tags())
        assert q.derive_schema_id() == p.derive_schema_id()
        assert q.data_domain == "FUNDAMENTAL.QUALITY"


# ---------------------------------------------------------------------------
# ② schema reward 聚合实现层结果 + 置信计数
# ---------------------------------------------------------------------------


def _registry(tmp_path=None):
    if tmp_path is None:
        return SchemaRegistry()
    return SchemaRegistry(str(tmp_path / "schema.sqlite3"))


class TestSchemaRewardAggregation:
    def test_impls_share_schema_and_stats_aggregate(self):
        reg = _registry()
        ident = reg.register_plan(_plan(data_domain="FUNDAMENTAL.QUALITY"))
        sid = ident.schema_id
        # 两个不同公式实现
        reg.register_implementation(
            implementation_id="impl_1", schema_id=sid, formula="rank(roe)"
        )
        reg.register_implementation(
            implementation_id="impl_2", schema_id=sid, formula="rank(ts_mean(roe, 5))"
        )
        st = reg.record_implementation_outcome(
            implementation_id="impl_1",
            outcome=ImplementationOutcome(reward=0.8, elite=True),
        )
        assert st.schema_id == sid
        assert st.n_impl == 2  # 两个实现都登记在该 schema 下
        st = reg.record_implementation_outcome(
            implementation_id="impl_2",
            outcome=ImplementationOutcome(reward=0.5),
        )
        assert st.n_impl == 2
        assert st.n_eval == 2
        assert st.n_elite == 1
        assert st.n_success == 2
        assert st.elite_rate == 0.5
        assert st.has_evidence is True
        # 置信计数：两个实现 → aggregate 在 (0.5+0.8)/2≈0.65 基础上向 0.5 收缩
        assert 0.5 < st.aggregate_reward <= 0.65

    def test_aggregate_shrinks_to_neutral_with_few_impls(self):
        """极少实现 → 收缩接近中性（Part G #29：缺样本绝不当最好）。"""
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(implementation_id="only", schema_id=sid)
        st = reg.record_implementation_outcome(
            implementation_id="only", outcome=ImplementationOutcome(reward=1.0, elite=True)
        )
        # n_impl=1、k=64 → reliability≈0.124 → conf≈0.56（远离 1.0，接近中性）
        assert st.aggregate_reward < 0.7
        assert st.aggregate_reward > 0.5

    def test_aggregate_approaches_raw_with_many_impls(self):
        """多实现 → 收缩弱化，aggregate 接近实现层均值。"""
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        for i in range(40):
            impl_id = f"m{i}"
            reg.register_implementation(implementation_id=impl_id, schema_id=sid)
            reg.record_implementation_outcome(
                implementation_id=impl_id,
                outcome=ImplementationOutcome(reward=0.8, elite=(i % 2 == 0)),
            )
        st = reg.stats_for(sid)
        assert st.n_impl == 40
        assert st.n_elite == 20
        # 40 个实现 → 收缩显著弱化（reliability≈0.62）；aggregate 明显高于 0.5
        assert st.aggregate_reward > 0.6
        assert st.aggregate_reward < 0.75

    def test_schema_stats_independent_of_formula_layer(self):
        """registry 不复制公式层/身份层：查不到公式参数/身份（只存 formula 文本）。"""
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(
            implementation_id="i1", schema_id=sid, formula="rank(roe)", meta={"param": 5}
        )  # meta 走 JSON 列（见 registry._SCHEMA implementations.meta）
        impls = reg.implementations_of(sid)
        assert len(impls) == 1
        assert impls[0]["formula"] == "rank(roe)"
        # 公式层身份（canonical_hash 等）不在 schema registry 表结构内
        cols = {k for impl in impls for k in impl}
        assert "canonical_ast_hash" not in cols


class TestFailedImplementationDoesNotFailSchema:
    def test_single_failed_impl_schema_not_failed(self):
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(implementation_id="ok1", schema_id=sid)
        reg.register_implementation(implementation_id="bad1", schema_id=sid)
        st = reg.record_implementation_outcome(
            implementation_id="ok1", outcome=ImplementationOutcome(reward=0.7)
        )
        assert st.n_impl == 2
        # bad1 FE 校验失败：mark_implementation_failed（不标 schema failed）
        st = reg.mark_implementation_failed(
            implementation_id="bad1", failure_reason="FE validation failed"
        )
        assert st.n_failed_impl == 1
        assert st.is_failed is False  # 单个失败实现 ≠ schema failed
        assert st.aggregate_reward > 0.5  # 剩余成功实现仍贡献分数

    def test_explicit_mark_schema_failed(self):
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(implementation_id="x", schema_id=sid)
        reg.record_implementation_outcome(
            implementation_id="x", outcome=ImplementationOutcome(reward=0.6)
        )
        st = reg.mark_schema_failed(sid, reason="direction died")
        assert st.is_failed is True

    def test_all_impls_failed_flips_schema_failed(self):
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(implementation_id="b1", schema_id=sid)
        reg.register_implementation(implementation_id="b2", schema_id=sid)
        reg.mark_implementation_failed(implementation_id="b1", failure_reason="bad dsl")
        st = reg.mark_implementation_failed(implementation_id="b2", failure_reason="bad dsl")
        assert st.n_failed_impl == 2
        assert st.is_failed is True  # 全部实现失败 → schema 视为 failed

    def test_implementation_record_failed_property(self):
        rec = ImplementationRecord(
            implementation_id="r", schema_id="S", formula="x", status="failed"
        )
        assert rec.failed is True
        assert ImplementationRecord(
            implementation_id="r2", schema_id="S", formula="x"
        ).failed is False


class TestSchemaStatsContract:
    def test_defaults_neutral(self):
        st = SchemaStats(schema_id="S")
        assert st.aggregate_reward == 0.5
        assert st.elite_rate is None
        assert st.success_rate is None
        assert st.has_evidence is False

    def test_as_dict_round_trip(self):
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(implementation_id="d1", schema_id=sid)
        st = reg.record_implementation_outcome(
            implementation_id="d1", outcome=ImplementationOutcome(reward=0.6, elite=True)
        )
        d = st.as_dict()
        assert d["n_impl"] == 1
        assert d["elite_rate"] == 1.0
        assert d["schema_id"] == sid
        assert set(d) >= {
            "schema_id", "version", "n_impl", "n_eval", "n_success", "n_elite",
            "n_failed_impl", "aggregate_reward", "elite_rate", "success_rate",
            "is_failed", "updated_at",
        }


class TestRegistryPersistence:
    def test_file_db_reopen_keeps_stats(self, tmp_path):
        # _registry(tmp_path) 内部拼接 schema.sqlite3（不能把整个 db 路径传进去）
        reg = _registry(tmp_path)
        db = tmp_path / "schema.sqlite3"
        import os as _os

        assert _os.path.isfile(db), f"after first reg, schema.sqlite3 is dir={_os.path.isdir(db)}"
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        reg.register_implementation(implementation_id="p1", schema_id=sid)
        reg.record_implementation_outcome(
            implementation_id="p1", outcome=ImplementationOutcome(reward=0.75, elite=True)
        )
        reg.close()
        assert _os.path.isfile(db), f"after close, schema.sqlite3 is dir={_os.path.isdir(db)}"
        # 重开（新连接）→ 统计仍在
        reg2 = SchemaRegistry(str(db))
        st = reg2.stats_for(sid)
        assert st.n_impl == 1
        assert st.n_elite == 1
        assert reg2.plan_for(sid) is not None
        reg2.close()

    def test_memory_default_isolation(self):
        a = SchemaRegistry()
        b = SchemaRegistry()
        ia = a.register_plan(_plan(event="ev_a"))
        b.register_plan(_plan(event="ev_b"))
        assert a.schema_ids() == [ia.schema_id]  # 内存库相互隔离

    def test_saturation_supports_retriever(self):
        reg = _registry()
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        assert reg.saturation(sid) is None  # 无实现 → None（不编 0）
        reg.register_implementation(implementation_id="s1", schema_id=sid)
        # 统计行在 record_outcome 后才生成（实现登记 ≠ 已评估）
        reg.record_implementation_outcome(
            implementation_id="s1", outcome=ImplementationOutcome(reward=0.6)
        )
        sat = reg.saturation(sid)
        assert sat is not None and 0 < sat < 1.0


# ---------------------------------------------------------------------------
# ③ 语义字段对照 DA taxonomy 校验，非法即拒
# ---------------------------------------------------------------------------


class TestSemanticValidation:
    def test_valid_domain_passes(self):
        p = _plan(data_domain="FUNDAMENTAL.QUALITY")
        validate_schema_plan(p)
        assert validate_domain_token("PRICE", DEFAULT_DOMAIN_VOCABULARY) == "PRICE"

    def test_case_insensitive_domain_normalize(self):
        assert validate_domain_token("price", DEFAULT_DOMAIN_VOCABULARY) == "PRICE"
        assert validate_domain_token("Fundamental.Quality", DEFAULT_DOMAIN_VOCABULARY) == "FUNDAMENTAL.QUALITY"
        assert validate_domain_token("fUNDAMENTAL.gROWTH", DEFAULT_DOMAIN_VOCABULARY) == "FUNDAMENTAL.GROWTH"

    def test_invalid_domain_rejected(self):
        # 自造/不在 DA 词表的 domain（如 VOLATILITY / FUNDAMENTAL.VOLATILITY）→ 拒
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(data_domain="VOLATILITY"))
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(data_domain="FUNDAMENTAL.VOLATILITY"))
        with pytest.raises(SchemaValidationError):
            validate_domain_token("SIZE", DEFAULT_DOMAIN_VOCABULARY)  # DA 词表无裸 SIZE

    def test_invalid_horizon_rejected(self):
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(horizon="fortnightly"))
        # 空 horizon = 放行
        validate_schema_plan(_plan(horizon=""))

    def test_invalid_direction_rejected(self):
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(direction="sideways"))
        validate_schema_plan(_plan(direction="short"))

    def test_invalid_normalization_rejected(self):
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(normalization="log_return"))
        validate_schema_plan(_plan(normalization="cross_section"))

    def test_invalid_output_and_tradability_rejected(self):
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(output="weird"))
        with pytest.raises(SchemaValidationError):
            validate_schema_plan(_plan(tradability="concentrated"))
        validate_schema_plan(_plan(output="rank", tradability="liquid"))

    def test_free_text_dims_do_not_validate(self):
        # Event/Context/Qualities 是自由文本，不校验词表
        validate_schema_plan(_plan(event="任意机制描述", context="bull", qualities="quality"))

    def test_unknown_or_empty_allowed(self):
        validate_schema_plan(SchemaPlan(data_domain="PRICE"))
        validate_schema_plan(SchemaPlan())  # 全空 → 放行（unknown schema）


# ---------------------------------------------------------------------------
# 契约：ImplementationRecord / 计数可查询
# ---------------------------------------------------------------------------


class TestImplementationOutcomeContract:
    def test_outcome_failed_flag(self):
        assert ImplementationOutcome(reward=0.9).failed is False
        assert ImplementationOutcome(reward=0.0, failed=True).failed is True

    def test_registry_counters_queryable(self, tmp_path):
        reg = SchemaRegistry(str(tmp_path / "s.sqlite3"))
        ident = reg.register_plan(_plan())
        sid = ident.schema_id
        for i in range(3):
            reg.register_implementation(implementation_id=f"q{i}", schema_id=sid)
        reg.record_implementation_outcome(
            implementation_id="q0", outcome=ImplementationOutcome(reward=0.9, elite=True)
        )
        reg.mark_implementation_failed(implementation_id="q1", failure_reason="invalid")
        summary = reg.summary()
        assert len(summary) == 1
        row = summary[0]
        assert row["n_impl"] == 3
        assert row["n_elite"] == 1
        assert row["n_failed_impl"] == 1
        assert row["schema_id"] == sid
