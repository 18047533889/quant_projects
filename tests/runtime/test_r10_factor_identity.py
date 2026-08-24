# -*- coding: utf-8 -*-
"""R10 #16 / #17 / #47 / #48 —— FactorSemanticIdentity + materializer checkpoint 身份。

覆盖：
- ``FactorSemanticIdentity.identity_digest``：确定性；任一字段变化 → digest 变化
  （source contract / operator contract / field contract / calendar / universe /
   price_basis / dialect_version）。
- ``compute_factor_identity``：从计划（IR/PlanNode）+ ctx 推导身份，覆盖字段生效。
- #17: production materialize 提供 ``__no_hash_provided__``（或什么都不提供）→
  ``FactorIdentityMismatch``。
- #47: 断点续写绑定身份 —— source-snapshot 指纹变化时该分区不得被 resume 跳过；
  checkpoint 指纹完全匹配时可跳过；指纹缺失（旧 checkpoint）在 production 下重算。
- #48: production incremental 全 NaN 输出写 tombstone（is_valid=0）清除旧有限值，
  即使调用方未传 ``null_overwrite=True``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.factor_identity import (
    NO_FACTOR_IDENTITY,
    FactorIdentityMismatch,
    FactorSemanticIdentity,
    checkpoint_fingerprint,
    checkpoint_fingerprint_matches,
    compute_factor_identity,
    partition_input_fingerprint,
)
from factor_engine.storage.materializer import ParquetMaterializer


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _identity(**overrides):
    """构造一个确定性基础身份；``overrides`` 覆盖任意字段。"""
    base = dict(
        ir_hash="ir-hash-1",
        operator_contract_hash="op-hash-1",
        field_contract_hash="field-hash-1",
        source_contract_hash="src-hash-1",
        source_dependency_hash="src-dep-hash-1",
        market="A",
        calendar="SSE",
        timezone="Asia/Shanghai",
        universe="ALL",
        price_basis="RAW",
        pit_policy="enforce",
        decision_time_policy="eod",
        dialect="native",
        dialect_version="1.0",
    )
    base.update(overrides)
    return FactorSemanticIdentity(**base)


def _series(rows):
    """rows: [(date_str, asset, value)] -> MultiIndex(timestamp, instrument) Series。"""
    frame = pd.DataFrame(
        [(pd.Timestamp(d), a, v) for d, a, v in rows],
        columns=["timestamp", "instrument", "value"],
    )
    return frame.set_index(["timestamp", "instrument"])["value"]


def _ir_node():
    return IRNode(
        op="ts_mean",
        attrs={"d": 20, "min_periods": 1},
        inputs=(IRNode(op="column", attrs={"name": "close"}),),
    )


# ---------------------------------------------------------------------------
# FactorSemanticIdentity.identity_digest
# ---------------------------------------------------------------------------


class TestIdentityDigest:
    def test_digest_is_deterministic(self):
        a = _identity()
        b = _identity()
        assert a.identity_digest() == b.identity_digest()
        assert a.identity_digest() == _identity().identity_digest()

    def test_digest_changes_when_source_contract_changes(self):
        assert (
            _identity(source_contract_hash="other-src").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_when_operator_contract_changes(self):
        assert (
            _identity(operator_contract_hash="other-op").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_when_field_contract_changes(self):
        assert (
            _identity(field_contract_hash="other-field").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_when_calendar_changes(self):
        assert (
            _identity(calendar="SZSE").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_when_universe_changes(self):
        assert (
            _identity(universe="CSI300").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_when_price_basis_changes(self):
        assert (
            _identity(price_basis="CONTINUOUS").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_when_dialect_version_changes(self):
        assert (
            _identity(dialect_version="2.0").identity_digest()
            != _identity().identity_digest()
        )

    def test_digest_changes_for_every_semantic_field(self):
        """遍历所有可空语义字段：逐一变化 → digest 必须不同。"""
        base = _identity()
        for field in (
            "market",
            "calendar",
            "timezone",
            "universe",
            "price_basis",
            "pit_policy",
            "decision_time_policy",
            "dialect",
            "dialect_version",
        ):
            mutated = _identity(**{field: f"changed-{field}"})
            assert mutated.identity_digest() != base.identity_digest(), (
                f"identity digest did not change for field {field!r}"
            )

    def test_to_dict_from_dict_roundtrip_preserves_digest(self):
        identity = _identity()
        restored = FactorSemanticIdentity.from_dict(identity.to_dict())
        assert restored == identity
        assert restored.identity_digest() == identity.identity_digest()


# ---------------------------------------------------------------------------
# compute_factor_identity
# ---------------------------------------------------------------------------


class TestComputeFactorIdentity:
    def test_computes_from_ir_with_ctx_overrides(self):
        plan = _ir_node()
        ctx = {
            "market": "A",
            "calendar": "SSE",
            "timezone": "Asia/Shanghai",
            "universe": "ALL",
            "price_basis": "RAW",
            "dialect": "native",
            "dialect_version": "1.0",
        }
        identity = compute_factor_identity(plan, ctx=ctx)
        # ir_hash 来自计划结构；覆盖字段应生效。
        assert identity.ir_hash == compute_ir_hash_of(plan)
        assert identity.market == "A"
        assert identity.calendar == "SSE"
        assert identity.universe == "ALL"
        assert identity.price_basis == "RAW"
        assert identity.dialect_version == "1.0"
        # 确定性
        assert (
            compute_factor_identity(plan, ctx=ctx).identity_digest()
            == identity.identity_digest()
        )

    def test_ctx_overrides_direct_hash_fields(self):
        plan = _ir_node()
        ctx = {
            "ir_hash": "custom-ir",
            "operator_contract_hash": "custom-op",
            "field_contract_hash": "custom-field",
            "source_contract_hash": "custom-src",
            "source_dependency_hash": "custom-dep",
        }
        identity = compute_factor_identity(plan, ctx=ctx)
        assert identity.ir_hash == "custom-ir"
        assert identity.operator_contract_hash == "custom-op"
        assert identity.field_contract_hash == "custom-field"
        assert identity.source_contract_hash == "custom-src"
        assert identity.source_dependency_hash == "custom-dep"

    def test_semantic_fields_read_from_plan_semantic_attrs(self):
        plan = IRNode(
            op="ts_mean",
            attrs={"d": 5},
            inputs=(IRNode(op="column", attrs={"name": "close"}),),
            semantic_attrs={"universe_id": "CSI300", "price_basis": "CONTINUOUS"},
        )
        identity = compute_factor_identity(plan)
        assert identity.universe == "CSI300"
        assert identity.price_basis == "CONTINUOUS"

    def test_source_contract_hash_from_data_source_config(self):
        plan = _ir_node()
        identity = compute_factor_identity(plan, ctx={"data_source_config": {"type": "x"}})
        assert identity.source_contract_hash != compute_factor_identity(
            plan, ctx={"data_source_config": {"type": "y"}}
        ).source_contract_hash


def compute_ir_hash_of(plan):
    from factor_engine.storage.catalog import compute_ir_hash

    return compute_ir_hash(plan)


# ---------------------------------------------------------------------------
# #47 checkpoint 指纹辅助
# ---------------------------------------------------------------------------


class TestCheckpointFingerprint:
    def test_matches_equal_fingerprints(self):
        fp = checkpoint_fingerprint(
            identity_digest="d",
            source_snapshot="s1",
            source_dependency_hash="dep",
            partition_input_fingerprint="p1",
            run_generation="g1",
        )
        assert checkpoint_fingerprint_matches(dict(fp), fp)

    def test_mismatch_on_any_component(self):
        base = checkpoint_fingerprint(
            identity_digest="d",
            source_snapshot="s1",
            source_dependency_hash="dep",
            partition_input_fingerprint="p1",
            run_generation="g1",
        )
        for key in base:
            changed = dict(base)
            changed[key] = "other"
            assert not checkpoint_fingerprint_matches(base, changed), key

    def test_partition_input_fingerprint_deterministic_and_sensitive(self):
        frame1 = pd.DataFrame(
            {"datetime": pd.to_datetime(["2024-01-01"]), "asset": ["A"], "value": [1.0]}
        )
        frame2 = pd.DataFrame(
            {"datetime": pd.to_datetime(["2024-01-01"]), "asset": ["A"], "value": [1.0]}
        )
        assert partition_input_fingerprint(frame1) == partition_input_fingerprint(frame2)
        frame3 = pd.DataFrame(
            {"datetime": pd.to_datetime(["2024-01-01"]), "asset": ["A"], "value": [2.0]}
        )
        assert partition_input_fingerprint(frame1) != partition_input_fingerprint(frame3)


# ---------------------------------------------------------------------------
# #17 production materialize 禁止 __no_hash_provided__
# ---------------------------------------------------------------------------


class TestProductionIdentityGate:
    def test_production_materialize_without_identity_raises(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])
        with pytest.raises(FactorIdentityMismatch):
            mat.materialize(
                factor_id="prod_no_identity",
                result=series,
                production=True,
            )

    def test_production_materialize_with_explicit_sentinel_raises(self, tmp_path):
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])
        with pytest.raises(FactorIdentityMismatch):
            mat.materialize(
                factor_id="prod_no_identity",
                result=series,
                ast_hash=NO_FACTOR_IDENTITY,
                production=True,
            )

    def test_production_local_write_rejected(self, tmp_path):
        """#收官轮 P0（Integration）：production + direct-local 落盘 → 统一
        orchestrator 拒绝（guard 不再只在 LocalParquetWriteTarget，materialize()
        主路径也能绕过去）。production 必须走 staging→publish。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])
        with pytest.raises(ValueError, match="production 禁止 direct-local"):
            mat.materialize(
                factor_id="prod_with_identity",
                result=series,
                ast_hash="abc123",
                production=True,
            )

    def test_production_materialize_with_identity_to_staging_succeeds(self, tmp_path, monkeypatch):
        """production + identity 走 staging（非 direct-local）成功，权威水位线 defer。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])

        class _FakeStore:
            def upsert(self, dataset, table, **kwargs):
                return {"rows_upserted": table.num_rows}

        monkeypatch.setattr("data_access.get_store", lambda: _FakeStore())
        out = mat.materialize(
            factor_id="prod_with_identity",
            result=series,
            ast_hash="abc123",
            production=True,
            write_target="staging",
        )
        assert out["rows_written"] == 1
        assert out["watermark_deferred"] is True
        assert out["watermark"] is None

    def test_research_materialize_without_identity_still_allowed(self, tmp_path):
        """research（非 production）保留 sentinel 兜底（向后兼容）。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])
        out = mat.materialize(
            factor_id="research_no_identity",
            result=series,
            production=False,
        )
        assert out["rows_written"] == 1
        info = mat.catalog.get_factor_info("research_no_identity")
        assert info["ast_hash"] == NO_FACTOR_IDENTITY


# ---------------------------------------------------------------------------
# #47 断点续写绑定身份
# ---------------------------------------------------------------------------


class TestResumeBoundToIdentity:
    def test_resume_skips_when_fingerprint_matches(self, tmp_path):
        """同一指纹 + status=success → resume 跳过该分区。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])
        out1 = mat.materialize(
            factor_id="resume_match",
            result=series,
            ast_hash="h1",
            data_snapshot_id="snap-1",
        )
        assert out1["partitions"] == [2024]
        assert mat.catalog.get_partition_checkpoint_by_key("resume_match", "year=2024")[
            "status"
        ] == "success"

        out2 = mat.materialize(
            factor_id="resume_match",
            result=series,
            ast_hash="h1",
            data_snapshot_id="snap-1",
            resume=True,
        )
        assert out2["partitions_skipped"] == [2024]
        assert out2["partitions"] == []

    def test_resume_not_skipped_when_source_snapshot_changed(self, tmp_path):
        """source-snapshot 指纹变化 → 该分区必须重算，不得被 resume 跳过。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series = _series([("2024-01-15", "A", 1.0)])
        mat.materialize(
            factor_id="snap_change",
            result=series,
            ast_hash="h1",
            data_snapshot_id="snap-1",
        )
        ck = mat.catalog.get_partition_checkpoint_by_key("snap_change", "year=2024")
        assert ck["status"] == "success"

        out = mat.materialize(
            factor_id="snap_change",
            result=series,
            ast_hash="h1",
            data_snapshot_id="snap-2",
            resume=True,
        )
        assert out["partitions_skipped"] == []
        assert 2024 in out["partitions"]

    def test_resume_recomputes_when_partition_input_changed(self, tmp_path):
        """同一分区输入数据变化 → 分区输入指纹不同 → 重算。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        series1 = _series([("2024-01-15", "A", 1.0)])
        mat.materialize(
            factor_id="part_input_change",
            result=series1,
            ast_hash="h1",
            data_snapshot_id="snap-1",
        )
        # 同分区、不同值 → 输入指纹变化
        series2 = _series([("2024-01-15", "A", 99.0)])
        out = mat.materialize(
            factor_id="part_input_change",
            result=series2,
            ast_hash="h1",
            data_snapshot_id="snap-1",
            resume=True,
        )
        assert out["partitions_skipped"] == []
        df = pd.read_parquet(
            tmp_path / "factors" / "part_input_change" / "year=2024" / "data.parquet"
        )
        assert df["value"].iloc[0] == pytest.approx(99.0)

    def test_production_missing_fingerprint_recomputes(self, tmp_path):
        """旧 checkpoint 无指纹文件：production 下必须重算而非跳过（R10 #47）。

        #收官轮 P0（Integration）：production + direct-local 已由统一 orchestrator
        拒绝（走 staging→publish），故此处直接对 ``_checkpoint_skippable`` 判定
        逻辑做单元断言——无指纹 sidecar 时 production → 不可跳过（重算）。
        """
        mat = ParquetMaterializer(lake_root=tmp_path)
        checkpoint = {"status": "success"}
        partition_dir = tmp_path / "factors" / "legacy_ck" / "year=2024"
        partition_dir.mkdir(parents=True)
        fingerprint = checkpoint_fingerprint(
            identity_digest="digest",
            source_snapshot="snap-1",
            source_dependency_hash="dep",
            partition_input_fingerprint="inp",
            run_generation="0",
        )
        # 无 .identity.json（旧 checkpoint）：production 必须重算
        assert (
            ParquetMaterializer._checkpoint_skippable(
                checkpoint,
                fingerprint,
                production=True,
                partition_dir=partition_dir,
            )
            is False
        )
        # research 保留旧行为（status==success 即跳过）
        assert (
            ParquetMaterializer._checkpoint_skippable(
                checkpoint,
                fingerprint,
                production=False,
                partition_dir=partition_dir,
            )
            is True
        )


# ---------------------------------------------------------------------------
# #48 all-NaN incremental → tombstone（无需 null_overwrite）
# ---------------------------------------------------------------------------


class TestAllNaNIncrementalTombstone:
    def test_all_nan_incremental_clears_stale_value_without_null_overwrite(self, tmp_path):
        """全 NaN 重算 + force_tombstones 必须写 tombstone 清除旧有限值（R10 #48）。

        #收官轮 P0（Integration）：production + direct-local 由 orchestrator 拒绝，
        本地 lake 的 tombstone 落盘用 research 模式 + 显式 force_tombstones 验证
        （tombstone 行为本身不依赖 production 标志）。
        """
        mat = ParquetMaterializer(lake_root=tmp_path)
        mat.materialize(
            factor_id="nan_incr",
            result=_series([("2024-01-15", "A", 1.0)]),
            ast_hash="h1",
        )
        out = mat.materialize(
            factor_id="nan_incr",
            result=_series([("2024-01-15", "A", np.nan)]),
            ast_hash="h1",
            force_tombstones=True,
        )
        assert out["rows_written"] == 1
        df = pd.read_parquet(
            tmp_path / "factors" / "nan_incr" / "year=2024" / "data.parquet"
        )
        assert len(df) == 1
        assert pd.isna(df["value"].iloc[0])
        assert df["is_valid"].iloc[0] == 0

    def test_all_nan_incremental_auto_detects_tombstones(self, tmp_path):
        """production + lineage mode=incremental → 自动写 tombstone（不传 force_tombstones）。

        #收官轮 P0：auto-detect 逻辑直接对 ``_resolve_force_tombstones`` 断言
        （production 判定不依赖 direct-local 落盘）。
        """
        mat = ParquetMaterializer(lake_root=tmp_path)
        lineage_incremental = {
            "run_id": "run-incr-1",
            "factor_id": "nan_incr2",
            "ast_hash": "h1",
            "extra": {"mode": "incremental"},
        }
        assert (
            ParquetMaterializer._resolve_force_tombstones(
                None,
                production=True,
                run_lineage=lineage_incremental,
            )
            is True
        )
        assert (
            ParquetMaterializer._resolve_force_tombstones(
                None,
                production=False,
                run_lineage=lineage_incremental,
            )
            is False
        )
        # 显式 force_tombstones 优先于 production 推断
        assert (
            ParquetMaterializer._resolve_force_tombstones(
                True,
                production=False,
                run_lineage=lineage_incremental,
            )
            is True
        )
        # 本地 lake 落盘验证（research + 显式 force_tombstones）
        mat.materialize(
            factor_id="nan_incr2",
            result=_series([("2024-01-15", "A", 1.0)]),
            ast_hash="h1",
        )
        out = mat.materialize(
            factor_id="nan_incr2",
            result=_series([("2024-01-15", "A", np.nan)]),
            ast_hash="h1",
            force_tombstones=True,
        )
        assert out["force_tombstones"] is True
        df = pd.read_parquet(
            tmp_path / "factors" / "nan_incr2" / "year=2024" / "data.parquet"
        )
        assert len(df) == 1
        assert pd.isna(df["value"].iloc[0])
        assert df["is_valid"].iloc[0] == 0

    def test_research_all_nan_without_tombstone_intent_still_skips(self, tmp_path):
        """research 普通全 NaN（无 incremental 标记）保持 rows_written=0（历史行为）。"""
        mat = ParquetMaterializer(lake_root=tmp_path)
        out = mat.materialize(
            factor_id="nan_skip",
            result=_series([("2024-01-15", "A", np.nan)]),
            ast_hash="h1",
            production=False,
        )
        assert out["rows_written"] == 0
        assert out["watermark"] is None
