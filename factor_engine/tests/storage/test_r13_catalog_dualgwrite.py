# -*- coding: utf-8 -*-
"""R13 审计收口：catalog typed-JSON + deleted-factor reuse + dual-write 统一契约。

覆盖：
  * NEW-P0-44  生产全量定义 typed JSON 序列化（未知对象 → CatalogSerializationError）
  * NEW-P0-45  catalog JSON 损坏 → CatalogCorruptionError（不再伪装成 {}）
  * NEW-P1-74  CatalogJsonField[T]：schema_version / strict decode / migration / checksum
  * NEW-P0-56/57  deleted-factor 复用守卫（墓碑 + rebuild=True）+ 权威 catalog 存全量 SHA-256
  * NEW-P0-59  Parquet vs ClickHouse factor_version 统一（同一 full digest 派生）
  * NEW-P0-60  tombstone（deleted_keys）传播到 ClickHouse
  * NEW-P1-61  generation / transaction_id 共享 token 贯穿 commit 路径
"""

from __future__ import annotations

import pandas as pd
import pytest

from runtime.factor_identity import FactorSemanticIdentity
from runtime.reconcile.dual_write_service import (
    DualWriteRequest,
    MaterializationDelta,
    _series_with_tombstones,
    append_clickhouse_to_summary,
)
from storage.catalog import (
    CatalogJsonField,
    FactorCatalog,
    _parse_json_field,
    catalog_strict_dumps,
)
from storage.exceptions import (
    CatalogCorruptionError,
    CatalogSerializationError,
    FactorRetiredError,
)


def _mk_series(
    dates=("2024-01-02", "2024-01-03"),
    assets=("AAA", "BBB"),
):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(list(dates)), list(assets)],
        names=["timestamp", "instrument"],
    )
    return pd.Series([float(i) for i in range(len(idx))], index=idx)


# ---------------------------------------------------------------------------
# NEW-P0-44 typed JSON serialization
# ---------------------------------------------------------------------------


def test_strict_dumps_rejects_callable_in_production(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    with pytest.raises(CatalogSerializationError, match="typed JSON"):
        cat.register(
            "f_strict",
            author="t",
            frequency="1d",
            ast_hash="a" * 64,
            data_source_config={"fn": lambda x: x},
            production=True,
        )


def test_strict_dumps_rejects_enum_like_object(tmp_path):
    import enum

    class _E(enum.Enum):
        A = "a"

    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    with pytest.raises(CatalogSerializationError):
        cat.register(
            "f_enum",
            author="t",
            frequency="1d",
            ast_hash="b" * 64,
            data_source_config={"kind": _E.A},
            production=True,
        )
    # research 保留历史容错（default=str 字符串化，不 raise）
    cat.register(
        "f_enum",
        author="t",
        frequency="1d",
        ast_hash="b" * 64,
        data_source_config={"kind": _E.A},
        production=False,
    )
    info = cat.get_factor_info("f_enum")
    assert "kind" in _parse_json_field(info["data_source_json"])


def test_catalog_strict_dumps_top_level():
    assert catalog_strict_dumps({"a": [1, 2, None], "b": "x"}) == (
        '{"a": [1, 2, null], "b": "x"}'
    )


# ---------------------------------------------------------------------------
# NEW-P0-45 corrupt JSON → CatalogCorruptionError
# ---------------------------------------------------------------------------


def test_parse_json_corrupt_raises_in_strict(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    # 直接往 factor_run.extra_json 写坏 JSON
    cat._conn.execute(
        "INSERT INTO factor_run "
        "(run_id, factor_id, ast_hash, created_at, extra_json) "
        "VALUES ('run1', 'f1', 'h', '2024-01-01', '{broken')"
    )
    cat._conn.commit()
    # research（缺省）：容错返回 {}
    assert _parse_json_field("{broken") == {}
    # production / strict：抛 CatalogCorruptionError，绝不伪装成 config 缺失
    with pytest.raises(CatalogCorruptionError):
        _parse_json_field("{broken", strict=True)


def test_read_corrupt_catalog_field_production_raises(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.register(
        "f_corrupt",
        author="t",
        frequency="1d",
        ast_hash="c" * 64,
        data_source_config={"ds": "ok"},
    )
    # 模拟落库后 data_source_json 被外部写坏
    cat._conn.execute(
        "UPDATE factor_registry SET data_source_json = '{not json' "
        "WHERE factor_id = 'f_corrupt'"
    )
    cat._conn.commit()
    with pytest.raises(CatalogCorruptionError):
        _parse_json_field(
            cat.get_factor_info("f_corrupt")["data_source_json"], strict=True
        )
    # list_dual_write_failures 在生产下遇到坏 extra_json 也应 fail-closed
    from runtime import production_policy

    cat._conn.execute(
        "INSERT INTO factor_run "
        "(run_id, factor_id, ast_hash, created_at, extra_json) "
        "VALUES ('run_bad', 'f_corrupt', 'h', '2024-01-01', '{broken')"
    )
    cat._conn.commit()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(production_policy, "is_production_mode", lambda mode=None: True)
    try:
        with pytest.raises(CatalogCorruptionError):
            cat.list_dual_write_failures(factor_id="f_corrupt")
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# NEW-P1-74 CatalogJsonField[T]
# ---------------------------------------------------------------------------


def test_catalog_json_field_roundtrip_with_schema_and_checksum():
    field = CatalogJsonField({"a": 1, "b": [1, 2]}, schema_version=2)
    raw = field.dumps()
    loaded = CatalogJsonField.loads(raw, schema_version=2, strict=True)
    assert loaded.value == {"a": 1, "b": [1, 2]}
    assert loaded.schema_version == 2
    assert "checksum" in raw


def test_catalog_json_field_migration_legacy_bare_payload():
    legacy = {"universe": "CSI300", "freq": "1d"}  # 无信封 legacy 裸 JSON
    raw = __import__("json").dumps(legacy)
    loaded = CatalogJsonField.loads(
        raw,
        schema_version=3,
        strict=True,
        migrate=lambda v: {**v, "migrated": True},
    )
    assert loaded.value == {"universe": "CSI300", "freq": "1d", "migrated": True}
    assert loaded.schema_version == 3


def test_catalog_json_field_checksum_mismatch_raises():
    import json

    field = CatalogJsonField({"a": 1})
    raw = json.loads(field.dumps())
    raw["value"] = {"a": 999}  # 篡改
    with pytest.raises(CatalogCorruptionError, match="checksum"):
        CatalogJsonField.loads(json.dumps(raw), strict=True)


def test_catalog_json_field_corrupt_raises_in_strict():
    with pytest.raises(CatalogCorruptionError):
        CatalogJsonField.loads("{broken", strict=True)
    # research：返回 None 值（不 raise）
    assert CatalogJsonField.loads("{broken").value is None


# ---------------------------------------------------------------------------
# NEW-P0-56/57 deleted-factor reuse guard + full SHA-256 in authoritative catalog
# ---------------------------------------------------------------------------


def test_register_stores_full_semantic_digest(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    full = "d" * 64
    cat.register(
        "f_full",
        author="t",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest=full,
    )
    assert cat.get_factor_info("f_full")["factor_version"] == full
    assert len(cat.get_factor_info("f_full")["factor_version"]) == 64


def test_register_upgrades_legacy_16char_to_full(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    full = "e" * 64
    cat.register(
        "f_upgrade",
        author="t",
        frequency="1d",
        ast_hash=full,
    )
    # legacy 16 位前缀升级为 full SHA-256（NEW-P0-57 权威 catalog 存全量）
    assert cat.get_factor_info("f_upgrade")["factor_version"] == full


def test_register_preserves_full_digest_when_16char_prefix_collides(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    v1 = "ab" * 32  # 64 hex
    v2 = "ab" * 16 + "cd" * 16  # 与 v1 同 16 位前缀、full digest 不同
    assert v1[:16] == v2[:16] and v1 != v2
    cat.register("f_coll", author="t", frequency="1d", ast_hash="a" * 64,
                 semantic_identity_digest=v1)
    cat.register("f_coll", author="t", frequency="1d", ast_hash="a" * 64,
                 semantic_identity_digest=v2)
    assert cat.get_factor_info("f_coll")["factor_version"] == v2


def test_delete_factor_tombstone_blocks_reuse(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.register(
        "f_reuse",
        author="t",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest="f" * 64,
    )
    cat.delete_factor("f_reuse")
    assert cat.is_factor_retired("f_reuse") is True
    assert cat.get_factor_info("f_reuse") is None
    # 直接复用 → 拒绝（旧世代物理分区仍在磁盘上）
    with pytest.raises(FactorRetiredError, match="f_reuse"):
        cat.register(
            "f_reuse",
            author="t",
            frequency="1d",
            ast_hash="a" * 64,
            semantic_identity_digest="f" * 64,
        )
    # 显式 rebuild=True → 清除墓碑，允许新世代
    cat.register(
        "f_reuse",
        author="t",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest="f" * 64,
        rebuild=True,
    )
    assert cat.is_factor_retired("f_reuse") is False
    assert cat.get_factor_info("f_reuse") is not None


def test_deleted_factor_reuse_must_not_expose_old_partitions(tmp_path):
    """端到端：materialize → delete_factor → 再次 materialize 必须被墓碑拒绝，
    旧分区（year=2024）绝不可能与新数据混列。"""
    from storage.materialize.materializer import ParquetMaterializer

    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake, staging_dataset="stg")
    s = _mk_series()
    mat.materialize(
        "f_old",
        result=s,
        ir_node=None,
        ast_hash="a" * 64,
        author="t",
        frequency="1d",
    )
    assert (lake / "factors" / "f_old").exists()
    mat.catalog.delete_factor("f_old")
    assert mat.catalog.is_factor_retired("f_old") is True
    # 复用同一 factor_id 物化 → 被墓碑拒绝，无法写入新世代数据
    with pytest.raises(FactorRetiredError):
        mat.materialize(
            "f_old",
            result=_mk_series(dates=("2026-01-02", "2026-01-03")),
            ir_node=None,
            ast_hash="b" * 64,
            author="t",
            frequency="1d",
        )


# ---------------------------------------------------------------------------
# NEW-P0-59 Parquet vs ClickHouse factor_version 统一
# ---------------------------------------------------------------------------


def _fake_ch_target(captured):
    """返回一个记录 (factor_id, series, factor_version) 的假 ClickHouseWriteTarget。"""
    from storage.write_targets import ClickHouseWriteTarget as RealCH

    class FakeCH:
        def __init__(self, *a, **k):
            self.table = k.get("table") or "factor_values"

        def write_factor_series(self, factor_id, result, *, factor_version="", **kw):
            captured["factor_id"] = factor_id
            captured["series"] = result
            captured["factor_version"] = factor_version
            return {
                "factor_id": factor_id,
                "table": self.table,
                "rows_written": len(result),
                "database": "default",
            }

    return FakeCH


def test_dual_write_version_unified_across_sinks(tmp_path, monkeypatch):
    """同一 AST、universe=CSI300 vs CSI500 → 两侧（Parquet/CH）版本一致且互不相同。"""
    import storage.write_targets

    captured = {}
    monkeypatch.setattr(
        storage.write_targets, "ClickHouseWriteTarget", _fake_ch_target(captured)
    )

    ast_hash = "a" * 64
    identity_csi300 = FactorSemanticIdentity(
        ir_hash=ast_hash,
        operator_contract_hash="op1",
        field_contract_hash="fd1",
        source_contract_hash="sc1",
        source_dependency_hash="sd1",
        universe="CSI300",
        frequency="1d",
    )
    identity_csi500 = FactorSemanticIdentity(
        ir_hash=ast_hash,
        operator_contract_hash="op1",
        field_contract_hash="fd1",
        source_contract_hash="sc1",
        source_dependency_hash="sd1",
        universe="CSI500",
        frequency="1d",
    )
    full_300 = identity_csi300.identity_digest()
    full_500 = identity_csi500.identity_digest()
    assert full_300 != full_500

    # 统一请求：full digest 派生 16 位显示版本（Parquet 与 CH 同源）
    req_300 = DualWriteRequest(
        factor_id="f300",
        upserts=_mk_series(),
        ast_hash=ast_hash,
        semantic_identity_digest=full_300,
    )
    req_500 = DualWriteRequest(
        factor_id="f500",
        upserts=_mk_series(),
        ast_hash=ast_hash,
        semantic_identity_digest=full_500,
    )
    assert req_300.factor_version != req_500.factor_version
    # Parquet 侧（MaterializeMetadata）与 CH 侧都取同一 16 位前缀
    assert req_300.factor_version == identity_csi300.identity_digest()[:16]
    assert req_500.factor_version == identity_csi500.identity_digest()[:16]

    # CH 写入实际收到的 version == 请求的统一 version
    summary300 = append_clickhouse_to_summary(
        {"factor_id": "f300"},
        factor_id="f300",
        result=_mk_series(),
        ast_hash=ast_hash,
        write_target="clickhouse",
        delta=req_300.to_delta(),
    )
    assert captured["factor_version"] == req_300.factor_version
    assert summary300["clickhouse"]["rows_written"] == len(_mk_series())

    captured.clear()
    summary500 = append_clickhouse_to_summary(
        {"factor_id": "f500"},
        factor_id="f500",
        result=_mk_series(),
        ast_hash=ast_hash,
        write_target="clickhouse",
        delta=req_500.to_delta(),
    )
    # 两个 universe 的 CH 版本互不相同，且各自与请求（= Parquet 侧）一致
    assert captured["factor_version"] == req_500.factor_version
    assert captured["factor_version"] != req_300.factor_version
    assert summary500["clickhouse"]["rows_written"] == len(_mk_series())


# ---------------------------------------------------------------------------
# NEW-P0-60 tombstone propagation to ClickHouse
# ---------------------------------------------------------------------------


def test_dual_write_tombstones_reach_clickhouse(tmp_path, monkeypatch):
    import storage.write_targets

    captured = {}
    monkeypatch.setattr(
        storage.write_targets, "ClickHouseWriteTarget", _fake_ch_target(captured)
    )

    deleted_key = (pd.Timestamp("2024-01-02"), "AAA")
    delta = MaterializationDelta(
        upserts=_mk_series(),
        tombstones=(deleted_key,),
        semantic_identity_digest="g" * 64,
        generation="gen-1",
        transaction_id="txn-1",
    )
    summary = append_clickhouse_to_summary(
        {"factor_id": "f_del"},
        factor_id="f_del",
        result=_mk_series(),
        ast_hash="a" * 64,
        write_target="clickhouse",
        delta=delta,
    )
    ch_series = captured["series"]
    # (2024-01-02, AAA) 在 CH 侧必须是 NaN（tombstone 覆盖旧有限值）
    assert ch_series.loc[deleted_key] != ch_series.loc[deleted_key]  # NaN
    # 其它键保持原值
    assert ch_series.loc[(pd.Timestamp("2024-01-03"), "BBB")] > 0
    # NEW-P1-61：generation / transaction_id 贯穿到 summary
    assert summary["run_generation"] == "gen-1"
    assert summary["transaction_id"] == "txn-1"


def test_series_with_tombstones_replaces_not_appends():
    s = _mk_series()
    out = _series_with_tombstones(s, ((pd.Timestamp("2024-01-02"), "AAA"),))
    # 已存在的键：NaN 覆盖，不增加行数
    assert len(out) == len(s)
    assert out.loc[(pd.Timestamp("2024-01-02"), "AAA")] != out.loc[
        (pd.Timestamp("2024-01-02"), "AAA")
    ]  # NaN


# ---------------------------------------------------------------------------
# NEW-P1-61 shared generation / commit token
# ---------------------------------------------------------------------------


def test_materialization_delta_carries_shared_token():
    delta = MaterializationDelta(
        upserts=_mk_series(),
        generation="gen-9",
        transaction_id="txn-9",
        semantic_identity_digest="h" * 64,
    )
    assert delta.generation == "gen-9"
    assert delta.transaction_id == "txn-9"
    assert delta.factor_version("a" * 64) == ("h" * 64)[:16]
    assert delta.has_tombstones is False


def test_dual_write_request_to_delta_preserves_token_and_version():
    req = DualWriteRequest(
        factor_id="f_tok",
        upserts=_mk_series(),
        ast_hash="a" * 64,
        semantic_identity_digest="i" * 64,
        generation="gen-10",
        transaction_id="txn-10",
        tombstones=((pd.Timestamp("2024-01-02"), "BBB"),),
    )
    d = req.to_delta()
    assert d.generation == "gen-10"
    assert d.transaction_id == "txn-10"
    assert d.semantic_identity_digest == "i" * 64
    assert d.factor_version("a" * 64) == req.factor_version
    assert d.has_tombstones is True
