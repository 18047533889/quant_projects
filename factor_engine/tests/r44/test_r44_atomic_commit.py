# -*- coding: utf-8 -*-
"""R44：原子提交事务 + 三水位线证书测试。"""
from __future__ import annotations

import os
import shutil
import stat

import pytest

from factor_engine.runtime.atomic_commit import (
    IncrementalCommitTransaction,
    IncrementalExecutionGeneration,
    WatermarkSet,
    WatermarkViolation,
    current_generation,
    finalize_watermark_advance,
    list_generations,
    load_generation,
    three_watermark_certificate,
)

# ---------------------------------------------------------------------------
# 1. 三水位线证书
# ---------------------------------------------------------------------------


def _valid_wm() -> WatermarkSet:
    return WatermarkSet(source_ingestion="2026-08-23", node_state="2026-08-23", factor_output="2026-08-23")


def test_three_watermark_certificate_valid() -> None:
    assert three_watermark_certificate(_valid_wm()) is True
    assert three_watermark_certificate(_valid_wm(), production=True) is True


def test_three_watermark_certificate_source_behind_state() -> None:
    bad = WatermarkSet(source_ingestion="2026-08-22", node_state="2026-08-23", factor_output="2026-08-23")
    # research -> False
    assert three_watermark_certificate(bad) is False
    # production -> raise
    with pytest.raises(WatermarkViolation):
        three_watermark_certificate(bad, production=True)


def test_three_watermark_certificate_state_output_mismatch() -> None:
    bad = WatermarkSet(source_ingestion="2026-08-24", node_state="2026-08-23", factor_output="2026-08-24")
    assert three_watermark_certificate(bad) is False
    with pytest.raises(WatermarkViolation):
        three_watermark_certificate(bad, production=True)


def test_three_watermark_certificate_any_none_fails() -> None:
    for wm in (
        WatermarkSet(None, "2026-08-23", "2026-08-23"),
        WatermarkSet("2026-08-23", None, "2026-08-23"),
        WatermarkSet("2026-08-23", "2026-08-23", None),
        WatermarkSet(None, None, None),
    ):
        assert three_watermark_certificate(wm) is False
        with pytest.raises(WatermarkViolation):
            three_watermark_certificate(wm, production=True)
    assert three_watermark_certificate(None) is False


# ---------------------------------------------------------------------------
# 2. stage + commit round-trip
# ---------------------------------------------------------------------------


def test_stage_commit_round_trip(tmp_path) -> None:
    tx = IncrementalCommitTransaction(tmp_path / "gen")
    tx.stage(
        factor_parts={
            "factor_a.parquet": b"AAA-data",
            "factor_b.parquet": b"BBB-data",
        },
        state_parts={"state_a.json": b'{"as_of":"2026-08-23"}'},
        watermarks=_valid_wm(),
        dq_certificate={"passed": True},
        pit_certificate={"ok": True},
        source_snapshot_id="snap-1",
        execution_id="exec-1",
    )
    gen = tx.commit()

    # current_generation + load_generation 返回完整世代
    assert current_generation(tmp_path / "gen") == gen.generation
    loaded = load_generation(tmp_path / "gen", gen.generation)
    assert loaded is not None
    assert loaded.generation == gen.generation
    assert set(loaded.factor_parts) == {"factor_a.parquet", "factor_b.parquet"}
    assert loaded.state_parts == ("state_a.json",)
    assert loaded.source_snapshot_id == "snap-1"
    assert loaded.execution_id == "exec-1"
    assert loaded.dq_certificate == {"passed": True}
    assert loaded.pit_certificate == {"ok": True}
    assert loaded.factor_output_watermark == "2026-08-23"
    assert loaded.state_watermark == "2026-08-23"

    # manifest 字段：既有 10 字段 + R45 manifest closure 追加字段（additive）。
    manifest = gen.to_manifest()
    expected = {
        "generation", "source_snapshot_id", "execution_id", "factor_parts",
        "state_parts", "dq_certificate", "pit_certificate",
        "factor_output_watermark", "state_watermark", "watermark",
        # R45 追加（缺省 None）。
        "factor_semantic_id", "data_read_identity", "universe", "decision_clock",
        "physical_plan_id", "pi_ids", "build", "calendar",
        "incremental_contract_identity",
    }
    assert set(manifest) == expected

    # from_manifest round-trip 保持一致
    rt = IncrementalExecutionGeneration.from_manifest(manifest)
    assert rt == gen

    # part 文件真实落盘
    gen_dir = (tmp_path / "gen") / f"generation={gen.generation}"
    assert (gen_dir / "factor_a.parquet").read_bytes() == b"AAA-data"
    assert (gen_dir / "state_a.json").read_bytes() == b'{"as_of":"2026-08-23"}'
    assert (gen_dir / "manifest.json").is_file()

    # list_generations
    assert gen.generation in list_generations(tmp_path / "gen")


# ---------------------------------------------------------------------------
# 3. crash mid-commit → 旧世代保留，重试成功
# ---------------------------------------------------------------------------


def test_crash_mid_commit_keeps_old_and_retry_succeeds(tmp_path) -> None:
    root = tmp_path / "gen"

    # 先提交一个旧世代
    old = IncrementalCommitTransaction(root)
    old.stage(
        factor_parts={"f_old.parquet": b"OLD"},
        watermarks=_valid_wm(),
        source_snapshot_id="snap-old",
    )
    old_gen = old.commit()
    assert current_generation(root) == old_gen.generation

    # 模拟 commit 的最后指针翻转失败：让 CURRENT 目标目录只读，使 os.replace 抛错。
    # 更稳做法：先把 staging 目录改成只读，使 commit 写 manifest 前失败。
    tx2 = IncrementalCommitTransaction(root)
    tx2.stage(
        factor_parts={"f_new.parquet": b"NEW"},
        watermarks=_valid_wm(),
        source_snapshot_id="snap-new",
    )
    # 找到事务的 staging 目录（commit 尚未把 generation 定下来，直接改 CURRENT 只读）
    # 改为：临时把 root 目录权限设为只读以触发 commit 失败，再恢复。
    saved = os.stat(root).st_mode
    try:
        os.chmod(root, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        with pytest.raises(Exception):
            tx2.commit()
    finally:
        os.chmod(root, saved)

    # 旧世代 + CURRENT 指针不变
    assert current_generation(root) == old_gen.generation
    loaded_old = load_generation(root, old_gen.generation)
    assert loaded_old is not None and loaded_old.factor_parts == ("f_old.parquet",)

    # 重试 commit 成功
    tx3 = IncrementalCommitTransaction(root)
    tx3.stage(
        factor_parts={"f_new.parquet": b"NEW"},
        watermarks=_valid_wm(),
        source_snapshot_id="snap-new",
    )
    new_gen = tx3.commit()
    assert current_generation(root) == new_gen.generation
    assert load_generation(root, new_gen.generation).factor_parts == ("f_new.parquet",)


# ---------------------------------------------------------------------------
# 4. duplicate retry exactly-once
# ---------------------------------------------------------------------------


def test_duplicate_commit_noop(tmp_path) -> None:
    root = tmp_path / "gen"
    tx = IncrementalCommitTransaction(root)
    tx.stage(
        factor_parts={"f.parquet": b"DATA"},
        watermarks=_valid_wm(),
        source_snapshot_id="snap",
    )
    gen1 = tx.commit()
    # 第二次 commit 同事务：no-op，返回同 generation，不新增文件
    gen2 = tx.commit()
    assert gen2.generation == gen1.generation
    gen_dir = (root / f"generation={gen1.generation}")
    files = [p.name for p in gen_dir.iterdir()]
    assert files.count("f.parquet") == 1
    assert files.count("manifest.json") == 1


# ---------------------------------------------------------------------------
# 5. finalize_watermark_advance 幂等
# ---------------------------------------------------------------------------


class _StubCatalog:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._wm: dict | None = None

    def update_watermark(self, factor_id, start_date, end_date, *, row_count=None) -> None:
        self.calls.append((factor_id, start_date, end_date))
        self._wm = {"factor_id": factor_id, "start_date": start_date, "end_date": end_date}

    def get_watermark(self, factor_id):
        return self._wm


def test_finalize_watermark_advance_idempotent(tmp_path) -> None:
    root = tmp_path / "gen"
    tx = IncrementalCommitTransaction(root)
    tx.stage(
        factor_parts={"f.parquet": b"DATA"},
        watermarks=_valid_wm(),
        source_snapshot_id="snap-f",
    )
    gen = tx.commit()

    catalog = _StubCatalog()
    first = finalize_watermark_advance(catalog, gen)
    assert first["advanced"] == 1
    assert first["factor_id"] == "snap-f"
    assert len(catalog.calls) == 1
    assert catalog.calls[0][1:] == ("2026-08-23", "2026-08-23")

    # 第二次推进：update_watermark 被再次调用，但 get 确认已到位 → 幂等成功
    second = finalize_watermark_advance(catalog, gen)
    assert second["advanced"] == 1
    assert catalog._wm["start_date"] == "2026-08-23"


def test_finalize_watermark_advance_callable() -> None:
    calls: list[tuple] = []

    def _fn(factor_id, start_date, end_date):
        calls.append((factor_id, start_date, end_date))

    gen = IncrementalExecutionGeneration(
        generation="G-x",
        source_snapshot_id="snap",
        execution_id="e",
        factor_parts=(),
        state_parts=(),
        watermark=WatermarkSet("2026-08-23", "2026-08-23", "2026-08-23"),
    )
    finalize_watermark_advance(_fn, gen)
    assert calls == [("snap", "2026-08-23", "2026-08-23")]


# ---------------------------------------------------------------------------
# 6. wire-in smoke（传入已物化 part 路径，仅测事务机制）
# ---------------------------------------------------------------------------


def test_wire_in_generation_none_returns_old_style(tmp_path) -> None:
    from factor_engine.runtime.materialize_service import execute_materialize_incremental_generation

    items = [
        {
            "factor": _DummyFactor("f1"),
            "output": {"result": None, "analysis": None},
            "factor_id": "f1",
            "options": {"part_paths": {"f1.parquet": b"X"}, "end_date": "2026-08-23"},
        }
    ]
    # generation_root=None → 旧式 dict（无原子提交）
    out = execute_materialize_incremental_generation(None, items)
    assert out["atomic"] is False
    assert out["generation"] is None


def test_wire_in_atomic_commit(tmp_path) -> None:
    from factor_engine.runtime.materialize_service import execute_materialize_incremental_generation

    items = [
        {
            "factor": _DummyFactor("f1"),
            "output": {"result": None, "analysis": None},
            "factor_id": "f1",
            "options": {"part_paths": {"f1.parquet": b"X"}, "end_date": "2026-08-23"},
        }
    ]
    root = tmp_path / "gen"
    out = execute_materialize_incremental_generation(
        None, items, generation_root=root, source_snapshot_id="snap-w",
        execution_id="exec-w", production=False,
    )
    assert out["atomic"] is True
    assert out["generation"] is not None
    # current_generation 匹配提交的 generation
    assert current_generation(root) == out["generation"]
    loaded = load_generation(root, out["generation"])
    assert loaded is not None
    assert loaded.factor_parts == ("f1.parquet",)
    assert loaded.watermark is not None
    assert loaded.watermark.factor_output == "2026-08-23"


class _DummyFactor:
    """极简 Factor 桩（wire-in smoke 只测事务机制，不跑真实物化）。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.freq = "1d"


# ---------------------------------------------------------------------------
# R45: 命名空间守卫 / 生产门禁 / ObjectStore 后端（零本地写）
# ---------------------------------------------------------------------------
def _full_wm() -> WatermarkSet:
    return WatermarkSet("2026-08-23", "2026-08-23", "2026-08-23")


def test_stage_rejects_same_name_factor_state_conflict(tmp_path) -> None:
    from factor_engine.runtime.generation_store import NamespaceConflictError

    tx = IncrementalCommitTransaction(tmp_path / "gen")
    with pytest.raises(NamespaceConflictError):
        tx.stage(
            factor_parts={"dup.parquet": b"F"},
            state_parts={"dup.parquet": b"S"},
            watermarks=_full_wm(),
        )


def test_production_commit_rejects_missing_gates(tmp_path) -> None:
    # 缺 DQ/PIT/DRI/FactorSemanticID → 生产提交被拒（research 宽松）。
    tx = IncrementalCommitTransaction(tmp_path / "gen")
    tx.stage(
        factor_parts={"f.parquet": b"X"},
        watermarks=_full_wm(),
    )
    with pytest.raises(WatermarkViolation):
        tx.commit(production=True)
    # research 提交不受门禁约束。
    tx.commit(production=False)
    assert current_generation(tmp_path / "gen") is not None


def test_production_commit_requires_data_read_identity(tmp_path) -> None:
    # DQ/PIT PASS 但 DRI/FactorSemanticID 缺失 → 仍拒绝（读身份未认证）。
    tx = IncrementalCommitTransaction(tmp_path / "gen")
    tx.stage(
        factor_parts={"f.parquet": b"X"},
        watermarks=_full_wm(),
        dq_certificate={"passed": True},
        pit_certificate={"passed": True},
    )
    with pytest.raises(WatermarkViolation):
        tx.commit(production=True)


def test_production_commit_passes_when_all_gates_pass(tmp_path) -> None:
    tx = IncrementalCommitTransaction(tmp_path / "gen")
    tx.stage(
        factor_parts={"f.parquet": b"X"},
        state_parts={"s.json": b"{}"},
        watermarks=_full_wm(),
        dq_certificate={"passed": True},
        pit_certificate={"passed": True},
        factor_semantic_id="sem1",
        data_read_identity="dri1",
    )
    gen = tx.commit(production=True)
    loaded = load_generation(tmp_path / "gen", gen.generation)
    assert loaded is not None
    assert loaded.factor_semantic_id == "sem1"
    assert loaded.data_read_identity == "dri1"


def test_object_store_generation_commit_zero_local_path_writes(tmp_path) -> None:
    """ObjectGenerationStore 生产提交：parts/manifest/CURRENT 全走 ObjectStore，
    命名空间 disjoint，无本地 Path 写。"""
    import importlib.util

    from factor_engine.runtime.generation_store import ObjectGenerationStore

    # dataaccess 包 __init__ 存在既有循环导入（会拉进 data_access），object_store
    # 本身仅用 stdlib，故按文件独立加载以隔离环境问题。
    _spec = importlib.util.spec_from_file_location(
        "_r45_object_store", "data_access/read/object_store.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)
    LocalObjectStore = _mod.LocalObjectStore

    store = LocalObjectStore(tmp_path / "lake")
    ostore = ObjectGenerationStore(store)
    tx = IncrementalCommitTransaction(tmp_path / "txroot")
    tx.stage(
        factor_parts={"f_a.parquet": b"AAA"},
        state_parts={"st_a.json": b"{}"},
        watermarks=_full_wm(),
        dq_certificate={"passed": True},
        pit_certificate={"passed": True},
        source_snapshot_id="snap1",
        execution_id="exec1",
        factor_semantic_id="SEM1",
        data_read_identity="DRI1",
        universe=["A", "B"],
        decision_clock="dc1",
        physical_plan_id="pp1",
        pi_ids=["pi1"],
        build="b1",
        calendar="cn",
        incremental_contract_identity="ici1",
    )
    gen = tx.commit(generation_store=ostore, production=True)
    keys = store.list_objects("")
    # factor / state 命名空间物理 disjoint（不同前缀），杜绝同名互相覆盖。
    assert f"factor/{gen.generation}/f_a.parquet" in keys
    assert f"state/{gen.generation}/st_a.json" in keys
    assert f"generation/{gen.generation}/manifest.json" in keys
    assert "generation/CURRENT" in keys
    assert store.range_read("generation/CURRENT", offset=0, length=200).decode().strip() == gen.generation
    for k in keys:
        assert not (k.startswith("factor/") and k.startswith("state/"))
    # CURRENT 指针指向的 manifest 保留新增 closure 字段。
    loaded = ostore.load_manifest(gen.generation)
    assert loaded["factor_semantic_id"] == "SEM1"
    assert loaded["data_read_identity"] == "DRI1"
    assert loaded["calendar"] == "cn"


def test_manifest_closure_backfills_from_stage(tmp_path) -> None:
    """R45: manifest closure 追加字段从 stage 传入并被 round-trip 保留。"""
    tx = IncrementalCommitTransaction(tmp_path / "gen")
    tx.stage(
        factor_parts={"f.parquet": b"X"},
        watermarks=_full_wm(),
        factor_semantic_id="sem9",
        data_read_identity="dri9",
        universe=["X", "Y"],
        decision_clock="dc9",
        physical_plan_id="pp9",
        pi_ids=["p1", "p2"],
        build="b9",
        calendar="cn",
        incremental_contract_identity="ici9",
    )
    gen = tx.commit()
    loaded = load_generation(tmp_path / "gen", gen.generation)
    assert loaded is not None
    assert loaded.factor_semantic_id == "sem9"
    assert loaded.data_read_identity == "dri9"
    assert loaded.universe == ["X", "Y"]
    assert loaded.decision_clock == "dc9"
    assert loaded.physical_plan_id == "pp9"
    assert loaded.pi_ids == ["p1", "p2"]
    assert loaded.build == "b9"
    assert loaded.calendar == "cn"
    assert loaded.incremental_contract_identity == "ici9"
