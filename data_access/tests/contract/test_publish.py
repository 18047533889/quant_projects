"""
PR3 publish_from_staging 的端到端 contract 测试。

覆盖：
    - 基本发布：staging → published，读得回
    - 有旧版本时归档到 _archive/，_archive 下还能读出历史内容
    - 校验失败的各种 raise：staging 空、access_mode 不对、schema 不一致、params 不匹配
    - 并发锁：已有锁文件时第二个 publish 直接 raise
    - copy 失败场景：published 原状态不变
    - 审计日志：每次 publish 无论成败都留一行
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.exceptions import DataError, ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore
from data_access import publish as publish_mod


@pytest.fixture
def publish_store(tmp_path, monkeypatch):
    """搭一个带 staging + published 配对的 store，便于做 publish 流程测试。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lake_root = tmp_path / "lake"
    lake_root.mkdir()

    monkeypatch.setenv("TEST_WORKSPACE", str(workspace))
    monkeypatch.setenv("TEST_LAKE_ROOT", str(lake_root))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "shw")
    monkeypatch.setenv("QUANT_OPERATOR", "shw@test")
    monkeypatch.setenv("QUANT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))

    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        # 配对 1：参数化（factor_id），hive 分区 —— 模拟 factor_lake
        factors_pub:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: ${TEST_LAKE_ROOT}/factors/{factor_id}
          glob_template: "year=*/*.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true

        factors_stg:
          kind: parametric
          access_mode: staging
          layout: hive
          root_template: ${TEST_WORKSPACE}/staging/${RUN_NAMESPACE}/factors/{factor_id}
          glob_template: "year=*/*.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true

        # 配对 2：参数化但 layout 不一致 —— 用于校验失败测试
        bad_layout_pub:
          kind: parametric
          access_mode: published
          layout: plain
          root_template: ${TEST_LAKE_ROOT}/bad/{x}
          glob_template: "**/*.parquet"
          params_schema:
            x: str
          time_column: datetime
          instrument_column: asset
          union_by_name: true

        # 再加一个 access_mode 错的：namespaced 不能当 staging 源
        not_staging:
          kind: parametric
          access_mode: namespaced
          layout: plain
          root_template: ${TEST_WORKSPACE}/users/${RUN_NAMESPACE}/ns/{x}
          glob_template: "**/*.parquet"
          params_schema:
            x: str
          time_column: datetime
          instrument_column: asset
          union_by_name: true
    """).strip() + "\n", encoding="utf-8")

    reset_store()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store, tmp_path, lake_root
    engine.close()
    reset_store()


def _sample_factor(n: int = 4, year: int = 2024) -> pa.Table:
    return pa.table({
        "datetime": pd.to_datetime([f"{year}-0{i+1}-01" for i in range(n)]),
        "asset": [f"SYM{i:02d}" for i in range(n)],
        "value": [i * 1.5 for i in range(n)],
        "year": [year] * n,
    })


# ---- 基本发布 ---------------------------------------------------------------

def test_publish_basic_flow(publish_store):
    """写 staging → publish → 读 published 能拿到同样的行数。"""
    store, _, lake_root = publish_store

    store.write_arrow(
        "factors_stg", _sample_factor(4),
        factor_id="mom_3d", mode="overwrite", partition_by=["year"],
    )

    result = store.publish_from_staging(
        "factors_stg", "factors_pub", factor_id="mom_3d",
    )

    assert result["rows"] == 4
    assert Path(result["target_path"]) == lake_root / "factors" / "mom_3d"
    assert result["archive_path"] is None   # 首次发布，没有旧版本可归档
    assert Path(result["target_path"]).exists()
    assert "manifest_path" in result
    from data_access.publish_manifest import read_publish_manifest

    manifest = read_publish_manifest(Path(result["target_path"]))
    assert manifest is not None
    assert manifest["rows"] == 4

    # 读 published 拿得到
    got = store.read_frame("factors_pub", factor_id="mom_3d", columns=["asset", "value"])
    assert len(got) == 4


def test_publish_archives_old_version(publish_store):
    """第二次 publish 同一个 factor_id：旧 published 应该被归档到 _archive/。"""
    store, _, lake_root = publish_store

    # 第一次：4 行
    store.write_arrow(
        "factors_stg", _sample_factor(4),
        factor_id="mom_3d", mode="overwrite", partition_by=["year"],
    )
    store.publish_from_staging("factors_stg", "factors_pub", factor_id="mom_3d")

    # 第二次：重新写 staging 成 2 行，再发布
    store.write_arrow(
        "factors_stg", _sample_factor(2),
        factor_id="mom_3d", mode="overwrite", partition_by=["year"],
    )
    result = store.publish_from_staging(
        "factors_stg", "factors_pub", factor_id="mom_3d",
    )

    assert result["rows"] == 2
    # 新版本已在位
    got = store.read_frame("factors_pub", factor_id="mom_3d", columns=["asset"])
    assert len(got) == 2

    # 归档目录应该存在，且归档里是旧的 4 行
    archive_dir = Path(result["archive_path"])
    assert archive_dir.exists()
    assert archive_dir.parent.name == "_archive"
    # 旧版本归档能被 pyarrow 重新读
    import pyarrow.parquet as pq
    archived_rows = sum(
        pq.read_metadata(str(p)).num_rows for p in archive_dir.rglob("*.parquet")
    )
    assert archived_rows == 4


def test_publish_preserves_hive_layout(publish_store):
    """发布前后 hive 分区结构必须保留（year=YYYY/）。"""
    store, _, lake_root = publish_store
    tbl = pa.table({
        "datetime": pd.to_datetime(["2023-01-01", "2024-01-01"]),
        "asset": ["AAPL", "MSFT"],
        "value": [1.0, 2.0],
        "year": [2023, 2024],
    })
    store.write_arrow(
        "factors_stg", tbl, factor_id="m1",
        mode="overwrite", partition_by=["year"],
    )
    store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    pub_root = lake_root / "factors" / "m1"
    assert (pub_root / "year=2023").exists()
    assert (pub_root / "year=2024").exists()


# ---- 校验拒绝 ---------------------------------------------------------------

def test_publish_rejects_empty_staging(publish_store):
    """staging 目录空 / 不存在 → DataError，不允许发布空数据。"""
    store, _, _ = publish_store
    with pytest.raises(DataError, match="没有可发布"):
        store.publish_from_staging("factors_stg", "factors_pub", factor_id="never_written")


def test_publish_rejects_non_staging_source(publish_store):
    """namespaced 数据集不能当 staging 来源。"""
    store, _, _ = publish_store
    with pytest.raises(ValidationError, match="access_mode"):
        store.publish_from_staging("not_staging", "factors_pub", factor_id="mom_3d")


def test_publish_rejects_non_published_target(publish_store):
    """target 必须是 published。用 staging 当 target 会 raise。"""
    store, _, _ = publish_store
    with pytest.raises(ValidationError, match="publish"):
        store.publish_from_staging("factors_stg", "factors_stg", factor_id="mom_3d")


def test_publish_rejects_layout_mismatch(publish_store):
    """staging 是 hive，target 是 plain → schema 层面不兼容。"""
    store, _, _ = publish_store
    with pytest.raises(ValidationError, match="layout"):
        store.publish_from_staging("factors_stg", "bad_layout_pub", factor_id="m1")


def test_publish_missing_params_raises(publish_store):
    """publish_from_staging 没传 factor_id → raise。"""
    store, _, _ = publish_store
    store.write_arrow(
        "factors_stg", _sample_factor(2),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )
    with pytest.raises(ValidationError, match="factor_id"):
        store.publish_from_staging("factors_stg", "factors_pub")


# ---- 并发锁 -----------------------------------------------------------------

def test_publish_lock_conflict(publish_store):
    """已有锁文件时 publish fail-fast，不等。"""
    store, _, lake_root = publish_store
    store.write_arrow(
        "factors_stg", _sample_factor(2),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )

    # 手动创建锁文件（模拟另一个 publish 进程正在跑）
    target_parent = lake_root / "factors"
    target_parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_parent / ".publish.m1.lock"
    lock_path.write_text("pid=99999\n")

    with pytest.raises(ValidationError, match="并发冲突"):
        store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    # 锁文件应该还在（不能被 publish fail 时删掉，否则就失去意义）
    assert lock_path.exists()


def test_publish_releases_lock_on_success(publish_store):
    store, _, lake_root = publish_store
    store.write_arrow(
        "factors_stg", _sample_factor(2),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )
    store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    lock_path = lake_root / "factors" / ".publish.m1.lock"
    assert not lock_path.exists(), "publish 成功后锁文件应该被清理"


# ---- 回滚 ------------------------------------------------------------------

def test_publish_rollback_on_rename_failure(publish_store, monkeypatch):
    """candidate → final 的 rename 失败后，archive 要回滚成 published；且锁要释放。"""
    store, _, lake_root = publish_store

    # 先发一版作为"旧版本"
    store.write_arrow(
        "factors_stg", _sample_factor(4),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )
    store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    # 再准备新一版
    store.write_arrow(
        "factors_stg", _sample_factor(2),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )

    # 让第二次 rename（candidate → final）失败
    real_rename = os.rename
    call_count = {"n": 0}

    def flaky_rename(src, dst):
        call_count["n"] += 1
        # 第一次 rename（old_published → archive）让过；第二次失败
        if call_count["n"] == 2:
            raise OSError("simulated candidate→final rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(publish_mod.os, "rename", flaky_rename)

    with pytest.raises(OSError, match="simulated"):
        store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    # 发布失败后，旧版本（4 行）应该还能读出来
    got = store.read_frame("factors_pub", factor_id="m1", columns=["asset"])
    assert len(got) == 4

    # 锁文件也要清干净
    lock_path = lake_root / "factors" / ".publish.m1.lock"
    assert not lock_path.exists()


def test_publish_candidate_cleaned_up_on_copy_failure(publish_store, monkeypatch):
    """copytree 失败的话 candidate 目录要被清掉，不能留垃圾。"""
    store, _, lake_root = publish_store

    store.write_arrow(
        "factors_stg", _sample_factor(2),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )

    def boom(src, dst, *args, **kwargs):
        # 故意不拷贝，制造 candidate 不存在的情况，然后 raise
        raise OSError("simulated copy failure")

    monkeypatch.setattr(publish_mod.shutil, "copytree", boom)

    with pytest.raises(OSError, match="simulated copy"):
        store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    # 父目录下不能残留 .publish_candidate.* 和锁文件
    target_parent = lake_root / "factors"
    candidates = list(target_parent.glob(".publish_candidate.*"))
    locks = list(target_parent.glob(".publish.m1.lock"))
    assert candidates == []
    assert locks == []


# ---- 审计 ------------------------------------------------------------------

def test_publish_records_audit_on_success(publish_store, tmp_path):
    store, root, _ = publish_store
    audit_path = root / "audit.jsonl"

    store.write_arrow(
        "factors_stg", _sample_factor(3),
        factor_id="m1", mode="overwrite", partition_by=["year"],
    )
    store.publish_from_staging("factors_stg", "factors_pub", factor_id="m1")

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    publish_lines = [json.loads(l) for l in lines if json.loads(l)["op"] == "publish"]
    assert len(publish_lines) == 1
    rec = publish_lines[0]
    assert rec["ok"] is True
    assert rec["dataset"] == "factors_pub"
    assert rec["rows"] == 3
    assert rec["extra"]["source"] == "factors_stg"
    assert rec["namespace"] == "shw"
    assert rec["operator"] == "shw@test"


def test_publish_records_audit_on_failure(publish_store, tmp_path):
    """即使 publish 失败也要记审计日志（ok=false + error）。"""
    store, root, _ = publish_store
    audit_path = root / "audit.jsonl"

    # staging 为空 → DataError
    with pytest.raises(DataError):
        store.publish_from_staging("factors_stg", "factors_pub", factor_id="not_written")

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    publish_lines = [json.loads(l) for l in lines if json.loads(l)["op"] == "publish"]
    assert len(publish_lines) == 1
    rec = publish_lines[0]
    assert rec["ok"] is False
    assert "没有可发布" in rec["error"]


# ---- 回归：成功分支不得误删 CWD ---------------------------------------------

def test_publish_does_not_wipe_cwd_on_success(publish_store, tmp_path, monkeypatch):
    """
    回归：publish 成功分支曾经写 `candidate_dir = Path()` 当 sentinel，
    Path() 等价于 PosixPath('.')，exists()==True，finally 里的
    shutil.rmtree(candidate_dir) 就把当前工作目录整个递归删了。
    事故：pytest 在仓库根跑 publish 成功路径时仓库被清空两次（2026-04-19）。
    修复：改成 candidate_dir = None 作为 sentinel。

    本测试的断言：在一个可控的 CWD 里跑一次成功 publish，CWD 下的哨兵
    文件/目录必须完好无损。如果有人把 sentinel 改回 Path()，此测试会 FAIL。
    """
    store, _, _ = publish_store

    # 建一个独立 CWD，里面放哨兵文件 + 哨兵子目录
    fake_cwd = tmp_path / "fake_cwd"
    fake_cwd.mkdir()
    sentinel_file = fake_cwd / "do_not_delete_me.txt"
    sentinel_file.write_text("regression guard", encoding="utf-8")
    sentinel_subdir = fake_cwd / "precious_subdir"
    sentinel_subdir.mkdir()
    (sentinel_subdir / "nested.txt").write_text("still here", encoding="utf-8")

    monkeypatch.chdir(fake_cwd)

    store.write_arrow(
        "factors_stg", _sample_factor(3),
        factor_id="regression_guard", mode="overwrite", partition_by=["year"],
    )
    result = store.publish_from_staging(
        "factors_stg", "factors_pub", factor_id="regression_guard",
    )
    assert result["rows"] == 3  # 确认 publish 真的走到了成功分支

    # 关键断言：CWD 和里面的哨兵全都要在
    assert fake_cwd.exists(), "publish 成功后 CWD 被删了——rmtree(Path()) bug 又回来了"
    assert sentinel_file.exists(), "CWD 下的哨兵文件被误删"
    assert sentinel_file.read_text(encoding="utf-8") == "regression guard"
    assert sentinel_subdir.exists()
    assert (sentinel_subdir / "nested.txt").exists()
