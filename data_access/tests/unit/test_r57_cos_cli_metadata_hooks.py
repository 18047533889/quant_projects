# -*- coding: utf-8 -*-
"""R57 —— 跨 domain（跨账号）COS bucket 的 LIST/HEAD 元数据通路修复。

背景：server C 上 ``qs-cold`` 等数据 bucket 归属其他账号（跨 domain）。本机
静态密钥对它们 IAM 不可见——boto3 / DuckDB httpfs 一律 ``NoSuchBucket``/404，
``ResolvedObject.content_length`` 恒 None → ``snapshot.total_bytes=None`` →
governor 按未知成本（保守上界 2^62-1）必然拒绝 → 「读 remote 永远被拒」。

修法：store 的 ``SourceSnapshotResolver`` 接上 LIST/HEAD 钩子；boto3 首选，
跨账号 bucket 落回 COS CLI 网关（``cos_cli_ls``/``cos_cli_head``，解析 coscli
``ls`` 人类可读表格）。本文件用 mock CLI / mock boto3 锁住行为，不触网。
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_access.core.engine import DuckDBEngine
from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from data_access.runtime.cache_manager import reset_cache_manager
    from data_access.runtime.resource_governor import reset_global_governor
    from data_access.read.query_cache import reset_query_cache

    reset_query_cache()
    reset_cache_manager()
    reset_global_governor()
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_REMOTE_SNAPSHOT_META", raising=False)
    yield
    reset_query_cache()
    reset_cache_manager()
    reset_global_governor()


def _store(tmp_path: Path) -> DataAccessStore:
    root = tmp_path / "d"
    root.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
    ds:
      kind: static
      access_mode: staging
      layout: plain
      root: "{root}"
      glob: "part-*.parquet"
      time_column: ts
      instrument_column: sym
      schema:
        ts: date
        sym: string
        val: double
    """
        ),
        encoding="utf-8",
    )
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=1))


_CLI_LS_TABLE = """                              KEY                              |   TYPE   |       LAST MODIFIED       |                ETAG                |   SIZE    | RESTORESTATUS
---------------------------------------------------------------+----------+---------------------------+------------------------------------+-----------+----------------
  clean_data/ashare/lqtp_data/StockDailyBar/2024-01-02.parquet | STANDARD | 2026-06-15T16:25:37+08:00 | "1b5f84ef3349aae942de60f3faf24680" | 205.82 KB |
  clean_data/ashare/lqtp_data/StockDailyBar/2024-01-03.parquet | STANDARD | 2026-06-15T16:25:37+08:00 | "29a08f273ee9d04cde1faa41fc42f18e" | 208.43 KB |
--------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                                                                            TOTAL OBJECTS:  | 2592
"""


# ===========================================================================
# cos_cli_ls / cos_cli_head 解析器（纯函数，无网络）
# ===========================================================================

class TestCliLsParser:
    def test_size_units(self):
        from data_access.cos.remote import _cli_size_to_bytes

        assert _cli_size_to_bytes("0 B") == 0
        assert _cli_size_to_bytes("123") == 123
        assert _cli_size_to_bytes("205.82 KB") == 210760  # ceil（宁高估不低估）
        assert _cli_size_to_bytes("5.93 MB") == 6218056
        assert _cli_size_to_bytes("1 GB") == 1024**3
        assert _cli_size_to_bytes("abc") is None
        assert _cli_size_to_bytes("") is None
        assert _cli_size_to_bytes("12 XB") is None  # 未知单位 fail-closed

    def test_parse_table_rows_and_skip_headers(self):
        from data_access.cos.remote import _parse_cos_cli_ls_output

        rows = _parse_cos_cli_ls_output(_CLI_LS_TABLE)
        assert len(rows) == 2
        assert rows[0]["key"].endswith("2024-01-02.parquet")
        assert rows[0]["size"] == 210760
        assert rows[0]["etag"] == "1b5f84ef3349aae942de60f3faf24680"
        assert rows[0]["last_modified"] == "2026-06-15T16:25:37+08:00"
        # 表头 / 分隔线 / TOTAL OBJECTS 一律跳过
        assert all("TOTAL" not in r["key"] and r["key"] != "KEY" for r in rows)

    def test_parse_garbage_returns_empty(self):
        from data_access.cos.remote import _parse_cos_cli_ls_output

        assert _parse_cos_cli_ls_output("") == []
        assert _parse_cos_cli_ls_output("not a table") == []

    def test_cos_cli_ls_invokes_cli_and_parses(self, monkeypatch):
        import data_access.cos.remote as cr

        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return MagicMock(returncode=0, stdout=_CLI_LS_TABLE)

        monkeypatch.setattr(cr.subprocess, "run", fake_run)
        rows = cr.cos_cli_ls("s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/")
        assert len(rows) == 2
        from data_access.cos.mirror import COS_CLI

        assert seen["cmd"][0] == COS_CLI
        assert seen["cmd"][1] == "ls"
        # s3:// 自动归一 cos://
        assert seen["cmd"][2].startswith("cos://qs-cold/")

    def test_cos_cli_ls_failure_returns_empty(self, monkeypatch):
        import subprocess

        import data_access.cos.remote as cr

        monkeypatch.setattr(
            cr.subprocess,
            "run",
            MagicMock(side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)),
        )
        assert cr.cos_cli_ls("s3://qs-cold/x/") == []
        monkeypatch.setattr(
            cr.subprocess, "run", MagicMock(returncode=1, stdout="", stderr="denied")
        )
        assert cr.cos_cli_ls("s3://qs-cold/x/") == []

    def test_cos_cli_head_single_object(self, monkeypatch):
        import data_access.cos.remote as cr

        monkeypatch.setattr(
            cr, "cos_cli_ls", lambda uri, **kw: (
                [{"key": "a.parquet", "size": 10, "etag": "e" * 32,
                  "last_modified": "2026-06-15T16:25:37+08:00"}]
                if uri.rstrip("/").endswith("a.parquet") else []
            )
        )
        head = cr.cos_cli_head("s3://qs-cold/prefix/a.parquet")
        assert head and head["size"] == 10
        assert cr.cos_cli_head("s3://qs-cold/prefix/missing.parquet") is None
        assert cr.cos_cli_head("s3://qs-cold/prefix/") is None  # 目录 URI 拒绝


# ===========================================================================
# store 钩子：boto3 失败 → CLI fallback（mock，无网络）
# ===========================================================================

class TestStoreCosHooks:
    def test_local_path_short_circuits(self, tmp_path):
        store = _store(tmp_path)
        assert store._cos_list_objects("/local/dir/") == []
        assert store._cos_head_object("/local/file.parquet") is None

    def test_unauthorized_bucket_returns_empty(self, tmp_path):
        store = _store(tmp_path)
        # 不在 allowed_s3_prefixes 白名单 → 不 LIST（授权语义与 authorize_s3_path 一致）
        assert store._cos_list_objects("s3://attacker-bucket/x/") == []

    def test_list_boto3_success(self, tmp_path):
        store = _store(tmp_path)
        fake = MagicMock()
        fake._s3.return_value.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "clean_data/ashare/lqtp_data/StockDailyBar/a.parquet",
                 "ETag": '"etag-a"', "Size": 100, "LastModified": "2026-01-01"},
            ],
            "IsTruncated": False,
        }
        with patch("data_access.read.object_store.COSObjectStore", return_value=fake):
            objs = store._cos_list_objects(
                "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/"
            )
        assert len(objs) == 1
        assert objs[0].uri == (
            "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/a.parquet"
        )
        assert objs[0].content_length == 100
        assert objs[0].etag == "etag-a"
        assert objs[0].source == "exact_list"

    def test_list_falls_back_to_cli_on_boto3_failure(self, tmp_path, monkeypatch):
        import data_access.cos.remote as cr

        store = _store(tmp_path)
        # boto3 层抛 NoSuchBucket（跨账号 IAM 不可见的实际表现）
        broken = MagicMock()
        broken._s3.return_value.list_objects_v2.side_effect = Exception("NoSuchBucket")
        monkeypatch.setattr(
            cr, "cos_cli_ls",
            lambda uri, **kw: [
                {"key": "clean_data/ashare/lqtp_data/StockDailyBar/b.parquet",
                 "size": 2048, "etag": "e" * 32, "last_modified": None}
            ],
        )
        with patch("data_access.read.object_store.COSObjectStore", return_value=broken):
            objs = store._cos_list_objects(
                "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/"
            )
        assert len(objs) == 1
        assert objs[0].content_length == 2048
        assert objs[0].source == "exact_list"

    def test_list_cli_prefix_gets_trailing_slash(self, tmp_path, monkeypatch):
        """coscli ls 无尾斜杠返回 DIR 行 → store 必须补 ``/`` 再 LIST。"""
        import data_access.cos.remote as cr

        store = _store(tmp_path)
        broken = MagicMock()
        broken._s3.return_value.list_objects_v2.side_effect = Exception("NoSuchBucket")
        captured = {}

        def spy(uri, **kw):
            captured["uri"] = uri
            return []

        monkeypatch.setattr(cr, "cos_cli_ls", spy)
        with patch("data_access.read.object_store.COSObjectStore", return_value=broken):
            store._cos_list_objects(
                "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar"
            )
        assert captured["uri"].endswith("StockDailyBar/")

    def test_head_cli_fallback(self, tmp_path, monkeypatch):
        """boto3 HEAD 被 gate/无凭证关闭时 → CLI 网关 fallback。"""
        import data_access.cos.remote as cr
        from data_access.read import read_contract

        store = _store(tmp_path)
        monkeypatch.setenv("DATA_ACCESS_REMOTE_SNAPSHOT_META", "1")
        # boto3 HEAD 失败（None）→ CLI fallback
        monkeypatch.setattr(
            read_contract,
            "_remote_object_meta",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            cr, "cos_cli_head",
            lambda uri, **kw: {"key": "x.parquet", "size": 4096,
                               "etag": "e" * 32, "last_modified": None},
        )
        obj = store._cos_head_object(
            "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/x.parquet"
        )
        assert obj is not None
        assert obj.content_length == 4096
        assert obj.source == "exact_head"
        assert obj.uri == (
            "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/x.parquet"
        )

    def test_head_glob_rejected(self, tmp_path, monkeypatch):
        import data_access.cos.remote as cr

        store = _store(tmp_path)
        called = []
        monkeypatch.setattr(
            cr, "cos_cli_head", lambda uri, **kw: called.append(uri) or None
        )
        assert store._cos_head_object("s3://qs-cold/x/*.parquet") is None
        assert called == []  # 通配路径绝不进 CLI

    def test_total_bytes_populated_after_hooks(self, tmp_path, monkeypatch):
        """端到端最小链：LIST 钩子回填 size → snapshot.total_bytes 非 None
        （governor 据此做真实 scan-bytes 准入，不再按未知成本拒绝）。"""
        import data_access.cos.remote as cr

        store = _store(tmp_path)
        paths = [
            "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/a.parquet",
            "s3://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/b.parquet",
        ]
        monkeypatch.setattr(
            cr, "cos_cli_ls",
            lambda uri, **kw: [
                {"key": "clean_data/ashare/lqtp_data/StockDailyBar/a.parquet",
                 "size": 100, "etag": "a" * 32, "last_modified": None},
                {"key": "clean_data/ashare/lqtp_data/StockDailyBar/b.parquet",
                 "size": 200, "etag": "b" * 32, "last_modified": None},
            ],
        )
        snap = store._pipeline.resolve_snapshot("ds", paths=paths)
        assert snap.object_count == 2
        assert snap.total_bytes == 300
        from data_access.snapshot.fidelity import SnapshotFidelity

        assert snap.fidelity == SnapshotFidelity.REMOTE_VERSION_ID