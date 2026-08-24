"""core/storage.py + core/duckdb_capabilities.py 单测。"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from data_access.core.duckdb_capabilities import (
    detect_duckdb_capabilities,
    get_duckdb_capabilities,
)
from data_access.core.duckdb_config import DuckDBConfig, apply_pragmas
from data_access.core.storage import (
    StorageSpec,
    is_remote_storage,
    parse_storage_spec,
    resolve_storage_for_dataset,
)
from data_access.registry import load_registry


def test_capabilities_detection():
    caps = get_duckdb_capabilities()
    assert caps.version == (1, 5)
    assert caps.supports_create_secret is True
    assert detect_duckdb_capabilities().version_str == "1.5"


def test_pragma_skip_deprecated_object_cache():
    con = duckdb.connect(":memory:")
    apply_pragmas(con, DuckDBConfig(threads=2, enable_object_cache=True))
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 2


def test_parse_storage_spec():
    spec = parse_storage_spec({"type": "s3", "uri": "s3://b/k"})
    assert spec.type == "s3"
    assert spec.is_remote
    spec2 = parse_storage_spec("local")
    assert spec2.type == "local"
    assert not spec2.is_remote
    with pytest.raises(Exception):
        parse_storage_spec({"type": "nfs"})


def test_resolve_storage_for_dataset(tmp_path):
    (tmp_path / "datasets.yaml").write_text(
        f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
  storage: {{type: s3, uri: s3://bucket/x}}
b:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
""",
        encoding="utf-8",
    )
    reg = load_registry(tmp_path / "datasets.yaml")
    spec_a = resolve_storage_for_dataset(reg.get("a"))
    assert spec_a.type == "s3"
    assert is_remote_storage(reg.get("a")) is True
    spec_b = resolve_storage_for_dataset(reg.get("b"))
    assert spec_b.type == "local"


def test_s3_secret_via_create_secret():
    from data_access.cos.remote import S3Credentials
    from data_access.cos.s3_duckdb import apply_s3_credentials

    con = duckdb.connect(":memory:")
    creds = S3Credentials(
        access_key_id="ak", secret_access_key="sk",
        endpoint="cos.ap-guangzhou.myqcloud.com", region="ap-guangzhou",
    )
    apply_s3_credentials(con, creds)
    row = con.execute(
        "SELECT name FROM duckdb_secrets() WHERE name='data_access_cos'"
    ).fetchone()
    assert row is not None and "data_access_cos" in row[0]
