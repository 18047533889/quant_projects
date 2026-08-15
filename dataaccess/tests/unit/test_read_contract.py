# -*- coding: utf-8 -*-
"""ReadResult / DataSnapshot / params 校验单元测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.exceptions import ValidationError
from data_access.registry.params_validation import validate_params, ParamSpec
from data_access.read.read_contract import (
    FileVersion,
    build_data_snapshot,
    build_file_manifest,
    canonicalize_params,
    file_manifest_hash,
    file_versions_from_manifest,
    schema_hash_from_decl,
)


def test_canonicalize_params_stable():
    a = canonicalize_params({"factor_id": "mom_3d", "version": 1})
    b = canonicalize_params({"version": 1, "factor_id": "mom_3d"})
    assert a == b


def test_identity_digests_are_full_sha256():
    assert len(schema_hash_from_decl({"x": "int"})) == 64
    assert len(file_manifest_hash([FileVersion(path="a", size=1)])) == 64


def test_canonicalize_params_rejects_non_string_key_even_without_collision():
    with pytest.raises(ValidationError, match="keys must be strings"):
        canonicalize_params({1: "int"})


def test_canonicalize_params_rejects_nested_non_string_key():
    with pytest.raises(ValidationError, match="keys must be strings"):
        canonicalize_params({"outer": {1: "int"}})


def test_canonicalize_params_rejects_unknown_types():
    class Unknown:
        pass

    with pytest.raises(ValidationError, match="Cannot encode|unsupported parameter value"):
        canonicalize_params({"value": Unknown()})


def test_file_manifest_rejects_unknown_metadata_types():
    class Unknown:
        pass

    with pytest.raises(ValidationError, match="Cannot encode"):
        file_manifest_hash([FileVersion(path="a", last_modified=Unknown())])


def test_validate_params_rejects_path_traversal():
    specs = {"factor_id": ParamSpec(name="factor_id", type="str", path_segment=True)}
    with pytest.raises(ValidationError, match="非法路径"):
        validate_params("factor_lake", specs, {"factor_id": "../escape"})


def test_snapshot_params_are_recursively_immutable_and_bound_to_canonical_bytes():
    original = {"nested": {"values": [1, {"name": "before"}]}}
    snap = build_data_snapshot(
        dataset="ds",
        registry_hash="reg1",
        schema={"x": "int"},
        paths=[],
        params=original,
        files=(),
    )
    identity = snap.snapshot_id
    canonical_bytes = snap.params_canonical_bytes

    original["nested"]["values"][1]["name"] = "after"
    original["nested"]["values"].append(2)

    assert snap.snapshot_id == identity
    assert snap.params_canonical_bytes == canonical_bytes
    assert snap.to_dict()["params"] == {
        "nested": {"values": [1, {"name": "before"}]}
    }
    with pytest.raises(TypeError):
        snap.params[0][1][0] = "mutated"


def test_remote_wildcard_never_calls_object_head_in_either_manifest_builder():
    wildcard = "s3://bucket/table/*.parquet"
    manifest = SimpleNamespace(files=())
    with patch(
        "data_access.read.read_contract._remote_snapshot_meta_enabled",
        return_value=True,
    ), patch(
        "data_access.read.read_contract._remote_object_meta",
        side_effect=AssertionError("wildcard must not be sent to object HEAD"),
    ) as remote_head:
        direct = build_file_manifest([wildcard])
        from_manifest = file_versions_from_manifest(manifest, [wildcard])

    remote_head.assert_not_called()
    assert direct == (FileVersion(path=wildcard),)
    assert from_manifest == direct


def test_build_data_snapshot_accepts_prebuilt_files(tmp_path: Path):
    root = tmp_path / "lake"
    root.mkdir()
    path = root / "a.parquet"
    pq.write_table(pa.table({"x": [1]}), path)
    from data_access.read.read_contract import build_file_manifest

    files = build_file_manifest([str(path)])
    snap = build_data_snapshot(
        dataset="ds",
        registry_hash="reg1",
        schema={"x": "int"},
        paths=["/unused/glob/*.parquet"],
        files=files,
    )
    assert snap.files == files
    assert snap.file_manifest_hash


def test_build_data_snapshot_differs_by_params(tmp_path: Path):
    root = tmp_path / "lake"
    root.mkdir()
    pq.write_table(pa.table({"x": [1]}), root / "a.parquet")

    snap_a = build_data_snapshot(
        dataset="ds",
        registry_hash="reg1",
        schema={"x": "int"},
        paths=[str(root / "*.parquet")],
        params={"factor_id": "a"},
    )
    snap_b = build_data_snapshot(
        dataset="ds",
        registry_hash="reg1",
        schema={"x": "int"},
        paths=[str(root / "*.parquet")],
        params={"factor_id": "b"},
    )
    assert snap_a.snapshot_id != snap_b.snapshot_id


def test_read_asof_filters_upper_bound(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    pq.write_table(
        pa.table({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]}),
        root / "data.parquet",
    )
    config = tmp_path / "datasets.yaml"
    config.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  root: {root}
  glob: "**/*.parquet"
  time_column: x
  instrument_column: y
  schema:
    x: int
    y: double
""",
        encoding="utf-8",
    )

    from data_access.core.engine import DuckDBEngine
    from data_access.registry import load_registry
    from data_access.store import DataAccessStore

    store = DataAccessStore(load_registry(str(config)), DuckDBEngine())
    result = store.read_asof("ds", as_of=2, columns=["x", "y"])
    assert result.table.num_rows == 2
    assert result.snapshot.snapshot_id


def test_read_result_roundtrip(tmp_path: Path):
    root = tmp_path / "data2"
    root.mkdir()
    pq.write_table(
        pa.table({"x": [1, 2], "y": [3.0, 4.0]}),
        root / "data.parquet",
    )
    config = tmp_path / "datasets2.yaml"
    config.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  root: {root}
  glob: "**/*.parquet"
  time_column: x
  instrument_column: y
  schema:
    x: int
    y: double
""",
        encoding="utf-8",
    )

    from data_access.core.engine import DuckDBEngine
    from data_access.registry import load_registry
    from data_access.store import DataAccessStore

    store = DataAccessStore(load_registry(str(config)), DuckDBEngine())
    result = store.read_result("ds", columns=["x", "y"])
    assert result.table.num_rows == 2
    assert result.snapshot.snapshot_id
    assert result.stats.rows == 2
