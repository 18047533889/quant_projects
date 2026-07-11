# -*- coding: utf-8 -*-
"""ReadResult / DataSnapshot / params 校验单元测试。"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.exceptions import ValidationError
from data_access.params_validation import validate_params, ParamSpec
from data_access.read_contract import build_data_snapshot, canonicalize_params


def test_canonicalize_params_stable():
    a = canonicalize_params({"factor_id": "mom_3d", "version": 1})
    b = canonicalize_params({"version": 1, "factor_id": "mom_3d"})
    assert a == b


def test_validate_params_rejects_path_traversal():
    specs = {"factor_id": ParamSpec(name="factor_id", type="str", path_segment=True)}
    with pytest.raises(ValidationError, match="非法路径"):
        validate_params("factor_lake", specs, {"factor_id": "../escape"})


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

    from data_access.engine import DuckDBEngine
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

    from data_access.engine import DuckDBEngine
    from data_access.registry import load_registry
    from data_access.store import DataAccessStore

    store = DataAccessStore(load_registry(str(config)), DuckDBEngine())
    result = store.read_result("ds", columns=["x", "y"])
    assert result.table.num_rows == 2
    assert result.snapshot.snapshot_id
    assert result.stats.rows == 2
