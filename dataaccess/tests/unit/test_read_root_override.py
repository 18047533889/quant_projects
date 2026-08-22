"""read_root 灰度切读与 bucket_values 元参数过滤。"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def bucket_dataset_store(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    table = pa.table({"x": [1, 2], "y": [3.0, 4.0]})
    pq.write_table(table, root / "data.parquet")

    override = root / "alt_layout"
    override.mkdir()
    pq.write_table(pa.table({"x": [9], "y": [9.0]}), override / "data.parquet")

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

    registry = load_registry(str(config))
    store = DataAccessStore(registry, DuckDBEngine())
    return store, "ds", str(override)


def test_read_root_override_dev_only(bucket_dataset_store, monkeypatch):
    store, name, override = bucket_dataset_store
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    paths = store._resolve_paths(
        store._registry.get(name),
        {"read_root": override},
    )
    assert "alt_layout" in paths[0]


def test_read_root_blocked_in_production(bucket_dataset_store, monkeypatch):
    store, name, override = bucket_dataset_store
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    with pytest.raises(Exception, match="read_root"):
        store._resolve_paths(
            store._registry.get(name),
            {"read_root": override},
        )


def test_bucket_values_does_not_break_static_resolve(bucket_dataset_store):
    store, name, _override = bucket_dataset_store
    paths = store._resolve_paths(
        store._registry.get(name),
        {"bucket_values": [0, 1]},
    )
    assert paths
