# -*- coding: utf-8
"""run_many_from_config / write_targets 扩展测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.write_targets import LocalParquetWriteTarget, resolve_write_target


def _write_factor_yaml(tmp_path: Path, name: str, expr: str, root: Path) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""
factor:
  name: {name}
  expr: {expr}
data_source:
  type: parquet_kline
  root: {root}
  timestamp_column: datetime
  instrument_column: asset
  fields:
    close: close
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_local_parquet_write_target_upserts_partition(tmp_path):
    target = LocalParquetWriteTarget(lake_root=tmp_path / "lake")
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "asset": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )
    result = target.write_factor_frame("f1", frame)
    assert result["rows_written"] == 2
    part = tmp_path / "lake" / "factors" / "f1" / "year=2024" / "data.parquet"
    assert part.is_file()


def test_resolve_write_target_local_and_staging():
    local = resolve_write_target("local", lake_root="/tmp/lake")
    assert isinstance(local, LocalParquetWriteTarget)
    staging = resolve_write_target("staging:custom_staging")
    assert staging.dataset == "custom_staging"


def test_run_many_from_config_groups_by_data_scope(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "close": [1.0],
        }
    ).to_parquet(root / "data.parquet")

    cfg_a = _write_factor_yaml(tmp_path, "fa", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "fb", "close * 2", root)
    out = FactorEngine.run_many_from_config([cfg_a, cfg_b])
    assert set(out["results"]) == {"fa", "fb"}
    assert "fa" in out["runs"] and "fb" in out["runs"]


def test_materialize_many_from_config_invokes_each(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "close": [1.0],
        }
    ).to_parquet(root / "data.parquet")

    cfg_a = _write_factor_yaml(tmp_path, "ma", "close", root)
    cfg_b = _write_factor_yaml(tmp_path, "mb", "close", root)
    seen: list[str] = []

    def _fake_materialize(self, factor, **kwargs):
        seen.append(factor.name)
        return {"materialization": {"factor_id": factor.name, "rows_written": 1}}

    monkeypatch.setattr(FactorEngine, "materialize", _fake_materialize)
    out = FactorEngine.materialize_many_from_config([cfg_a, cfg_b], batch_run=False)
    assert set(seen) == {"ma", "mb"}
    mats = out["materializations"]
    assert mats["ma"]["materialization"]["factor_id"] == "ma"
    assert mats["mb"]["materialization"]["factor_id"] == "mb"


def test_clickhouse_write_target_delegates_to_materializer(monkeypatch):
    from factor_engine.storage.write_targets import ClickHouseWriteTarget

    captured: dict = {}

    class FakeSummary:
        factor_id = "f1"
        rows_written = 3
        table = "factor_values"
        database = "default"

    class FakeCHMat:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def materialize(self, **kwargs):
            captured["materialize"] = kwargs
            return FakeSummary()

    monkeypatch.setattr(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer",
        FakeCHMat,
    )
    target = ClickHouseWriteTarget(table="fv", host="h1")
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"]),
            "asset": ["A"],
            "value": [1.0],
        }
    )
    out = target.write_factor_frame("f1", frame, factor_version="v1")
    assert out["rows_written"] == 3
    assert captured["init"]["table"] == "fv"
    assert captured["materialize"]["factor_id"] == "f1"
