# -*- coding: utf-8 -*-
"""R14 #1：DataAccess factor_matrix 读路径跟随 FactorEngine generation 指针。

外部 AI 复查第二轮 P0：FactorEngine 把 matrix 写成
``freq=1d/generation/<gid>/year=*/month=*/data.parquet``，而 DataAccess registry
的 glob 还是 ``year=*/month=*/data.parquet``——读路径完全找不到数据；且
``_matrix_available_columns`` 只认 registry schema（datetime/asset），动态因子列
永远判 MatrixCoverageMiss → matrix 路由永远 fallback factor-major。

R14 #1 修复（只改 DataAccess 读侧）：
  * ``ParametricDataset.resolve_paths``：generation_pointer 数据集先读
    ``manifest.json`` 的 ``generation``，只解析 ``generation/<gid>/<glob>`` 那一代
    （绝不 ``generation/*`` 通配，那会把 current+previous 混读）；
  * manifest.generation 指向缺失目录 → ``DataError`` fail-closed（绝不 legacy
    fallback 把 previous/orphan 数据捞出来）；
  * 动态因子列来自 ``manifest["factors"].keys()``（不依赖 registry schema）；
  * generation id 纳入 read 审计参数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import DataError
from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")


def _store(tmp_path) -> tuple[DataAccessStore, Path]:
    yaml_cfg = tmp_path / "datasets.yaml"
    yaml_cfg.write_text(
        f"""
factor_matrix:
  kind: parametric
  access_mode: published
  layout: hive
  root_template: "{tmp_path}/matrix/universe={{universe}}/freq={{frequency}}"
  authorized_root: "{tmp_path}/matrix"
  glob_template: "year=*/month=*/data.parquet"
  partition_columns: [year, month]
  storage_format: long
  params_schema:
    universe: str
    frequency: str
  time_column: datetime
  instrument_column: asset
  hive_partitioning: true
  union_by_name: true
  schema:
    datetime: timestamp
    asset: string
  generation_pointer: true
""",
        encoding="utf-8",
    )
    store = DataAccessStore(load_registry(yaml_cfg), DuckDBEngine(threads=2))
    return store, tmp_path / "matrix"


def _write_generation(
    base: Path, gid: str, value: float, *, factors: tuple[str, ...] = ("f",)
) -> None:
    """写一代：``generation/<gid>/year=2024/month=1/data.parquet``（宽表列）。"""
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-15"]),
            "asset": ["A"],
        }
    )
    for f in factors:
        df[f] = value
    part = base / "generation" / gid / "year=2024" / "month=1"
    part.mkdir(parents=True)
    df.to_parquet(part / "data.parquet", index=False)


def _write_manifest(base: Path, gid: str, *, factors: tuple[str, ...] = ("f",)) -> None:
    base.mkdir(parents=True, exist_ok=True)
    manifest = {
        "universe": "u",
        "frequency": "1d",
        "factors": {f: {"semantic_digest": "d" * 16} for f in factors},
        "generation": gid,
        "manifest_version": 1,
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_values(store: DataAccessStore, *, universe: str = "u") -> pd.DataFrame:
    handle = store.read_factors(
        ["f"],
        layout="wide",
        universe=universe,
        frequency="1d",
        route_matrix_threshold=1,
    )
    return handle.to_arrow().to_pandas()


def test_r14_da_reads_current_generation(tmp_path):
    store, root = _store(tmp_path)
    base = root / "universe=u" / "freq=1d"
    _write_generation(base, "GEN_A", 1.0)
    _write_manifest(base, "GEN_A")
    frame = _read_values(store)
    assert list(frame["f"]) == [1.0]


def test_r14_da_reads_only_current_after_pointer_flip(tmp_path):
    store, root = _store(tmp_path)
    base = root / "universe=u" / "freq=1d"
    # gen A 与 gen B 同时在盘（B 是更新的发布代）
    _write_generation(base, "GEN_A", 1.0)
    _write_generation(base, "GEN_B", 2.0)
    _write_manifest(base, "GEN_A")
    assert list(_read_values(store)["f"]) == [1.0]
    # 切指针到 B → 只读 B（A 还在盘上但不可见）
    _write_manifest(base, "GEN_B")
    assert list(_read_values(store)["f"]) == [2.0]


def test_r14_da_ignores_orphan_generation(tmp_path):
    store, root = _store(tmp_path)
    base = root / "universe=u" / "freq=1d"
    _write_generation(base, "GEN_CURRENT", 1.0)
    _write_generation(base, "GEN_ORPHAN", 999.0)  # 未被 manifest 引用
    _write_manifest(base, "GEN_CURRENT")
    assert list(_read_values(store)["f"]) == [1.0]


def test_r14_da_missing_generation_dir_hard_fails(tmp_path):
    store, root = _store(tmp_path)
    base = root / "universe=u" / "freq=1d"
    # 先有 legacy 诱饵（base 直接放分区），再让 manifest 指向缺失目录——绝不能
    # fallback 读诱饵。
    legacy = base / "year=2024" / "month=1" / "data.parquet"
    legacy.parent.mkdir(parents=True)
    pd.DataFrame(
        {"datetime": pd.to_datetime(["2024-01-15"]), "asset": ["A"], "f": [777.0]}
    ).to_parquet(legacy, index=False)
    _write_manifest(base, "GEN_MISSING")
    with pytest.raises(DataError, match="generation"):
        _read_values(store)


def test_r14_da_available_columns_from_manifest(tmp_path):
    store, root = _store(tmp_path)
    base = root / "universe=u" / "freq=1d"
    _write_generation(base, "GEN_A", 1.0, factors=("f1", "f2", "f3"))
    _write_manifest(base, "GEN_A", factors=("f1", "f2", "f3"))
    cols = store._matrix_available_columns(universe="u", frequency="1d")
    # registry schema(datetime/asset) + manifest 动态因子列
    assert {"datetime", "asset", "f1", "f2", "f3"} <= set(cols)
    # 精确投影走 matrix 路由（不再 MatrixCoverageMiss fallback）
    handle = store.read_factors(
        ["f2"],
        layout="wide",
        universe="u",
        frequency="1d",
        route_matrix_threshold=1,
    )
    frame = handle.to_arrow().to_pandas()
    assert set(frame.columns) == {"datetime", "asset", "f2"}
