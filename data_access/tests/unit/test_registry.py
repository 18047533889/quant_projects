"""
registry 单元测试：YAML 加载、字段校验、参数化数据集解析。
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from data_access.exceptions import ValidationError
from data_access.registry import (
    ParametricDataset,
    StaticDataset,
    load_registry,
)


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_load_static_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("DA_TEST_ROOT", str(tmp_path / "data"))
    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        demo_day:
          kind: static
          access_mode: published
          layout: plain
          root: ${DA_TEST_ROOT}/day
          time_column: align_time
          instrument_column: ticker
    """)

    reg = load_registry(yaml_path)
    ds = reg.get("demo_day")
    assert isinstance(ds, StaticDataset)
    assert ds.access_mode == "published"
    assert ds.time_column == "align_time"
    assert str(ds.root) == str((tmp_path / "data" / "day").resolve())


def test_load_parametric_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("DA_LAKE", str(tmp_path / "lake"))
    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        my_lake:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: ${DA_LAKE}/factors/{factor_id}
          glob_template: "year=*/data.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
    """)

    reg = load_registry(yaml_path)
    ds = reg.get("my_lake")
    assert isinstance(ds, ParametricDataset)
    paths = ds.resolve_paths(factor_id="mom_5d")
    assert len(paths) == 1
    assert "mom_5d/year=*/data.parquet" in paths[0]


def test_parametric_missing_required_param(tmp_path):
    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        my_lake:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: /tmp/{factor_id}
          glob_template: "data.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
    """)
    reg = load_registry(yaml_path)
    with pytest.raises(ValidationError, match="缺参数"):
        reg.get("my_lake").resolve_paths()  # 没传 factor_id


def test_parametric_extra_param_rejected(tmp_path):
    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        my_lake:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: /tmp/{factor_id}
          glob_template: "data.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
    """)
    reg = load_registry(yaml_path)
    with pytest.raises(ValidationError, match="未登记参数"):
        reg.get("my_lake").resolve_paths(factor_id="x", oops="y")


def test_invalid_access_mode(tmp_path):
    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        bad:
          kind: static
          access_mode: world_readable
          layout: plain
          root: /tmp/foo
          time_column: t
          instrument_column: i
    """)
    with pytest.raises(ValidationError, match="access_mode"):
        load_registry(yaml_path)


def test_unknown_dataset_gives_helpful_error(tmp_path):
    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        known:
          kind: static
          access_mode: published
          layout: plain
          root: /tmp/foo
          time_column: t
          instrument_column: i
    """)
    reg = load_registry(yaml_path)
    with pytest.raises(ValidationError) as exc_info:
        reg.get("unknown")
    msg = str(exc_info.value)
    assert "unknown" in msg
    assert "known" in msg  # 错误信息要列出已注册的数据集


def test_missing_config_file():
    with pytest.raises(ValidationError, match="不存在"):
        load_registry("/nonexistent/path/datasets.yaml")


def test_namespaced_root_expands_run_namespace(tmp_path, monkeypatch):
    """${RUN_NAMESPACE} 占位符要替换成 resolve_namespace() 的值。"""
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "testuser")
    # 清 namespace cache
    from data_access import namespace as ns_mod
    ns_mod._git_branch.cache_clear()

    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        my_runs:
          kind: parametric
          access_mode: namespaced
          layout: plain
          root_template: /tmp/users/${RUN_NAMESPACE}/{job_id}
          glob_template: "**/*.parquet"
          params_schema:
            job_id: str
          time_column: timestamp
          instrument_column: symbol
    """)
    reg = load_registry(yaml_path)
    paths = reg.get("my_runs").resolve_paths(job_id="job42")
    assert "/users/testuser/" in paths[0]
    assert "job42" in paths[0]
