# -*- coding: utf-8 -*-
"""store 对象存储路由 contract 测试（UPSTREAM_FIX_PLAN 问题二）。

覆盖：
    - write_arrow → object_store 数据集：不可变 generation 上传 + CURRENT 翻转
    - publish_from_staging → object_store published：staging 内容单代次发布
    - 失败语义：上传失败 CURRENT 不变；读端沿 CURRENT→manifest→精确对象
"""
from __future__ import annotations

import json
from textwrap import dedent
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import DataError, ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore, reset_store


@pytest.fixture
def object_store_env(tmp_path: Path, monkeypatch):
    """带 object_store published + staging 配对数据集的 store。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    objects_root = tmp_path / "objects"
    monkeypatch.setenv("DA_OBJECT_STORE_LOCAL_ROOT", str(objects_root))
    monkeypatch.setenv("TEST_STAGING_ROOT", str(workspace / "staging"))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "shw")
    monkeypatch.setenv("QUANT_OPERATOR", "shw@test")
    monkeypatch.setenv("QUANT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))

    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        factor_pool_stg:
          kind: parametric
          access_mode: staging
          layout: hive
          root_template: ${TEST_STAGING_ROOT}/${RUN_NAMESPACE}/factors/{factor_id}
          glob_template: "year=*/*.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true

        factor_pool:
          kind: parametric
          access_mode: published
          layout: hive
          storage:
            type: object_store
            uri: cos://qs-cold/factor_pool/{factor_id}
            current_key: CURRENT.json
            layout_version: 1
          root_template: ${TEST_STAGING_ROOT}/unused/{factor_id}
          glob_template: "year=*/*.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
    """).strip() + "\n", encoding="utf-8")

    reset_store()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store, tmp_path, objects_root
    engine.close()
    reset_store()


def _sample(n: int = 4, year: int = 2024) -> pa.Table:
    return pa.table({
        "datetime": pd.to_datetime([f"{year}-0{i+1}-01" for i in range(n)]),
        "asset": [f"SYM{i:02d}" for i in range(n)],
        "value": [i * 1.0 for i in range(n)],
        "year": [year] * n,
    })


def _current_of(objects_root: Path, prefix: str) -> str | None:
    from data_access.read.object_store import LocalObjectStore

    st = LocalObjectStore(objects_root)
    key = f"{prefix}/CURRENT.json"
    head = st.head_object(key)
    if head is None:
        return None
    blob = st.range_read(key, offset=0, length=int(head["size"]))
    return json.loads(blob.decode("utf-8"))["generation_id"]


def test_write_arrow_object_store_publishes_generation(object_store_env):
    store, _tmp, objects_root = object_store_env
    result = store.write_arrow(
        "factor_pool", _sample(4), factor_id="mom_3d", mode="overwrite"
    )
    assert result["rows"] == 4
    assert result["generation_id"]
    assert result["path"] == "s3://qs-cold/factor_pool/mom_3d"
    gid = _current_of(objects_root, "factor_pool/mom_3d")
    assert gid == result["generation_id"]
    # 精确对象存在（year=2024 分区文件已上传）
    from data_access.read.object_store import LocalObjectStore

    st = LocalObjectStore(objects_root)
    keys = [k for k in st.list_objects("factor_pool/mom_3d")]
    assert any(k.endswith(".parquet") for k in keys)
    assert f"factor_pool/mom_3d/{result['generation_id']}/_manifest.json" in keys


def test_write_arrow_object_store_failed_upload_keeps_current(
    object_store_env, monkeypatch
):
    store, _tmp, objects_root = object_store_env
    r1 = store.write_arrow("factor_pool", _sample(3), factor_id="f1")
    old_gid = _current_of(objects_root, "factor_pool/f1")

    # 第二次写：add_object 阶段炸 → abort → CURRENT 不变
    from data_access.write.object_store_generation_publisher import (
        ObjectStoreGenerationPublisher,
    )

    orig_add = ObjectStoreGenerationPublisher.add_object

    def boom(self, generation_id, key, data, **kw):
        raise OSError("simulated upload failure")

    monkeypatch.setattr(ObjectStoreGenerationPublisher, "add_object", boom)
    with pytest.raises(OSError):
        store.write_arrow("factor_pool", _sample(2), factor_id="f1")
    monkeypatch.setattr(ObjectStoreGenerationPublisher, "add_object", orig_add)
    assert _current_of(objects_root, "factor_pool/f1") == old_gid
    assert old_gid == r1["generation_id"]


def test_publish_from_staging_object_store(object_store_env):
    store, _tmp, objects_root = object_store_env
    store.write_arrow(
        "factor_pool_stg", _sample(4), factor_id="m2",
        mode="overwrite", partition_by=["year"],
    )
    result = store.publish_from_staging("factor_pool_stg", "factor_pool", factor_id="m2")
    assert result["rows"] == 4
    assert result["target_path"] == "s3://qs-cold/factor_pool/m2"
    assert result["generation_id"]
    gid = _current_of(objects_root, "factor_pool/m2")
    assert gid == result["generation_id"]


def test_publish_from_staging_object_store_empty_staging_raises(object_store_env):
    store, _tmp, _objects_root = object_store_env
    with pytest.raises((DataError, ValidationError)):
        store.publish_from_staging("factor_pool_stg", "factor_pool", factor_id="empty")


def test_read_side_resolves_current_only(object_store_env):
    """读端语义：CURRENT→manifest→精确对象；未完成 generation 不可见。"""
    store, _tmp, objects_root = object_store_env
    store.write_arrow("factor_pool", _sample(4), factor_id="r1")
    from data_access.write.t8_publish import resolve_current_objects

    store_backend = __import__("data_access.read.object_store", fromlist=["LocalObjectStore"]).LocalObjectStore(objects_root)
    keys = resolve_current_objects(store_backend, "factor_pool/r1")
    assert keys, "CURRENT 指向的 generation 必须可解析出精确对象列表"
    assert all("*" not in k and "?" not in k for k in keys)
