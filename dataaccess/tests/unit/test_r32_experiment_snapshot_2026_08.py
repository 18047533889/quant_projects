"""R32 P0-078..082: ExperimentSnapshot 分离 source/execution/security ID。

旧实现：单一 snapshot_id 混合所有依赖（source data + code build + security scope）。
新实现：三个独立字段 + snapshot_id 向后兼容：
    - source_snapshot_id: 数据源快照（datasets, calendar, universe, semantic_contract）
    - execution_context_id: 执行上下文（code_build_sha, registry）
    - security_scope_id: 安全范围（security_scope_digest）
"""
from __future__ import annotations

from unittest.mock import Mock, patch

from data_access.r30.experiment_snapshot import ExperimentDataSnapshot


def _mock_store():
    """创建不触发真实 calendar 加载的 mock store。"""
    store = Mock()
    store.manifest_version = Mock(return_value=None)
    store.contract_ir_fingerprint = Mock(return_value="semantic_abc")
    store.registry_fingerprint = Mock(return_value="registry_xyz")
    return store


def test_experiment_snapshot_has_separated_ids():
    """ExperimentDataSnapshot 有三个独立 ID + 向后兼容的 snapshot_id。"""
    store = _mock_store()

    # 跳过 calendar 加载
    with patch("data_access.r30.experiment_snapshot.CalendarSnapshot.from_store", return_value=None):
        snapshot = ExperimentDataSnapshot.build(
            store=store,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v1", "ds2": "v2"},
            calendar=None,
            universe=None,
        )

    # R32: 三个独立 ID 必须存在
    assert hasattr(snapshot, "source_snapshot_id")
    assert hasattr(snapshot, "execution_context_id")
    assert hasattr(snapshot, "security_scope_id")
    assert snapshot.source_snapshot_id
    assert snapshot.execution_context_id
    assert snapshot.security_scope_id

    # 向后兼容：snapshot_id 仍存在
    assert snapshot.snapshot_id

    # 三个 ID 都不同（分别聚合不同依赖）
    assert snapshot.source_snapshot_id != snapshot.execution_context_id
    assert snapshot.source_snapshot_id != snapshot.security_scope_id
    assert snapshot.execution_context_id != snapshot.security_scope_id


def test_experiment_snapshot_source_id_independent_of_code():
    """source_snapshot_id 不受 code_build_sha / registry 变化影响。"""
    store1 = _mock_store()
    store1.registry_fingerprint = Mock(return_value="registry_v1")

    store2 = _mock_store()
    store2.registry_fingerprint = Mock(return_value="registry_v2")

    with patch("data_access.r30.experiment_snapshot.CalendarSnapshot.from_store", return_value=None):
        snap1 = ExperimentDataSnapshot.build(
            store=store1,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v1"},
        )

        snap2 = ExperimentDataSnapshot.build(
            store=store2,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v1"},
        )

    # source_snapshot_id 不变（数据源未变）
    assert snap1.source_snapshot_id == snap2.source_snapshot_id

    # execution_context_id 改变（registry 变了）
    assert snap1.execution_context_id != snap2.execution_context_id

    # 向后兼容的 snapshot_id 改变（包含全部依赖）
    assert snap1.snapshot_id != snap2.snapshot_id


def test_experiment_snapshot_execution_context_id_independent_of_data():
    """execution_context_id 不受 datasets 变化影响。"""
    store = _mock_store()

    with patch("data_access.r30.experiment_snapshot.CalendarSnapshot.from_store", return_value=None):
        snap1 = ExperimentDataSnapshot.build(
            store=store,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v1"},
        )

        snap2 = ExperimentDataSnapshot.build(
            store=store,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v2"},  # 数据版本变化
        )

    # execution_context_id 不变（代码/registry 未变）
    assert snap1.execution_context_id == snap2.execution_context_id

    # source_snapshot_id 改变（数据变了）
    assert snap1.source_snapshot_id != snap2.source_snapshot_id


def test_experiment_snapshot_to_dict_has_all_ids():
    """to_dict() 输出包含所有四个 ID 字段。"""
    store = _mock_store()

    with patch("data_access.r30.experiment_snapshot.CalendarSnapshot.from_store", return_value=None):
        snapshot = ExperimentDataSnapshot.build(
            store=store,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v1"},
        )

    d = snapshot.to_dict()
    assert "snapshot_id" in d
    assert "source_snapshot_id" in d
    assert "execution_context_id" in d
    assert "security_scope_id" in d
    assert d["snapshot_id"]
    assert d["source_snapshot_id"]
    assert d["execution_context_id"]
    assert d["security_scope_id"]


def test_experiment_snapshot_security_scope_id_stable():
    """security_scope_id 只聚合 security_scope_digest。"""
    store = _mock_store()

    with patch("data_access.r30.experiment_snapshot.CalendarSnapshot.from_store", return_value=None):
        snap1 = ExperimentDataSnapshot.build(
            store=store,
            experiment_id="exp123",
            market="ashare",
            datasets={"ds1": "v1"},
        )

        snap2 = ExperimentDataSnapshot.build(
            store=store,
            experiment_id="exp456",  # 不同 experiment_id
            market="us",  # 不同市场
            datasets={"ds2": "v2"},  # 不同 datasets
        )

    # security_scope_id 只依赖 security_scope_digest（与实验/数据/市场无关）
    # 如果 store 的 security_scope 相同，则 security_scope_id 相同
    assert snap1.security_scope_id == snap2.security_scope_id
