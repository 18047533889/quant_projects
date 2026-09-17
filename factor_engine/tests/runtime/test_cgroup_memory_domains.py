"""A process cap must never be paired with shared-cgroup usage."""
from dataclasses import replace

import pytest

from factor_engine.runtime import resource_governor as governor
from factor_engine.runtime.resource_broker import ResourceBroker

GIB = 1024**3


def hierarchy(tmp_path, monkeypatch, leaf_limit="max", parent_limit="max", parent_current=40):
    parent = tmp_path / "group"
    leaf = parent / "worker"
    leaf.mkdir(parents=True)
    for path, limit, current in ((leaf, leaf_limit, 30), (parent, parent_limit, parent_current)):
        (path / "memory.max").write_text(str(limit))
        (path / "memory.current").write_text(str(current * GIB))
    monkeypatch.setattr(governor, "_cgroup_v2_ancestor_dirs", lambda: (leaf, parent))
    return leaf, parent


def test_unlimited_leaf_does_not_hide_finite_ancestor_limit(tmp_path, monkeypatch):
    hierarchy(tmp_path, monkeypatch, parent_limit=48 * GIB)
    assert governor._read_cgroup_v2_max() == 48 * GIB


@pytest.mark.parametrize("parent_limit, parent_current, expected", [
    ("max", 40, 7 * GIB),
    (48 * GIB, 40, 7 * GIB),
    (48 * GIB, 47, GIB),
    (48 * GIB, 48, 0),
])
def test_live_headroom_pairs_each_limit_with_its_own_usage(
    tmp_path, monkeypatch, parent_limit, parent_current, expected
):
    hierarchy(tmp_path, monkeypatch, parent_limit=parent_limit, parent_current=parent_current)
    monkeypatch.setattr(governor, "_host_mem_available_bytes", lambda: 40 * GIB)
    monkeypatch.setattr(governor, "process_family_rss_bytes", lambda: GIB)
    assert governor.live_memory_headroom_bytes(
        hard_limit=8 * GIB, min_host_reserve_gb=0, min_host_reserve_fraction=0
    ) == expected


def test_broker_unlimited_shared_group_does_not_zero_process_budget(tmp_path, monkeypatch):
    hierarchy(tmp_path, monkeypatch)
    broker = ResourceBroker(hard_memory_limit=8 * GIB, cpu_slots=1,
                            min_host_reserve_gb=0, min_host_reserve_fraction=0)
    snapshot = replace(broker.snapshot(), cgroup_memory_current=30 * GIB,
                       host_mem_available=40 * GIB, host_mem_available_known=True,
                       process_rss=GIB, process_family_rss=GIB,
                       process_family_pss=GIB, rlimit_as_remaining=None)
    monkeypatch.setattr(broker, "_refresh", lambda force=False: snapshot)
    assert broker.execution_budget() == int(0.8 * 7 * GIB)
    assert snapshot.live_headroom == 7 * GIB


def test_finite_cgroup_with_unreadable_usage_cannot_be_treated_as_unlimited(tmp_path, monkeypatch):
    leaf, _ = hierarchy(tmp_path, monkeypatch, leaf_limit=8 * GIB)
    (leaf / "memory.current").unlink()
    assert governor._cgroup_v2_memory_remaining_bytes() == 0


def test_zero_snapshot_is_not_revived_by_second_cgroup_probe(tmp_path, monkeypatch):
    hierarchy(tmp_path, monkeypatch, parent_limit=48 * GIB, parent_current=48)
    broker = ResourceBroker(hard_memory_limit=8 * GIB, cpu_slots=1,
                            min_host_reserve_gb=0, min_host_reserve_fraction=0)
    snapshot = replace(broker.snapshot(), cgroup_memory_remaining=0,
                       host_mem_available=40 * GIB, host_mem_available_known=True,
                       process_rss=GIB, process_family_rss=GIB,
                       process_family_pss=GIB, rlimit_as_remaining=None)
    monkeypatch.setattr(broker, "_refresh", lambda force=False: snapshot)
    monkeypatch.setattr(governor, "_cgroup_v2_memory_remaining_bytes", lambda: None)
    assert snapshot.live_headroom == 0
    assert broker.execution_budget() == 0


@pytest.mark.parametrize("unknown", [None, 0])
def test_unknown_process_family_rss_denies_live_headroom(tmp_path, monkeypatch, unknown):
    hierarchy(tmp_path, monkeypatch)
    monkeypatch.setattr(governor, "_host_mem_available_bytes", lambda: 40 * GIB)
    monkeypatch.setattr(governor, "process_family_rss_bytes", lambda: unknown)
    assert governor.live_memory_headroom_bytes(
        hard_limit=8 * GIB, min_host_reserve_gb=0, min_host_reserve_fraction=0
    ) == 0
