"""ResourceBroker must preserve whether process-family RSS is trustworthy."""
from __future__ import annotations

from dataclasses import replace
import resource
import sys
from types import SimpleNamespace

import pytest

from factor_engine.runtime import resource_broker
from factor_engine.runtime.resource_broker import ResourceBroker


class _NoSuchProcess(Exception):
    pass


class _AccessDenied(Exception):
    pass


class _Process:
    def __init__(self, pid: int, rss: int, children=(), memory_error=None):
        self.pid = pid
        self._rss = rss
        self._children = list(children)
        self._memory_error = memory_error

    def memory_info(self):
        if self._memory_error is not None:
            raise self._memory_error
        return SimpleNamespace(rss=self._rss, pss=0)

    def children(self, recursive=True):
        return list(self._children)


def _install_psutil(monkeypatch, root):
    fake = SimpleNamespace(
        Process=lambda: root,
        NoSuchProcess=_NoSuchProcess,
        ZombieProcess=_NoSuchProcess,
        AccessDenied=_AccessDenied,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)


def test_process_family_rss_marks_full_sampling_failure_unknown(monkeypatch):
    _install_psutil(monkeypatch, _Process(1, 0, memory_error=_AccessDenied()))
    monkeypatch.setattr(
        resource,
        "getrusage",
        lambda _who: SimpleNamespace(ru_maxrss=123),
    )

    assert resource_broker._process_family_rss() == (123 * 1024, False)


@pytest.mark.parametrize(
    ("child_error", "expected_known"),
    [(_NoSuchProcess(), True), (_AccessDenied(), False)],
)
def test_process_family_rss_distinguishes_child_exit_from_permission_failure(
    monkeypatch, child_error, expected_known
):
    child = _Process(2, 0, memory_error=child_error)
    _install_psutil(monkeypatch, _Process(1, 10_000, children=[child]))

    assert resource_broker._process_family_rss() == (10_000, expected_known)


def _controlled_snapshot(broker, *, rss: int, known: bool):
    return replace(
        broker.snapshot(),
        hard_memory_limit=8 * 1024**3,
        cgroup_memory_remaining=7 * 1024**3,
        host_mem_available=40 * 1024**3,
        host_mem_available_known=True,
        process_rss=rss,
        process_family_rss=rss,
        process_family_pss=rss,
        process_family_rss_known=known,
        rlimit_as_remaining=None,
    )


def test_unknown_family_rss_fails_closed_for_headroom_and_execution(monkeypatch):
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=1,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    snapshot = _controlled_snapshot(broker, rss=123 * 1024, known=False)
    monkeypatch.setattr(broker, "_refresh", lambda force=False: snapshot)

    assert snapshot.live_headroom == 0
    assert broker.execution_budget() == 0


def test_known_zero_family_rss_keeps_normal_eighty_percent_budget(monkeypatch):
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=1,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    snapshot = _controlled_snapshot(broker, rss=0, known=True)
    monkeypatch.setattr(broker, "_refresh", lambda force=False: snapshot)

    assert snapshot.live_headroom == 7 * 1024**3
    assert broker.execution_budget() == int(0.8 * 7 * 1024**3)
