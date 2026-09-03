# -*- coding: utf-8 -*-
"""P0#8 (100k GO §11): scheduler forced-thread vs HybridExecutor classifier —
single-authority execution policy.

Conflict: the adaptive scheduler called ``executor.submit(..., prefer="thread")``
on task / fusion / microbatch paths, unconditionally overriding HybridExecutor's
backend classifier (pandas/GIL-bound → process, native → thread). GO §11: the
scheduler must NOT unconditionally override the executor classifier.

Resolution (single authority):
  - default (no env) → ``prefer=None`` → HybridExecutor classifier is authoritative
    (pandas_numpy/research_python → process; duckdb_sql/polars → thread).
  - ``FACTOR_ENGINE_SCHEDULER=thread|process`` → scheduler-level env override is
    authoritative (explicit ops force).
  - the effective policy is surfaced in ``scheduler_stats["execution_policy"]``
    (telemetry-visible) and logged at construction.
"""
from __future__ import annotations

import os

import pytest

from factor_engine.runtime.adaptive_batch_scheduler import (
    AdaptiveBatchScheduler,
    _resolve_execution_policy,
)
from factor_engine.runtime.resource_broker import ResourceBroker


def _broker() -> ResourceBroker:
    return ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)


class _Ctx:
    """Module-level ctx so it is picklable when process policy is forced."""
    run_mode = "research"
    runtime_stats = {}


def _execute_root(task) -> str:
    return f"v:{task.factor_name}"


def _materialize_shared(sid, node) -> None:
    return None


def test_policy_resolver_default_is_classifier(monkeypatch):
    """(a) no env → None → HybridExecutor classifier is the authority."""
    monkeypatch.delenv("FACTOR_ENGINE_SCHEDULER", raising=False)
    assert _resolve_execution_policy() is None


@pytest.mark.parametrize("val", ["thread", "process"])
def test_policy_resolver_env_override(monkeypatch, val):
    """(b) FACTOR_ENGINE_SCHEDULER=thread|process → scheduler authority."""
    monkeypatch.setenv("FACTOR_ENGINE_SCHEDULER", val)
    assert _resolve_execution_policy() == val


def test_policy_resolver_adaptive_is_classifier(monkeypatch):
    """'adaptive' (the default value) must NOT force — classifier authority."""
    monkeypatch.setenv("FACTOR_ENGINE_SCHEDULER", "adaptive")
    assert _resolve_execution_policy() is None


def test_scheduler_default_policy_is_classifier(monkeypatch):
    """(a) default scheduler → prefer=None → classifier decides per backend."""
    monkeypatch.delenv("FACTOR_ENGINE_SCHEDULER", raising=False)
    sched = AdaptiveBatchScheduler(broker=_broker())
    try:
        assert sched._execution_policy is None
        assert sched._scheduler_stats.get("execution_policy") is None
    finally:
        sched.executor.shutdown(wait=True)


def test_scheduler_env_override_policy(monkeypatch):
    """(b) FACTOR_ENGINE_SCHEDULER=thread → scheduler overrides classifier."""
    monkeypatch.setenv("FACTOR_ENGINE_SCHEDULER", "thread")
    sched = AdaptiveBatchScheduler(broker=_broker())
    try:
        assert sched._execution_policy == "thread"
    finally:
        sched.executor.shutdown(wait=True)


def test_policy_telemetry_visible(monkeypatch):
    """(c) effective policy is telemetry-visible in scheduler_stats."""
    monkeypatch.setenv("FACTOR_ENGINE_SCHEDULER", "process")
    sched = AdaptiveBatchScheduler(broker=_broker())
    try:
        # scheduler_stats is rebuilt per run(); assert the field is present in
        # the run() output path by checking the attribute + a run() smoke.
        assert sched._execution_policy == "process"
        # run() populates scheduler_stats with execution_policy.
        from factor_engine.planner.physical_factor_dag import (
            TASK_ROOT,
            PhysicalFactorDAG,
            PhysicalFactorTask,
        )
        from factor_engine.runtime.task_resource_contract import TaskResourceContract

        dag = PhysicalFactorDAG()
        dag.add_task(PhysicalFactorTask(
            task_id="root:A", op="add", task_type=TASK_ROOT,
            resource_contract=TaskResourceContract(peak_memory_bytes=32 * 1024**2, cpu_tokens=1),
            factor_name="A",
        ))
        dag.roots = ("root:A",)

        out = sched.run(dag, backend=None, ctx=_Ctx(),
                        execute_root=_execute_root,
                        materialize_shared=_materialize_shared)
        assert out["scheduler_stats"]["execution_policy"] == "process"
        assert out["results"] == {"A": "v:A"}
    finally:
        sched.executor.shutdown(wait=True)
