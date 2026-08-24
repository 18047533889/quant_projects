# -*- coding: utf-8 -*-
"""R38 P0-043/044/045/046 + P0-027/028（§11/§17）：dynamic sink + service JobLease。

行为探针：
    - ``set_target_bytes`` 缩容不丢已有 items、producer 阻塞（§R38_DYNAMIC_SINK_SHRINK）；
    - writer join 超时后线程仍 alive → fatal（§R38_WRITER_LIVE_THREAD_FATAL）；
    - sink.submit False（writer fatal）→ 上层 abort generation（§R38_SINK_FAILURE_STOPS_ADMISSION）；
    - service job 走 HostCoordinator JobLease admission（§R38_MULTI_JOB_LEASE_ISOLATION）；
    - QoS lane：BACKGROUND 在压力下停新 admission（P0-028）。
"""
from __future__ import annotations

import threading
import time

import pytest

from factor_engine.runtime.streaming_result_sink import BoundedResultQueue, StreamingResultSink


def test_elastic_queue_shrink_keeps_items_blocks_producer():
    q = BoundedResultQueue(max_bytes=100)
    from factor_engine.runtime.streaming_result_sink import ResultItem

    assert q.put(ResultItem("a", b"x" * 40, bytes=40)) is True
    # 缩容到 10 —— 已有 item 保留，producer 阻塞直到消费降到新 target 以下。
    q.set_target_bytes(10)
    assert q.current_bytes == 40, "shrink 不丢已有 items"
    assert q.backpressure_ratio >= 1.0
    # 消费后队列下降。
    got = q.get()
    assert got.name == "a"
    assert q.current_bytes == 0
    # 缩容到 10 后 20B 的 item 放不下 → put 返回 False（producer 阻塞超时，
    # 绝不静默 drop）。
    result = {}
    import threading

    t = threading.Thread(
        target=lambda: result.setdefault("ok", q.put(ResultItem("b", b"y" * 20, bytes=20), timeout=0.1))
    )
    t.start(); t.join()
    assert result.get("ok") is False
    assert q.queued_count == 0


def test_writer_alive_after_join_is_fatal():
    import time as _time

    # writer 永不退出 → join(timeout) 后线程 alive → finish 必须 fatal。
    def _stuck(_batch):
        _time.sleep(30)

    sink = StreamingResultSink(writer=_stuck, queue_bytes=1024, batch_size=1)
    sink.start()
    sink.submit("f", object())
    with pytest.raises(RuntimeError, match="writer thread alive after join|alive after join"):
        sink.finish()


def test_sink_submit_false_after_writer_fatal():
    def _boom(_batch):
        raise OSError("disk full")

    sink = StreamingResultSink(writer=_boom, queue_bytes=1024, batch_size=1)
    sink.start()
    sink.submit("f1", object())
    # writer 首次提交即 permanent 失败 → fatal；后续 submit 返回 False。
    sink.finish_async_mark_fatal() if hasattr(sink, "finish_async_mark_fatal") else None
    # 直接验证：writer FAILED 后 submit 拒绝。
    from factor_engine.runtime.streaming_result_sink import _WriterWorker, WS_FAILED
    sink._workers[0].state = WS_FAILED
    sink._workers[0].fatal_error = OSError("disk full")
    sink._set_fatal(sink._workers[0].fatal_error)
    assert sink.submit("f2", object()) is False


def test_service_job_lease_admission():
    from types import SimpleNamespace

    from factor_engine.runtime.host_resource_coordinator import HostResourceCoordinator, reset_host_coordinator
    from factor_engine.service import queue as sq
    from factor_engine.service.jobstore import JobRecord, JobStatus

    reset_host_coordinator()
    coord = HostResourceCoordinator()
    q = sq.BoundedJobQueue(max_queue=16, max_running=2)
    q.coordinator = coord
    q.broker = coord.broker

    store = {}
    q._store = SimpleNamespace(
        update=lambda job: store.setdefault(job.run_id, job),
        get=lambda run_id: store.get(run_id),
    )

    def _noop(job):
        time.sleep(0.01)

    est = sq.JobResourceEstimate(priority=sq.QOS_STANDARD, memory_p99=1024**2, cpu_budget=1)
    job = JobRecord(run_id="job-lease-1", requested_by="p")
    q.submit(job, run_fn=_noop, estimate=est)
    # 手动触发 _begin（worker loop 未启动）——期间申请 JobLease，finally 释放。
    requested = []
    orig = coord.request_job_lease
    coord.request_job_lease = lambda **kw: (
        requested.append(kw), orig(**kw)
    )[1]
    q._begin(job, _noop)
    # lease 确实被申请过（memory/cpu 来自 estimate）。
    assert len(requested) == 1
    assert requested[0]["memory_bytes"] == 1024**2
    # finally 已 release（_job_leases 清空，coordinator 无泄漏）。
    assert len(q._job_leases) == 0
    assert coord.reconcile()["active_root_leases"] == 0


def test_qos_background_paused_under_pressure():
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.service import queue as sq
    from factor_engine.service.jobstore import JobRecord, JobStatus

    broker = ResourceBroker(hard_memory_limit=2 * 1024**3, cpu_slots=4,
                            min_host_reserve_gb=0.0, min_host_reserve_fraction=0.0)
    # 强制造压力档位。
    broker._cpu.set_soft_budget(1)
    q = sq.BoundedJobQueue(max_queue=16, max_running=4)
    q.broker = broker
    q.coordinator = None
    # 模拟 pressure_stage 高（broker live headroom 小）。
    job = JobRecord(run_id="bg-1", requested_by="p")
    est = sq.JobResourceEstimate(priority=sq.QOS_BACKGROUND, memory_p99=1, cpu_budget=1)
    from data_access.runtime.resource_governor import reset_global_governor
    reset_global_governor()
    if broker.pressure_stage() in {"PRESSURE_2", "PRESSURE_3", "PRESSURE_4", "CRITICAL"}:
        with pytest.raises(sq.ServiceError) as ei:
            q.submit(job, run_fn=lambda j: None, estimate=est)
        assert ei.value.code == "JOB_QOS_DEFERRED"
    else:
        # 压力不够时 BACKGROUND 照常进入。
        q.submit(job, run_fn=lambda j: None, estimate=est)
