# -*- coding: utf-8 -*-
"""R27-199: ResourceBroker admission 审计（memory/CPU/IO/spill token + live headroom）。"""
from __future__ import annotations

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.resource_governor import live_memory_headroom_bytes
    from factor_engine.runtime.task_resource_contract import TaskResourceContract

    broker = ResourceBroker()
    snap = broker.snapshot()
    admitted = broker.reserve(
        TaskResourceContract(peak_memory_bytes=16 * 1024**2, cpu_tokens=1),
        task_id="probe",
    )
    broker.release(
        TaskResourceContract(peak_memory_bytes=16 * 1024**2, cpu_tokens=1),
        task_id="probe",
    )
    big_blocked = broker.can_admit(
        TaskResourceContract(peak_memory_bytes=broker.hard_memory_limit)
    )
    print("live_headroom:", snap.live_headroom)
    print("host_mem_available:", snap.host_mem_available)
    print("process_family_rss:", snap.process_family_rss)
    print("pressure_stage:", broker.pressure_stage())
    print("recommended_concurrency:", broker.recommended_concurrency())
    print("small_task_admitted:", admitted)
    print("oversize_task_blocked:", big_blocked is False)
    print("live_headroom_numeric:", isinstance(live_memory_headroom_bytes(), int))
    assert admitted is True and big_blocked is False
    print("R27_RESOURCE_ADMISSION_OK")


if __name__ == "__main__":
    main()
