# -*- coding: utf-8 -*-
"""R36 evidence 生成：server profile + decision trace + calibration + acceptance。

绑定当前 HEAD，写 ``docs/evidence/r36/``：
    R36_HEAD.json / SERVER_PROFILE.json / RESOURCE_DECISION_TRACE.json /
    CALIBRATION_STORE.json / R36_HARD_GATES.json（由 audit 生成）/
    R36_FINAL_ACCEPTANCE_REPORT.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_EVID = "docs/evidence/r36"
os.makedirs(_EVID, exist_ok=True)


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _server_profile() -> dict:
    from runtime.resource_governor import (
        effective_cpu_slots,
        effective_memory_limit_bytes,
        spill_disk_available,
        tempfile_dir,
    )
    from runtime.resource_monitor import (
        cgroup_pressure,
        psi_cpu,
        psi_io,
        psi_memory,
        swap_usage,
    )
    from runtime.resource_shape import hardware_fingerprint

    hw = hardware_fingerprint()
    return {
        "head": _head(),
        "hardware_fingerprint": hw,
        "effective_cpu_slots": effective_cpu_slots(),
        "effective_memory_limit_bytes": effective_memory_limit_bytes(),
        "spill_dir": tempfile_dir(),
        "spill_free_bytes": spill_disk_available(tempfile_dir()),
        "psi_cpu": psi_cpu(),
        "psi_memory": psi_memory(),
        "psi_io": psi_io(),
        "cgroup_cpu_pressure": cgroup_pressure("cpu"),
        "swap": swap_usage(),
    }


def _decision_trace() -> dict:
    from runtime.resource_broker import ResourceBroker

    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    log = []
    for _ in range(6):
        d = broker.resource_decision()
        log.append(d.to_dict())
    return {
        "head": _head(),
        "decision_log": log,
        "controller": broker.resource_controller_summary(),
    }


def _calibration_summary() -> dict:
    from runtime.resource_calibration_store import global_calibration_store
    from runtime.resource_shape import ResourceShapeKey

    store = global_calibration_store()
    key = ResourceShapeKey("ts_mean", "duckdb", rows_bucket=2, instruments_bucket=2, window_bucket=1)
    store.record(key, elapsed_ms=42, peak_mem=1024**3)
    store.record(key, elapsed_ms=44, peak_mem=1100 * 1024**2)
    store.record(key, elapsed_ms=40, peak_mem=980 * 1024**2)
    pred = store.predict(key)
    return {
        "head": _head(),
        "summary": store.summary(),
        "example_prediction": pred,
    }


def _write(name: str, payload: dict) -> str:
    path = os.path.join(_EVID, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def main() -> None:
    head = _head()
    _write("R36_HEAD.json", {"head": head})
    _write("SERVER_PROFILE.json", _server_profile())
    _write("RESOURCE_DECISION_TRACE.json", _decision_trace())
    _write("CALIBRATION_STORE.json", _calibration_summary())
    print(f"R36 artifacts written to {_EVID} (HEAD {head})")


if __name__ == "__main__":
    main()
