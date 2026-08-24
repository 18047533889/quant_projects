# -*- coding: utf-8 -*-
"""R27-199: 外部负载感知审计（R27-130/131/237）。"""
from __future__ import annotations
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)

def main() -> None:
    from factor_engine.runtime.resource_governor import MemoryGovernor, live_memory_headroom_bytes
    gov = MemoryGovernor(process_budget_bytes=16 * 1024**3, duckdb_budget_bytes=1024**3)
    stage = gov.external_pressure_stage()
    assert stage in {"normal", "stop_warmup", "throttle", "critical"}
    assert isinstance(live_memory_headroom_bytes(), int)
    print("external_pressure_stage:", stage)
    print("live_headroom:", live_memory_headroom_bytes())
    print("R27_EXTERNAL_PRESSURE_OK")

if __name__ == "__main__":
    main()
