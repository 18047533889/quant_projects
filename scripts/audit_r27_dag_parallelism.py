# -*- coding: utf-8 -*-
"""R27-199: DAG 并行审计 —— shared node 真实依赖并行 + as_completed 流式。"""
from __future__ import annotations
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from scripts.audit_r27_acceptance import probe_dag_parallelism, probe_scheduler_runs

def main() -> None:
    dag = probe_dag_parallelism()
    sched = probe_scheduler_runs()
    print("dag:", dag)
    print("scheduler:", {k: sched[k] for k in ("ok","done","results","broker_stage") if k in sched})
    assert dag["shared_before_roots"]
    assert sched.get("ok")
    print("R27_DAG_PARALLELISM_OK")

if __name__ == "__main__":
    main()
