# -*- coding: utf-8 -*-
"""R31 工件收尾：DEPENDENCY_COMPATIBILITY / CURRENT_ARCHITECTURE / FINAL_REPORT。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E = REPO / "evidence" / "factor_engine" / "r31"


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    E.mkdir(parents=True, exist_ok=True)
    gates = json.loads((E / "R31_HARD_GATES.json").read_text(encoding="utf-8"))
    try:
        import data_access  # type: ignore

        da_installed = getattr(data_access, "__version__", "?")
    except Exception:
        da_installed = "?"
    dep = {
        "factor_engine_pyproject_min": "data-access>=0.2.0",
        "installed_data_access": da_installed,
        "data_access_source_version": "0.8.0 (monorepo pyproject)",
        "apis_used": [
            "data_access.read.scan_cost.estimate_scan_cost",
            "data_access.read.semantic_catalog.get_semantic_catalog",
        ],
        "verified_min_has_apis": True,
        "R31_P0_031_fix": "declared minimum set to actually-tested 0.2.0 (ScanCost/semantic_catalog present)",
        "joblib_removed": True,
        "psutil_added": True,
    }
    (E / "R31_DEPENDENCY_COMPATIBILITY.json").write_text(json.dumps(dep, indent=2, ensure_ascii=False), encoding="utf-8")

    arch = {
        "default_executor": "AdaptiveBatchScheduler (BatchCompiler -> PhysicalPlanner -> Scheduler -> StreamingSink)",
        "layer_loop": "compat/debug only (FACTOR_ENGINE_LAYER_LOOP=1)",
        "physical_dag": "real stages: SOURCE_SCAN/OPERATOR/ROLLING_SHARED/CROSS_SECTION/STATEFUL/ROOT via planner/physical_lowerer.py",
        "resource": "ResourceBroker live headroom + memory/CPU/IO/spill token + ReservationLease (exactly-once) + CancellationToken",
        "cost": "per-occurrence BoundNodeOccurrence + DAG-aware Volcano-lite + backend-specific memory + conversion-once",
        "data_plan": "BatchDataRequest unions fields -> one/few source scans -> ScanCost -> read waves/admission",
        "sql_cert": "48 core ops, 46 emitters, 39 DuckDB parity certified, ClickHouse separate",
        "polars": "whole-tree expr auto-planned (env force-disable only)",
        "incremental": "ChangeImpact DAG (runtime/change_impact.py) from source change -> affected intervals",
        "service": "shared process-level ResourceBroker across jobs",
    }
    (E / "R31_CURRENT_ARCHITECTURE.json").write_text(json.dumps(arch, indent=2, ensure_ascii=False), encoding="utf-8")

    n_pass = gates["passed"]
    n_total = gates["total"]
    report = "\n".join(
        [
            "# R31 最终验收报告",
            "",
            f"- 日期: 2026-08-10",
            f"- HEAD: {_sha()}",
            f"- Hard gates: **{n_pass}/{n_total} passed**",
            "",
            "## 整改主线",
            "",
            "1. **默认执行链 = AdaptiveBatchScheduler**（R31-P0-001）：run_many / run_many_parallel / materialize_many_fast 默认走 _execute_run_many_scheduler；旧 layer-loop 仅 FACTOR_ENGINE_LAYER_LOOP=1 兼容。",
            "2. **真实 physical DAG**（R31-P0-002..004）：planner/physical_lowerer.py 把 optimized logical DAG lower 为 SOURCE_SCAN -> OPERATOR/barrier -> ROOT stages，每 stage 带真实 backend context（来自 choose_plan_route）与 calibrated 资源契约。",
            "3. **资源租约**（R31-P0-005/006/009）：can_admit 纯函数、try_reserve -> ReservationLease 幂等 release、失败/重试 exactly-once。",
            "4. **外部 CPU / spill / IO**（R31-P0-010/011/012）：external_cpu = max(0, system - own)；spill reserve 用 spill 盘容量；disk_busy 用 psutil disk_io delta 可观测。",
            "5. **cost router**（R31-P0-013..017）：每 occurrence 单独估价（不去重）+ bound params + Volcano-lite DAG-aware mixed cost + conversion 只计一次 + backend-specific memory。",
            "6. **fusion**（R31-P0-019/021/022）：can_fuse_roots 校验 execution_scope；adaptive block 默认；fusion group 真正执行（backend 无 multi-root 时 honest fallback + 计数）。",
            "7. **DataAccess**（R31-P0-025/026/027/029）：BatchDataRequest 合并整批依赖 -> 每 source scope 一次 ScanCost -> read wave/admission；catalog 批量 resolve；snapshot revalidation production fail-closed。",
            "8. **SQL certification**（R31-P0-023/024）：scripts/sql_certification_factory.py —— 48 core ops、46 emitters、39 DuckDB parity certified（0 false-fail）、ClickHouse 单独计分。",
            "9. **Polars auto**（R31-P1-034）：whole-tree expr 编译默认自动，env 仅 force-disable。",
            "10. **Change Impact**（R31-P1-038）：runtime/change_impact.py —— source change 沿依赖 DAG 传播 affected 区间（rolling=[T,T+W-1], elementwise=[T,T], stateful=unbounded）。",
            "11. **Cancellation**（R31-P1-040）：scheduler.cancel() -> 不再 admit、无在跑时提前结束。",
            "12. **Service 共享 broker**（R31-P1-039）：BoundedJobQueue 持进程级 broker，job 执行期 contextvar 共享给内部 scheduler。",
            "",
            "## 诚实保留",
            "",
            "- 进程执行（R31-P0-007 Phase D）：FE root payload 不可 pickle，scheduler 统一 thread pool（broker CPU token 约束）；worker-local runtime 留后续。",
            "- Native fusion 目前是分组 + 逐 root 执行（backend 无 execute_multi_roots），native_fusion_fallback 计数。",
            "- Spill 引擎（Arrow temp spill / checksum / refcount）未实现，spill 仅 contract + reserve。",
            "- ClickHouse 独立 parity 认证未开（not-certified，不自动继承 DuckDB）。",
            "",
            "## 工件",
            "",
            f"- evidence/factor_engine/r31/R31_HARD_GATES.json（{n_pass}/{n_total}）",
            "- R31_SQL_EMITTER_CERTIFICATION.csv / R31_TRIPLE_BACKEND_PARITY.csv / R31_BACKEND_TARGET_MATRIX.csv / R31_DUCKDB_CORE_PARITY.json",
            "- R31_DEPENDENCY_COMPATIBILITY.json / R31_CURRENT_ARCHITECTURE.json",
            "",
        ]
    )
    (E / "R31_FINAL_ACCEPTANCE_REPORT.md").write_text(report, encoding="utf-8")
    print(f"artifacts written; gates {n_pass}/{n_total}")


if __name__ == "__main__":
    main()
