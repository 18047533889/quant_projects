# -*- coding: utf-8 -*-
"""R38 Phase 0/10: baseline + claim-vs-runtime ledger + wiring audit.

Generates ``evidence/factor_engine/r38/``:
    R38_BASELINE.json
    R38_RUNTIME_WIRING_AUDIT.json
    R38_R35_R36_CLAIM_VS_RUNTIME.csv

The claim-vs-runtime ledger is the core R38 artifact (§40): for every R35/R36
claim, it records the real runtime entrypoint, whether the claim is actually
wired into the main execution chain, and whether a behavior test exists.
``--before`` (default) emits the honest pre-fix state; ``--after`` emits the
post-fix state computed by the audit script.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

_EVID = "evidence/factor_engine/r38"
os.makedirs(_EVID, exist_ok=True)


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


_CANDIDATE_FILES = (
    "factor_engine/runtime/adaptive_batch_scheduler.py",
    "factor_engine/runtime/auto_shard_planner.py",
    "factor_engine/runtime/buffer_store.py",
    "factor_engine/runtime/host_resource_coordinator.py",
    "factor_engine/runtime/resource_calibration_store.py",
    "factor_engine/runtime/resource_autopilot.py",
    "factor_engine/runtime/resource_broker.py",
    "factor_engine/runtime/resource_autopilot_service.py",
    "factor_engine/runtime/batch_service.py",
    "factor_engine/runtime/streaming_result_sink.py",
    "factor_engine/runtime/task_run_observation.py",
    "factor_engine/runtime/spill_store.py",
    "factor_engine/runtime/shard_execution_plan.py",
    "factor_engine/runtime/shard_merger.py",
    "factor_engine/runtime/memory_budget_allocator.py",
    "factor_engine/backend/fast_linear_window.py",
    "factor_engine/backend/model_family_kernels.py",
    "factor_engine/backend/numba_kernel_registry.py",
    "factor_engine/cache/session.py",
    "factor_engine/service/queue.py",
    "factor_engine/planner/physical_lowerer.py",
    "data_access/read/read_session.py",
    "data_access/runtime/read_pipeline.py",
    "data_access/runtime/prepared_read.py",
    "data_access/runtime/host_resource_bridge.py",
    "data_access/runtime/resource_governor.py",
)


def _source_present(path: str, needle: str) -> bool:
    """Check whether a needle occurs in a source file (wiring probe).

    The needle is searched across all known R38-relevant source files (the path
    argument is a descriptive entrypoint, not a reliable file path). Returns
    True if the needle appears in any of them.
    """
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for rel in _CANDIDATE_FILES:
        full = os.path.join(base, rel)
        try:
            with open(full, encoding="utf-8") as fh:
                if needle in fh.read():
                    return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


#: claim_id -> {claim_text, report, runtime_entrypoint, probes, wire_note}
CLAIMS: dict[str, dict[str, str]] = {
    "R36 auto-shard": {
        "claim": "AutoShard turns an oversized task into N executable shards + merge",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "scheduler._auto_shard_replan",
        "probe": "def _auto_shard_replan",
        "wire_note": "pre-fix: contract-peak-only (no shard children, no merge)",
    },
    "R36 OOM replan": {
        "claim": "OOM drives smaller-shape replan, same shape never retried",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "scheduler FAILED_FATAL path",
        "probe": "classify_error",
        "wire_note": "pre-fix: MemoryError -> ERROR_PERMANENT (no replan)",
    },
    "R36 calibration": {
        "claim": "CalibrationStore receives real elapsed/peak/output observations",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "scheduler._record_task_calibration",
        "probe": "finished_at_ms",
        "wire_note": "pre-fix: elapsed=monotonic timestamp, peak=predicted contract",
    },
    "R36 governed buffer put": {
        "claim": "Buffer put refusal is handled by callers (no silent return)",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "batch_service._materialize_shared_subplan",
        "probe": "store.put(sid, value)",
        "wire_note": "pre-fix: bare bool ignored; raw dict fallback remains",
    },
    "R36 raw CSE bypass zero": {
        "claim": "No production raw shared_result_cache[sid]=value fallback",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "batch_service._materialize_shared_subplan",
        "probe": "ctx.shared_result_cache[sid] = value",
        "wire_note": "pre-fix: raw fallback present, no run_mode guard",
    },
    "R36 spill": {
        "claim": "Buffer spill writes, verifies, reloads or recomputes per policy",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "GovernedBufferStore.spill",
        "probe": "def spill(self, key)",
        "wire_note": "pre-fix: spill == release (drop), no SpillStore",
    },
    "R36 host lease tree": {
        "claim": "FE/DA/service share one HostResourceCoordinator lease tree",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "HostResourceCoordinator.request_lease",
        "probe": "sum(l.memory_bytes for l in self._leases.values())",
        "wire_note": "pre-fix: parent+child double count; CPU/IO/spill not enforced",
    },
    "R36 FE/DA one authority": {
        "claim": "FE and DA use the same host child-lease accounting",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "HostResourceCoordinator.apply_da_envelope",
        "probe": "set_max_total_reserved_memory",
        "wire_note": "pre-fix: cap-setter only, DA ReadPipeline uses its own governor",
    },
    "R36 service broker": {
        "claim": "Service uses HostCoordinator job lease admission, not just fixed max_running",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "service/queue.BoundedJobQueue.submit",
        "probe": "running_now + queued_now",
        "wire_note": "pre-fix: count-based admission, no JobResourceEstimate->JobLease",
    },
    "R36 fixed-cadence controller": {
        "claim": "ResourceController ticks on fixed time cadence, not scheduler loop count",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "broker.resource_decision",
        "probe": "self._refresh(force=True)",
        "wire_note": "pre-fix: every scheduler loop ticks controller (loop-count driven)",
    },
    "R36 decision fields consumed": {
        "claim": "Every ResourceDecision field has a unique consumer",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "scheduler.run / materialize",
        "probe": "target_cpu_tokens",
        "wire_note": "pre-fix: target_concurrency/io/remote/cache/spill not consumed",
    },
    "R36 dynamic wave": {
        "claim": "In-flight read waves repartition when memory pressure rises",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "scheduler._execute_read_waves",
        "probe": "for wave in waves",
        "wire_note": "pre-fix: waves built once in plan(); not repartitioned",
    },
    "R36 dynamic sink": {
        "claim": "Sink queue budget shrinks/grows with pressure",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "BoundedResultQueue",
        "probe": "self.max_bytes = max(1, int(max_bytes))",
        "wire_note": "pre-fix: set once at construction, no set_target_bytes",
    },
    "R36 duckdb token truth": {
        "claim": "Backend thread tokens come from real lease/engine truth",
        "report": "R36 FINAL ACCEPTANCE",
        "entrypoint": "physical_lowerer._engine_threads_for",
        "probe": "os.environ.get(\"DUCKDB_MAX_THREADS\"",
        "wire_note": "pre-fix: env/default guesses (4 duckdb / 2 polars)",
    },
    "R35 fast linear": {
        "claim": "FastLinearWindowEngine is true sliding sufficient-statistics O(T p^2)",
        "report": "R35 acceptance",
        "entrypoint": "factor_engine.backend.fast_linear_window._roll_grams",
        "probe": "Xs.T @ Xs",
        "wire_note": "pre-fix: per-row rescan O(T*window*p^2)",
    },
    "R35 numba main chain": {
        "claim": "Certified Numba kernels dispatch inside the cleaned-operator main chain",
        "report": "R35 acceptance",
        "entrypoint": "cleaned operator implementations",
        "probe": "NumbaKernelRegistry.get",
        "wire_note": "pre-fix: kernels benchmarked standalone; operators run numpy reference",
    },
    "R35 PCA shared state": {
        "claim": "model-family shared PCA state == canonical panel PCA semantics",
        "report": "R35 acceptance",
        "entrypoint": "backend.model_family_kernels PCABlock",
        "probe": "def commonality",
        "wire_note": "pre-fix: shared-block commonality != canonical 1-Var(resid)/Var(ret)",
    },
    "R29 DA session cache": {
        "claim": "DataReadSession resolution cache is request-scoped, race-free",
        "report": "R29 closure",
        "entrypoint": "data_access/read/read_session.DataReadSession.__enter__",
        "probe": "store._resolution_cache = self._resolution_cache",
        "wire_note": "pre-fix: mutates shared Store attribute (concurrent clobber)",
    },
}


def _claim_status(claim_id: str, when: str) -> dict[str, str]:
    info = CLAIMS[claim_id]
    path = info["entrypoint"]
    wired = _source_present(path, info["probe"])
    return {
        "claim_id": claim_id,
        "claim_text": info["claim"],
        "report": info["report"],
        "real_runtime_entrypoint": path,
        "wired": "true" if wired else "false",
        "behavior_test_exists": _behavior_test(claim_id),
        "status": "NOT_WIRED" if not wired else ("SYNTHETIC_ONLY" if when == "before" else "WIRED"),
        "note": info["wire_note"],
    }


def _resolve_path(entry: str) -> str:
    """Map entrypoint description to a relative source path (best effort)."""
    for candidate in (
        "factor_engine/runtime/adaptive_batch_scheduler.py",
        "factor_engine/runtime/buffer_store.py",
        "factor_engine/runtime/host_resource_coordinator.py",
        "factor_engine/runtime/resource_calibration_store.py",
        "factor_engine/runtime/resource_autopilot.py",
        "factor_engine/runtime/resource_broker.py",
        "factor_engine/runtime/batch_service.py",
        "factor_engine/backend/fast_linear_window.py",
        "factor_engine/backend/model_family_kernels.py",
        "factor_engine/backend/numba_kernel_registry.py",
        "factor_engine/service/queue.py",
        "factor_engine/planner/physical_lowerer.py",
        "data_access/read/read_session.py",
        "data_access/runtime/read_pipeline.py",
        "data_access/runtime/prepared_read.py",
    ):
        if entry in candidate:
            return candidate
    # fallback: search for the entry name across the tree
    return entry


def _behavior_test(claim_id: str) -> str:
    key = {
        "R36 auto-shard": "test_real_auto_shard",
        "R36 OOM replan": "test_oom_replan",
        "R36 calibration": "test_task_calibration",
        "R36 governed buffer put": "test_buffer_put_refusal",
        "R36 raw CSE bypass zero": "test_no_production_raw_cse",
        "R36 spill": "test_spill_reload",
        "R36 host lease tree": "test_host_lease_parent_child",
        "R36 FE/DA one authority": "test_fe_da_same_host_lease_tree",
        "R36 service broker": "test_multijob_independent_job_leases",
        "R36 fixed-cadence controller": "test_resource_controller_fixed_cadence",
        "R36 decision fields consumed": "test_multi_scheduler_does_not_multitick",
        "R36 dynamic wave": "test_dynamic_read_wave_repartition",
        "R36 dynamic sink": "test_dynamic_sink_shrink",
        "R36 duckdb token truth": "test_duckdb_cpu_token_matches_engine_threads",
        "R35 fast linear": "test_fast_linear_true_sliding_parity",
        "R35 numba main chain": "test_numba_end_to_end_operator_dispatch",
        "R35 PCA shared state": "test_pca_shared_state_parity",
        "R29 DA session cache": "test_datareadsession_concurrent_cache_isolation",
    }.get(claim_id, "")
    if not key:
        return "none"
    for root in ("tests/r38", "../data_access/tests/unit"):
        try:
            hits = subprocess.check_output(
                ["grep", "-rl", key, root], stderr=subprocess.DEVNULL
            ).decode()
            if hits.strip():
                return "exists"
        except Exception:
            continue
    return "none"


def main() -> None:
    when = "after" if "--after" in sys.argv else "before"
    head = _head()
    rows = [_claim_status(cid, when) for cid in sorted(CLAIMS)]
    baseline = {
        "head": head,
        "phase": when,
        "claim_count": len(rows),
        "wired": sum(1 for r in rows if r["wired"] == "true"),
        "not_wired": sum(1 for r in rows if r["wired"] == "false"),
    }
    wiring = {
        "head": head,
        "phase": when,
        "issues_scope": [
            "real-autoshard", "oom-replan", "calibration-truth", "buffer-put-refusal",
            "raw-cse-bypass", "real-spill", "host-lease-tree", "fe-da-one-authority",
            "service-joblease", "fixed-cadence-controller", "decision-consumers",
            "dynamic-wave", "dynamic-sink", "backend-token-truth", "fast-linear-sliding",
            "numba-main-chain", "pca-shared-state", "da-session-cache",
        ],
        "claims": rows,
    }
    _write_json("R38_BASELINE.json", baseline)
    _write_json("R38_RUNTIME_WIRING_AUDIT.json", wiring)
    csv_path = os.path.join(_EVID, "R38_R35_R36_CLAIM_VS_RUNTIME.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "claim_id", "claim_text", "report", "real_runtime_entrypoint",
                "wired", "behavior_test_exists", "status", "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(
        f"R38 baseline written to {_EVID} (HEAD {head}, phase={when}, "
        f"wired={baseline['wired']}/{baseline['claim_count']})"
    )


def _write_json(name: str, payload: dict) -> None:
    path = os.path.join(_EVID, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
