#!/usr/bin/env python3
"""Execute evidence-backed QAT groups and map GOLD-40 / E2E-A..H coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


GROUPS = {
    "QAT-01": [
        "quant_evaluator/tests/test_metamorphic.py",
        "quant_evaluator/tests/test_consistency.py",
    ],
    "QAT-02": [
        "quant_evaluator/tests/test_v3_drawdown_contracts.py",
        "quant_evaluator/tests/test_portfolio_stats_registry.py",
        "quant_evaluator/tests/test_daily_quantile_artifact_v3.py",
        "quant_evaluator/tests/test_v3_statistical_evidence.py",
    ],
    "QAT-03": [
        "quant_evaluator/tests/test_v3_gpu_semantics.py",
        "quant_evaluator/tests/test_gpu_parity.py",
        "quant_evaluator/tests/test_turnover_validity.py",
    ],
    "QAT-05": [
        "quant_evaluator/tests/test_v3_input_contracts.py",
        "quant_evaluator/tests/test_qe_serialization.py",
        "factor_assets/tests/contracts/test_evidence_invalidation_v3.py",
        "factor_assets/tests/contracts/test_fingerprint_spec_v3.py",
    ],
    "QAT-06": [
        "data_access/tests/unit/test_r30_change_lineage_2026_08.py",
        "data_access/tests/unit/test_r32_change_impact_p0.py",
        "quant_platform/tests/test_generation_coordinator.py",
        "quant_evaluator/tests/test_consistency.py",
    ],
}

E2E = {
    "E2E-A": {
        "tests": ["factor_optimizer/tests/search/test_qe_adapter_v3.py", "factor_assets/tests/test_library_promotion_gate.py"],
        "status": "PARTIAL", "gap": "no single trace spans DSL field catalog through admission",
    },
    "E2E-B": {
        "tests": ["factor_optimizer/tests/search/test_supervised_parameter_v3.py", "quant_evaluator/tests/test_shape_evidence.py"],
        "status": "PARTIAL", "gap": "parent-child public reevaluation trace is not unified",
    },
    "E2E-C": {
        "tests": ["factor_optimizer/tests/search/test_supervised_parameter_v3.py", "quant_evaluator/tests/test_daily_quantile_artifact_v3.py"],
        "status": "PARTIAL", "gap": "Q20-to-liquidity/exposure-to-hinge chain is not one executable trace",
    },
    "E2E-D": {
        "tests": ["quant_evaluator/tests/test_exposure_evidence.py", "factor_optimizer/tests/search/test_winner_set_v3.py"],
        "status": "PARTIAL", "gap": "neutralization candidate execution is not bound to final noninferiority verdict",
    },
    "E2E-E": {
        "tests": ["data_access/tests/unit/test_r30_change_lineage_2026_08.py", "factor_preprocess/tests/test_v3_preprocess_contracts.py"],
        "status": "PARTIAL", "gap": "announcement revision replay is not connected to a model feature artifact trace",
    },
    "E2E-F": {
        "tests": ["factor_optimizer/tests/search/test_durable_campaign_store_v3.py", "quant_platform/tests/test_generation_coordinator.py"],
        "status": "PARTIAL", "gap": "failed materialized-value GC receipt is outside the combined trace",
    },
    "E2E-G": {
        "tests": ["factor_assets/tests/test_incremental_cluster_assign.py", "factor_assets/tests/test_library_promotion_gate.py"],
        "status": "PARTIAL", "gap": "cluster-version change is not connected to deployed model-column compatibility",
    },
    "E2E-H": {
        "tests": ["factor_optimizer/tests/search/test_execution_plan_checkpoint_v3.py", "factor_preprocess/tests/test_v3_preprocess_contracts.py", "tests/modeling/test_r41_monitoring_ledger_contracts.py"],
        "status": "PARTIAL", "gap": "no one trace proves optimizer-free daily restart and formal-pointer non-flip",
    },
}

GOLD40 = {
    "status": "PARTIAL",
    "tests": [
        "factor_assets/tests/contracts/test_evidence_invalidation_v3.py",
        "factor_preprocess/tests/test_v3_preprocess_contracts.py",
        "factor_optimizer/tests/search/test_execution_plan_checkpoint_v3.py",
        "quant_evaluator/tests/test_v3_gpu_semantics.py",
    ],
    "covered": [
        "dependency identity invalidation",
        "public FP max_lag mapping and frozen recipe identity",
        "checkpoint offline replay identity",
        "real-GPU execution status recorded separately",
    ],
    "gap": "no complete historical evaluation->health->cluster->feature migration/rollback replay",
}


def execute(repo: Path, targets: list[str]) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "factor_optimizer"), str(repo / "factor_preprocess"), str(repo)])
    command = [sys.executable, "-m", "pytest", "-q", *targets]
    result = subprocess.run(command, cwd=repo, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"command": command, "exit_code": result.returncode,
            "status": "PASS" if result.returncode == 0 else "BLOCKED",
            "output": result.stdout[-12000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); repo = args.repo.resolve()
    results = {name: execute(repo, targets) for name, targets in GROUPS.items()}
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "QAT-01-02-03-05-06-and-delivery-map", "repo": str(repo),
        "groups": results,
        "gpu_device_probe": {"exit_code": gpu.returncode, "output": gpu.stdout.strip(),
                             "certification": "REAL_DEVICE" if gpu.returncode == 0 else "BLOCKED"},
        "GOLD-40": GOLD40, "E2E": E2E,
        "status": "PASS_WITH_EXPLICIT_PARTIAL_E2E" if all(v["status"] == "PASS" for v in results.values()) else "BLOCKED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"],
                      "groups": {k: v["status"] for k, v in results.items()},
                      "gpu": payload["gpu_device_probe"], "GOLD-40": GOLD40["status"],
                      "E2E": {k: v["status"] for k, v in E2E.items()}}, indent=2))
    return 0 if all(v["status"] == "PASS" for v in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
