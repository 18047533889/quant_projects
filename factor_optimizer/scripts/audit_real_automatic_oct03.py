"""Bounded real automatic TRAIN/VALIDATION selection, with TEST unscored."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_real_training_methods_oct03 import (
    check_environment, check_resource_headroom, load_real_source,
    validate_source, source_fingerprint, write_report_exclusive,
)
from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch
from quant_evaluator.contracts.factor_batch import FactorBatch


def validate_result(result, batch, expected_split):
    """Reject malformed output and any claim of scored TEST before reporting."""
    output = result.optimized
    if not isinstance(output, FactorBatch):
        raise ValueError("automatic output must be a FactorBatch")
    if output.factor_ids != batch.factor_ids or output.values.shape != batch.values.shape:
        raise ValueError("automatic output factor IDs/order or shape changed")
    if not np.array_equal(output.time_axis.values, batch.time_axis.values):
        raise ValueError("automatic output time axis changed")
    if not np.array_equal(output.asset_axis.values, batch.asset_axis.values):
        raise ValueError("automatic output asset axis changed")
    if np.isinf(output.values).any():
        raise ValueError("automatic output contains infinity")
    if output.validity is None or not np.array_equal(output.validity, np.isfinite(output.values)):
        raise ValueError("automatic output validity does not match finite values")
    if result.execution_mode != "research_only" or result.test_evaluated is not False:
        raise ValueError("automatic result is not research-only with TEST unscored")
    if result.split != expected_split:
        raise ValueError("automatic result split differs from prescribed split")
    refs = output.context_refs or {}
    if refs.get("optimization_mode") != "research_only" or refs.get("split") != expected_split.identity:
        raise ValueError("automatic output context does not bind the research split")
    if set(result.factors) != set(batch.factor_ids):
        raise ValueError("automatic result factor mapping changed")
    for key, item in result.factors.items():
        if item.factor_id != key:
            raise ValueError("automatic factor result identity differs from mapping key")


def run_audit(*, source_loader=load_real_source, auto_runner=optimize_factor_batch,
              resource_check=check_resource_headroom, config=None):
    """Use existing optimizer and lineage; never search TEST or publish factors."""
    from factor_optimizer.research_batch import automatic_time_split
    check_environment()
    resource_check()
    batch, labels, provenance, lineages = source_loader()
    validate_source(batch, labels, provenance)
    config = config or BatchOptimizationConfig()
    split = automatic_time_split(labels, config)
    before = source_fingerprint(batch, labels, provenance)
    result = auto_runner(batch, labels, config=config, allow_research=True, lineages=lineages)
    validate_result(result, batch, split)
    after = source_fingerprint(batch, labels, provenance)
    if before != after:
        raise RuntimeError("automatic optimizer mutated its bound source")
    report = {
        "audit": "real-automatic-selection-v1", "source": provenance,
        "source_fingerprints": {"before": before, "after": after},
        "batch_shape": list(batch.values.shape), "test_evaluated": False,
        "split": {"identity": split.identity, "train_days": len(split.train_indices),
                  "validation_days": len(split.validation_indices),
                  "test_days_reserved": len(split.test_indices)},
        "selection_semantics": "TRAIN searches; VALIDATION checks only the frozen winner; TEST labels unscored",
        "limitations": "One real factor; research diagnostics, not production admission or guaranteed improvement. Source fingerprints cover data/provenance, not complete runtime closure.",
        "automatic": {},
    }
    for key in batch.factor_ids:
        item = result.factors[key]
        report["automatic"][key] = {
            "status": item.status, "selected_family": item.selected_family,
            "reason": item.reason, "plan_identity": item.plan_identity,
            "train_gain": item.train_gain, "validation_lower_bound": item.validation_lower_bound,
            "validation_candidate_identity": item.validation_candidate_identity,
            "validation_coverage": item.validation_coverage,
            "materialization_error": item.materialization_error,
            "candidate_budget": dict((item.training_diagnostics or {}).get("candidate_budget", {})),
            "baseline_diagnostics": dict(item.baseline_diagnostics or {}),
            "joint_diagnostics": dict(item.joint_diagnostics or {}),
            "candidates": [dict(row) for row in item.candidates],
        }
    # Enforce the same report envelope even for callers that do not save it.
    from audit_real_training_methods_oct03 import MAX_REPORT_BYTES
    if len(json.dumps(report, allow_nan=False, ensure_ascii=False).encode()) > MAX_REPORT_BYTES:
        raise ValueError("automatic report exceeds 1 MiB")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError("refusing to overwrite automatic audit report")
    report = run_audit()
    write_report_exclusive(args.report, report)
    print(json.dumps({"report_path": str(args.report), "test_evaluated": False,
                     "automatic": {key: {field: row[field] for field in
                        ("status", "selected_family", "reason", "candidate_budget")}
                        for key, row in report["automatic"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
