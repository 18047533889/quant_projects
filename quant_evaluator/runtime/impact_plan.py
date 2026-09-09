"""Read-only V5 impact query; never modifies factor values or production pointers.

Input: JSON array of existing FA EvidenceRef.to-dataclass field projections.
The existing FA metric-version invalidation planner remains the authority.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path

from factor_assets.contracts.evidence_ref import EvidenceRef, plan_metric_version_invalidation
from quant_evaluator.registry.metrics import get_metric, resolve_alias, list_metrics


def plan_v5_impact(evidence_refs, *, change_kind="METRIC_SEMANTICS", changed_metrics=None):
    if change_kind not in {"METRIC_SEMANTICS", "GRADING_POLICY", "COST_POLICY", "CLUSTER_RESEARCH_LABEL", "FACTOR_VALUE_SEMANTICS"}:
        raise ValueError("Unknown scoped change kind")
    normalized = tuple(replace(ref,metric_name=resolve_alias(ref.metric_name)) for ref in evidence_refs)
    ids = tuple(resolve_alias(mid) for mid in changed_metrics) if changed_metrics is not None else tuple(
        mid for mid in list_metrics() if get_metric(mid).metric_version == "3.0.0")
    if change_kind == "METRIC_SEMANTICS":
        versions = {mid:get_metric(mid).metric_version for mid in ids}
        plan = plan_metric_version_invalidation(normalized, versions)
        affected = plan.stale_evidence_ids
        raw = asdict(plan)
    else:
        affected = tuple(ref.evidence_id for ref in normalized)
        raw = {"stale_evidence_ids": affected, "affected_factor_ids": sorted({ref.factor_id for ref in normalized})}
    actions = {
        "METRIC_SEMANTICS": ("RECOMPUTE_AFFECTED_EVALUATIONS", "RESCORE_DEPENDENT_HEALTH_AND_SELECTION"),
        "GRADING_POLICY": ("RESCORE_STORED_RAW_METRICS",),
        "COST_POLICY": ("REPRICE_REFERENCED_EXECUTION_TRACES", "REEVALUATE_DEPENDENT_SELECTION"),
        "CLUSTER_RESEARCH_LABEL": ("UPDATE_RESEARCH_CLUSTER_EVIDENCE_ONLY",),
        "FACTOR_VALUE_SEMANTICS": ("REMATERIALIZE_EXPLICITLY_SCOPED_FACTOR_VALUES", "REEVALUATE_DEPENDENCIES"),
    }[change_kind]
    return {**raw, "change_kind":change_kind, "actions":actions if affected else (),
            "rematerialize_factor_values": bool(affected) and change_kind == "FACTOR_VALUE_SEMANTICS",
            "production_pointer_mutation": False, "delete_old_artifacts": False,
            "rollback": "Retain immutable old evaluation/model/cluster refs; explicit release authority required",
            "dry_run": True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_json",type=Path)
    parser.add_argument("--change-kind",default="METRIC_SEMANTICS")
    args=parser.parse_args()
    records=json.loads(args.evidence_json.read_text())
    refs=tuple(EvidenceRef(**item) for item in records)
    print(json.dumps(plan_v5_impact(refs,change_kind=args.change_kind),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
