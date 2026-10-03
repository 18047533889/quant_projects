"""Compare F1/F4/F16 automatic selection on one strict, paired real cohort.

The factors, dates, assets, labels, and lineage are loaded once at F16. Smaller
panels are immutable nested prefixes of that same cohort; TEST labels remain
reserved and unscored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path("/home/sunhaiwei/quant_projects")
EXAMPLES = ROOT / "factor_optimizer/examples"
FACTOR_COUNT = 16
PREFIX_SIZES = (1, 4, 16)
MAX_FACTOR_BYTES = 128 * 1024**2
MAX_BATCH_BYTES = 2 * 1024**3
MIN_MEMORY_BYTES = 16 * 1024**3
MIN_DISK_BYTES = 2 * 1024**3
MAX_REPORT_BYTES = 1 * 1024**2


def _canonical_json(value):
    """Encode only ordinary JSON values; reject NaN and unsupported objects."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _validate_cohort(batch, provenance, lineages):
    if not isinstance(provenance, Mapping):
        raise ValueError("cohort provenance must be a mapping")
    if len(batch.factor_ids) != FACTOR_COUNT:
        raise ValueError("strict F16 cohort did not retain exactly 16 factors")
    if tuple(provenance.get("retained_factor_ids", ())) != batch.factor_ids:
        raise ValueError("cohort retained-factor provenance differs from batch order")
    if provenance.get("quarantined_factors") != []:
        raise ValueError("strict F16 cohort must not contain quarantined factors")
    if not isinstance(lineages, dict) or set(lineages) != set(batch.factor_ids):
        raise ValueError("cohort lineage keys must exactly match retained factor IDs")
    sources = provenance.get("sources")
    if not isinstance(sources, list) or len(sources) != FACTOR_COUNT:
        raise ValueError("cohort must have exactly one bound source per factor")
    if any(not isinstance(row, dict) for row in sources):
        raise ValueError("every cohort source row must be a mapping")
    manifest_uri = provenance.get("manifest_uri")
    match = re.fullmatch(
        r"cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/([0-9a-f]{64})/landing_manifest\.json",
        manifest_uri if isinstance(manifest_uri, str) else "")
    if match is None:
        raise ValueError("cohort manifest URI is not content-addressed")
    manifest_digest = match.group(1)
    if [row.get("factor") for row in sources] != list(batch.factor_ids):
        raise ValueError("source rows must preserve the retained factor ordering")
    total = 0
    for row in sources:
        if row.get("manifest_sha256") != manifest_digest:
            raise ValueError("source manifest SHA256 differs from cohort manifest URI")
        size = row.get("downloaded_bytes")
        if type(size) is not int or not 0 < size <= MAX_FACTOR_BYTES:
            raise ValueError("each downloaded source exceeds the 128 MiB object cap")
        total += size
    if total > MAX_BATCH_BYTES:
        raise ValueError("total downloaded source bytes exceed 2 GiB")
    if batch.values.nbytes > 512 * 1024**2:
        raise ValueError("declared cohort panel exceeds 512 MiB")
    return total


def _leaf_source(cohort_provenance, factor_id):
    source = next(row for row in cohort_provenance["sources"]
                  if row.get("factor") == factor_id)
    # Reuse the established strict one-factor source validator while binding
    # each leaf report back to the complete F16 cohort identity.
    leaf = dict(cohort_provenance)
    leaf["sources"] = [source]
    leaf["retained_factor_ids"] = [factor_id]
    leaf["cohort_factor_ids"] = list(cohort_provenance["retained_factor_ids"])
    leaf["cohort_manifest_uri"] = cohort_provenance["manifest_uri"]
    return leaf


def _summarize_factor(item):
    ledger = [dict(row) for row in item.candidates]
    encoded = _canonical_json(ledger).encode("utf-8")
    statuses, families = Counter(), Counter()
    for row in ledger:
        status, family = row.get("status"), row.get("family")
        if not isinstance(status, str) or not status:
            raise ValueError("candidate ledger row has no valid status")
        if not isinstance(family, str) or not family:
            raise ValueError("candidate ledger row has no valid family")
        statuses[status] += 1
        families[family] += 1
    if item.plan is None or item.plan_identity != item.plan.identity:
        raise ValueError("selected plan identity is missing or changed")
    if item.materialization_error is not None or item.status == "materialization_failed":
        raise ValueError("optimizer returned a materialization failure")
    family_status_counts, family_reason_counts, reason_counts = (
        _candidate_diagnostic_counts(ledger))
    return {
        "status": item.status,
        "selected_family": item.selected_family,
        "reason": item.reason,
        "plan_identity": item.plan_identity,
        "train_gain": item.train_gain,
        "validation_lower_bound": item.validation_lower_bound,
        "validation_candidate_identity": item.validation_candidate_identity,
        "validation_coverage": item.validation_coverage,
        "candidate_count": len(ledger),
        "candidate_status_counts": dict(sorted(statuses.items())),
        "candidate_family_counts": dict(sorted(families.items())),
        "ineligible_reason_counts": reason_counts,
        "candidate_family_status_counts": family_status_counts,
        "ineligible_family_reason_counts": family_reason_counts,
        "joint_validation": _joint_validation_details(
            getattr(item, "joint_diagnostics", None)),
        "candidate_ledger_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _compare_prefix_selections(prefixes):
    """Compare complete per-factor summaries across every shared prefix."""
    prefix_names = [f"F{size}" for size in PREFIX_SIZES]
    reference_order = prefixes[prefix_names[-1]]["factor_ids"]
    comparisons = {}
    for factor_id in reference_order:
        observed = [(name, prefixes[name]["selection"][factor_id])
                    for name in prefix_names
                    if factor_id in prefixes[name]["selection"]]
        comparison_count = max(0, len(observed) - 1)
        summaries = [summary for _, summary in observed]
        encoded = [_canonical_json(summary) for summary in summaries]
        matched = (all(value == encoded[0] for value in encoded[1:])
                   if comparison_count else None)
        differing_fields = []
        differences = {}
        if comparison_count:
            fields = sorted(set().union(*(summary.keys() for summary in summaries)))
            for field in fields:
                values = {name: summary.get(field) for name, summary in observed}
                if (any(field not in summary for summary in summaries)
                        or len({_canonical_json(value) for value in values.values()}) != 1):
                    differing_fields.append(field)
                    differences[field] = values
        comparisons[factor_id] = {
            "compared_prefixes": [name for name, _ in observed],
            "comparison_count": comparison_count,
            "matched": matched,
            "differing_fields": differing_fields,
            "differences": differences,
        }
    compared_factor_count = sum(row["comparison_count"] > 0
                                for row in comparisons.values())
    comparison_count = sum(row["comparison_count"] for row in comparisons.values())
    return {
        "all_matched": compared_factor_count > 0 and all(
            row["matched"] for row in comparisons.values()
            if row["comparison_count"] > 0),
        "compared_factor_count": compared_factor_count,
        "comparison_count": comparison_count,
        "factors": comparisons,
    }


def run_audit(*, source_loader=None, auto_runner=None, resource_check=None,
              config=None):
    from audit_real_training_methods_oct03 import (
        MAX_REPORT_BYTES as HELPER_MAX_REPORT_BYTES,
        check_environment, check_resource_headroom, source_fingerprint,
        validate_source,
    )
    from audit_real_automatic_oct03 import validate_result
    from factor_optimizer.research_batch import (
        BatchOptimizationConfig, automatic_time_split, optimize_factor_batch,
    )
    from quant_evaluator.contracts.factor_batch import FactorBatch

    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    from cos_batch_audit import load_cos_sample

    loader = load_cos_sample if source_loader is None else source_loader
    runner = optimize_factor_batch if auto_runner is None else auto_runner
    check_environment()
    resource_admission = (resource_check if resource_check is not None else
        lambda: check_resource_headroom(minimum_memory_bytes=MIN_MEMORY_BYTES,
                                        minimum_disk_bytes=MIN_DISK_BYTES))
    resource_admission()
    cohort, labels, provenance, lineages = loader(
        n_factors=FACTOR_COUNT, n_assets=256, include_lineages=True,
        max_factor_bytes=MAX_FACTOR_BYTES, max_batch_factor_bytes=MAX_BATCH_BYTES,
        coverage_policy="strict")
    loaded_cohort_bytes = _validate_cohort(cohort, provenance, lineages)
    config = config or BatchOptimizationConfig()
    cohort_fingerprint = source_fingerprint(cohort, labels, provenance)
    split = automatic_time_split(labels, config)
    if split.identity != provenance.get("asset_selection_split"):
        raise ValueError("optimizer split differs from TRAIN asset-selection split")
    for factor_id in cohort.factor_ids:
        factor_index = cohort.factor_ids.index(factor_id)
        one = FactorBatch((factor_id,), cohort.time_axis, cohort.asset_axis,
                          cohort.values[:, :, factor_index:factor_index+1],
                          validity=(None if cohort.validity is None else
                                    cohort.validity[:, :, factor_index:factor_index+1]))
        validate_source(one, labels, _leaf_source(provenance, factor_id),
                        max_factor_bytes=MAX_FACTOR_BYTES,
                        max_batch_factor_bytes=MAX_BATCH_BYTES)
        del one
    rows = {}
    for size in PREFIX_SIZES:
        resource_admission()
        started = time.perf_counter()
        ids = cohort.factor_ids[:size]
        # FactorBatch detaches and freezes each prefix while reusing the exact
        # cohort AxisRef instances and the exact same LabelBundle object.
        prefix = FactorBatch(ids, cohort.time_axis, cohort.asset_axis,
                             cohort.values[:, :, :size],
                             validity=(None if cohort.validity is None
                                       else cohort.validity[:, :, :size]),
                             context_refs=cohort.context_refs)
        prefix_provenance = dict(provenance)
        prefix_provenance["retained_factor_ids"] = list(ids)
        prefix_provenance["sources"] = [
            row for row in provenance["sources"] if row["factor"] in ids]
        before = source_fingerprint(prefix, labels, prefix_provenance)
        leaf_lineages = {factor_id: lineages[factor_id] for factor_id in ids}
        result = runner(prefix, labels, config=config, allow_research=True,
                        lineages=leaf_lineages)
        validate_result(result, prefix, split)
        after = source_fingerprint(prefix, labels, prefix_provenance)
        if before != after or source_fingerprint(cohort, labels, provenance) != cohort_fingerprint:
            raise RuntimeError("optimizer mutated the shared cohort or prefix source")
        rows[f"F{size}"] = {
            "factor_ids": list(ids),
            "batch_shape": list(prefix.values.shape),
            "source_fingerprint_before": before,
            "source_fingerprint_after": after,
            "elapsed_seconds": time.perf_counter() - started,
            "process_cumulative_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * 1024,
            "selection": {factor_id: _summarize_factor(result.factors[factor_id])
                          for factor_id in ids},
        }
        del result, prefix
    report = {
        "audit": "real-scaling-cohort-oct03-v1",
        "cohort_source": provenance,
        "cohort_source_fingerprint": cohort_fingerprint,
        "cohort_factor_ids": list(cohort.factor_ids),
        "prefixes": rows,
        "selection_consistency": _compare_prefix_selections(rows),
        "split": {"identity": split.identity,
                  "train_days": len(split.train_indices),
                  "validation_days_reserved": len(split.validation_indices),
                  "test_days_reserved": len(split.test_indices)},
        "test_evaluated": False,
        "comparison_semantics": "F1/F4/F16 nested prefixes of one strict F16 cohort; same axes, dates, assets, labels, split, and per-factor lineage",
        "limits": {"loaded_cohort_bytes": loaded_cohort_bytes,
                   "factor_count": FACTOR_COUNT, "asset_count": 256,
                   "date_count": 500, "max_factor_object_bytes": MAX_FACTOR_BYTES,
                   "max_batch_factor_bytes": MAX_BATCH_BYTES,
                   "declared_panel_array_budget_bytes": 512 * 1024**2,
                   "minimum_available_memory_admission_bytes": MIN_MEMORY_BYTES,
                   "minimum_disk_headroom_bytes": MIN_DISK_BYTES,
                   "report_limit_bytes": min(MAX_REPORT_BYTES, HELPER_MAX_REPORT_BYTES)},
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if len(encoded.encode("utf-8")) > min(MAX_REPORT_BYTES, HELPER_MAX_REPORT_BYTES):
        raise ValueError("JSON report exceeds the 1 MiB output limit")
    return report


def main(argv=None):
    from audit_real_training_methods_oct03 import write_report_exclusive
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")
    report = run_audit()
    write_report_exclusive(args.report, report)
    print(json.dumps({"report_path": str(args.report), "test_evaluated": False,
                      "prefixes": {key: len(value["factor_ids"])
                                   for key, value in report["prefixes"].items()}},
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0


MAX_DIAGNOSTIC_TEXT = 240
_JOINT_METRICS = ("rank_ic", "rank_icir", "sharpe", "max_drawdown",
                  "turnover", "worst_block_sharpe")
_JOINT_VALIDATION_FIELDS = ("objective", "policy", "comparison_status")


def _bounded_text(value):
    if not isinstance(value, str):
        return "<missing>"
    return value if len(value) <= MAX_DIAGNOSTIC_TEXT else value[:MAX_DIAGNOSTIC_TEXT]


def _bounded_reason(value):
    if not isinstance(value, str):
        return "<missing>"
    if len(value) <= MAX_DIAGNOSTIC_TEXT:
        return value
    return value[:MAX_DIAGNOSTIC_TEXT - 65] + "#" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _joint_validation_details(diagnostics):
    """Keep only bounded, JSON-safe selection context and validation metrics."""
    if diagnostics is None:
        return {}
    if not isinstance(diagnostics, Mapping):
        raise ValueError("joint diagnostics must be a mapping")
    details = {}
    for field in _JOINT_VALIDATION_FIELDS:
        value = diagnostics.get(field)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"joint diagnostic {field} must be text")
            details[field] = _bounded_text(value)
    for name in ("validation_raw", "validation_candidate"):
        metrics = diagnostics.get(name)
        if metrics is None:
            continue
        if not isinstance(metrics, Mapping):
            raise ValueError(f"joint diagnostic {name} must be a mapping")
        snapshot = {}
        for metric in _JOINT_METRICS:
            value = metrics.get(metric)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"joint diagnostic metric {metric} must be numeric")
                snapshot[metric] = value
        details[name] = snapshot
    unavailable = diagnostics.get("unavailable_metrics")
    if unavailable is not None:
        if not isinstance(unavailable, (tuple, list)):
            raise ValueError("joint unavailable_metrics must be a sequence")
        details["unavailable_metrics"] = sorted({
            metric for metric in unavailable
            if isinstance(metric, str) and metric in _JOINT_METRICS
        })
    for field in ("unavailable_code", "unavailable_role", "unavailable_reason"):
        value = diagnostics.get(field)
        if value is not None:
            details[field] = _bounded_text(value)
    _canonical_json(details)  # reject NaN and unsupported values
    return details


def _candidate_diagnostic_counts(ledger):
    family_statuses, family_reasons, reasons = {}, {}, Counter()
    for row in ledger:
        family, status = row["family"], row["status"]
        if len(family) > MAX_DIAGNOSTIC_TEXT or len(status) > MAX_DIAGNOSTIC_TEXT:
            raise ValueError("candidate family or status exceeds diagnostic bound")
        family_statuses.setdefault(family, Counter())[status] += 1
        if row.get("status") == "ineligible":
            reason = _bounded_reason(row.get("reason"))
            family_reasons.setdefault(family, Counter())[reason] += 1
            reasons[reason] += 1
    return ({family: dict(sorted(counts.items()))
             for family, counts in sorted(family_statuses.items())},
            {family: dict(sorted(counts.items()))
             for family, counts in sorted(family_reasons.items())},
            dict(sorted(reasons.items())))

if __name__ == "__main__":
    raise SystemExit(main())
