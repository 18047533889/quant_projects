"""Verify report-backed qualification provider runs."""
from dataclasses import asdict



def verify_provider_run(bundle, receipt, report, oracle_bundle, *, run_index,
                        require_cache_hit):
    """Verify one live provider-backed auto run against typed report evidence."""
    from collections.abc import Mapping

    from quant_evaluator.runtime.source_profile_output_identity import (
        matches_profile_output_identity, metric_output_identity,
    )
    from quant_evaluator.runtime.source_profile_report_schema import (
        LINEAR_SHAPE_METRICS, LINEAR_SHAPE_REPORT_KIND,
    )
    from quant_evaluator.scripts.benchmark_real_cos_linear_shape_profile_abba import (
        _verify_auto,
    )

    if getattr(report, "kind", None) != LINEAR_SHAPE_REPORT_KIND:
        raise ValueError("provider report kind is not linear-shape")
    if type(run_index) is not int or run_index not in (0, 1):
        raise ValueError("provider run index must be 0 or 1")
    if type(require_cache_hit) is not bool or require_cache_hit != (run_index == 1):
        raise ValueError("provider run cache requirement does not match its run index")
    if not isinstance(receipt, Mapping):
        raise ValueError("provider run receipt is missing")
    oom = receipt.get("oom_retries")
    if type(oom) is not int or oom != 0:
        raise ValueError("provider run requires integer zero OOM retries")
    if len(report.records) != 2:
        raise ValueError("linear-shape report profile records are incomplete")
    context = report.records[0].context
    qualification = report.qualification
    winner = qualification.winning_backend
    if winner not in ("cpu", "cuda"):
        raise ValueError("linear-shape report winner is invalid")
    profile = report.records[0].cuda if winner == "cuda" else report.records[0].cpu
    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("provider output metadata is missing")
    expected_origin = "process_cache" if require_cache_hit else "report_candidate"
    expected_provider_status = "not_checked" if require_cache_hit else "candidate_validated"
    expected_cache_status = "cache_hit" if require_cache_hit else "supplied"
    if (metadata.get("source_qualification_origin") != expected_origin
            or metadata.get("source_qualification_provider_status")
            != expected_provider_status):
        raise ValueError("provider qualification origin or provider status differs")
    if metadata.get("source_qualification_cache_status") != expected_cache_status:
        raise ValueError("provider qualification cache status differs")
    if not matches_profile_output_identity(
            bundle, profile, context, metrics=LINEAR_SHAPE_METRICS,
            expected_factor_ids=oracle_bundle.factor_ids):
        raise ValueError("provider output differs from selected profile output identity")

    verification = _verify_auto(
        bundle, receipt, qualification, context, oracle_bundle,
        run_index=run_index, require_cache_hit=require_cache_hit)
    identities = {}
    for metric in LINEAR_SHAPE_METRICS:
        identity = metric_output_identity(
            bundle.scalar_metrics[metric], bundle.observation_counts[metric])
        identities[metric] = {
            "values_sha256": identity.values_sha256,
            "finite_mask_sha256": identity.finite_mask_sha256,
            "observation_counts_sha256": identity.observation_counts_sha256,
        }
    evidence = {
        "run_index": run_index,
        "backend_used": verification["backend_used"],
        "qualification_applied": verification["qualification_applied"],
        "qualification_status": verification["qualification_status"],
        "qualification_winner": verification["qualification_winner"],
        "effective_max_tile_size": dict(qualification.actual_tile_sizes)[winner],
        "oom_retries": oom,
        "request_content_sha256": context.request_content_sha256,
        "executable_source_sha256": context.executable_source_sha256,
        "runtime_fingerprint_sha256": context.runtime_fingerprint_sha256,
        "correctness_digest_sha256": qualification.correctness_digest_sha256,
        "context": asdict(context),
        "source_qualification_origin": expected_origin,
        "source_qualification_provider_status": expected_provider_status,
        "source_qualification_cache_status": metadata["source_qualification_cache_status"],
        "output_matches_independent_oracle": True,
        "metric_identities": identities,
    }
    return evidence
