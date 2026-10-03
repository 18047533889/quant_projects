"""Behavioral checks for provider-candidate and cache-hit F48 verification."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import json

from quant_evaluator.runtime.source_profile_output_identity import metric_output_identity
from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.scripts.source_profile_report_reader import parse_source_profile_report
from quant_evaluator.scripts.linear_shape_provider_verification import verify_provider_run
from test_source_linear_shape_report_reader_oct04 import METRICS, _payload


def _fixture(winner="cpu"):
    report = parse_source_profile_report(json.dumps(_payload()))
    context = report.records[0].context
    factor_ids = tuple(f"factor-{index}" for index in range(48))
    values = {metric: np.arange(48, dtype=np.float64) for metric in METRICS}
    counts = {metric: np.ones(48, dtype=np.int64) for metric in METRICS}
    bundle = BatchEvaluationBundle(
        factor_ids=factor_ids, label_id="fixture", scalar_metrics=values,
        series_metrics={}, metadata={
            "source_qualification_applied": True,
            "source_qualification_status": "qualified_current_source",
            "source_qualification_winner": winner,
            "source_qualification_cache_status": "supplied",
            "source_qualification_origin": "report_candidate",
            "source_qualification_provider_status": "candidate_validated",
        }, observation_counts=counts)
    oracle_values = {metric: values[metric].copy() for metric in METRICS}
    oracle_counts = {metric: counts[metric].copy() for metric in METRICS}
    oracle = BatchEvaluationBundle(
        factor_ids=factor_ids, label_id="fixture", scalar_metrics=oracle_values,
        series_metrics={}, observation_counts=oracle_counts,
        metadata={"source_request_fingerprint": context.request_content_sha256})
    profile = report.records[0].cuda if winner == "cuda" else report.records[0].cpu
    outputs = tuple(replace(
        old, values_sha256=identity.values_sha256,
        finite_mask_sha256=identity.finite_mask_sha256,
        observation_counts_sha256=identity.observation_counts_sha256,
        coverage_expected=48, coverage_observed=48)
        for old, identity in ((old, metric_output_identity(values[old.metric_id],
                                                          counts[old.metric_id]))
                              for old in profile.outputs))
    selected = replace(profile, outputs=outputs)
    record = replace(report.records[0], **{winner: selected})
    qualification = replace(report.qualification, winning_backend=winner)
    report = replace(report, records=(record, report.records[1]),
                     qualification=qualification)
    receipt = {"context_before": context, "context_after": context,
               "backend_used": winner,
               "effective_max_tile_size": dict(qualification.actual_tile_sizes)[winner],
               "oom_retries": 0}
    return bundle, receipt, report, oracle


def test_valid_report_candidate_and_process_cache_runs_return_sanitized_hash_evidence():
    bundle, receipt, report, oracle = _fixture()
    evidence = verify_provider_run(bundle, receipt, report, oracle,
                                   run_index=0, require_cache_hit=False)
    context = report.records[0].context
    assert evidence["context"] == asdict(context)
    assert evidence["qualification_applied"] is True
    assert evidence["qualification_status"] == "qualified_current_source"
    assert evidence["qualification_winner"] == report.qualification.winning_backend
    assert evidence["effective_max_tile_size"] == dict(
        report.qualification.actual_tile_sizes)[report.qualification.winning_backend]
    assert evidence["oom_retries"] == 0
    assert evidence["request_content_sha256"] == context.request_content_sha256
    assert evidence["executable_source_sha256"] == context.executable_source_sha256
    assert evidence["runtime_fingerprint_sha256"] == context.runtime_fingerprint_sha256
    assert evidence["correctness_digest_sha256"] == report.qualification.correctness_digest_sha256
    assert evidence["source_qualification_origin"] == "report_candidate"
    assert evidence["source_qualification_provider_status"] == "candidate_validated"
    assert evidence["source_qualification_cache_status"] == "supplied"
    assert tuple(evidence["metric_identities"]) == METRICS
    assert set(evidence["metric_identities"][METRICS[0]]) == {
        "values_sha256", "finite_mask_sha256", "observation_counts_sha256"}

    assert all(not np.shares_memory(bundle.scalar_metrics[name],
                                   oracle.scalar_metrics[name]) for name in METRICS)
    bundle.metadata.update(source_qualification_origin="process_cache",
                           source_qualification_provider_status="not_checked",
                           source_qualification_cache_status="cache_hit")
    evidence = verify_provider_run(bundle, receipt, report, oracle,
                                   run_index=1, require_cache_hit=True)
    assert evidence["source_qualification_origin"] == "process_cache"
    assert evidence["source_qualification_cache_status"] == "cache_hit"


@pytest.mark.parametrize("mutation,match", [
    ("context", "context"), ("winner", "profile"),
    ("origin", "origin"), ("provider_status", "provider status"),
    ("value", "profile output"), ("count", "profile output"),
    ("oracle_disagreement", "oracle mismatch"), ("oom", "OOM"),
    ("cache_status_missing", "cache status"),
    ("cache_status_none", "cache status"),
    ("cache_status_hit", "cache status"),
    ("cache_status_unknown", "cache status"),
    ("oom_boolean", "OOM"), ("oom_missing", "OOM"), ("oom_negative", "OOM"),
])
def test_provider_run_rejects_unbound_or_mismatching_evidence(mutation, match):
    bundle, receipt, report, oracle = _fixture()
    if mutation == "context":
        receipt["context_after"] = None
    elif mutation == "winner":
        receipt["backend_used"] = "cuda" if receipt["backend_used"] == "cpu" else "cpu"
    elif mutation == "origin":
        bundle.metadata["source_qualification_origin"] = "none"
    elif mutation == "provider_status":
        bundle.metadata["source_qualification_provider_status"] = "provider_error"
    elif mutation == "cache_status_missing":
        del bundle.metadata["source_qualification_cache_status"]
    elif mutation == "cache_status_none":
        bundle.metadata["source_qualification_cache_status"] = None
    elif mutation == "cache_status_hit":
        bundle.metadata["source_qualification_cache_status"] = "cache_hit"
    elif mutation == "cache_status_unknown":
        bundle.metadata["source_qualification_cache_status"] = "unknown"
    elif mutation == "value":
        bundle.scalar_metrics[METRICS[0]][0] += 1
    elif mutation == "count":
        bundle.observation_counts[METRICS[0]][0] += 1
    elif mutation == "oracle_disagreement":
        oracle.scalar_metrics[METRICS[0]][0] += 1
    elif mutation == "oom_boolean":
        receipt["oom_retries"] = True
    elif mutation == "oom_missing":
        del receipt["oom_retries"]
    elif mutation == "oom_negative":
        receipt["oom_retries"] = -1
    elif mutation == "oom":
        receipt["oom_retries"] = 1
    with pytest.raises(ValueError, match=match):
        verify_provider_run(bundle, receipt, report, oracle,
                            run_index=0, require_cache_hit=False)


def test_provider_run_rejects_non_linear_report_kind():
    bundle, receipt, report, oracle = _fixture()
    with pytest.raises(ValueError, match="report kind"):
        verify_provider_run(bundle, receipt, replace(report, kind="other"), oracle,
                            run_index=0, require_cache_hit=False)


def test_cuda_winning_profile_is_checked_with_its_actual_tile_and_output_identity():
    bundle, receipt, report, oracle = _fixture("cuda")
    evidence = verify_provider_run(bundle, receipt, report, oracle,
                                   run_index=0, require_cache_hit=False)
    assert evidence["backend_used"] == "cuda"
