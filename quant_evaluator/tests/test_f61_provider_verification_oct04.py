"""Provider receipts bind fresh physical F61 runs to the qualified profile."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib

import numpy as np
import pytest

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_profile_report_schema import (
    F61_ALL15_SCHEMA, F61_ALL24_SCHEMA,
)
from quant_evaluator.scripts import benchmark_real_cos_source_batch as source_batch
from quant_evaluator.scripts.source_profile_measurement import (
    build_counterbalanced_profile_record,
)
from quant_evaluator.scripts.source_profile_report_reader import SourceProfileReport
from quant_evaluator.runtime.source_route_profiles import validate_source_route_profile_qualification
from quant_evaluator.scripts.verify_real_cos_f61_provider import verify_provider_run
from quant_evaluator.tests.test_f61_source_profile_report_reader import _fixture


METRICS = F61_ALL24_SCHEMA.metric_ids
SHAPE = (2586, 5461, 61)


def _run_receipt(bundle, context, backend, width, seconds, *, effective_cap=16):
    ranges = tuple((i, min(i + width, 61)) for i in range(0, 61, width))
    return {
        "backend_requested": backend, "backend_used": backend,
        "source_auto_policy": "qualified_only",
        "source_request_fingerprint": context.request_content_sha256,
        "factor_ids_sha256": hashlib.sha256(
            __import__("json").dumps(list(bundle.factor_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "seconds": float(seconds), "total_wall_seconds": float(seconds),
        "timing_scope": "evaluate_factor_source_batch_wall_v1", "oom_retries": 0,
        "actual_source_tile_size": width,
        "actual_gpu_factor_tile_size": width if backend == "cuda" else None,
        "effective_max_tile_size": effective_cap, "execution_schedule_scope": "zero_oom_source_equals_compute_v1",
        "factor_tiles_processed": len(ranges), "tile_ranges": ranges,
        "compute_tile_ranges": ranges,
        "execution_schedule_sha256": stable_content_hex(tag="SourceExecutionSchedule.v1", fields={
            "backend": backend, "factor_count": 61, "source_width": width,
            "source_ranges": ranges, "compute_ranges": ranges, "oom_retries": 0,
        }),
    }


def _fixture_run(winner="cpu", version="24"):
    metrics = F61_ALL15_SCHEMA.metric_ids if version == "15" else METRICS
    context = _fixture(metrics)[0].context
    factors = tuple(f"factor-{i}" for i in range(61))
    counts = {metric: np.full(61, 10, dtype=np.int64) for metric in metrics}
    scalars = {m: np.full(61, 0.25, dtype=np.float64)
               for m in metrics if m not in {"rank_ic_series", "pearson_ic_series"}}
    series = {m: np.full((2586, 61), 0.25, dtype=np.float64)
              for m in ("rank_ic_series", "pearson_ic_series")}

    def bundle(backend, *, auto=False):
        metadata = {"backend_used": backend,
                    "source_request_fingerprint": context.request_content_sha256}
        if auto:
            metadata.update({"source_auto_policy": "qualified_only",
                             "source_qualification_status": "qualified_current_source",
                             "source_qualification_applied": True,
                             "source_qualification_winner": winner,
                             "source_qualification_origin": "report_candidate",
                             "source_qualification_provider_status": "candidate_validated",
                             "source_qualification_cache_status": "supplied"})
        return BatchEvaluationBundle(factors, "label", deepcopy(scalars), deepcopy(series),
                                     {}, metadata, deepcopy(counts))

    cpu, cuda = bundle("cpu"), bundle("cuda")
    cpu_receipt = _run_receipt(cpu, context, "cpu", 16, 1.0 if winner == "cpu" else 2.0)
    cuda_receipt = _run_receipt(cuda, context, "cuda", 4, 2.0 if winner == "cpu" else 1.0)
    comparison = source_batch.compare(cpu, cuda, metrics, expected_days=2586)
    records = (
        build_counterbalanced_profile_record(
            cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_receipt,
            cuda_run_receipt=cuda_receipt, context=context,
            comparison_report=comparison, execution_order=("cpu", "cuda"),
            cpu_correctness_validated=True, cuda_correctness_validated=True),
        build_counterbalanced_profile_record(
            cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_receipt,
            cuda_run_receipt=cuda_receipt, context=context,
            comparison_report=comparison, execution_order=("cuda", "cpu"),
            cpu_correctness_validated=True, cuda_correctness_validated=True),
    )
    qualification = validate_source_route_profile_qualification(records, expected_context=context)
    assert qualification.winning_backend == winner
    report = SourceProfileReport(f"real_cos_profile_abba_f61_all{version}.v1", "complete",
                                 "a" * 64, records, qualification)
    auto_bundle = bundle(winner, auto=True)
    selected = records[0].cpu if winner == "cpu" else records[0].cuda
    auto_width = selected.actual_tile_size
    auto_receipt = _run_receipt(auto_bundle, context, winner, auto_width, selected.seconds,
                                effective_cap=auto_width)
    auto_receipt.update({"backend_requested": "auto", "source_auto_policy": "qualified_only",
                         "context_before": context, "context_after": context})
    oracle_bundle = bundle("cpu")
    return auto_bundle, auto_receipt, report, oracle_bundle


@pytest.mark.parametrize("winner", ["cpu", "cuda"])
@pytest.mark.parametrize("version", ["15", "24"])
def test_first_and_cached_provider_receipts_bind_physical_runs_to_fixed_f61_profile(winner, version):
    bundle, receipt, report, oracle = _fixture_run(winner, version)
    assert type(report) is SourceProfileReport
    assert report.status == "complete"
    assert report.records[0].context.request_shape == SHAPE
    assert report.records[0].context.requested_tile_size == 16
    first = verify_provider_run(bundle, receipt, report, oracle, run_index=0,
                                require_cache_hit=False)
    assert first["run_index"] == first["oracle_report"]["run_index"] == 0
    assert first["auto_protocol_run_index"] == 4
    assert first["source_qualification_origin"] == "report_candidate"
    assert first["source_qualification_provider_status"] == "candidate_validated"
    assert first["source_qualification_cache_status"] == first["cache_status"] == "supplied"
    bundle.metadata.update(source_qualification_origin="process_cache",
                           source_qualification_provider_status="not_checked",
                           source_qualification_cache_status="cache_hit")
    second = verify_provider_run(bundle, receipt, report, oracle, run_index=1,
                                 require_cache_hit=True)
    assert second["run_index"] == second["oracle_report"]["run_index"] == 1
    assert second["auto_protocol_run_index"] == 5
    assert second["source_qualification_origin"] == "process_cache"
    assert second["source_qualification_provider_status"] == "not_checked"
    assert second["source_qualification_cache_status"] == "cache_hit"
    assert second["cache_status"] == "cache_hit"
    assert second["context"] == asdict(report.records[0].context)


@pytest.mark.parametrize("mutation", [
    "bool_index", "bad_index", "policy", "context", "output_within_tolerance",
    "wrong_shape", "wrong_kind",
])
def test_provider_rejects_invalid_receipts_only_after_valid_baseline(mutation):
    bundle, receipt, report, oracle = _fixture_run()
    baseline = verify_provider_run(bundle, receipt, report, oracle, run_index=0,
                                   require_cache_hit=False)
    assert baseline["oracle_report"]["pass"] is True
    if mutation == "bool_index":
        args = dict(run_index=True, require_cache_hit=False)
    elif mutation == "bad_index":
        args = dict(run_index=2, require_cache_hit=False)
    else:
        args = dict(run_index=0, require_cache_hit=False)
        if mutation == "policy":
            receipt["source_auto_policy"] = "legacy_measured"
        elif mutation == "context":
            context = report.records[0].context
            receipt["context_after"] = replace(
                context, request_content_sha256="0" * 64)
        elif mutation == "output_within_tolerance":
            bundle.scalar_metrics["rank_ic"][0] += 1e-12
        elif mutation == "wrong_shape":
            wrong_context = replace(report.records[0].context,
                                    request_shape=(2586, 5461, 60))
            wrong_records = tuple(replace(record, context=wrong_context)
                                  for record in report.records)
            report = replace(report, records=wrong_records)
        elif mutation == "wrong_kind":
            report = replace(report, kind="real_cos_profile_abba_f61_all15.v1")
    with pytest.raises((ValueError, TypeError)):
        verify_provider_run(bundle, receipt, report, oracle, **args)
