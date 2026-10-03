"""Mock-only contracts for the bounded real-F16 cache A/B harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_raw_summary_cache_oct04.py"
SPEC = importlib.util.spec_from_file_location("benchmark_raw_summary_cache_oct04", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def _cohort():
    from factor_optimizer.research_batch import automatic_time_split

    times, assets, factors = 500, 256, 16
    factor_ids = tuple(f"mock_{i:02d}" for i in range(factors))
    time_values = np.arange(times, dtype=np.int64)
    asset_values = np.asarray([f"asset-{i:03d}" for i in range(assets)])
    ta = AxisRef("time", "int", times, time_values)
    aa = AxisRef("asset", "str", assets, asset_values)
    values = np.arange(times * assets * factors, dtype=np.float32).reshape(
        times, assets, factors)
    batch = FactorBatch(factor_ids, ta, aa, values)
    label_values = np.full((times, assets), .01, dtype=np.float32)
    labels = LabelBundle(
        "mock-label", label_values, 1,
        decision_time=tuple(time_values),
        label_start_time=tuple(time_values + 1),
        label_end_time=tuple(time_values + 2),
        asset_axis=aa,
    )
    split = automatic_time_split(labels)
    sources = []
    for i, factor_id in enumerate(factor_ids):
        digest = f"{i + 1:064x}"
        sources.append({
            "factor": factor_id,
            "uri": f"cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values/{digest}/{factor_id}.parquet",
            "etag": f"etag-{i}", "sha256": digest,
            "downloaded_bytes": 1024 + i, "manifest_sha256": "a" * 64,
            "source_status": "evaluated_optimization_pending",
            "expression": f"fixture-{i}",
        })
    provenance = {
        "manifest_uri": "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/" + "a" * 64 + "/landing_manifest.json",
        "retained_factor_ids": list(factor_ids), "quarantined_factors": [],
        "sources": sources, "asset_selection_split": split.identity,
        "asset_selection_train_days": len(split.train_indices),
        "max_factor_bytes": 128 * 1024**2,
        "max_batch_factor_bytes": 2 * 1024**3,
    }
    lineages = {factor_id: {"treatment": "raw"} for factor_id in factor_ids}
    return batch, labels, provenance, lineages


def _fake_result(batch, labels, config):
    from factor_optimizer.research_batch import automatic_time_split

    split = automatic_time_split(labels, config)
    output_values = np.asarray(batch.values, dtype=np.float64).copy()
    output = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, output_values,
        validity=np.isfinite(output_values),
        context_refs={"optimization_mode": "research_only", "split": split.identity},
    )
    factors = {}
    for factor_id in batch.factor_ids:
        factors[factor_id] = SimpleNamespace(
            factor_id=factor_id, status="raw_retained", selected_family="NO_OP_RAW",
            reason="test fixture", plan_identity=f"plan-{factor_id}",
            plan=SimpleNamespace(identity=f"plan-{factor_id}"),
            train_gain=None, validation_lower_bound=None,
            validation_candidate_identity=None, validation_coverage=1.0,
            materialization_error=None,
            candidates=({"family": "NO_OP_RAW", "status": "raw_retained"},),
            joint_diagnostics={"comparison_status": "NOT_REQUIRED", "value": 1.0},
        )
    return SimpleNamespace(
        optimized=output, factors=factors, split=split,
        execution_mode="research_only", test_evaluated=False,
    )

def _record_two_raw_summary_calls():
    from factor_optimizer import research_summary_cache

    panel = np.column_stack((np.linspace(-.1, .1, 60),
                             np.sin(np.arange(60)) * .01,
                             np.full(60, .4)))
    cache = research_summary_cache.RawMetricSummaryCache()
    cache.summarize(panel)
    cache.summarize(panel.copy())



def test_mock_benchmark_reuses_one_strict_cohort_in_abba_and_reports_exact_parity(
        tmp_path, monkeypatch):
    from factor_optimizer import research_fitness, research_summary_cache

    report_path = tmp_path / "raw-summary.json"
    cohort = _cohort()
    load_calls, resource_calls, run_calls = [], [], []
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"code": "stable"})
    monkeypatch.setattr(BENCH, "runtime_context", lambda: {"runtime": "stable"})

    def loader(**kwargs):
        load_calls.append(kwargs)
        return cohort

    def resource_check():
        resource_calls.append(1)

    def runner(batch, labels, *, config, allow_research, lineages):
        run_calls.append((batch, labels, lineages,
                          research_summary_cache.RawMetricSummaryCache.__name__))
        cache = research_summary_cache.RawMetricSummaryCache()
        panel = np.column_stack((np.linspace(-.1, .1, 60),
                                 np.sin(np.arange(60)) * .01,
                                 np.full(60, .4)))
        cache.summarize(panel)
        cache.summarize(panel.copy())
        research_fitness.summarize(panel)
        return _fake_result(batch, labels, config)

    result = BENCH.run_benchmark(
        report=report_path, source_loader=loader, auto_runner=runner,
        resource_check=resource_check,
        environment_check=lambda: None,
    )
    saved = json.loads(report_path.read_text())

    assert len(load_calls) == 1
    assert load_calls[0]["n_factors"] == 16
    assert load_calls[0]["n_assets"] == 256
    assert load_calls[0]["coverage_policy"] == "strict"
    assert all(call[0] is cohort[0] and call[1] is cohort[1] for call in run_calls)
    assert [call[3] for call in run_calls] == [
        "ObservedSummaryCache",
        "UncachedSummaryCache", "ObservedSummaryCache",
        "ObservedSummaryCache", "UncachedSummaryCache",
    ]
    for run in saved["runs"] + [saved["warmup"]]:
        counts = run["summarize_counts"]
        assert counts["raw_summary_requests"] == 2
        assert counts["raw_summary_failures"] == 0
        assert counts["raw_summary_computes"] + counts["raw_summary_cache_hits"] == 2
    assert len(resource_calls) == 6
    assert result["equivalence"]["all_outputs_and_ledgers_identical"] is True
    assert saved["scope"]["dates"] == 500
    assert saved["scope"]["assets"] == 256
    assert saved["scope"]["full_market_or_multiyear_claim"] is False
    assert [run["mode"] for run in saved["runs"]] == [
        "uncached", "cached", "cached", "uncached",
    ]
    assert saved["runs"][0]["summarize_counts"]["raw_summary_computes"] == 2
    assert saved["runs"][1]["summarize_counts"]["raw_summary_computes"] == 1
    assert saved["runs"][1]["summarize_counts"]["raw_summary_cache_hits"] == 1
    assert saved["warmup"]["mode"] == "cached"
    assert saved["paired_timing"]["pairs"][1]["order"] == ["cached", "uncached"]
    assert saved["source_hashes_before"] == saved["source_hashes_after"]
    assert saved["runtime_fingerprint_before"] == saved["runtime_fingerprint_after"]
    assert report_path.stat().st_size < BENCH.MAX_REPORT_BYTES


def test_existing_report_refused_before_resource_or_cos_load(tmp_path):
    path = tmp_path / "keep.json"
    path.write_text("unchanged")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        BENCH.run_benchmark(
            report=path,
            source_loader=lambda **kwargs: pytest.fail("source load started"),
            resource_check=lambda: pytest.fail("resource check started"),
        )
    assert path.read_text() == "unchanged"


def test_resource_failure_prevents_source_load_and_report(tmp_path):
    path = tmp_path / "not-created.json"

    def insufficient():
        raise MemoryError("not enough admitted RAM")

    with pytest.raises(MemoryError, match="not enough"):
        BENCH.run_benchmark(
            report=path,
            source_loader=lambda **kwargs: pytest.fail("source load started"),
            resource_check=insufficient,
            environment_check=lambda: None,
        )
    assert not path.exists()



def test_default_source_loader_forwards_strict_f16_kwargs(monkeypatch):
    from types import ModuleType
    import sys

    received = []
    expected = {
        "n_factors": 16, "n_assets": 256, "include_lineages": True,
        "max_factor_bytes": 128 * 1024**2,
        "max_batch_factor_bytes": 2 * 1024**3,
        "coverage_policy": "strict",
    }
    marker = object()
    fake_cos = ModuleType("cos_batch_audit")
    fake_cos.load_cos_sample = lambda **kwargs: (received.append(kwargs), marker)[1]
    monkeypatch.setitem(sys.modules, "cos_batch_audit", fake_cos)
    monkeypatch.setattr(BENCH.sys, "path", list(BENCH.sys.path))

    loaded = BENCH._default_source_loader(**expected)

    assert loaded is marker
    assert received == [expected]



def test_cache_class_and_summarizer_are_restored_when_runner_raises(tmp_path, monkeypatch):
    from factor_optimizer import research_fitness, research_summary_cache

    cohort = _cohort()
    original_cache = research_summary_cache.RawMetricSummaryCache
    original_summarize = research_fitness.summarize
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"code": "stable"})
    monkeypatch.setattr(BENCH, "runtime_context", lambda: {"runtime": "stable"})

    def failing_runner(*args, **kwargs):
        raise RuntimeError("optimizer fixture failure")

    with pytest.raises(RuntimeError, match="optimizer fixture failure"):
        BENCH.run_benchmark(
            report=tmp_path / "runner-failure.json",
            source_loader=lambda **kwargs: cohort,
            auto_runner=failing_runner,
            resource_check=lambda: None,
            environment_check=lambda: None,
        )

    assert research_summary_cache.RawMetricSummaryCache is original_cache
    assert research_fitness.summarize is original_summarize



@pytest.mark.parametrize("field", ["ledger", "validity", "plan_identity"])
def test_benchmark_rejects_output_ledger_validity_or_plan_drift(
        field, tmp_path, monkeypatch):
    cohort = _cohort()
    calls = []
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"code": "stable"})
    monkeypatch.setattr(BENCH, "runtime_context", lambda: {"runtime": "stable"})

    def runner(batch, labels, *, config, **kwargs):
        result = _fake_result(batch, labels, config)
        _record_two_raw_summary_calls()
        calls.append(1)
        if len(calls) == 2:
            factor_id = batch.factor_ids[0]
            if field == "ledger":
                result.factors[factor_id].candidates = (
                    {"family": "NO_OP_RAW", "status": "different"},)
            elif field == "validity":
                object.__setattr__(
                    result.optimized, "validity",
                    result.optimized.validity.astype(np.uint8),
                )
            else:
                result.factors[factor_id].plan_identity += "-drift"
        return result

    expected = ("selected plan identity" if field == "plan_identity"
                else "results differ")
    with pytest.raises((RuntimeError, ValueError), match=expected):
        BENCH.run_benchmark(
            report=tmp_path / f"drift-{field}.json",
            source_loader=lambda **kwargs: cohort,
            auto_runner=runner, resource_check=lambda: None,
            environment_check=lambda: None,
        )
    assert len(calls) == (2 if field == "plan_identity" else 5)
    assert not (tmp_path / f"drift-{field}.json").exists()



@pytest.mark.parametrize("changed_snapshot", ["source", "runtime"])
def test_benchmark_aborts_on_source_or_runtime_drift_between_optimizer_windows(
        changed_snapshot, tmp_path, monkeypatch):
    cohort = _cohort()
    calls = []
    counters = {"source": 0, "runtime": 0}

    def source_hashes():
        counters["source"] += 1
        value = "changed" if changed_snapshot == "source" and counters["source"] >= 3 else "stable"
        return {"code": value}

    def runtime_context():
        counters["runtime"] += 1
        value = "changed" if changed_snapshot == "runtime" and counters["runtime"] >= 3 else "stable"
        return {"runtime": value}

    monkeypatch.setattr(BENCH, "source_hashes", source_hashes)
    monkeypatch.setattr(BENCH, "runtime_context", runtime_context)

    def runner(batch, labels, *, config, **kwargs):
        calls.append(1)
        return _fake_result(batch, labels, config)

    expected = ("source files changed during optimizer call"
                if changed_snapshot == "source"
                else "runtime changed during optimizer call")
    with pytest.raises(RuntimeError, match=expected):
        BENCH.run_benchmark(
            report=tmp_path / f"{changed_snapshot}-drift.json",
            source_loader=lambda **kwargs: cohort,
            auto_runner=runner, resource_check=lambda: None,
            environment_check=lambda: None,
        )

    assert len(calls) == 1
    assert not (tmp_path / f"{changed_snapshot}-drift.json").exists()



def test_report_size_cap_refuses_output_before_exclusive_write(tmp_path, monkeypatch):
    cohort = _cohort()
    report = tmp_path / "oversized.json"
    monkeypatch.setattr(BENCH, "MAX_REPORT_BYTES", 128)
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"code": "stable"})
    monkeypatch.setattr(BENCH, "runtime_context", lambda: {"runtime": "stable"})

    def runner(batch, labels, *, config, **kwargs):
        _record_two_raw_summary_calls()
        return _fake_result(batch, labels, config)

    with pytest.raises(ValueError, match="strictly smaller than 1 MiB"):
        BENCH.run_benchmark(
            report=report,
            source_loader=lambda **kwargs: cohort,
            auto_runner=runner,
            resource_check=lambda: None,
            environment_check=lambda: None,
        )

    assert not report.exists()


def test_exclusive_report_write_preserves_file_created_after_preflight(tmp_path, monkeypatch):
    cohort = _cohort()
    report = tmp_path / "exclusive-race.json"
    calls = []
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"code": "stable"})
    monkeypatch.setattr(BENCH, "runtime_context", lambda: {"runtime": "stable"})

    def runner(batch, labels, *, config, **kwargs):
        if not report.exists():
            report.write_text("created after preflight")
        calls.append(1)
        _record_two_raw_summary_calls()
        return _fake_result(batch, labels, config)

    with pytest.raises(FileExistsError):
        BENCH.run_benchmark(
            report=report,
            source_loader=lambda **kwargs: cohort,
            auto_runner=runner, resource_check=lambda: None,
            environment_check=lambda: None,
        )

    assert len(calls) == 5
    assert report.read_text() == "created after preflight"
