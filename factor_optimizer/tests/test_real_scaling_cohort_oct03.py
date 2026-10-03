"""Mock-only contracts for the paired F1/F4/F16 real-cohort audit.

These tests exercise source binding, orchestration, and report guards. The
fake optimizer runner is not evidence that the optimizer algorithm is correct.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "real_scaling_cohort_audit", SCRIPTS / "audit_real_scaling_cohort_oct03.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
SOURCE_HELPERS = importlib.import_module("audit_real_training_methods_oct03")


def sample_cohort():
    """Build the script's bounded F16 contract without any COS reads."""
    from factor_optimizer.research_batch import automatic_time_split
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    time_count, asset_count, factor_count = 500, 256, 16
    factor_ids = tuple(f"mock_{i:02d}" for i in range(factor_count))
    times = np.arange(time_count, dtype=np.int64)
    assets = np.asarray([f"a{i:03d}.SZ" for i in range(asset_count)])
    time_axis = AxisRef("time", "int", time_count, times)
    asset_axis = AxisRef("asset", "str", asset_count, assets)
    values = np.arange(time_count * asset_count * factor_count,
                       dtype=np.float32).reshape(time_count, asset_count, factor_count)
    batch = FactorBatch(factor_ids, time_axis, asset_axis, values)
    label_values = np.full((time_count, asset_count), .01, dtype=np.float32)
    labels = LabelBundle(
        "mock-forward-return", label_values, 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2), asset_axis=asset_axis,
    )
    split = automatic_time_split(labels)
    sources = []
    for i, factor_id in enumerate(factor_ids):
        sha = f"{i + 1:064x}"
        sources.append({
            "factor": factor_id,
            "uri": f"{SOURCE_HELPERS.FACTOR_POOL}/{sha}/{factor_id}.parquet",
            "etag": f"mock-etag-{i}", "sha256": sha,
            "downloaded_bytes": 1024 + i, "bytes": 1024 + i,
            "manifest_sha256": "a" * 64,
            "source_status": "evaluated_optimization_pending",
            "expression": f"mock_factor_{i}",
        })
    provenance = {
        "manifest_uri": (SOURCE_HELPERS.MANIFEST_PREFIX + "a" * 64
                         + "/landing_manifest.json"),
        "retained_factor_ids": list(factor_ids), "quarantined_factors": [],
        "sources": sources,
        "asset_selection_split": split.identity,
        "asset_selection_train_days": len(split.train_indices),
        "max_factor_bytes": AUDIT.MAX_FACTOR_BYTES,
        "max_batch_factor_bytes": AUDIT.MAX_BATCH_BYTES,
    }
    lineages = {factor_id: {"treatment": "raw", "expression": f"expr-{i}"}
                for i, factor_id in enumerate(factor_ids)}
    return batch, labels, provenance, lineages


def fake_result(batch, labels, config, *, test_evaluated=False, candidates=None):
    """Return a contract-valid report stub; does not run optimizer logic."""
    from factor_optimizer.research_batch import automatic_time_split
    from quant_evaluator.contracts.factor_batch import FactorBatch

    split = automatic_time_split(labels, config)
    values = np.asarray(batch.values, dtype=np.float64).copy()
    output = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, values,
        validity=np.isfinite(values),
        context_refs={"optimization_mode": "research_only", "split": split.identity},
    )
    factors = {}
    for factor_id in batch.factor_ids:
        plan = SimpleNamespace(identity=f"plan-{factor_id}")
        ledger = candidates if candidates is not None else (
            {"family": "NO_OP_RAW", "status": "raw_retained"},
        )
        factors[factor_id] = SimpleNamespace(
            factor_id=factor_id, status="raw_retained", selected_family="NO_OP_RAW",
            reason="test stub", plan_identity=plan.identity, plan=plan,
            train_gain=None, validation_lower_bound=None,
            validation_candidate_identity=None, validation_coverage=1.0,
            candidates=ledger, materialization_error=None,
        )
    return SimpleNamespace(
        optimized=output, factors=factors, split=split,
        execution_mode="research_only", test_evaluated=test_evaluated,
    )


def _install_non_cos_guards(monkeypatch):
    monkeypatch.setattr(SOURCE_HELPERS, "check_environment", lambda: None)
    fake_cos = ModuleType("cos_batch_audit")
    fake_cos.load_cos_sample = lambda **kwargs: pytest.fail(
        "the audit must use the injected fake loader"
    )
    monkeypatch.setitem(sys.modules, "cos_batch_audit", fake_cos)


def _call(batch, labels, provenance, lineages, *, runner, config=None,
          source_loader=None, resource_check=None):
    return AUDIT.run_audit(
        source_loader=source_loader or (lambda **kwargs: (
            batch, labels, provenance, lineages)),
        auto_runner=runner,
        resource_check=resource_check or (lambda: None), config=config,
    )


def test_run_audit_loads_one_strict_f16_and_runs_nested_prefixes(monkeypatch):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig()
    loader_calls, runner_calls, events = [], [], []

    def loader(**kwargs):
        loader_calls.append(kwargs)
        events.append("load")
        return batch, labels, provenance, lineages

    def resource_check():
        events.append("resource")

    def runner(prefix, target, **kwargs):
        runner_calls.append((prefix, target, kwargs))
        return fake_result(prefix, target, kwargs["config"])

    report = _call(batch, labels, provenance, lineages, runner=runner,
                   config=config, source_loader=loader,
                   resource_check=resource_check)

    assert events[0] == "resource" and events[1] == "load"
    assert events.count("resource") == 4
    assert loader_calls == [{
        "n_factors": 16, "n_assets": 256, "include_lineages": True,
        "max_factor_bytes": AUDIT.MAX_FACTOR_BYTES,
        "max_batch_factor_bytes": AUDIT.MAX_BATCH_BYTES, "coverage_policy": "strict",
    }]
    assert [len(prefix.factor_ids) for prefix, _, _ in runner_calls] == [1, 4, 16]
    assert [prefix.factor_ids for prefix, _, _ in runner_calls] == [
        batch.factor_ids[:size] for size in (1, 4, 16)
    ]
    for prefix, target, kwargs in runner_calls:
        assert target is labels
        assert prefix.time_axis is batch.time_axis
        assert prefix.asset_axis is batch.asset_axis
        assert kwargs["allow_research"] is True
        assert kwargs["config"] is config
        assert kwargs["lineages"] == {
            factor_id: lineages[factor_id] for factor_id in prefix.factor_ids
        }
        np.testing.assert_array_equal(
            prefix.values, batch.values[:, :, :len(prefix.factor_ids)]
        )
    assert tuple(report["prefixes"]) == ("F1", "F4", "F16")
    assert report["test_evaluated"] is False
    consistency = report["selection_consistency"]
    assert consistency["all_matched"] is True
    assert consistency["compared_factor_count"] == 4
    assert consistency["comparison_count"] == 5
    assert set(consistency["factors"]) == set(batch.factor_ids)
    assert consistency["factors"][batch.factor_ids[0]] == {
        "compared_prefixes": ["F1", "F4", "F16"],
        "comparison_count": 2, "matched": True,
        "differing_fields": [], "differences": {},
    }
    assert consistency["factors"][batch.factor_ids[3]]["compared_prefixes"] == ["F4", "F16"]
    assert consistency["factors"][batch.factor_ids[3]]["comparison_count"] == 1
    assert consistency["factors"][batch.factor_ids[3]]["matched"] is True
    # These factors exist only in F16; keep them visible, but do not claim a
    # cross-batch comparison was made for them.
    singleton = consistency["factors"][batch.factor_ids[4]]
    assert singleton == {
        "compared_prefixes": ["F16"],
        "comparison_count": 0, "matched": None,
        "differing_fields": [], "differences": {},
    }
    assert report["cohort_source_fingerprint"] == SOURCE_HELPERS.source_fingerprint(
        batch, labels, provenance
    )
    for row in report["prefixes"].values():
        assert row["source_fingerprint_before"] == row["source_fingerprint_after"]
        assert row["elapsed_seconds"] >= 0
        assert row["process_cumulative_peak_rss_bytes"] >= 0


@pytest.mark.parametrize("damage", [
    "factor_order", "source_size", "last_source_uri", "manifest_mismatch",
])
def test_run_audit_rejects_malformed_cohort_sources_before_runner(monkeypatch, damage):
    _install_non_cos_guards(monkeypatch)
    batch, labels, provenance, lineages = sample_cohort()
    bad = dict(provenance)
    bad["sources"] = [dict(row) for row in provenance["sources"]]
    if damage == "factor_order":
        bad["sources"][0]["factor"] = "wrong-factor"
    elif damage == "source_size":
        bad["sources"][0]["downloaded_bytes"] = 0
    elif damage == "last_source_uri":
        bad["sources"][-1]["uri"] = "cos://unauthorized/not-a-bound-source.parquet"
    else:
        bad["sources"][0]["manifest_sha256"] = "c" * 64
    runner_calls = []

    def runner(*args, **kwargs):
        runner_calls.append((args, kwargs))
        return fake_result(args[0], args[1], kwargs["config"])

    with pytest.raises(ValueError, match="source rows|downloaded source|factor source URI|manifest SHA256"):
        _call(batch, labels, bad, lineages, runner=runner)
    assert runner_calls == []


@pytest.mark.parametrize(
    "mismatch,expected_field",
    [
        ("plan", "plan_identity"),
        ("gain", "train_gain"),
        ("ledger", "candidate_ledger_sha256"),
    ],
)
def test_run_audit_reports_exact_selection_mismatch_across_prefixes(
        monkeypatch, mismatch, expected_field):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig()
    runner_calls = []

    def runner(prefix, target, **kwargs):
        runner_calls.append(len(prefix.factor_ids))
        result = fake_result(prefix, target, kwargs["config"])
        item = result.factors[batch.factor_ids[0]]
        if mismatch == "gain":
            # JSON canonical comparison must retain the sign bit of zero.
            item.train_gain = 0.0 if len(prefix.factor_ids) == 4 else -0.0
        elif len(prefix.factor_ids) == 4:
            if mismatch == "plan":
                item.plan_identity = "different-plan-identity"
                item.plan.identity = item.plan_identity
            else:
                item.candidates = ({
                    "family": "NO_OP_RAW", "status": "raw_retained",
                    "fixture_marker": "ledger-mismatch",
                },)
        return result

    report = _call(batch, labels, provenance, lineages, runner=runner,
                   config=config)

    assert runner_calls == [1, 4, 16]
    consistency = report["selection_consistency"]
    assert consistency["all_matched"] is False
    assert consistency["compared_factor_count"] == 4
    assert consistency["comparison_count"] == 5
    mismatch_row = consistency["factors"][batch.factor_ids[0]]
    assert mismatch_row["compared_prefixes"] == ["F1", "F4", "F16"]
    assert mismatch_row["comparison_count"] == 2
    assert mismatch_row["matched"] is False
    assert mismatch_row["differing_fields"] == [expected_field]
    assert set(mismatch_row["differences"]) == {expected_field}
    if mismatch == "gain":
        gains = mismatch_row["differences"]["train_gain"]
        assert {prefix: value.hex() for prefix, value in gains.items()} == {
            "F1": "-0x0.0p+0", "F4": "0x0.0p+0", "F16": "-0x0.0p+0",
        }
    assert consistency["factors"][batch.factor_ids[1]]["matched"] is True


def test_run_audit_rejects_custom_split_that_differs_from_asset_selection(monkeypatch):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig(warmup_bars=31)
    runner_calls = []

    def runner(*args, **kwargs):
        runner_calls.append((args, kwargs))
        return fake_result(args[0], args[1], kwargs["config"])

    with pytest.raises(ValueError, match="optimizer split differs from TRAIN asset-selection split"):
        _call(batch, labels, provenance, lineages, runner=runner, config=config)
    assert runner_calls == []


def test_run_audit_rejects_claimed_test_scoring(monkeypatch):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig()

    def runner(prefix, target, **kwargs):
        return fake_result(prefix, target, kwargs["config"], test_evaluated=True)

    with pytest.raises(ValueError, match="TEST unscored"):
        _call(batch, labels, provenance, lineages, runner=runner, config=config)


def test_run_audit_rejects_mutated_prefix_source(monkeypatch):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig()

    def mutate(prefix, target, **kwargs):
        result = fake_result(prefix, target, kwargs["config"])
        changed = np.array(prefix.values, copy=True)
        changed[0, 0, 0] += 1
        object.__setattr__(prefix, "values", changed)
        return result

    with pytest.raises(RuntimeError, match="mutated the shared cohort or prefix source"):
        _call(batch, labels, provenance, lineages, runner=mutate, config=config)


def test_run_audit_rejects_nan_candidate_ledger_and_report_over_cap(monkeypatch):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig()

    def nan_runner(prefix, target, **kwargs):
        return fake_result(prefix, target, kwargs["config"], candidates=(
            {"family": "NO_OP_RAW", "status": "raw_retained", "score": float("nan")},
        ))

    with pytest.raises(ValueError, match="Out of range float"):
        _call(batch, labels, provenance, lineages, runner=nan_runner, config=config)

    monkeypatch.setattr(SOURCE_HELPERS, "MAX_REPORT_BYTES", 1)
    normal_runner = lambda prefix, target, **kwargs: fake_result(
        prefix, target, kwargs["config"]
    )
    with pytest.raises(ValueError, match="1 MiB output limit"):
        _call(batch, labels, provenance, lineages, runner=normal_runner, config=config)


def test_main_refuses_existing_report_before_running_loader(monkeypatch, tmp_path):
    _install_non_cos_guards(monkeypatch)
    from factor_optimizer.research_batch import BatchOptimizationConfig

    batch, labels, provenance, lineages = sample_cohort()
    config = BatchOptimizationConfig()
    path = tmp_path / "existing-report.json"
    path.write_text("existing", encoding="utf-8")
    loader_calls = []
    original_run_audit = AUDIT.run_audit

    def loader(**kwargs):
        loader_calls.append(kwargs)
        return batch, labels, provenance, lineages

    def fake_runner(prefix, target, **kwargs):
        return fake_result(prefix, target, kwargs["config"])

    def run_with_injected_sources():
        return original_run_audit(
            source_loader=loader, auto_runner=fake_runner,
            resource_check=lambda: None, config=config,
        )

    # If main proceeds into the audit, this injected path records the load but
    # never touches COS or runs the production optimizer.
    monkeypatch.setattr(AUDIT, "run_audit", run_with_injected_sources)
    with pytest.raises(FileExistsError):
        AUDIT.main(["--report", str(path)])
    assert loader_calls == []


def test_summarize_factor_includes_bounded_joint_validation_and_ineligible_counts():
    candidates = (
        {"family": "A", "status": "ineligible", "reason": "raw-relative floor failed"},
        {"family": "A", "status": "ineligible", "reason": "raw-relative floor failed"},
        {"family": "B", "status": "train_evaluated", "reason": "ignored"},
    )
    item = SimpleNamespace(
        candidates=candidates, plan=SimpleNamespace(identity="plan-x"),
        plan_identity="plan-x", materialization_error=None, status="selected",
        selected_family="B", reason="selected", train_gain=0.1,
        validation_lower_bound=0.02, validation_candidate_identity="candidate-x",
        validation_coverage=1.0,
        joint_diagnostics={
            "objective": "joint", "policy": "joint.v1",
            "comparison_status": "SUPERIOR",
            "baseline_train_raw": {"sharpe": 1.0},
            "validation_raw": {"rank_ic": 0.1, "sharpe": -0.2, "ignored": "large"},
            "validation_candidate": {"rank_ic": 0.2, "sharpe": 0.3},
        },
    )

    summary = AUDIT._summarize_factor(item)

    assert summary["candidate_family_status_counts"] == {
        "A": {"ineligible": 2}, "B": {"train_evaluated": 1},
    }
    assert summary["ineligible_family_reason_counts"] == {
        "A": {"raw-relative floor failed": 2},
    }
    assert summary["joint_validation"] == {
        "objective": "joint", "policy": "joint.v1", "comparison_status": "SUPERIOR",
        "validation_raw": {"rank_ic": 0.1, "sharpe": -0.2},
        "validation_candidate": {"rank_ic": 0.2, "sharpe": 0.3},
    }
    assert len(AUDIT._canonical_json(summary["joint_validation"]).encode("utf-8")) < 2048


def test_summarize_factor_rejects_non_json_joint_validation_metric():
    item = SimpleNamespace(
        candidates=({"family": "A", "status": "raw_retained"},),
        plan=SimpleNamespace(identity="plan-x"), plan_identity="plan-x",
        materialization_error=None, status="raw_retained", selected_family="NO_OP_RAW",
        reason="raw", train_gain=None, validation_lower_bound=None,
        validation_candidate_identity=None, validation_coverage=None,
        joint_diagnostics={"validation_raw": {"sharpe": float("nan")}},
    )
    with pytest.raises(ValueError):
        AUDIT._summarize_factor(item)

def test_long_ineligible_reasons_remain_distinguishable_when_bounded():
    prefix = "shared-prefix-" * 30
    counts = AUDIT._candidate_diagnostic_counts((
        {"family": "A", "status": "ineligible", "reason": prefix + "first-tail"},
        {"family": "A", "status": "ineligible", "reason": prefix + "second-tail"},
    ))[2]

    assert len(counts) == 2
    assert sorted(counts.values()) == [1, 1]
    assert all(len(reason) <= AUDIT.MAX_DIAGNOSTIC_TEXT for reason in counts)
