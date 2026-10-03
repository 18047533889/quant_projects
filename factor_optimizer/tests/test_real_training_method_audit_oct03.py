"""Mock-source contracts for the real COS method-audit entry point."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "real_training_method_audit",
    Path(__file__).parents[1] / "scripts/audit_real_training_methods_oct03.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def sample():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    n, a = 500, 256
    times = np.arange(n)
    assets = np.asarray([f"a{i}.SZ" for i in range(a)])
    ta = AxisRef("time", "int", n, times)
    aa = AxisRef("asset", "str", a, assets)
    values = np.arange(n*a, dtype=np.float32).reshape(n, a, 1)
    batch = FactorBatch(("mock",), ta, aa, values)
    labels = LabelBundle("mock-return", np.full((n, a), .01, dtype=np.float32), 1,
        decision_time=tuple(times), label_start_time=tuple(times+1),
        label_end_time=tuple(times+2), asset_axis=aa)
    from factor_optimizer.research_batch import automatic_time_split
    split = automatic_time_split(labels)
    source_hash = "b" * 64
    provenance = {
        "manifest_uri": AUDIT.MANIFEST_PREFIX + "a" * 64 + "/landing_manifest.json",
        "sources": [{"factor": "mock", "uri": f"{AUDIT.FACTOR_POOL}/{source_hash}/mock.parquet",
            "etag": "mock-etag", "sha256": source_hash, "downloaded_bytes": 123,
            "manifest_sha256": "c" * 64, "source_status": "evaluated_optimization_pending",
            "expression": "ts_delta(col('volume'), 5)"}],
        "retained_factor_ids": ["mock"], "asset_selection_split": split.identity,
        "asset_selection_train_days": len(split.train_indices), "max_factor_bytes": AUDIT.MAX_FACTOR_BYTES,
        "max_batch_factor_bytes": AUDIT.MAX_BATCH_BYTES,
    }
    return batch, labels, provenance


def test_server_c_environment_guard_is_fail_closed():
    with pytest.raises(RuntimeError, match="ASHARE_PARQUET_ROOT"):
        AUDIT.check_environment({})
    AUDIT.check_environment({"ASHARE_PARQUET_ROOT": "/home/sunhaiwei/cos_data",
        "DATA_ACCESS_COS_CLI": "admin-cos",
        "DATA_ACCESS_COS_CACHE_ROOT": str(AUDIT.CACHE_ROOT)})


def test_resource_gate_rejects_low_memory_or_disk(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="RAM"):
        AUDIT.check_resource_headroom(memory_bytes=AUDIT.MIN_AVAILABLE_MEMORY_BYTES - 1,
                                      cache_root=tmp_path)
    monkeypatch.setattr(AUDIT.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="free"):
        AUDIT.check_resource_headroom(memory_bytes=AUDIT.MIN_AVAILABLE_MEMORY_BYTES,
                                      cache_root=tmp_path)


def test_source_validation_rejects_bad_hash_size_and_shape():
    batch, labels, provenance = sample()
    AUDIT.validate_source(batch, labels, provenance)
    bad = dict(provenance)
    bad["sources"] = [dict(provenance["sources"][0], downloaded_bytes=AUDIT.MAX_FACTOR_BYTES + 1)]
    with pytest.raises(ValueError, match="8 MiB"):
        AUDIT.validate_source(batch, labels, bad)
    bad = dict(provenance)
    bad["sources"] = [dict(provenance["sources"][0], sha256="z" * 64)]
    with pytest.raises(ValueError, match="SHA256"):
        AUDIT.validate_source(batch, labels, bad)


def test_validate_source_rejects_shifted_and_permuted_same_shape_axes():
    from quant_evaluator.contracts.factor_batch import AxisRef
    batch, labels, provenance = sample()
    shifted_times = np.asarray(batch.time_axis.values) + 1
    shifted_time_axis = AxisRef("time", "int", len(shifted_times), shifted_times)
    with pytest.raises(ValueError, match="time axis"):
        AUDIT.validate_source(replace(batch, time_axis=shifted_time_axis), labels, provenance)
    reversed_assets = np.asarray(batch.asset_axis.values)[::-1].copy()
    permuted_asset_axis = AxisRef("asset", "str", len(reversed_assets), reversed_assets)
    with pytest.raises(ValueError, match="asset axis"):
        AUDIT.validate_source(replace(batch, asset_axis=permuted_asset_axis), labels, provenance)
    missing_label_axis = SimpleNamespace(values=labels.values,
        decision_time=labels.decision_time, asset_axis=None, validity=labels.validity)
    with pytest.raises(ValueError, match="asset axes are required"):
        AUDIT.validate_source(batch, missing_label_axis, provenance)


def test_source_fingerprint_detects_value_mutation():
    batch, labels, provenance = sample()
    object.__setattr__(batch, "values", batch.values.copy())
    before = AUDIT.source_fingerprint(batch, labels, provenance)
    batch.values[0, 0, 0] += 1
    after = AUDIT.source_fingerprint(batch, labels, provenance)
    assert before != after


def test_source_fingerprint_covers_label_asset_axis_and_timing_metadata():
    from quant_evaluator.contracts.factor_batch import AxisRef
    batch, labels, provenance = sample()
    before = AUDIT.source_fingerprint(batch, labels, provenance)
    shifted_start = tuple(value + 1 for value in labels.label_start_time)
    shifted_end = tuple(value + 1 for value in labels.label_end_time)
    changed_timing = replace(labels, label_start_time=shifted_start,
                             label_end_time=shifted_end)
    assert AUDIT.source_fingerprint(batch, changed_timing, provenance) != before
    reversed_assets = np.asarray(labels.asset_axis.values)[::-1].copy()
    changed_axis = replace(labels, asset_axis=AxisRef("asset", "str",
        len(reversed_assets), reversed_assets))
    assert AUDIT.source_fingerprint(batch, changed_axis, provenance) != before


def test_report_has_fixed_exploratory_semantics_and_unscored_holdouts():
    batch, labels, provenance = sample()
    methods = [{"family": "CAUSAL_SMOOTHING", "status": "executed"},
        {"family": "NEUTRALIZATION", "status": "requires_additional_inputs_or_control",
         "reason": "historical exposure unavailable"}]
    report = AUDIT.build_report(batch, labels, provenance, methods,
        fingerprints={"before": "d"*64, "after": "d"*64})
    assert report["source"] == provenance
    assert report["source_fingerprints"]["before"] == report["source_fingerprints"]["after"]
    assert report["test_evaluated"] is False
    assert report["method_audit_partition"] == "TRAIN only"
    assert "purged TRAIN indices only" in report["finite_fraction_scope"]
    assert "no fitted optimizer" in report["method_semantics"]
    assert report["method_status_counts"] == {"executed": 1,
        "requires_additional_inputs_or_control": 1}
    assert report["split"]["train_days"] > 0 and report["split"]["test_days_reserved"] > 0


@pytest.mark.parametrize("has_failure, expected", [(False, 0), (True, 1)])
def test_run_audit_mock_source_checks_fingerprint_and_failure(monkeypatch, has_failure, expected):
    batch, labels, provenance = sample()
    rows = [{"family": "RAW", "status": "executed"}]
    if has_failure:
        rows.append({"family": "broken", "status": "failed", "reason": "bad execution"})
    observed = []
    monkeypatch.setattr(AUDIT, "check_environment", lambda: None)
    monkeypatch.setattr(AUDIT, "check_resource_headroom", lambda: None)
    report = AUDIT.run_audit(source_loader=lambda: (batch, labels, provenance, {}),
        method_runner=lambda b, y, *, config: observed.append((b, y, config)) or rows)
    assert observed[0][:2] == (batch, labels)
    assert report["source"] == provenance and report["methods"] == rows
    assert report["test_evaluated"] is False
    assert report["source_fingerprints"]["before"] == report["source_fingerprints"]["after"]
    assert (1 if report["method_status_counts"].get("failed", 0) else 0) == expected


def test_run_audit_rejects_mutating_method_runner(monkeypatch):
    batch, labels, provenance = sample()
    monkeypatch.setattr(AUDIT, "check_environment", lambda: None)
    monkeypatch.setattr(AUDIT, "check_resource_headroom", lambda: None)
    def mutate(b, y, *, config):
        object.__setattr__(b, "values", b.values.copy())
        b.values[0, 0, 0] += 1
        return [{"family": "bad", "status": "executed"}]
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        AUDIT.run_audit(source_loader=lambda: (batch, labels, provenance, {}),
                        method_runner=mutate)


def test_report_is_exclusive_and_capped(tmp_path):
    report = {"safe": True}
    out = tmp_path / "audit.json"
    assert AUDIT.write_report_exclusive(out, report) == out
    with pytest.raises(FileExistsError):
        AUDIT.write_report_exclusive(out, report)
    oversized = {"payload": "x" * AUDIT.MAX_REPORT_BYTES}
    with pytest.raises(ValueError, match="1 MiB"):
        AUDIT.write_report_exclusive(tmp_path / "large.json", oversized)
    assert not (tmp_path / "large.json").exists()


def test_main_prints_concise_summary_and_writes_full_report(monkeypatch, tmp_path, capsys):
    batch, labels, provenance = sample()
    report = AUDIT.build_report(batch, labels, provenance,
        [{"family": "RAW", "status": "executed"}],
        fingerprints={"before": "d"*64, "after": "d"*64})
    monkeypatch.setattr(AUDIT, "run_audit", lambda: report)
    output = tmp_path / "new.json"
    assert AUDIT.main(["--report", str(output)]) == 0
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    assert "methods" not in summary and "source" not in summary
    assert json.loads(output.read_text())["methods"][0]["family"] == "RAW"
