"""Tiny safety and equivalence checks for the output-buffer benchmark."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "factor_optimizer/scripts/benchmark_output_buffer_oct04.py"
SPEC = importlib.util.spec_from_file_location("output_buffer_benchmark_oct04", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def _fresh_child(mode: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--child", mode, "--factors", "2",
         "--times", "3", "--assets", "4", "--seed", "20261004"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=30,
    )
    return json.loads(result.stdout)


def test_tiny_fresh_children_produce_equivalent_readonly_outputs():
    old = _fresh_child("old_list_stack")
    new = _fresh_child("factor_major")

    assert old["fixture"]["cells"] == 24
    assert old["mode"] == "old_list_stack"
    assert new["mode"] == "factor_major"
    assert old["shape"] == new["shape"] == [3, 4, 2]
    assert old["factor_ids"] == new["factor_ids"]
    assert old["dtype"] == new["dtype"] == "float64"
    assert old["values_sha256"] == new["values_sha256"]
    assert old["validity_sha256"] == new["validity_sha256"]
    assert old["values_readonly"] and old["validity_readonly"]
    assert new["values_readonly"] and new["validity_readonly"]
    assert BENCH.compare_records([old, new]) == {"equivalent": True, "records_compared": 2}


def test_fixture_and_equivalence_guards_reject_bad_inputs():
    with pytest.raises(ValueError, match="positive integers"):
        BENCH.validate_case(0, 3, 4)
    with pytest.raises(ValueError, match="positive integers"):
        BENCH.validate_case(True, 3, 4)
    with pytest.raises(ValueError, match="limit"):
        BENCH.validate_case(BENCH.MAX_INPUT_CELLS + 1, 1, 1)

    required = BENCH.validate_case(2, 3, 4)["required_available_bytes"]
    with pytest.raises(MemoryError, match="below required"):
        BENCH.validate_case(2, 3, 4, available_bytes=required - 1)

    with pytest.raises(ValueError, match="both"):
        BENCH.compare_records([_fresh_child("old_list_stack")])
    incomplete = _fresh_child("factor_major")
    del incomplete["values_sha256"]
    with pytest.raises(ValueError, match="missing required"):
        BENCH.compare_records([_fresh_child("old_list_stack"), incomplete])
    mismatched = _fresh_child("factor_major")
    mismatched["values_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="equivalence failed"):
        BENCH.compare_records([_fresh_child("old_list_stack"), mismatched])


def test_report_is_strict_json_bounded_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(BENCH, "MAX_REPORT_BYTES", 32)
    oversized = tmp_path / "oversized.json"
    with pytest.raises(ValueError, match="strictly smaller"):
        BENCH.write_report_exclusive(oversized, {"payload": "x" * 32})
    with pytest.raises(ValueError):
        BENCH.write_report_exclusive(tmp_path / "nan.json", {"bad": float("nan")})
    assert not oversized.exists()

    report = tmp_path / "report.json"
    assert BENCH.write_report_exclusive(report, {"ok": True}) == report
    original = report.read_bytes()
    assert original == b'{"ok":true}\n'
    with pytest.raises(FileExistsError):
        BENCH.write_report_exclusive(report, {"ok": False})
    assert report.read_bytes() == original


def test_benchmark_rejects_existing_report_before_work_and_bounds_pairs(tmp_path, monkeypatch):
    report = tmp_path / "existing.json"
    report.write_text("keep me")
    monkeypatch.setattr(BENCH, "source_hashes", lambda: pytest.fail("work began"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        BENCH.run_benchmark(report=report, factors=1, times=1, assets=1, pairs=1)

    for pairs in (0, 11, True):
        with pytest.raises(ValueError, match="1..10"):
            BENCH.run_benchmark(report=tmp_path / f"{pairs}.json", pairs=pairs)

def test_tiny_benchmark_writes_paired_report_and_checks_each_child_admission(tmp_path, monkeypatch):
    report = tmp_path / "tiny-report.json"
    available_calls = []

    def available():
        available_calls.append(1)
        return 1_000_000_000_000

    monkeypatch.setattr(BENCH, "available_memory_bytes", available)
    result = BENCH.run_benchmark(
        report=report, factors=2, times=3, assets=4, seed=20261004, pairs=1,
    )

    saved = json.loads(report.read_text())
    assert result["equivalence"] == {"equivalent": True, "records_compared": 2}
    assert saved["scope"] == "output_assembly_only"
    assert saved["full_optimizer_speed_claim"] is False
    assert len(saved["paired_runs"]) == 1
    assert saved["paired_runs"][0]["order"] == ["old_list_stack", "factor_major"]
    assert saved["runtime_before"] == saved["runtime_after"]
    assert len(available_calls) == 3  # parent preflight plus each fresh child admission


def test_tiny_benchmark_refuses_memory_shortfall_before_child_launch(tmp_path, monkeypatch):
    required = BENCH.validate_case(1, 1, 1)["required_available_bytes"]
    calls = iter((required + 1024, required - 1))
    monkeypatch.setattr(BENCH, "available_memory_bytes", lambda: next(calls))
    monkeypatch.setattr(
        BENCH.subprocess, "run", lambda *args, **kwargs: pytest.fail("child launched"),
    )

    with pytest.raises(MemoryError, match="below required"):
        BENCH.run_benchmark(
            report=tmp_path / "not-written.json", factors=1, times=1, assets=1, pairs=1,
        )


def test_tiny_benchmark_rejects_runtime_drift_and_preserves_no_report(tmp_path, monkeypatch):
    report = tmp_path / "runtime-drift.json"
    contexts = iter(({"runtime": "before"}, {"runtime": "after"}))
    monkeypatch.setattr(BENCH, "runtime_context", lambda: next(contexts))
    monkeypatch.setattr(BENCH, "source_hashes", lambda: {"source": "unchanged"})
    monkeypatch.setattr(BENCH, "_invoke_child", lambda mode, *args: _fresh_child(mode))

    with pytest.raises(RuntimeError, match="runtime changed"):
        BENCH.run_benchmark(
            report=report, factors=2, times=3, assets=4, pairs=1,
        )
    assert not report.exists()


def test_benchmark_rejects_existing_symlink_before_work(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_text("keep me")
    report_link = tmp_path / "existing-link.json"
    report_link.symlink_to(target)
    monkeypatch.setattr(BENCH, "source_hashes", lambda: pytest.fail("work began"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        BENCH.run_benchmark(
            report=report_link, factors=1, times=1, assets=1, pairs=1,
        )
    assert report_link.is_symlink()
    assert target.read_text() == "keep me"
