"""End-to-end checks for the bounded bootstrap benchmark CLI."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "factor_optimizer/benchmarks/benchmark_bootstrap_scratch_oct04.py"
SOURCE = PROJECT_ROOT / "factor_optimizer/factor_optimizer/research_bootstrap.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=20,
    )


def test_tiny_abba_run_emits_reproducible_json_and_exact_parity():
    result = _run("--sizes", "60", "--draws", "7", "--rounds", "1")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "moving-block-bootstrap-abba-v1"
    assert payload["benchmark"] == "moving_block_lower_bound"
    assert re.fullmatch(r"[0-9a-f]{64}", payload["source_sha256"])
    expected_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert payload["source_sha256"] == expected_hash
    assert payload["order"] == ["old", "scratch", "scratch", "old"]
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert (case["n"], case["bootstrap_draws"], case["rounds"]) == (60, 7, 1)
    assert case["parity"] is True
    assert len(case["raw_times_seconds"]["old"]) == 2
    assert len(case["raw_times_seconds"]["scratch"]) == 2
    assert case["median_seconds"]["old"] > 0
    assert case["median_seconds"]["scratch"] > 0


def test_cli_rejects_oversized_work_before_running_benchmark():
    result = _run("--sizes", "100000000000000000000", "--draws", "7", "--rounds", "1")
    assert result.returncode != 0
    assert "max 10000" in result.stderr
    assert result.stdout == ""


def test_cli_rejects_draw_and_round_budgets_above_hard_caps():
    draw_result = _run("--sizes", "60", "--draws", "5001", "--rounds", "1")
    round_result = _run("--sizes", "60", "--draws", "7", "--rounds", "6")
    assert draw_result.returncode != 0
    assert "max 5000" in draw_result.stderr
    assert round_result.returncode != 0
    assert "max 5" in round_result.stderr


def test_cli_rejects_oversized_cartesian_case_count():
    result = _run("--sizes", "60,61,62,63,64", "--draws", "1,2,3,4", "--rounds", "1")
    assert result.returncode != 0
    assert "max 16 cases" in result.stderr
    assert result.stdout == ""


def test_cli_rejects_aggregate_workload_before_benchmarking():
    result = _run("--sizes", "10000", "--draws", "5000", "--rounds", "5")
    assert result.returncode != 0
    assert "400000000" in result.stderr
    assert result.stdout == ""


def test_timed_scratch_mismatch_fails_closed(monkeypatch):
    spec = importlib.util.spec_from_file_location("bootstrap_benchmark_under_test", SCRIPT)
    benchmark = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(benchmark)
    values = np.random.default_rng(821).normal(size=60)
    values[::37] = np.nan
    original = benchmark._scratch_lower_bound
    calls = 0

    def mismatch_after_preflight(differences, *, bootstrap_draws):
        nonlocal calls
        calls += 1
        result = original(differences, bootstrap_draws=bootstrap_draws)
        return result if calls == 1 else result + 1.0

    monkeypatch.setattr(benchmark, "_scratch_lower_bound", mismatch_after_preflight)
    with pytest.raises(RuntimeError, match="during timed run"):
        benchmark._benchmark_case(values, bootstrap_draws=7, rounds=1)


def test_cli_rejects_n15_when_helpers_have_no_finite_lower_bound():
    result = _run("--sizes", "15", "--draws", "499", "--rounds", "1")
    assert result.returncode != 0
    assert result.stdout == ""
    assert "finite result" in result.stderr.lower()


def test_benchmark_case_rejects_none_reference_before_timing():
    spec = importlib.util.spec_from_file_location("bootstrap_none_reference", SCRIPT)
    benchmark = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(benchmark)
    values = np.random.default_rng(15).normal(size=15)
    with pytest.raises(RuntimeError, match="finite result"):
        benchmark._benchmark_case(values, bootstrap_draws=499, rounds=1)


def test_benchmark_case_rejects_nonfinite_result_during_timed_run(monkeypatch):
    spec = importlib.util.spec_from_file_location("bootstrap_timed_none", SCRIPT)
    benchmark = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(benchmark)
    values = np.random.default_rng(821).normal(size=60)
    values[::37] = np.nan
    original = benchmark._scratch_lower_bound
    calls = 0

    def return_none_after_preflight(differences, *, bootstrap_draws):
        nonlocal calls
        calls += 1
        return (original(differences, bootstrap_draws=bootstrap_draws)
                if calls == 1 else None)

    monkeypatch.setattr(benchmark, "_scratch_lower_bound", return_none_after_preflight)
    with pytest.raises(RuntimeError, match="finite result"):
        benchmark._benchmark_case(values, bootstrap_draws=7, rounds=1)


def test_aggregate_budget_counts_two_full_preflight_calls():
    spec = importlib.util.spec_from_file_location("bootstrap_budget_gate", SCRIPT)
    benchmark = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(benchmark)
    # 10,000 * 5,000 * (4 * 2 timed + 2 preflight) = 500M work units.
    with pytest.raises(ValueError, match="maximum allowed operations"):
        benchmark._validate_budget([10_000], [5_000], 2)
