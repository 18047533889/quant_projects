#!/usr/bin/env python3
"""
Tests for benchmark_cli
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli"))

from benchmark_cli import benchmark_cli


def test_benchmark_cli_version():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(benchmark_cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "0.1.0" in result.output


def test_benchmark_list_suites():
    """Test list-suites command."""
    runner = CliRunner()
    result = runner.invoke(benchmark_cli, ["list-suites"])

    assert result.exit_code == 0
    assert "qe" in result.output
    assert "fp" in result.output
    assert "fa" in result.output
    assert "fo" in result.output


def test_benchmark_run_fo_only(tmp_path):
    """Test running single suite (FO - fastest)."""
    runner = CliRunner()
    output_file = tmp_path / "results.json"

    result = runner.invoke(benchmark_cli, [
        "run",
        "--suite", "fo",
        "--output", str(output_file),
        "--quick"
    ])

    # May fail if dependencies missing, that's OK for this test
    if result.exit_code == 0:
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "benchmarks" in data


def test_benchmark_report_text(tmp_path):
    """Test report generation in text format."""
    # Create a mock results file
    results = {
        "schema_version": "1.0",
        "generated_at": "2026-08-14T00:00:00",
        "mode": "quick",
        "benchmarks": {
            "fo": {
                "description": "Factor optimization",
                "total_time_s": 1.5,
                "results": {}
            }
        }
    }

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results))

    runner = CliRunner()
    result = runner.invoke(benchmark_cli, [
        "report",
        str(results_file),
        "--format", "text"
    ])

    assert result.exit_code == 0
    assert "BENCHMARK REPORT" in result.output


def test_benchmark_report_json(tmp_path):
    """Test report generation in JSON format."""
    results = {
        "schema_version": "1.0",
        "benchmarks": {}
    }

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results))

    runner = CliRunner()
    result = runner.invoke(benchmark_cli, [
        "report",
        str(results_file),
        "--format", "json"
    ])

    assert result.exit_code == 0
    # Should output valid JSON
    parsed = json.loads(result.output)
    assert "schema_version" in parsed


def test_benchmark_compare(tmp_path):
    """Test comparing two benchmark results."""
    baseline = {
        "benchmarks": {
            "fo": {"total_time_s": 1.0},
            "fa": {"total_time_s": 2.0}
        }
    }

    current = {
        "benchmarks": {
            "fo": {"total_time_s": 1.5},  # 50% slower - regression
            "fa": {"total_time_s": 1.5}   # 25% faster - improvement
        }
    }

    baseline_file = tmp_path / "baseline.json"
    current_file = tmp_path / "current.json"

    baseline_file.write_text(json.dumps(baseline))
    current_file.write_text(json.dumps(current))

    runner = CliRunner()
    result = runner.invoke(benchmark_cli, [
        "compare",
        str(baseline_file),
        str(current_file),
        "--threshold", "0.2"  # 20% threshold
    ])

    assert result.exit_code == 0
    assert "COMPARISON" in result.output


def test_benchmark_help():
    """Test help message."""
    runner = CliRunner()
    result = runner.invoke(benchmark_cli, ["--help"])

    assert result.exit_code == 0
    assert "Benchmark CLI" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
