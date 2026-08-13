#!/usr/bin/env python3
"""
Tests for fo_cli (Factor Optimization CLI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli"))

from fo_cli import fo_cli


def test_fo_cli_version():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(fo_cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "0.1.0" in result.output


def test_fo_search_basic(tmp_path):
    """Test basic search command."""
    runner = CliRunner()
    output_file = tmp_path / "results.json"

    result = runner.invoke(fo_cli, [
        "search",
        "--n-trials", "10",
        "--budget", "50",
        "--output", str(output_file),
        "--seed", "42"
    ])

    assert result.exit_code == 0
    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert "config" in data
    assert "summary" in data
    assert "results" in data
    assert data["config"]["n_trials"] == 10


def test_fo_search_no_output():
    """Test search without output file."""
    runner = CliRunner()

    result = runner.invoke(fo_cli, [
        "search",
        "--n-trials", "5",
        "--budget", "10",
        "--seed", "42"
    ])

    assert result.exit_code == 0
    assert "Search completed" in result.output


def test_fo_analyze(tmp_path):
    """Test analyze command."""
    runner = CliRunner()

    # First run a search
    output_file = tmp_path / "results.json"
    result = runner.invoke(fo_cli, [
        "search",
        "--n-trials", "20",
        "--output", str(output_file)
    ])
    assert result.exit_code == 0

    # Then analyze the results
    result = runner.invoke(fo_cli, [
        "analyze",
        str(output_file),
        "--top-n", "5",
        "--metric", "ic"
    ])

    assert result.exit_code == 0
    assert "Statistics" in result.output


def test_fo_benchmark():
    """Test benchmark command."""
    runner = CliRunner()

    result = runner.invoke(fo_cli, ["benchmark", "--size", "100"])

    assert result.exit_code == 0
    assert "Benchmarking" in result.output
    assert "Cache size" in result.output


def test_fo_grammar():
    """Test grammar command."""
    runner = CliRunner()

    result = runner.invoke(fo_cli, ["grammar"])

    assert result.exit_code == 0
    assert "Mutation Grammar" in result.output
    assert "param_tweak" in result.output


def test_fo_help():
    """Test help message."""
    runner = CliRunner()
    result = runner.invoke(fo_cli, ["--help"])

    assert result.exit_code == 0
    assert "Factor Optimization CLI" in result.output


def test_fo_search_budget_exhaustion(tmp_path):
    """Test that search respects budget limit."""
    runner = CliRunner()
    output_file = tmp_path / "results.json"

    result = runner.invoke(fo_cli, [
        "search",
        "--n-trials", "100",
        "--budget", "10",  # Budget smaller than trials
        "--output", str(output_file)
    ])

    assert result.exit_code == 0

    data = json.loads(output_file.read_text())
    # Should stop at budget, not complete all trials
    assert data["summary"]["evaluated"] <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
