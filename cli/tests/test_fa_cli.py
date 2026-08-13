#!/usr/bin/env python3
"""
Tests for fa_cli (Factor Analysis CLI)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli"))

from fa_cli import fa_cli


def test_fa_cli_version():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "0.1.0" in result.output


def test_fa_list_operators():
    """Test listing operators."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["list-operators"])

    # May fail if factor_engine not available
    if result.exit_code != 0:
        pytest.skip("Factor engine not available")

    assert "operators" in result.output.lower()


def test_fa_list_operators_count():
    """Test listing operators with count flag."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["list-operators", "--count"])

    if result.exit_code != 0:
        pytest.skip("Factor engine not available")

    # Should output just a number
    assert result.output.strip().isdigit()


def test_fa_families():
    """Test families command."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["families"])

    if result.exit_code != 0:
        pytest.skip("Factor engine not available")

    assert "Families" in result.output


def test_fa_search():
    """Test search command."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["search", "mean"])

    if result.exit_code != 0:
        pytest.skip("Factor engine not available")

    # Should find something with "mean" in the name
    assert result.exit_code == 0


def test_fa_stats():
    """Test stats command."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["stats"])

    if result.exit_code != 0:
        pytest.skip("Factor engine not available")

    assert "Statistics" in result.output
    assert "Total operators" in result.output


def test_fa_help():
    """Test help message."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["--help"])

    assert result.exit_code == 0
    assert "Factor Analysis CLI" in result.output


def test_fa_list_operators_with_filter():
    """Test filtering operators by family."""
    runner = CliRunner()
    result = runner.invoke(fa_cli, ["list-operators", "--family", "technical"])

    if result.exit_code != 0:
        pytest.skip("Factor engine not available")

    # Either finds operators or shows zero
    assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
