#!/usr/bin/env python3
"""
Tests for qe_cli (Quant Evaluator CLI)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cli"))

from qe_cli import qe_cli


@pytest.fixture
def sample_factor_data(tmp_path):
    """Generate sample factor data."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    assets = [f"A{i:03d}" for i in range(20)]

    rng = np.random.default_rng(42)
    data = rng.standard_normal((50, 20))
    df = pd.DataFrame(data, index=dates, columns=assets)

    factor_file = tmp_path / "factor.csv"
    df.to_csv(factor_file)

    return factor_file


@pytest.fixture
def sample_label_data(tmp_path):
    """Generate sample label data."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    assets = [f"A{i:03d}" for i in range(20)]

    rng = np.random.default_rng(123)
    data = rng.standard_normal((50, 20)) * 0.02
    df = pd.DataFrame(data, index=dates, columns=assets)

    label_file = tmp_path / "label.csv"
    df.to_csv(label_file)

    return label_file


def test_qe_cli_version():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(qe_cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "0.1.0" in result.output


def test_qe_evaluate_basic(sample_factor_data, sample_label_data, tmp_path):
    """Test basic evaluation."""
    runner = CliRunner()
    output_file = tmp_path / "results.json"

    result = runner.invoke(qe_cli, [
        "evaluate",
        str(sample_factor_data),
        str(sample_label_data),
        "-m", "rank_ic",
        "-o", str(output_file),
        "-f", "json"
    ])

    assert result.exit_code == 0
    assert output_file.exists()

    import json
    results = json.loads(output_file.read_text())
    assert "rank_ic" in results


def test_qe_evaluate_multiple_metrics(sample_factor_data, sample_label_data, tmp_path):
    """Test evaluation with multiple metrics."""
    runner = CliRunner()
    output_file = tmp_path / "results.csv"

    result = runner.invoke(qe_cli, [
        "evaluate",
        str(sample_factor_data),
        str(sample_label_data),
        "-m", "rank_ic",
        "-m", "ic",
        "-m", "ir",
        "-o", str(output_file),
        "-f", "csv"
    ])

    assert result.exit_code == 0
    assert output_file.exists()


def test_qe_info(sample_factor_data):
    """Test info command."""
    runner = CliRunner()
    result = runner.invoke(qe_cli, ["info", str(sample_factor_data)])

    assert result.exit_code == 0
    assert "Shape" in result.output
    assert "Date range" in result.output
    assert "Coverage" in result.output


def test_qe_evaluate_text_format(sample_factor_data, sample_label_data):
    """Test text format output."""
    runner = CliRunner()

    result = runner.invoke(qe_cli, [
        "evaluate",
        str(sample_factor_data),
        str(sample_label_data),
        "-m", "rank_ic",
        "-f", "text"
    ])

    assert result.exit_code == 0
    assert "Evaluation Results" in result.output


def test_qe_help():
    """Test help message."""
    runner = CliRunner()
    result = runner.invoke(qe_cli, ["--help"])

    assert result.exit_code == 0
    assert "Quant Evaluator CLI" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
