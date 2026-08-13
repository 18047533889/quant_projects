#!/usr/bin/env python3
"""
Tests for fp_cli (Factor Preprocessing CLI)
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

from fp_cli import fp_cli


@pytest.fixture
def sample_data(tmp_path):
    """Generate sample factor data."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    assets = [f"A{i:03d}" for i in range(15)]

    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 15))
    # Add some NaNs and outliers
    data[5, 3] = np.nan
    data[10, 8] = 10.0  # outlier
    df = pd.DataFrame(data, index=dates, columns=assets)

    input_file = tmp_path / "input.csv"
    df.to_csv(input_file)

    return input_file


def test_fp_cli_version():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(fp_cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "0.1.0" in result.output


def test_fp_transform_zscore(sample_data, tmp_path):
    """Test z-score transformation."""
    runner = CliRunner()
    output_file = tmp_path / "output.csv"

    result = runner.invoke(fp_cli, [
        "transform",
        str(sample_data),
        str(output_file),
        "-t", "zscore",
        "--axis", "cs"
    ])

    assert result.exit_code == 0
    assert output_file.exists()

    df = pd.read_csv(output_file, index_col=0)
    assert df.shape == (30, 15)


def test_fp_transform_multiple(sample_data, tmp_path):
    """Test multiple transformations."""
    runner = CliRunner()
    output_file = tmp_path / "output.csv"

    result = runner.invoke(fp_cli, [
        "transform",
        str(sample_data),
        str(output_file),
        "-t", "fillna",
        "-t", "winsorize",
        "-t", "zscore",
        "--fillna-method", "mean",
        "--winsorize-std", "3.0"
    ])

    assert result.exit_code == 0
    assert output_file.exists()


def test_fp_transform_rank(sample_data, tmp_path):
    """Test rank transformation."""
    runner = CliRunner()
    output_file = tmp_path / "output.csv"

    result = runner.invoke(fp_cli, [
        "transform",
        str(sample_data),
        str(output_file),
        "-t", "rank",
        "--axis", "cs"
    ])

    assert result.exit_code == 0
    assert output_file.exists()

    df = pd.read_csv(output_file, index_col=0)
    # Rank should be between 0 and 1 (percentile rank)
    valid_vals = df.values[~np.isnan(df.values)]
    assert valid_vals.min() >= 0
    assert valid_vals.max() <= 1


def test_fp_outliers(sample_data):
    """Test outlier detection."""
    runner = CliRunner()

    result = runner.invoke(fp_cli, [
        "outliers",
        str(sample_data),
        "--top-n", "5"
    ])

    assert result.exit_code == 0
    assert "outliers" in result.output.lower()


def test_fp_stats(sample_data):
    """Test stats command."""
    runner = CliRunner()

    result = runner.invoke(fp_cli, ["stats", str(sample_data)])

    assert result.exit_code == 0
    assert "Statistics" in result.output
    assert "Shape" in result.output
    assert "Mean" in result.output


def test_fp_transform_neutralize(sample_data, tmp_path):
    """Test neutralization."""
    runner = CliRunner()
    output_file = tmp_path / "output.csv"

    result = runner.invoke(fp_cli, [
        "transform",
        str(sample_data),
        str(output_file),
        "-t", "neutralize"
    ])

    assert result.exit_code == 0
    assert output_file.exists()

    df = pd.read_csv(output_file, index_col=0)
    # After neutralization, cross-sectional mean should be near zero
    cross_mean = df.mean(axis=1).mean()
    assert abs(cross_mean) < 1e-10


def test_fp_help():
    """Test help message."""
    runner = CliRunner()
    result = runner.invoke(fp_cli, ["--help"])

    assert result.exit_code == 0
    assert "Factor Preprocessing CLI" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
