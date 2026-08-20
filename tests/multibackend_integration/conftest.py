# -*- coding: utf-8 -*-
"""Shared fixtures for multibackend integration tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_panel_data():
    """Sample panel data for testing (100 dates × 50 instruments)."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    instruments = [f"INST_{i:03d}" for i in range(50)]

    data = []
    for date in dates:
        for inst in instruments:
            data.append({
                "date": date,
                "instrument": inst,
                "close": 100.0 + np.random.randn() * 10.0,
                "volume": int(1_000_000 + np.random.randn() * 100_000),
                "returns": np.random.randn() * 0.02,
            })

    return pd.DataFrame(data)


@pytest.fixture
def large_panel_data():
    """Large panel data for stress testing (252 dates × 500 instruments)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    instruments = [f"INST_{i:04d}" for i in range(500)]

    data = []
    for date in dates:
        for inst in instruments:
            data.append({
                "date": date,
                "instrument": inst,
                "close": 100.0 + np.random.randn() * 10.0,
                "volume": int(1_000_000 + np.random.randn() * 100_000),
                "returns": np.random.randn() * 0.02,
            })

    return pd.DataFrame(data)


@pytest.fixture
def mock_source_snapshot(sample_panel_data, temp_data_dir):
    """Mock data source snapshot for testing."""
    from dataclasses import dataclass

    @dataclass
    class MockSourceSnapshot:
        snapshot_id: str
        data: pd.DataFrame
        data_dir: Path

        def get_data(self):
            return self.data

    return MockSourceSnapshot(
        snapshot_id="test_snapshot_001",
        data=sample_panel_data,
        data_dir=temp_data_dir,
    )
