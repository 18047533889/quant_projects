"""TRAIN coverage workspace admission precedes panel and price materialization."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_coverage_workspace_budget_rejects_before_training_reindex_and_price_read(
    monkeypatch,
):
    """Oversized coverage workspace must be refused before either expensive stage."""
    fixture_path = Path(__file__).with_name(
        "test_cos_loader_session_integration_oct04.py")
    spec = importlib.util.spec_from_file_location(
        "cos_coverage_admission_fixtures", fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)

    calendar = pd.bdate_range("2020-01-02", periods=502)
    price_store = fixtures._PriceStore(calendar)
    example, _, _ = fixtures._install_fakes(
        monkeypatch, n_days=500, price_store=price_store)

    # Isolate the coverage admission stage: decoded Arrow evidence and source
    # frame measurements are tiny, while the TRAIN coverage workspace is > 4 KiB.
    monkeypatch.setattr(example, "arrow_table_buffer_bytes", lambda _table: 1)
    monkeypatch.setattr(
        pd.DataFrame, "memory_usage",
        lambda self, *args, **kwargs: pd.Series([0], dtype="int64"),
    )

    reindex_calls = []

    def track_train_reindex(self, *args, **kwargs):
        reindex_calls.append((args, kwargs))
        pytest.fail("TRAIN panel reindex started before workspace admission")

    monkeypatch.setattr(pd.DataFrame, "reindex", track_train_reindex)
    price_reads = []

    def forbidden_price_read(*args, **kwargs):
        price_reads.append((args, kwargs))
        pytest.fail("price read started before TRAIN coverage workspace admission")

    monkeypatch.setattr(example, "read_session_vwap", forbidden_price_read)

    with pytest.raises(MemoryError, match="materialization budget exceeded"):
        example.load_cos_sample(
            n_factors=1, n_assets=20, n_days=500,
            materialization_budget_bytes=4 * 1024,
        )

    assert reindex_calls == []
    assert price_reads == []
