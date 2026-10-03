"""Incremental source admission contracts for COS decoded factor panels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from factor_optimizer.cohort_materialization import (
    admit_decoded_source,
    arrow_table_buffer_bytes,
    estimate_pandas_conversion_bytes,
)


def test_decoded_source_admission_counts_retained_arrow_and_conversion_allowance():
    allowance = estimate_pandas_conversion_bytes(120)
    assert allowance == 240
    assert admit_decoded_source(100, 120 + allowance, 500) == 460
    with pytest.raises(MemoryError, match="source conversion"):
        admit_decoded_source(100, 120 + allowance, 459)


@pytest.mark.parametrize("table", [object(), SimpleNamespace(get_total_buffer_size=lambda: True),
                                   SimpleNamespace(get_total_buffer_size=lambda: -1)])
def test_arrow_admission_requires_valid_buffer_evidence(table):
    with pytest.raises((TypeError, ValueError)):
        arrow_table_buffer_bytes(table)


def test_loader_rejects_over_budget_before_arrow_to_pandas(monkeypatch):
    helper_path = Path(__file__).with_name("test_cos_loader_session_integration_oct04.py")
    spec = importlib.util.spec_from_file_location("cos_loader_integration_fixtures", helper_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    calendar = __import__("pandas").bdate_range("2020-01-02", periods=502)
    price_store = fixtures._PriceStore(calendar)
    example, _, _ = fixtures._install_fakes(monkeypatch, n_days=500, price_store=price_store)

    class ArrowSpy:
        def get_total_buffer_size(self):
            return 1024

        def to_pandas(self):
            pytest.fail("Arrow conversion began before decoded-source admission")

    factor = SimpleNamespace(table=ArrowSpy(), source_uri="fixture", source_etag="e",
        content_sha256="a" * 64, downloaded_bytes=128)
    bound = SimpleNamespace(factor=factor, treatment_signature=(), manifest_sha256="b" * 64,
        source_status="evaluated_optimization_pending", expression="fixture")
    monkeypatch.setattr(example, "read_bound_factor", lambda *args, **kwargs: bound)
    with pytest.raises(MemoryError, match="materialization budget"):
        example.load_cos_sample(n_factors=1, n_assets=20, n_days=500,
                                materialization_budget_bytes=1)
