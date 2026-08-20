# -*- coding: utf-8
"""Fake batch-mirror registrations must not define production coverage."""
from __future__ import annotations

import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

pytest.importorskip("polars")


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


@pytest.mark.parametrize("canonical", ["nan_to_num", "protected_log", "rank_transform", "product", "quarter", "ttm", "yoy", "row_kurt", "row_skew", "row_prod"])
def test_removed_batch_mirror_names_are_not_daily(canonical: str):
    from cleaned_operators.operator_surface import DAILY_CANONICALS

    assert canonical not in DAILY_CANONICALS


def test_daily_polars_coverage_is_evidence_backed():
    from backend.fastpath_evidence import polars_executed_parity_canonicals
    from cleaned_operators.operator_surface import DAILY_CANONICALS

    assert DAILY_CANONICALS <= polars_executed_parity_canonicals()
