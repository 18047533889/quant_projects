# -*- coding: utf-8 -*-
"""R24-075..078: ``snapshot_now_only`` is caller INTENT, not proof.  Production
requires the source's snapshot validity metadata, and the historical/current
judgment is relative to the decision context, not machine datetime.now()."""
from __future__ import annotations

import pytest

from factor_engine.storage.sources.data_access_source import (
    DataAccessSource,
    HistoricalSnapshotBackfillError,
)


def _make_source(**overrides):
    kwargs = dict(
        dataset="ashare_stock_valuation_daily",
        read_mode="panel",
        snapshot_only=True,
        enforce_mining_gate=True,
        production=True,
        start_date="2024-01-01",
    )
    kwargs.update(overrides)
    return DataAccessSource(**kwargs)


def test_production_snapshot_now_only_requires_source_validity() -> None:
    # R24-075..077: a bare caller bool cannot self-prove snapshot_now_only.
    src = _make_source(snapshot_now_only=True)
    with pytest.raises(HistoricalSnapshotBackfillError, match="intent, not proof"):
        src._enforce_field_contract_gates({})


def test_snapshot_now_only_with_source_validity_is_honored() -> None:
    # R24-076: the runtime verifies against the source's snapshot validity.
    src = _make_source(
        snapshot_now_only=True, snapshot_valid_at="2025-01-01",
    )
    # No raise — validity is declared.
    src._enforce_field_contract_gates({})


def test_window_historical_relative_to_asof_not_machine_now() -> None:
    # R24-078: "historical" is judged against the asof decision context.
    src = _make_source(snapshot_now_only=False, start_date="2024-01-01")
    # A decision context BEFORE start is NOT a historical backfill.
    assert src._window_is_historical(asof="2023-12-01") is False
    # A decision context AFTER start IS historical.
    assert src._window_is_historical(asof="2025-06-01") is True


def test_production_mining_window_judgment_without_context_fails_closed() -> None:
    src = _make_source(snapshot_now_only=False, production=True, enforce_mining_gate=True)
    with pytest.raises(HistoricalSnapshotBackfillError, match="decision context"):
        src._window_is_historical()  # no asof, no snapshot_valid_at, production mining
