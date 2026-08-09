# -*- coding: utf-8 -*-
"""Round-11 §35 — US valuation helper must actually construct, and the
SnapshotOnlySourcePolicy (plan A) must govern it correctly."""
from __future__ import annotations

import pytest


def test_default_us_valuation_helper_constructs():
    """The default helper's config must be buildable (was previously rejected on
    two counts: 'current_only' join not in _ALLOWED_JOIN_METHODS and
    'snapshot_only'/'usage'/'notes' being unsupported options)."""
    from api.mining_integration import default_us_pv_valuation_data_source_config
    from storage.factory import build_data_source

    cfg = default_us_pv_valuation_data_source_config()
    assert cfg["joins"]["valuation"] == "current_only"
    source = build_data_source(cfg)
    assert source.__class__.__name__ == "CompositeDataSource"
    valuation = source.sources["valuation"]
    assert valuation.snapshot_only is True


def test_snapshot_source_asof_join_rejected_at_build():
    """A snapshot_only source joined asof/ffill would invent history — reject at
    build time."""
    from api.mining_integration import default_us_pv_valuation_data_source_config
    from storage.factory import build_data_source

    cfg = default_us_pv_valuation_data_source_config()
    for bad_method in ("asof_backward", "forward_fill"):
        bad = dict(cfg)
        bad["joins"] = {"valuation": bad_method}
        with pytest.raises(ValueError, match="snapshot_only"):
            build_data_source(bad)


def test_snapshot_only_production_historical_mining_fails_closed():
    """A snapshot_only source must hard-fail production historical mining over a
    historical window; current-snapshot research with snapshot_now_only passes."""
    from storage.sources.data_access_source import (
        DataAccessSource,
        HistoricalSnapshotBackfillError,
    )

    def _build(*, snapshot_now_only: bool):
        return DataAccessSource(
            dataset="us_stock_valuation_daily",
            fields={"pe": "price_to_earnings"},
            start_date="2020-01-01",
            enforce_mining_gate=True,
            snapshot_only=True,
            snapshot_now_only=snapshot_now_only,
        )

    # Production mining gate ON + historical window + snapshot_only + no
    # snapshot_now_only => hard fail.
    with pytest.raises(HistoricalSnapshotBackfillError):
        _build(snapshot_now_only=False)._enforce_field_contract_gates(
            {name: _plan() for name in ("pe",)}
        )
    # snapshot_now_only research path is allowed (no raise).
    _build(snapshot_now_only=True)._enforce_field_contract_gates(
        {name: _plan() for name in ("pe",)}
    )


def _plan():
    from storage.sources.field_plan import NormalizedFieldPlan

    return NormalizedFieldPlan(
        logical_concept="pe",
        physical_dataset="us_stock_valuation_daily",
        mining_allowed=True,
        current_snapshot_only=False,
        coverage="full_history",
    )
