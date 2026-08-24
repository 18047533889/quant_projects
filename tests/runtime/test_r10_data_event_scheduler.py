# -*- coding: utf-8 -*-
"""R10 #50–#54: rich dependency edges, per-factor engines, full definitions,
extended DataEvent, and production fail-closed lineage universe masks."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from factor_engine.runtime.dependency_catalog import DependencyCatalog, FactorDependencyEdge
from factor_engine.runtime.incremental_scheduler import (
    DataEvent,
    execute_incremental_updates_from_event,
    normalize_data_event,
    plan_updates_from_data_event,
)
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.materializer import ParquetMaterializer


# ---------------------------------------------------------------------------
# R10 #51 — each affected factor runs on its OWN engine reconstructed from the
# catalog's full spec; two factors with different data-source configs must not
# share an engine.
# ---------------------------------------------------------------------------

def test_different_data_source_configs_use_distinct_engines(tmp_path):
    lake = tmp_path / "lake"
    catalog = ParquetMaterializer(lake_root=lake).catalog
    dep = DependencyCatalog(catalog)

    specs = {
        "f_a": {"type": "parquet", "dataset": "ds_a", "path": "/data/a"},
        "f_b": {"type": "parquet", "dataset": "ds_b", "path": "/data/b"},
    }
    for fid, ds_cfg in specs.items():
        catalog.register(
            fid,
            author="tester",
            frequency="1d",
            ast_hash="h_" + fid,
            expression='col("close")',
            data_source_config=ds_cfg,
        )
        dep.record_factor_edges(
            fid,
            edges=[FactorDependencyEdge(fid, "shared_ds", "close")],
            lookback=2,
            frequency="1d",
            source_dataset="shared_ds",
        )
        dep.record_full_factor_definition(
            fid,
            expression='col("close")',
            frequency="1d",
            data_source_config=ds_cfg,
        )

    built: dict[str, dict] = {}

    def fake_engine_factory(full_def, *, lake_root=None, market=None, run_mode=None):
        fid = full_def["factor_id"]
        eng = MagicMock()
        eng.materialize_incremental.return_value = {
            "materialization": {"factor_id": fid, "rows_written": 1}
        }
        built[fid] = dict(full_def.get("data_source_config") or {})
        return eng

    event = DataEvent(
        dataset="shared_ds",
        column="close",
        updated_date="2026-07-09",
        field_id="close",
        revision_kind="update",
    )
    out = execute_incremental_updates_from_event(
        None,
        event,
        lake_root=lake,
        engine_factory=fake_engine_factory,
    )

    assert out["succeeded"] == 2
    assert out["failed"] == 0
    assert set(built) == {"f_a", "f_b"}
    # Each factor was rebuilt from its OWN data-source config.
    assert built["f_a"]["dataset"] == "ds_a"
    assert built["f_b"]["dataset"] == "ds_b"
    assert built["f_a"] != built["f_b"]


def test_factor_without_source_config_falls_back_to_caller_engine(tmp_path):
    """Legacy catalog rows (no data-source config) keep using the caller engine."""
    lake = tmp_path / "lake"
    catalog = ParquetMaterializer(lake_root=lake).catalog
    catalog.record_factor_dependency(
        "legacy_f",
        referenced_columns=["close"],
        lookback=1,
        source_dataset="mock_ds",
        frequency="1d",
    )
    catalog.register(
        "legacy_f",
        author="tester",
        frequency="1d",
        ast_hash="h_legacy",
        expression='col("close")',
    )

    caller = MagicMock()
    caller.materialize_incremental.return_value = {
        "materialization": {"factor_id": "legacy_f", "rows_written": 1}
    }
    out = execute_incremental_updates_from_event(
        caller,
        DataEvent(dataset="mock_ds", column="close", updated_date="2026-07-09"),
        lake_root=lake,
    )
    assert out["succeeded"] == 1
    assert "legacy_f" in out["materializations"]
    caller.materialize_incremental.assert_called_once()


# ---------------------------------------------------------------------------
# R10 #50 — a composite (multi-source) factor records one edge per source field.
# ---------------------------------------------------------------------------

def test_composite_factor_has_multiple_dependency_edges(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    dep = DependencyCatalog(catalog)
    dep.record_factor_edges(
        "composite_f",
        edges=[
            FactorDependencyEdge("composite_f", "StockDailyBar", "close"),
            FactorDependencyEdge(
                "composite_f", "StockValuationDaily", "pe_ttm",
                physical_field="pe_ttm", snapshot_semantics="asof",
            ),
        ],
        lookback=5,
        frequency="1d",
    )

    edges = dep.get_factor_dependency_edges("composite_f")
    assert len(edges) == 2
    assert {e.source_dataset for e in edges} == {"StockDailyBar", "StockValuationDaily"}
    assert {e.field_id for e in edges} == {"close", "pe_ttm"}
    asof = [e for e in edges if e.field_id == "pe_ttm"][0]
    assert asof.snapshot_semantics == "asof"
    assert asof.physical_field == "pe_ttm"

    # The flat legacy row mirrors the union of referenced fields.
    flat = catalog.get_factor_dependency("composite_f")
    assert set(flat["referenced_columns"]) == {"close", "pe_ttm"}


def test_composite_secondary_field_revision_triggers_recompute(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    dep = DependencyCatalog(catalog)
    dep.record_factor_edges(
        "composite_f",
        edges=[
            FactorDependencyEdge("composite_f", "StockDailyBar", "close"),
            FactorDependencyEdge("composite_f", "StockValuationDaily", "pe_ttm"),
        ],
        lookback=5,
        frequency="1d",
    )

    event = DataEvent(
        dataset="StockValuationDaily",
        column="pe_ttm",
        updated_date="2026-07-09",
        field_id="pe_ttm",
        revision_kind="revision",
        affected_start="2026-07-01",
        affected_end="2026-07-09",
    )
    rows = dep.factors_for_event(event)
    assert len(rows) == 1
    assert rows[0]["factor_id"] == "composite_f"
    assert rows[0]["field_id"] == "pe_ttm"
    assert len(rows[0]["edges"]) >= 1

    plans = plan_updates_from_data_event(dep, event, lookback_extra=2)
    assert len(plans) == 1
    assert plans[0].factor_id == "composite_f"
    assert plans[0].field_id == "pe_ttm"


# ---------------------------------------------------------------------------
# R10 #53 — extended DataEvent fields with backward compatibility.
# ---------------------------------------------------------------------------

def test_data_event_carries_revision_fields():
    ev = DataEvent(
        dataset="d",
        column="close",
        updated_date="2026-07-09",
        affected_start="2026-07-01",
        affected_end="2026-07-09",
        revision_kind="update",
        deleted_keys=("AAA", "BBB"),
    )
    assert ev.affected_start == "2026-07-01"
    assert ev.affected_end == "2026-07-09"
    assert ev.revision_kind == "update"
    assert ev.deleted_keys == ("AAA", "BBB")
    d = ev.to_dict()
    assert d["deleted_keys"] == ["AAA", "BBB"]
    assert d["field_id"] == "close"
    assert d["affected_start"] == "2026-07-01"

    # Original positional constructor still works.
    legacy = DataEvent("d", "close", "2026-07-09")
    assert legacy.field_id is None
    assert legacy.revision_kind is None
    assert legacy.dataset == "d"

    # Dict normalization forwards the new fields.
    ev3 = normalize_data_event(
        {
            "dataset": "d",
            "column": "close",
            "updated_date": "2026-07-09",
            "field_id": "close",
            "revision_kind": "revision",
            "affected_start": "2026-07-01",
            "deleted_keys": ["X"],
        }
    )
    assert ev3.revision_kind == "revision"
    assert ev3.deleted_keys == ("X",)
    assert ev3.affected_start == "2026-07-01"


# ---------------------------------------------------------------------------
# R10 #54 — a universe-mask application failure REJECTS materialization in
# production while research keeps the historical drop_reason path.
# ---------------------------------------------------------------------------

def test_lineage_universe_mask_production_raises_research_keeps_drop_reason():
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.runtime.lineage_service import build_materialize_lineage

    factor = parse_factor('col("close")', name="f_um")
    analysis = Analyzer().lower(factor.expr)
    panel = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.date_range("2024-01-01", periods=2),
        columns=["A", "B"],
    )
    # A mask component with a shape that does not match the panel forces the
    # universe-mask application to raise.
    bad_mask = {
        "close": pd.DataFrame(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            index=pd.date_range("2024-01-01", periods=3),
            columns=["A", "B", "C"],
        )
    }
    kwargs = dict(
        factor=factor,
        analysis=analysis,
        output={"result": panel},
        factor_id="f_um",
        expression='col("close")',
        data_source_config=None,
        data_source=None,
        mode="full",
        market="a_share",
        universe_mask=bad_mask,
    )
    # Production: the failure must propagate, not be swallowed into drop_reason.
    with pytest.raises(Exception):
        build_materialize_lineage(production=True, **kwargs)
    # Research: keep the historical drop_reason path.
    lineage = build_materialize_lineage(production=False, **kwargs)
    assert lineage.drop_reason is not None
    assert "universe_mask application failed" in lineage.drop_reason
