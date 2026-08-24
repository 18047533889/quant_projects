# -*- coding: utf-8 -*-
"""R21-PERF-TELEMETRY-PIID — bind performance telemetry to PhysicalImplementationID.

Spec (§21 / MB-P2-009 / FE-P0-004): every performance telemetry record must
bind:

    canonical, physical_implementation_id, bound_params, shape,
    source_residency, threads, actual_ttdc_ms, peak_rss_bytes, spill_bytes,
    transfer_bytes

This module verifies the contract at the region record and batch record level:

  (a) ``bind_piid_telemetry`` requires a non-empty PI-ID and writes every
      field onto the region record; ``to_dict`` round-trips the full field set.
  (b) ``BatchExecutionTelemetry`` aggregates the per-region PI-ID field set and
      round-trips it through ``to_dict``.
  (c) a real ``PhysicalImplementationID`` (derived from a complete
      ``PhysicalImplementationSpec``) binds end-to-end into the record.
"""
from __future__ import annotations

import pytest

from factor_engine.backend.telemetry_region import (
    PIID_TELEMETRY_FIELDS,
    BatchExecutionTelemetry,
    RegionTelemetry,
    bind_piid_telemetry,
)

#: Every one of the ten R21-PERF-TELEMETRY-PIID fields must survive a
#: ``to_dict()`` round trip on both region and batch records.
_REQUIRED = PIID_TELEMETRY_FIELDS


def _region() -> RegionTelemetry:
    return RegionTelemetry(
        region_id="R1",
        backend="polars",
        representation="polars_native",
        node_count=2,
        actual_compute_ms=12.5,
    )


# ---------------------------------------------------------------------------
# (a) bind_piid_telemetry + region record round trip
# ---------------------------------------------------------------------------

class TestBindPiidTelemetry:
    def test_bind_requires_non_blank_piid(self) -> None:
        region = _region()
        with pytest.raises(ValueError, match="physical_implementation_id"):
            bind_piid_telemetry(region, canonical="ts_mean", physical_implementation_id="")
        with pytest.raises(ValueError, match="physical_implementation_id"):
            bind_piid_telemetry(region, canonical="ts_mean", physical_implementation_id="   ")
        # unchanged (fail-closed: no partial binding)
        assert region.physical_implementation_id == ""

    def test_bind_writes_all_required_fields(self) -> None:
        region = bind_piid_telemetry(
            _region(),
            canonical="ts_mean",
            physical_implementation_id="pi:v3:abc",
            bound_params={"window": 3},
            shape={"rows": 10, "columns": 2},
            source_residency="polars_long",
            threads=2,
            actual_ttdc_ms=123.4,
            peak_rss_bytes=2048,
            spill_bytes=512,
            transfer_bytes=96,
        )
        assert region.canonical == "ts_mean"
        assert region.physical_implementation_id == "pi:v3:abc"
        assert region.bound_params == {"window": 3}
        assert region.shape == {"rows": 10, "columns": 2}
        assert region.source_residency == "polars_long"
        assert region.threads == 2
        assert region.actual_ttdc_ms == 123.4
        assert region.peak_rss_bytes == 2048
        assert region.spill_bytes == 512
        assert region.transfer_bytes == 96

    def test_bind_source_residency_requires_real_source(self) -> None:
        # R22 P0: source_residency must never be fabricated.  A region with no
        # dataset keeps the default "" (fail-closed).
        region = RegionTelemetry(
            region_id="R1", backend="polars", representation="polars_native", node_count=1
        )
        region.bind_source_residency(dataset="", snapshot_id="snap-1")
        assert region.source_residency == ""
        # A region with a real source carries a stable "backend:dataset" string.
        region.bind_source_residency(dataset="ashare_1d", snapshot_id="snap-1", market="A")
        assert region.source_residency == "polars:ashare_1d:snap-1:A"

    def test_bind_batch_source_residency_requires_real_source(self) -> None:
        batch = BatchExecutionTelemetry(batch_id="B1")
        batch.bind_batch_source_residency(dataset="", snapshot_id="snap-1")
        assert batch.source_residency == ""
        batch.bind_batch_source_residency(dataset="ashare_1d", snapshot_id="snap-1", market="A")
        assert batch.source_residency == "ashare_1d:snap-1:A"

    def test_region_to_dict_contains_all_required_fields(self) -> None:
        region = bind_piid_telemetry(
            _region(),
            canonical="ts_rank",
            physical_implementation_id="pi:v3:xyz",
            bound_params={"window": 10},
            shape={"rows": 100, "columns": 3},
            source_residency="duckdb_sql",
            threads=4,
            actual_ttdc_ms=45.6,
            peak_rss_bytes=1024,
            spill_bytes=0,
            transfer_bytes=32,
        )
        data = region.to_dict()
        for key in _REQUIRED:
            assert key in data, f"missing required telemetry field: {key}"
        assert data["physical_implementation_id"] == "pi:v3:xyz"
        assert data["bound_params"] == {"window": 10}
        assert data["shape"] == {"rows": 100, "columns": 3}
        assert data["source_residency"] == "duckdb_sql"
        assert data["threads"] == 4
        assert data["actual_ttdc_ms"] == 45.6
        assert data["peak_rss_bytes"] == 1024
        assert data["spill_bytes"] == 0
        assert data["transfer_bytes"] == 32


# ---------------------------------------------------------------------------
# (b) BatchExecutionTelemetry aggregation + round trip
# ---------------------------------------------------------------------------

class TestBatchPiidTelemetry:
    def test_batch_defaults_all_required_fields(self) -> None:
        batch = BatchExecutionTelemetry(batch_id="B1")
        data = batch.to_dict()
        for key in _REQUIRED:
            assert key in data, f"missing batch telemetry field: {key}"
        assert data["physical_implementation_ids"] == []
        assert data["bound_params"] is None
        assert data["shape"] is None
        assert data["source_residency"] == ""
        assert data["threads"] == 0
        assert data["peak_rss_bytes"] == 0
        assert data["spill_bytes"] == 0
        assert data["transfer_bytes"] == 0

    def test_batch_aggregates_piid_ids_and_required_fields(self) -> None:
        r1 = bind_piid_telemetry(
            _region(),
            canonical="ts_mean",
            physical_implementation_id="pi:v3:one",
            bound_params={"window": 3},
            shape={"rows": 10, "columns": 1},
            source_residency="polars_long",
            threads=2,
            actual_ttdc_ms=50.0,
            peak_rss_bytes=2048,
            spill_bytes=256,
            transfer_bytes=16,
        )
        r2 = bind_piid_telemetry(
            RegionTelemetry(
                region_id="R2", backend="duckdb_sql", representation="sql", node_count=1
            ),
            canonical="ts_rank",
            physical_implementation_id="pi:v3:two",
            bound_params={"window": 5},
            shape={"rows": 10, "columns": 1},
            source_residency="duckdb_sql",
            threads=2,
            actual_ttdc_ms=30.0,
            peak_rss_bytes=1024,
            spill_bytes=0,
            transfer_bytes=8,
        )
        batch = BatchExecutionTelemetry(
            batch_id="B1",
            regions=[r1, r2],
            total_ttdc_ms=80.0,
            total_transfer_bytes=24,
            total_spill_bytes=256,
            peak_memory_bytes=2048,
            physical_implementation_ids=("pi:v3:one", "pi:v3:two"),
            bound_params={"window": 3},
            shape={"rows": 10, "columns": 1},
            source_residency="polars_long",
            threads=2,
            peak_rss_bytes=2048,
            spill_bytes=256,
            transfer_bytes=24,
        )
        data = batch.to_dict()
        for key in _REQUIRED:
            assert key in data, f"missing batch telemetry field: {key}"
        assert data["physical_implementation_ids"] == ["pi:v3:one", "pi:v3:two"]
        assert data["bound_params"] == {"window": 3}
        assert data["shape"] == {"rows": 10, "columns": 1}
        assert data["source_residency"] == "polars_long"
        assert data["threads"] == 2
        assert data["actual_ttdc_ms"] == 80.0
        assert data["peak_rss_bytes"] == 2048
        assert data["spill_bytes"] == 256
        assert data["transfer_bytes"] == 24
        # per-region records also carry the full field set
        for region_dict in data["regions"]:
            for key in _REQUIRED:
                assert key in region_dict, f"missing region telemetry field: {key}"


# ---------------------------------------------------------------------------
# (c) real PhysicalImplementationID end to end
# ---------------------------------------------------------------------------

def _valid_spec():
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
    from factor_engine.backend.evidence_provenance import implementation_closure_hash_for

    closure = implementation_closure_hash_for("ts_mean")
    return PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        implementation_source_hash=closure,
        emitter_identity="polars.expr:v1",
        kernel_identity="polars.rolling:v1",
        parameter_domain_hash=closure,
        semantic_contract_hash=closure,
        implementation_closure_hash=closure,
    )


class TestPhysicalImplementationIdEndToEnd:
    def test_real_piid_binds_into_region_and_batch(self) -> None:
        spec = _valid_spec()
        assert spec.validation_errors() == ()
        piid = spec.physical_implementation_id
        assert piid is not None
        assert str(piid).startswith("pi:v3:")

        region = bind_piid_telemetry(
            _region(),
            canonical="ts_mean",
            physical_implementation_id=str(piid),
            bound_params={"window": 3},
            shape={"rows": 10, "columns": 1},
            source_residency="polars_long",
            threads=2,
            actual_ttdc_ms=11.5,
            peak_rss_bytes=2048,
            spill_bytes=64,
            transfer_bytes=12,
        )
        data = region.to_dict()
        assert data["physical_implementation_id"] == str(piid)
        assert data["canonical"] == "ts_mean"

        batch = BatchExecutionTelemetry(
            batch_id="B1",
            regions=[region],
            physical_implementation_ids=(str(piid),),
            peak_rss_bytes=2048,
        )
        batch_data = batch.to_dict()
        assert batch_data["physical_implementation_ids"] == [str(piid)]
        # batch derives the singular PI-ID when exactly one distinct region PI-ID exists
        assert batch_data["physical_implementation_id"] == str(piid)
        assert batch_data["canonical"] == "ts_mean"

    def test_piid_changes_when_spec_binding_changes(self) -> None:
        base = _valid_spec()
        changed = _valid_spec()
        # different implementation closure => different PI-ID
        changed = type(changed)(
            canonical=changed.canonical,
            backend=changed.backend,
            execution_kind=changed.execution_kind,
            implementation_source_hash=changed.implementation_source_hash,
            emitter_identity=changed.emitter_identity,
            kernel_identity="polars.rolling:v2",
            parameter_domain_hash=changed.parameter_domain_hash,
            semantic_contract_hash=changed.semantic_contract_hash,
            implementation_closure_hash="b" * 64,
        )
        assert base.physical_implementation_id is not None
        assert changed.physical_implementation_id is not None
        assert base.physical_implementation_id != changed.physical_implementation_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
