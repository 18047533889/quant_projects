# -*- coding: utf-8 -*-
"""Full-stack telemetry for region execution (MB-P2-009, MB-P2-010).

This module provides:
- MB-P2-009: Region/transfer/sort/spill/source scan full-chain telemetry
- MB-P2-010: ExplainPlan - explain why each region chose its backend
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: R21-PERF-TELEMETRY-PIID: canonical ordering of per-record PI-ID telemetry
#: fields shared by :class:`RegionTelemetry` and :class:`BatchExecutionTelemetry`.
#: Every record MUST bind: canonical, physical_implementation_id, bound_params,
#: shape, source residency, threads, actual TTDC, peak RSS, spill and transfer
#: bytes.  Keeping the tuple as a single authority means the JSON evidence and
#: the tests can enumerate the exact field contract once.
PIID_TELEMETRY_FIELDS: tuple[str, ...] = (
    "canonical",
    "physical_implementation_id",
    "bound_params",
    "shape",
    "source_residency",
    "threads",
    "actual_ttdc_ms",
    "peak_rss_bytes",
    "spill_bytes",
    "transfer_bytes",
)


@dataclass
class RegionTelemetry:
    """Telemetry for a single backend region execution (MB-P2-009).

    R21-PERF-TELEMETRY-PIID: every region record additionally binds
    ``physical_implementation_id`` (the ``pi:v3:...`` digest of the selected
    physical implementation), its ``canonical`` operator name, the bound
    parameter payload, the output shape, the data-source residency the region
    executed on, the concurrency it used, the actual TTDC it took, and its
    peak RSS / spill / transfer bytes.
    """

    region_id: str
    backend: str
    representation: str
    node_count: int

    # Timing breakdown
    actual_compute_ms: float = 0.0
    actual_scan_ms: float = 0.0
    actual_transfer_ms: float = 0.0
    actual_materialize_ms: float = 0.0
    actual_total_ms: float = 0.0

    # Resource usage
    peak_memory_bytes: int = 0
    scan_bytes: int = 0
    output_bytes: int = 0

    # Execution details
    actual_rows: int = 0
    fallback_applied: bool = False
    error_message: str = ""

    # R21-PERF-TELEMETRY-PIID: PI-ID binding fields (all default to None/empty
    # so legacy constructions keep working; a record that carries them is
    # production telemetry and must bind every field).
    canonical: str = ""
    physical_implementation_id: str = ""
    bound_params: dict[str, Any] | None = None
    shape: dict[str, Any] | None = None
    source_residency: str = ""
    threads: int = 0
    actual_ttdc_ms: float = 0.0
    peak_rss_bytes: int = 0
    spill_bytes: int = 0
    transfer_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "backend": self.backend,
            "representation": self.representation,
            "node_count": self.node_count,
            "actual_compute_ms": round(self.actual_compute_ms, 3),
            "actual_scan_ms": round(self.actual_scan_ms, 3),
            "actual_transfer_ms": round(self.actual_transfer_ms, 3),
            "actual_materialize_ms": round(self.actual_materialize_ms, 3),
            "actual_total_ms": round(self.actual_total_ms, 3),
            "peak_memory_bytes": self.peak_memory_bytes,
            "scan_bytes": self.scan_bytes,
            "output_bytes": self.output_bytes,
            "actual_rows": self.actual_rows,
            "fallback_applied": self.fallback_applied,
            "error_message": self.error_message,
            # R21-PERF-TELEMETRY-PIID binding
            "canonical": self.canonical,
            "physical_implementation_id": self.physical_implementation_id,
            "bound_params": dict(self.bound_params) if self.bound_params else None,
            "shape": dict(self.shape) if self.shape else None,
            "source_residency": self.source_residency,
            "threads": self.threads,
            "actual_ttdc_ms": round(self.actual_ttdc_ms, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "spill_bytes": self.spill_bytes,
            "transfer_bytes": self.transfer_bytes,
        }


@dataclass
class TransferTelemetry:
    """Telemetry for a transfer edge (MB-P2-009)."""

    edge_id: str
    source_backend: str
    target_backend: str
    source_representation: str
    target_representation: str

    # Transfer details
    actual_transfer_ms: float = 0.0
    actual_bytes: int = 0
    actual_rows: int = 0

    # Transformations applied
    sort_applied: bool = False
    sort_ms: float = 0.0
    repartition_applied: bool = False
    repartition_ms: float = 0.0
    reshape_applied: bool = False
    reshape_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_backend": self.source_backend,
            "target_backend": self.target_backend,
            "source_representation": self.source_representation,
            "target_representation": self.target_representation,
            "actual_transfer_ms": round(self.actual_transfer_ms, 3),
            "actual_bytes": self.actual_bytes,
            "actual_rows": self.actual_rows,
            "sort_applied": self.sort_applied,
            "sort_ms": round(self.sort_ms, 3),
            "repartition_applied": self.repartition_applied,
            "repartition_ms": round(self.repartition_ms, 3),
            "reshape_applied": self.reshape_applied,
            "reshape_ms": round(self.reshape_ms, 3),
        }


@dataclass
class SpillTelemetry:
    """Telemetry for spill operations (MB-P2-016)."""

    spill_id: str
    spill_bytes: int = 0
    spill_ms: float = 0.0
    restore_bytes: int = 0
    restore_ms: float = 0.0
    storage_path: str = ""
    cleanup_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "spill_id": self.spill_id,
            "spill_bytes": self.spill_bytes,
            "spill_ms": round(self.spill_ms, 3),
            "restore_bytes": self.restore_bytes,
            "restore_ms": round(self.restore_ms, 3),
            "storage_path": self.storage_path,
            "cleanup_applied": self.cleanup_applied,
        }


@dataclass
class BatchExecutionTelemetry:
    """Complete telemetry for batch execution (MB-P2-009)."""

    batch_id: str
    regions: list[RegionTelemetry] = field(default_factory=list)
    transfers: list[TransferTelemetry] = field(default_factory=list)
    spills: list[SpillTelemetry] = field(default_factory=list)

    # Aggregate metrics
    total_ttdc_ms: float = 0.0
    peak_memory_bytes: int = 0
    total_scan_bytes: int = 0
    total_transfer_bytes: int = 0
    total_spill_bytes: int = 0

    # Plan vs actual
    planned_backend_count: int = 0
    actual_backend_count: int = 0
    backend_switches: int = 0
    fallback_count: int = 0

    # R21-PERF-TELEMETRY-PIID: batch-level binding of the PI-ID telemetry
    # contract.  The singular ``canonical`` / ``physical_implementation_id``
    # hold the sole value when the batch executed exactly one distinct
    # canonical / PI-ID; ``physical_implementation_ids`` lists every distinct
    # PI-ID recorded across the regions.  Blank singular values default to the
    # derived distinct value in ``to_dict`` when there is exactly one.
    physical_implementation_ids: tuple[str, ...] = ()
    canonical: str = ""
    physical_implementation_id: str = ""
    bound_params: dict[str, Any] | None = None
    shape: dict[str, Any] | None = None
    source_residency: str = ""
    threads: int = 0
    peak_rss_bytes: int = 0
    spill_bytes: int = 0
    transfer_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        distinct_canonicals = sorted(
            {r.canonical for r in self.regions if getattr(r, "canonical", "")}
        )
        region_piids = sorted(
            {r.physical_implementation_id for r in self.regions if getattr(r, "physical_implementation_id", "")}
        )
        canonical = self.canonical or (distinct_canonicals[0] if len(distinct_canonicals) == 1 else "")
        piid = self.physical_implementation_id or (
            region_piids[0] if len(region_piids) == 1 else ""
        )
        piids = list(self.physical_implementation_ids) or region_piids
        return {
            "batch_id": self.batch_id,
            "regions": [r.to_dict() for r in self.regions],
            "transfers": [t.to_dict() for t in self.transfers],
            "spills": [s.to_dict() for s in self.spills],
            "total_ttdc_ms": round(self.total_ttdc_ms, 3),
            "peak_memory_bytes": self.peak_memory_bytes,
            "total_scan_bytes": self.total_scan_bytes,
            "total_transfer_bytes": self.total_transfer_bytes,
            "total_spill_bytes": self.total_spill_bytes,
            "planned_backend_count": self.planned_backend_count,
            "actual_backend_count": self.actual_backend_count,
            "backend_switches": self.backend_switches,
            "fallback_count": self.fallback_count,
            # R21-PERF-TELEMETRY-PIID binding
            "canonical": canonical,
            "physical_implementation_id": piid,
            "physical_implementation_ids": piids,
            "bound_params": dict(self.bound_params) if self.bound_params else None,
            "shape": dict(self.shape) if self.shape else None,
            "source_residency": self.source_residency,
            "threads": self.threads,
            # ``actual_ttdc_ms`` is the R21 required key; on the batch record it
            # aliases the legacy aggregate ``total_ttdc_ms`` so the contract is
            # uniform across region and batch records.
            "actual_ttdc_ms": round(self.total_ttdc_ms, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "spill_bytes": self.spill_bytes,
            "transfer_bytes": self.transfer_bytes,
        }


# MB-P2-010: ExplainPlan

@dataclass
class RegionExplanation:
    """Explanation of why a region chose its backend (MB-P2-010)."""

    region_id: str
    chosen_backend: str
    reason: str
    alternatives: list[tuple[str, str]] = field(default_factory=list)  # (backend, reason_rejected)

    # Decision factors
    node_count: int = 0
    native_expression_count: int = 0
    delegate_count: int = 0
    input_representation: str = ""
    estimated_compute_ms: float = 0.0
    estimated_transfer_ms: float = 0.0
    estimated_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "chosen_backend": self.chosen_backend,
            "reason": self.reason,
            "alternatives": [{"backend": b, "reason_rejected": r} for b, r in self.alternatives],
            "node_count": self.node_count,
            "native_expression_count": self.native_expression_count,
            "delegate_count": self.delegate_count,
            "input_representation": self.input_representation,
            "estimated_compute_ms": round(self.estimated_compute_ms, 3),
            "estimated_transfer_ms": round(self.estimated_transfer_ms, 3),
            "estimated_memory_mb": round(self.estimated_memory_mb, 3),
        }


@dataclass
class ExplainPlan:
    """Complete explanation of physical plan decisions (MB-P2-010).

    Example output:
        Region R2 → Polars
        - 43 native expressions
        - input already Arrow/Polars
        - no additional global sort
        - estimated 310MB
        - DuckDB saves only 4ms
        - boundary cost estimated 12ms
        => stay Polars
    """

    plan_id: str
    regions: list[RegionExplanation] = field(default_factory=list)
    total_regions: int = 0
    backend_distribution: dict[str, int] = field(default_factory=dict)
    total_backend_switches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "regions": [r.to_dict() for r in self.regions],
            "total_regions": self.total_regions,
            "backend_distribution": dict(self.backend_distribution),
            "total_backend_switches": self.total_backend_switches,
        }

    def format_text(self) -> str:
        """Format explanation as human-readable text."""
        lines = [f"Physical Plan Explanation: {self.plan_id}"]
        lines.append(f"Total Regions: {self.total_regions}")
        lines.append(f"Backend Distribution: {dict(self.backend_distribution)}")
        lines.append(f"Backend Switches: {self.total_backend_switches}")
        lines.append("")

        for r in self.regions:
            lines.append(f"Region {r.region_id} → {r.chosen_backend}")
            lines.append(f"  Reason: {r.reason}")
            if r.native_expression_count > 0:
                lines.append(f"  - {r.native_expression_count} native expressions")
            if r.delegate_count > 0:
                lines.append(f"  - {r.delegate_count} delegate operations")
            if r.input_representation:
                lines.append(f"  - input: {r.input_representation}")
            if r.estimated_compute_ms > 0:
                lines.append(f"  - estimated compute: {r.estimated_compute_ms:.1f}ms")
            if r.estimated_transfer_ms > 0:
                lines.append(f"  - estimated transfer: {r.estimated_transfer_ms:.1f}ms")
            if r.estimated_memory_mb > 0:
                lines.append(f"  - estimated memory: {r.estimated_memory_mb:.1f}MB")

            if r.alternatives:
                lines.append("  Alternatives rejected:")
                for backend, reason in r.alternatives:
                    lines.append(f"    - {backend}: {reason}")
            lines.append("")

        return "\n".join(lines)


def build_explain_plan(plan: Any, region_decisions: list[dict[str, Any]]) -> ExplainPlan:
    """Build ExplainPlan from physical region plan and decision metadata (MB-P2-010)."""
    from collections import Counter

    plan_id = getattr(plan, "plan_id", "unknown")
    regions = []

    backend_counts: Counter[str] = Counter()

    for decision in region_decisions:
        region_id = decision.get("region_id", "")
        chosen_backend = decision.get("chosen_backend", "")
        backend_counts[chosen_backend] += 1

        alternatives = decision.get("alternatives", [])

        regions.append(RegionExplanation(
            region_id=region_id,
            chosen_backend=chosen_backend,
            reason=decision.get("reason", ""),
            alternatives=[(a.get("backend", ""), a.get("reason", "")) for a in alternatives],
            node_count=decision.get("node_count", 0),
            native_expression_count=decision.get("native_expression_count", 0),
            delegate_count=decision.get("delegate_count", 0),
            input_representation=decision.get("input_representation", ""),
            estimated_compute_ms=decision.get("estimated_compute_ms", 0.0),
            estimated_transfer_ms=decision.get("estimated_transfer_ms", 0.0),
            estimated_memory_mb=decision.get("estimated_memory_mb", 0.0),
        ))

    backend_switches = getattr(plan, "backend_switch_count", 0)

    return ExplainPlan(
        plan_id=plan_id,
        regions=regions,
        total_regions=len(regions),
        backend_distribution=dict(backend_counts),
        total_backend_switches=backend_switches,
    )


# ---------------------------------------------------------------------------
# R21-PERF-TELEMETRY-PIID: bind PI-ID telemetry into a region record
# ---------------------------------------------------------------------------
def bind_piid_telemetry(
    region: RegionTelemetry,
    *,
    canonical: str,
    physical_implementation_id: str,
    bound_params: dict[str, Any] | None = None,
    shape: dict[str, Any] | None = None,
    source_residency: str = "",
    threads: int = 0,
    actual_ttdc_ms: float = 0.0,
    peak_rss_bytes: int = 0,
    spill_bytes: int = 0,
    transfer_bytes: int = 0,
) -> RegionTelemetry:
    """Bind the R21-PERF-TELEMETRY-PIID fields onto one region telemetry record.

    The ``physical_implementation_id`` must be a non-empty string (the
    ``pi:v3:...`` digest of the selected physical implementation); a blank PI-ID
    is rejected with ``ValueError`` so a production record can never be written
    without its PI-ID binding.  All other fields default to their zero value and
    are recorded as-is.
    """
    if not str(physical_implementation_id or "").strip():
        raise ValueError(
            "R21-PERF-TELEMETRY-PIID: physical_implementation_id is required "
            "on every performance telemetry record"
        )
    region.canonical = str(canonical or "")
    region.physical_implementation_id = str(physical_implementation_id)
    region.bound_params = dict(bound_params) if bound_params else None
    region.shape = dict(shape) if shape else None
    region.source_residency = str(source_residency or "")
    region.threads = max(0, int(threads or 0))
    region.actual_ttdc_ms = max(0.0, float(actual_ttdc_ms or 0.0))
    region.peak_rss_bytes = max(0, int(peak_rss_bytes or 0))
    region.spill_bytes = max(0, int(spill_bytes or 0))
    region.transfer_bytes = max(0, int(transfer_bytes or 0))
    return region
