#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARCH-P0-003: Single authority for physical I/O boundary policy.

This module defines the authoritative policy for which modules may perform
direct physical I/O operations. All tools (boundary checker, pre-commit hooks,
audit scripts) must reference this single source of truth.

Physical I/O operations include:
- duckdb.connect()
- pd.read_parquet(), pd.read_csv()
- pq.read_table(), pq.write_table()
- pl.read_parquet(), pl.scan_parquet()
- df.to_parquet(), df.to_csv()

These operations bypass DataAccess governance (PIT, credentials, schema,
resource accounting, audit trails) and must only be used in explicitly
authorized modules.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class ExemptionCategory(Enum):
    """Categories of physical I/O exemptions with clear responsibilities."""

    SOURCE_READ = "source_read"
    """DataAccess source layer - governed physical reads with PIT validation."""

    RESULT_PERSISTENCE = "result_persistence"
    """DataAccess persistence layer - governed atomic writes with versioning."""

    MAINTENANCE = "maintenance"
    """One-time maintenance scripts (migration, repair, backfill)."""

    TEST_FIXTURE = "test_fixture"
    """Test fixtures and isolated test harness code."""

    BENCHMARK = "benchmark"
    """Performance benchmark harness (isolated, non-production)."""

    MIGRATION = "migration"
    """Schema migration and legacy format conversion."""


class ExemptedModule(NamedTuple):
    """A module exempted from physical I/O boundary enforcement."""

    path: str
    """Relative path from repo root (e.g., 'storage/cache.py')."""

    category: ExemptionCategory
    """Category of exemption."""

    responsibility: str
    """Human-readable description of the module's responsibility."""


#: Single source of truth for physical I/O boundary policy.
#: All modules not in this list must use DataAccess abstraction.
PHYSICAL_IO_AUTHORITY_POLICY: list[ExemptedModule] = [
    # -------------------------------------------------------------------------
    # SOURCE_READ: DataAccess source layer (governed physical I/O boundary)
    # -------------------------------------------------------------------------
    ExemptedModule(
        path="storage/sources/kline_parquet_source.py",
        category=ExemptionCategory.SOURCE_READ,
        responsibility="DataAccess kline reader - governed parquet read with PIT",
    ),
    ExemptedModule(
        path="storage/sources/parquet_source.py",
        category=ExemptionCategory.SOURCE_READ,
        responsibility="DataAccess generic parquet reader - governed physical I/O",
    ),
    ExemptedModule(
        path="storage/sources/staging_loader.py",
        category=ExemptionCategory.SOURCE_READ,
        responsibility="DataAccess staging loader - ingestion with validation",
    ),
    ExemptedModule(
        path="storage/sources/lqtp_logical_source.py",
        category=ExemptionCategory.SOURCE_READ,
        responsibility="DataAccess LQTP source - logical query physical execution",
    ),
    # -------------------------------------------------------------------------
    # RESULT_PERSISTENCE: Storage layer (governed writes)
    # -------------------------------------------------------------------------
    ExemptedModule(
        path="storage/cache.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Factor cache - governed persistent result cache",
    ),
    ExemptedModule(
        path="storage/result_store.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Result store - governed factor result persistence",
    ),
    ExemptedModule(
        path="storage/delta_store.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Delta store - governed incremental update storage",
    ),
    ExemptedModule(
        path="storage/block_lake.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Block lake - governed versioned parquet blocks",
    ),
    ExemptedModule(
        path="storage/parquet_batch_writer.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Batch writer - governed low-level parquet streaming",
    ),
    ExemptedModule(
        path="storage/lake_version.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Lake versioning - governed snapshot management",
    ),
    ExemptedModule(
        path="storage/matrix_block_layout.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Matrix layout - governed wide-form storage",
    ),
    # -------------------------------------------------------------------------
    # RESULT_PERSISTENCE: Materializer subsystem
    # -------------------------------------------------------------------------
    ExemptedModule(
        path="storage/materialize/materializer.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Materializer - governed atomic factor result persistence",
    ),
    ExemptedModule(
        path="storage/materialize/factor_matrix_materializer.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Matrix materializer - governed wide-form persistence",
    ),
    ExemptedModule(
        path="storage/materialize/lake_publish.py",
        category=ExemptionCategory.RESULT_PERSISTENCE,
        responsibility="Lake publisher - governed multi-partition atomic commit",
    ),
    # -------------------------------------------------------------------------
    # MIGRATION: Schema migration and legacy conversion
    # -------------------------------------------------------------------------
    ExemptedModule(
        path="storage/schema_migration.py",
        category=ExemptionCategory.MIGRATION,
        responsibility="Schema migration - governed legacy format conversion",
    ),
]

#: Path patterns that are automatically exempted (tests, benchmarks).
#: These are broad categories that don't need individual module listing.
AUTOMATIC_EXEMPTION_PATTERNS: list[tuple[str, ExemptionCategory]] = [
    ("tests/", ExemptionCategory.TEST_FIXTURE),
    ("benchmarks/", ExemptionCategory.BENCHMARK),
]


def get_exempted_modules_by_category(
    category: ExemptionCategory,
) -> list[ExemptedModule]:
    """Get all exempted modules in a specific category."""
    return [m for m in PHYSICAL_IO_AUTHORITY_POLICY if m.category == category]


def is_path_exempted(relative_path: str) -> tuple[bool, ExemptionCategory | None, str]:
    """Check if a path is exempted from physical I/O boundary enforcement.

    Args:
        relative_path: Path relative to repo root (e.g., 'storage/cache.py')

    Returns:
        (is_exempted, category, reason): Whether path is exempted, its category,
            and human-readable reason.
    """
    # Check exact module matches
    for module in PHYSICAL_IO_AUTHORITY_POLICY:
        if relative_path == module.path:
            return True, module.category, module.responsibility

    # Check automatic exemption patterns
    for pattern, category in AUTOMATIC_EXEMPTION_PATTERNS:
        if pattern in relative_path:
            return True, category, f"Automatic exemption: {category.value}"

    return False, None, "Not in PHYSICAL_IO_AUTHORITY_POLICY"


def format_policy_report() -> str:
    """Format a human-readable report of the physical I/O authority policy."""
    lines = []
    lines.append("=" * 80)
    lines.append("Physical I/O Authority Policy (ARCH-P0-003)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Single source of truth for physical I/O boundary enforcement.")
    lines.append("")

    for category in ExemptionCategory:
        modules = get_exempted_modules_by_category(category)
        if not modules:
            continue

        lines.append(f"{category.value.upper().replace('_', ' ')}:")
        lines.append("-" * 80)
        for module in modules:
            lines.append(f"  {module.path}")
            lines.append(f"    → {module.responsibility}")
        lines.append("")

    lines.append("AUTOMATIC EXEMPTIONS:")
    lines.append("-" * 80)
    for pattern, category in AUTOMATIC_EXEMPTION_PATTERNS:
        lines.append(f"  {pattern} → {category.value}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Print policy report when run directly
    print(format_policy_report())
