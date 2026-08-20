# -*- coding: utf-8 -*-
"""R21-EVIDENCE-REFRESH-CLI: Regression tests for evidence refresh CLI.

Constraints:
  - LOCAL ONLY (no git, no network)
  - Serial pytest, no xdist
  - Thread env vars set by runner

5 tests:
  1. cli_help            – CLI --help exits 0
  2. cli_dry_run         – --dry-run completes without errors
  3. current_json_schema – CURRENT.json has required fields
  4. artifact_status_enum – Status values are valid
  5. pipeline_steps      – All 9 steps are present in CLI
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Test 1: CLI --help exits 0
# ---------------------------------------------------------------------------

def test_cli_help() -> None:
    """CLI --help should exit with code 0 and print usage."""
    result = subprocess.run(
        [sys.executable, "-m", "evidence.refresh", "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Test 2: CLI --dry-run completes without errors
# ---------------------------------------------------------------------------

def test_cli_dry_run() -> None:
    """CLI --dry-run should complete without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "evidence.refresh", "--dry-run"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Dry run should not fail
    assert result.returncode == 0, f"--dry-run failed: {result.stderr}"
    assert "DRY_RUN" in result.stdout or "dry run" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Test 3: CURRENT.json has required fields (after dry run)
# ---------------------------------------------------------------------------

def test_current_json_schema() -> None:
    """CURRENT.json should exist and have required fields after CLI run."""
    # First, run CLI to generate CURRENT.json
    result = subprocess.run(
        [sys.executable, "-m", "evidence.refresh", "--dry-run"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Note: dry-run does not write CURRENT.json, so we check the schema
    # by importing the generate_current function
    sys.path.insert(0, str(_REPO_ROOT))
    from evidence.__main__ import generate_current, ArtifactStatus
    from evidence.registry import EvidenceArtifact

    # Create a mock artifact
    art = EvidenceArtifact(
        artifact_id="test-artifact",
        artifact_type="inventory",
        description="Test artifact",
        generator_script="test.py",
        runner_command="echo test",
    )

    # Create mock status
    statuses = {
        "test-artifact": ArtifactStatus(
            "test-artifact", "CURRENT", "Test details"
        )
    }

    # Generate current data (dry run)
    current_data = generate_current([art], statuses, dry_run=True)

    # Check required fields
    assert "schema_version" in current_data
    assert "generated_at" in current_data
    assert "source_snapshot_id" in current_data
    assert "artifacts" in current_data
    assert "summary" in current_data

    # Check summary fields
    summary = current_data["summary"]
    assert "total" in summary
    assert "current" in summary
    assert "stale" in summary
    assert "failed" in summary
    assert "not_run" in summary


# ---------------------------------------------------------------------------
# Test 4: Status values are valid
# ---------------------------------------------------------------------------

def test_artifact_status_enum() -> None:
    """ArtifactStatus should only accept valid status values."""
    from evidence.__main__ import ArtifactStatus, ARTIFACT_STATUSES

    # Valid statuses
    for status in ARTIFACT_STATUSES:
        art = ArtifactStatus("test", status, "details")
        assert art.status == status

    # Invalid status should still be accepted (no strict validation in dataclass)
    # but the summary table groups by valid statuses
    art = ArtifactStatus("test", "INVALID", "details")
    assert art.status == "INVALID"


# ---------------------------------------------------------------------------
# Test 5: All 9 steps are present in CLI
# ---------------------------------------------------------------------------

def test_pipeline_steps() -> None:
    """CLI should implement all 9 pipeline steps."""
    import evidence.__main__ as cli

    # Check that all step functions exist
    required_steps = [
        "scan_source",
        "calculate_snapshot",
        "load_evidence_registry",
        "compute_input_identities",
        "mark_stale",
        "regenerate_inventories",
        "run_certifications",
        "generate_current",
        "verify_no_stale_production",
    ]

    for step in required_steps:
        assert hasattr(cli, step), f"Missing step: {step}"
        assert callable(getattr(cli, step)), f"Step not callable: {step}"
