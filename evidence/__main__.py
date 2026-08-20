# -*- coding: utf-8 -*-
"""Evidence refresh CLI: `python -m evidence.refresh` entry point.

Provides local-only evidence autopilot with the following steps:
1. scan_source → Merkle-tree source snapshot
2. calculate_snapshot → Source identity computation
3. load_registry → Load evidence artifact registry
4. compute_input_identities → SHA-256 digests for all inputs
5. mark_stale → Detect artifacts with changed inputs
6. regenerate_inventories → Regenerate inventory CSVs
7. run_certifications → Run certification verification
8. generate_current → Generate CURRENT.json as single-truth entry point
9. verify_no_stale_production → Final verification gate

Output: Summary table with CURRENT/STALE/FAILED/NOT_RUN per artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence.registry import (
    EvidenceArtifact,
    check_freshness,
    load_registry,
    sha256_files,
)
from evidence.source_snapshot import build_snapshot, write_snapshot


# ---------------------------------------------------------------------------
# Artifact status enum
# ---------------------------------------------------------------------------
ARTIFACT_STATUSES = {"CURRENT", "STALE", "FAILED", "NOT_RUN"}


@dataclass
class ArtifactStatus:
    """Status of a single artifact after refresh."""
    artifact_id: str
    status: str  # CURRENT | STALE | FAILED | NOT_RUN
    details: str = ""
    stale_inputs: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "status": self.status,
            "details": self.details,
            "stale_inputs": self.stale_inputs or [],
        }


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def scan_source(dry_run: bool = False) -> dict[str, Any]:
    """Step 1: Scan source tree and build Merkle snapshot."""
    if dry_run:
        return {"status": "DRY_RUN", "files": 0}

    snapshot = write_snapshot()
    return {
        "status": "OK",
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "merkle_root": snapshot["merkle_root"],
        "file_count": snapshot["file_count"],
    }


def calculate_snapshot(dry_run: bool = False) -> dict[str, Any]:
    """Step 2: Calculate source identity from snapshot."""
    if dry_run:
        return {"status": "DRY_RUN", "identity": "dry_run_identity"}

    from evidence.source_snapshot import load_snapshot
    snapshot = load_snapshot()
    if snapshot is None:
        return {"status": "FAILED", "error": "No snapshot found"}

    return {
        "status": "OK",
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "merkle_root": snapshot["merkle_root"],
    }


def load_evidence_registry(dry_run: bool = False) -> list[EvidenceArtifact]:
    """Step 3: Load evidence artifact registry."""
    if dry_run:
        return []

    return load_registry()


def compute_input_identities(
    artifacts: list[EvidenceArtifact], dry_run: bool = False
) -> dict[str, str]:
    """Step 4: Compute SHA-256 identities for all artifact inputs."""
    if dry_run:
        return {a.artifact_id: "dry_run" for a in artifacts}

    identities: dict[str, str] = {}
    for art in artifacts:
        identities[art.artifact_id] = art.recompute_identity()
    return identities


def mark_stale(
    artifacts: list[EvidenceArtifact], dry_run: bool = False
) -> dict[str, ArtifactStatus]:
    """Step 5: Mark artifacts as CURRENT or STALE based on input changes."""
    if dry_run:
        return {
            a.artifact_id: ArtifactStatus(a.artifact_id, "DRY_RUN")
            for a in artifacts
        }

    statuses: dict[str, ArtifactStatus] = {}
    for art in artifacts:
        report = check_freshness(art)
        if report.stale_count == 0:
            statuses[art.artifact_id] = ArtifactStatus(
                art.artifact_id, "CURRENT", "All inputs unchanged"
            )
        else:
            statuses[art.artifact_id] = ArtifactStatus(
                art.artifact_id,
                "STALE",
                f"{len(art.stale_inputs)} stale input(s)",
                art.stale_inputs,
            )
    return statuses


def regenerate_inventories(
    artifacts: list[EvidenceArtifact],
    statuses: dict[str, ArtifactStatus],
    dry_run: bool = False,
) -> dict[str, ArtifactStatus]:
    """Step 6: Regenerate inventory artifacts that are STALE."""
    if dry_run:
        return statuses

    for art in artifacts:
        if art.artifact_type != "inventory":
            continue
        if statuses[art.artifact_id].status != "STALE":
            continue

        try:
            # Run the generator command
            import subprocess
            result = subprocess.run(
                art.runner_command.split(),
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                # Recompute identity after regeneration
                new_identity = art.recompute_identity()
                art.evidence_input_identity = new_identity
                statuses[art.artifact_id] = ArtifactStatus(
                    art.artifact_id, "CURRENT", "Regenerated successfully"
                )
            else:
                statuses[art.artifact_id] = ArtifactStatus(
                    art.artifact_id, "FAILED", f"Generator failed: {result.stderr[:200]}"
                )
        except Exception as e:
            statuses[art.artifact_id] = ArtifactStatus(
                art.artifact_id, "FAILED", f"Exception: {str(e)[:200]}"
            )

    return statuses


def run_certifications(
    artifacts: list[EvidenceArtifact],
    statuses: dict[str, ArtifactStatus],
    dry_run: bool = False,
) -> dict[str, ArtifactStatus]:
    """Step 7: Run certification verification for certification artifacts."""
    if dry_run:
        return statuses

    for art in artifacts:
        if art.artifact_type != "certification":
            continue
        if statuses[art.artifact_id].status != "STALE":
            continue

        try:
            # Run the certification command
            import subprocess
            result = subprocess.run(
                art.runner_command.split(),
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Recompute identity after certification
                new_identity = art.recompute_identity()
                art.evidence_input_identity = new_identity
                statuses[art.artifact_id] = ArtifactStatus(
                    art.artifact_id, "CURRENT", "Certification passed"
                )
            else:
                statuses[art.artifact_id] = ArtifactStatus(
                    art.artifact_id, "FAILED", f"Certification failed: {result.stderr[:200]}"
                )
        except Exception as e:
            statuses[art.artifact_id] = ArtifactStatus(
                art.artifact_id, "FAILED", f"Exception: {str(e)[:200]}"
            )

    return statuses


def generate_current(
    artifacts: list[EvidenceArtifact],
    statuses: dict[str, ArtifactStatus],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Step 8: Generate CURRENT.json as single-truth entry point."""
    current_data = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_snapshot_id": "",
        "artifacts": {},
        "summary": {
            "total": len(artifacts),
            "current": 0,
            "stale": 0,
            "failed": 0,
            "not_run": 0,
        },
    }

    # Load source snapshot if available
    from evidence.source_snapshot import load_snapshot
    snapshot = load_snapshot()
    if snapshot:
        current_data["source_snapshot_id"] = snapshot["source_snapshot_id"]

    # Populate artifact statuses
    for art in artifacts:
        status = statuses[art.artifact_id]
        current_data["artifacts"][art.artifact_id] = status.to_dict()

        # Update summary counts
        if status.status == "CURRENT":
            current_data["summary"]["current"] += 1
        elif status.status == "STALE":
            current_data["summary"]["stale"] += 1
        elif status.status == "FAILED":
            current_data["summary"]["failed"] += 1
        elif status.status == "NOT_RUN":
            current_data["summary"]["not_run"] += 1

    if dry_run:
        return current_data

    # Write CURRENT.json
    current_path = _REPO_ROOT / "evidence" / "CURRENT.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(
        json.dumps(current_data, indent=2, sort_keys=False), encoding="utf-8"
    )

    return current_data


def verify_no_stale_production(
    artifacts: list[EvidenceArtifact],
    statuses: dict[str, ArtifactStatus],
    dry_run: bool = False,
) -> tuple[bool, list[str]]:
    """Step 9: Final verification gate - no stale production artifacts."""
    if dry_run:
        return True, []

    stale_production = []
    for art in artifacts:
        status = statuses[art.artifact_id]
        if status.status == "STALE" and art.artifact_type in ("inventory", "certification"):
            stale_production.append(f"{art.artifact_id} ({status.details})")

    passed = len(stale_production) == 0
    return passed, stale_production


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(
    artifacts: list[EvidenceArtifact],
    statuses: dict[str, ArtifactStatus],
) -> None:
    """Print a formatted summary table of artifact statuses."""
    # Group by status
    by_status: dict[str, list[ArtifactStatus]] = {s: [] for s in ARTIFACT_STATUSES}
    for art in artifacts:
        status = statuses[art.artifact_id]
        by_status[status.status].append(status)

    print("\n" + "=" * 80)
    print("EVIDENCE REFRESH SUMMARY")
    print("=" * 80)

    # Summary counts
    counts = {s: len(by_status[s]) for s in ARTIFACT_STATUSES}
    print(f"\n  CURRENT: {counts['CURRENT']:3d}  |  STALE: {counts['STALE']:3d}  |  "
          f"FAILED: {counts['FAILED']:3d}  |  NOT_RUN: {counts['NOT_RUN']:3d}")
    print(f"  TOTAL:   {len(artifacts):3d}")

    # Detail sections
    for status_name in ["STALE", "FAILED", "NOT_RUN", "CURRENT"]:
        items = by_status[status_name]
        if not items:
            continue

        print(f"\n--- {status_name} ({len(items)}) ---")
        for item in items:
            if item.status == "STALE" and item.stale_inputs:
                print(f"  {item.artifact_id}: {item.details}")
                for inp in item.stale_inputs[:5]:  # Limit to 5 for readability
                    print(f"    - {inp}")
                if len(item.stale_inputs) > 5:
                    print(f"    ... and {len(item.stale_inputs) - 5} more")
            else:
                print(f"  {item.artifact_id}: {item.details}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m evidence.refresh",
        description="Evidence refresh CLI: local-only evidence autopilot",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        default=False,
        help="Only refresh artifacts with changed inputs (default: refresh all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Refresh all artifacts (opposite of --changed-only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Determine refresh mode
    refresh_all = args.all or not args.changed_only

    start_time = time.time()
    print(f"Starting evidence refresh (mode: {'all' if refresh_all else 'changed-only'})")
    print(f"Dry run: {args.dry_run}")

    try:
        # Step 1: Scan source
        print("\n[1/9] Scanning source tree...")
        scan_result = scan_source(dry_run=args.dry_run)
        print(f"  Status: {scan_result['status']}")
        if scan_result["status"] == "OK":
            print(f"  Snapshot: {scan_result['source_snapshot_id']}")
            print(f"  Files: {scan_result['file_count']}")

        # Step 2: Calculate snapshot
        print("\n[2/9] Calculating source identity...")
        snapshot_result = calculate_snapshot(dry_run=args.dry_run)
        print(f"  Status: {snapshot_result['status']}")

        # Step 3: Load registry
        print("\n[3/9] Loading evidence registry...")
        artifacts = load_evidence_registry(dry_run=args.dry_run)
        print(f"  Loaded {len(artifacts)} artifacts")

        # Step 4: Compute input identities
        print("\n[4/9] Computing input identities...")
        identities = compute_input_identities(artifacts, dry_run=args.dry_run)
        print(f"  Computed {len(identities)} identities")

        # Step 5: Mark stale
        print("\n[5/9] Checking freshness...")
        statuses = mark_stale(artifacts, dry_run=args.dry_run)
        stale_count = sum(1 for s in statuses.values() if s.status == "STALE")
        print(f"  Stale: {stale_count}/{len(artifacts)}")

        # Step 6: Regenerate inventories (if not changed-only or if stale)
        if refresh_all or stale_count > 0:
            print("\n[6/9] Regenerating inventories...")
            statuses = regenerate_inventories(artifacts, statuses, dry_run=args.dry_run)
        else:
            print("\n[6/9] Skipping inventory regeneration (changed-only mode)")

        # Step 7: Run certifications (if not changed-only or if stale)
        if refresh_all or stale_count > 0:
            print("\n[7/9] Running certifications...")
            statuses = run_certifications(artifacts, statuses, dry_run=args.dry_run)
        else:
            print("\n[7/9] Skipping certification (changed-only mode)")

        # Step 8: Generate CURRENT.json
        print("\n[8/9] Generating CURRENT.json...")
        current_data = generate_current(artifacts, statuses, dry_run=args.dry_run)
        print(f"  Summary: {current_data['summary']}")

        # Step 9: Verify no stale production
        print("\n[9/9] Final verification gate...")
        passed, stale_production = verify_no_stale_production(
            artifacts, statuses, dry_run=args.dry_run
        )

        if passed:
            print("  PASSED: No stale production artifacts")
        else:
            print(f"  FAILED: {len(stale_production)} stale production artifact(s)")
            for item in stale_production:
                print(f"    - {item}")

        # Print summary table
        print_summary_table(artifacts, statuses)

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")

        return 0 if passed else 1

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
