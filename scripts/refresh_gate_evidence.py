#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VER-P0-01 — refresh per-gate gate evidence by ACTUALLY running pytest.

This script replaces the old hard-coded ``result: "passed" / count: N``
pattern (VER-P0-01).  Every evidence entry is produced by
``scripts/gate_runner.py``, which executes the exact pytest command via
subprocess, parses the JUnit XML, and derives status strictly:

    PASS  <=> exit_code == 0 AND junit.failures == 0 AND junit.errors == 0
              AND junit.tests > 0

No result/count in this output is written from configuration.  A gate that
cannot prove a pass stays NOT_RUN / BLOCKED / FAIL.

Output: evidence/verified_gate_evidence.json (read by
        scripts/gen_verification_manifest.py -> evidence/VerificationManifest.json)

LOCAL-ONLY: no git mutations, no network, no checkout/reset.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from gate_runner import GATE_SPECS, GateRunner, Status

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "evidence"
OUTPUT_PATH = EVIDENCE_DIR / "verified_gate_evidence.json"


def main() -> int:
    timeout = 600
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    runner = GateRunner()
    artifact = runner.build_artifact(list(GATE_SPECS), timeout=timeout)

    # Convert the raw run artifact into the manifest-consumer shape.  The
    # machine fields (command_hash / exit_code / counts / executed_at) are
    # carried verbatim from the runner — nothing is synthesized here.
    gates: list[dict] = []
    for g in artifact["gates"]:
        entries: list[dict] = []
        for e in g.get("evidence", []):
            entry = {
                "test": e["test"],
                "result": "passed" if e["status"] == Status.PASS else e["status"].lower(),
                "command_hash": e["command_hash"],
                "exit_code": e["exit_code"],
                "passed": e["passed"],
                "failed": e["failed"],
                "errors": e["errors"],
                "skipped": e["skipped"],
                "xfailed": e["xfailed"],
                "xpassed": e["xpassed"],
                "tests": e["tests"],
                "collected_tests": e["collected_tests"],
                "duration_sec": e["duration_sec"],
                "executed_at": e["executed_at"],
                "run_at": e["executed_at"],
            }
            if e.get("timed_out"):
                entry["note"] = "TIMEOUT"
            entries.append(entry)
        if entries:
            gates.append({"name": g["name"], "status": g["status"], "evidence": entries})
        else:
            gates.append({"name": g["name"], "status": g["status"], "evidence": []})

    payload = {
        "schema_version": 2,
        "generated_by": "refresh_gate_evidence",
        "git_sha": artifact.get("git_sha"),
        "timestamp": artifact.get("timestamp"),
        "gates": gates,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    statuses = ", ".join(f"{g['name']}={g['status']}" for g in gates)
    print(f"gates: {statuses or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
