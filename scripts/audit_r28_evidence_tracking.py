#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R28 §九十一..九十二 / §一百五十八: evidence must be GitHub-visible and tracked.

- No expected R28 artifact may be gitignored (``git check-ignore``).
- Every expected artifact must be tracked (``git ls-files --error-unmatch``).
- Evidence git_sha must equal current HEAD; canonical digest must match.

Run:  python3 scripts/audit_r28_evidence_tracking.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evidence" / "factor_engine" / "r28"

EXPECTED = [
    "R28_OPERATOR_INVENTORY.csv",
    "R28_OPERATOR_INVENTORY.json",
    "R28_CANONICAL_TEST_COVERAGE.csv",
    "R28_CANONICAL_TEST_COVERAGE.json",
    "R28_PER_CANONICAL_TEST_RESULTS.csv",
    "R28_PER_CANONICAL_TEST_RESULTS.json",
    "R28_FORBIDDEN_OPERATOR_AUDIT.json",
    "R28_STATIC_LOOKAHEAD_SCAN.csv",
    "R28_STATIC_LOOKAHEAD_SCAN.json",
    "R28_MODEL_CAUSALITY_MATRIX.csv",
    "R28_MODEL_CAUSALITY_MATRIX.json",
    "R28_PYTEST_SUMMARY.json",
    "R28_ARTIFACT_MANIFEST.json",
    "R28_FINAL_ACCEPTANCE_REPORT.md",
    "R28_CANONICAL_SET_DIGEST.txt",
]


def git(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *cmd], capture_output=True, text=True)


def _only_evidence_commits_since(manifest_sha: str) -> bool:
    """True when every file changed between the manifest code sha and HEAD is an
    evidence artifact (no code changed after the evidence was generated)."""
    out = git(["diff", "--name-only", manifest_sha, "HEAD"]).stdout.strip()
    if not out:
        return False
    return all("/evidence/factor_engine/r28/" in line for line in out.splitlines())


def main() -> int:
    problems = []
    for name in EXPECTED:
        p = OUT / name
        if not p.exists():
            problems.append(f"missing: {name}")
            continue
        rel = p.relative_to(REPO).as_posix()
        ignored = git(["check-ignore", rel])
        if ignored.returncode == 0:
            problems.append(f"gitignored: {name}")
        tracked = git(["ls-files", "--error-unmatch", rel])
        if tracked.returncode != 0:
            problems.append(f"not tracked: {name}")

    sha = git(["rev-parse", "HEAD"]).stdout.strip()
    manifest_path = OUT / "R28_ARTIFACT_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest_sha = manifest.get("git_sha")
        # The evidence is bound to the CODE sha that produced the tests.  The
        # evidence commit itself only touches evidence/factor_engine/r28; if HEAD is that
        # evidence commit, its parent is the code sha and the evidence is current.
        if manifest_sha == sha:
            ok_sha = True
        else:
            ok_sha = _only_evidence_commits_since(manifest_sha)
        if not ok_sha:
            problems.append(
                f"evidence git_sha {manifest_sha} != HEAD {sha} (STALE)"
            )
        digest = OUT / "R28_CANONICAL_SET_DIGEST.txt"
        if digest.exists():
            expected_digest = ""
            for line in digest.read_text().splitlines():
                if line.startswith("canonical_set_digest="):
                    expected_digest = line.split("=", 1)[1]
            if manifest.get("canonical_set_digest") != expected_digest:
                problems.append("evidence canonical digest != current digest (STALE)")

    ok = not problems
    print(f"R28 evidence tracking: git_sha={sha}")
    for p in problems:
        print(f"  !! {p}")
    if ok:
        print("  R28_EVIDENCE_FILES_NOT_GITIGNORED == TRUE")
        print("  R28_TEST_RESULTS_GITHUB_TRACKED == TRUE")
        print("  R28_EVIDENCE_GIT_SHA_CURRENT == TRUE")
        return 0
    print("  evidence-tracking gates FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
