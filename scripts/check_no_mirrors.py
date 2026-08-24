#!/usr/bin/env python
"""R49 governance gate: fail when a package is mirrored across ownership
boundaries.

The known mirror pattern this gate targets is `root/<pkg>/...` vs
`factor_engine/<pkg>/...`: if a top-level directory under `factor_engine/`
(excluding intentionally dual-home dirs) also exists at the repo root AND is
tracked by git, then every file under the root copy is compared for byte
identity against the corresponding `factor_engine/` file.  Any byte-identical
file is a mirror → FAIL.

Intentionally dual-home dirs are whitelisted (they are NOT mirrors; they are
the chosen dual-homed layout):

  scripts/  tests/  examples/  docs/  evidence/  benchmarks/

Exit code: 0 = clean (no mirror), 1 = mirror detected.  Usable in CI.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FE_DIR = REPO_ROOT / "factor_engine"

# Dirs intentionally dual-homed (root = dev source, factor_engine/ = wheel
# copy for some; root = authority for evidence).  Whitelisted by R49 spec.
INTENTIONAL_DUAL_HOME = {"scripts", "tests", "evidence"}

# Top-level FE names that are not package mirrors and must never be compared:
# `.claude/` holds a small FE-side agent/skill pointer tree (docs, not code);
# `__pycache__` is generated junk.  Both are excluded.
SKIP_TOP_LEVEL = {".claude", "__pycache__"}


def tracked_files(prefix: str) -> set[Path]:
    """Return the set of repo-relative tracked paths under ``prefix``.

    Uses `git ls-files` (fast, authoritative for "is tracked").  Falls back to
    a plain directory walk when git is unavailable (e.g. a bare source tarball
    with no .git), so the gate still works in a CI checkout.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", prefix],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
        paths = {Path(p) for p in out.decode("utf-8", "replace").splitlines() if p}
        if paths:
            return paths
    except (subprocess.CalledProcessError, OSError):
        pass
    # Fallback: walk the prefix on disk (untracked files included — a mirror
    # that has not been committed yet is still a mirror we want to flag).
    base = REPO_ROOT / prefix
    return {p.relative_to(REPO_ROOT) for p in base.rglob("*") if p.is_file()} if base.exists() else set()


def is_mirror(root_rel: Path, fe_rel: Path) -> bool:
    """True when two tracked files are byte-identical mirrors."""
    return root_rel != fe_rel and filecmp.cmp(REPO_ROOT / root_rel, REPO_ROOT / fe_rel, shallow=True)


def find_mirrors() -> list[tuple[Path, Path]]:
    """Return (root_rel, fe_rel) pairs that are byte-identical mirrors.

    Strategy (kept fast): walk `factor_engine/` one level deep for package
    dirs; only for names NOT in the whitelist, check whether the repo root has
    a tracked dir of the same name; only then compare files byte-for-byte.
    """
    mirrors: list[tuple[Path, Path]] = []

    if not FE_DIR.is_dir():
        return mirrors

    fe_top_dirs = sorted(d for d in FE_DIR.iterdir() if d.is_dir() and not d.name.startswith("__"))
    for fe_dir in fe_top_dirs:
        name = fe_dir.name
        if name in INTENTIONAL_DUAL_HOME or name in SKIP_TOP_LEVEL:
            continue
        # Root copy must exist AND be tracked for the mirror pattern to apply.
        root_dir = REPO_ROOT / name
        if not root_dir.is_dir():
            continue
        tracked_root = {p for p in tracked_files(name) if (REPO_ROOT / p).is_file()}
        if not tracked_root:
            continue

        fe_tracked = {p for p in tracked_files(f"factor_engine/{name}")}

        # Compare every tracked file under the root dir against its FE
        # counterpart when the counterpart exists.
        for root_file in sorted(tracked_root):
            fe_counterpart = Path("factor_engine") / root_file
            if fe_counterpart not in fe_tracked:
                continue
            if is_mirror(root_file, fe_counterpart):
                mirrors.append((root_file, fe_counterpart))

    return mirrors


def main() -> int:
    mirrors = find_mirrors()
    if not mirrors:
        print("OK: no duplicate mirror packages")
        return 0
    for root_rel, fe_rel in mirrors:
        print(f"FAIL: duplicate mirror detected: {root_rel} (root/{root_rel.parent} == factor_engine/{fe_rel.parent})")
    print(f"{len(mirrors)} mirror file(s) found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
