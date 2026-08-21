# -*- coding: utf-8 -*-
"""R21 — LocalSourceSnapshotIdentity: Merkle-tree integrity fingerprint for the source tree.

Scans the canonical source directories, hashes each file with SHA256, and builds a
Merkle tree whose root identifies the exact source state.  The snapshot is LOCAL-ONLY
(no git, no network) and deterministic for a fixed file set.

Output: evidence/current/source_snapshot.json
Format: {"source_snapshot_id": "src:v1:<root_hex[:16]>", "merkle_root": "<hex>",
         "trees": {"backend": "...", ...}, "files": [...], "generated_at": "..."}
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVIDENCE_DIR = Path(__file__).resolve().parents[0]          # evidence/
REPO_ROOT = EVIDENCE_DIR.parent                              # quant_projects/
OUTPUT_PATH = EVIDENCE_DIR / "current" / "source_snapshot.json"

# Directories to scan (relative to REPO_ROOT)
SCAN_DIRS: tuple[str, ...] = (
    "backend",
    "cleaned_operators",
    "mining",
    "planner",
    "runtime",
    "market",
    "fields",
)

# Directory / pattern exclusions (relative to each scanned dir root)
EXCLUDE_DIRS: frozenset[str] = frozenset({
    ".git",
    "evidence",
    "docs",
    "build",
    "__pycache__",
    "logs",
    "temp",
    "data",
    # R22-EVIDENCE-CONTAMINATION: test-certified / fixture / synthetic subtrees
    # are never production evidence; recursive source scans must skip them.
    "_test_certified",
    "fixtures",
    "synthetic",
})

EXCLUDE_SUFFIXES: frozenset[str] = frozenset({".pyc", ".py.pre_lazy_opt"})

# Also skip anything inside evidence/generated or docs/generated anywhere
_EXACT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "evidence/generated",
    "docs/generated",
})


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Merkle helpers
# ---------------------------------------------------------------------------

def _merkle_pair(left: str, right: str) -> str:
    """Combine two child hashes into a parent hash (lexicographic order)."""
    combined = (left + right).encode("utf-8")
    return _sha256_bytes(combined)


def _merkle_root(hashes: list[str]) -> str:
    """Compute Merkle root over a list of leaf hashes (sorted for determinism)."""
    if not hashes:
        return _sha256_bytes(b"")  # empty tree → deterministic hash
    leaves = sorted(hashes)
    level = leaves
    while len(level) > 1:
        next_level: list[str] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                next_level.append(_merkle_pair(level[i], level[i + 1]))
            else:
                next_level.append(level[i])  # odd element promoted
        level = next_level
    return level[0]


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------

def _should_skip_dir(rel: str) -> bool:
    """True if any component of the relative path matches an excluded dir."""
    parts = Path(rel).parts
    for p in parts:
        if p in EXCLUDE_DIRS:
            return True
    # exact relative-path exclusions
    for excl in _EXACT_EXCLUDE_DIRS:
        if rel == excl or rel.startswith(excl + "/"):
            return True
    return False


@dataclass
class FileEntry:
    rel_path: str   # relative to REPO_ROOT
    sha256: str


def scan_source_tree(
    repo_root: Path | None = None,
    scan_dirs: tuple[str, ...] | None = None,
) -> list[FileEntry]:
    """Walk the source dirs and return FileEntry for every included file."""
    root = repo_root or REPO_ROOT
    dirs = scan_dirs or SCAN_DIRS
    entries: list[FileEntry] = []

    for top_dir in dirs:
        abs_top = root / top_dir
        if not abs_top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(abs_top):
            # Compute rel path from repo root for exclusion checks
            rel_dir = os.path.relpath(dirpath, root)
            # Filter dirnames in-place to avoid descending into excluded dirs
            dirnames[:] = [
                d for d in dirnames
                if not _should_skip_dir(os.path.join(rel_dir, d))
            ]
            for fname in sorted(filenames):
                # suffix check
                if any(fname.endswith(s) for s in EXCLUDE_SUFFIXES):
                    continue
                abs_file = Path(dirpath) / fname
                rel_file = os.path.relpath(abs_file, root)
                # double-check full relative path exclusions
                if _should_skip_dir(rel_file):
                    continue
                file_hash = _sha256_file(abs_file)
                entries.append(FileEntry(rel_path=rel_file, sha256=file_hash))

    # Sort for determinism
    entries.sort(key=lambda e: e.rel_path)
    return entries


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot(
    repo_root: Path | None = None,
    scan_dirs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the full snapshot dict (ready to write to JSON)."""
    root = repo_root or REPO_ROOT
    entries = scan_source_tree(root, scan_dirs)

    # Per-tree hashes: group entries by their top-level directory
    tree_buckets: dict[str, list[str]] = {d: [] for d in (scan_dirs or SCAN_DIRS)}
    for e in entries:
        top = e.rel_path.split(os.sep)[0]
        if top in tree_buckets:
            tree_buckets[top].append(e.sha256)

    tree_roots: dict[str, str] = {}
    all_leaf_hashes: list[str] = []
    for d in (scan_dirs or SCAN_DIRS):
        hashes = sorted(tree_buckets.get(d, []))
        tree_roots[d] = _merkle_root(hashes)
        all_leaf_hashes.extend(hashes)

    merkle_root = _merkle_root(all_leaf_hashes)
    short_id = merkle_root[:16]

    file_list = [{"path": e.rel_path, "sha256": e.sha256} for e in entries]

    return {
        "schema_version": 1,
        "source_snapshot_id": f"src:v1:{short_id}",
        "merkle_root": merkle_root,
        "tree_count": len(tree_roots),
        "file_count": len(entries),
        "trees": tree_roots,
        "files": file_list,
        "scan_dirs": list(scan_dirs or SCAN_DIRS),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def write_snapshot(
    out_path: Path | None = None,
    repo_root: Path | None = None,
    scan_dirs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build, write, and return the snapshot."""
    snapshot = build_snapshot(repo_root, scan_dirs)
    target = out_path or OUTPUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=False), encoding="utf-8")
    return snapshot


def load_snapshot(path: Path | None = None) -> dict[str, Any] | None:
    """Load an existing snapshot from disk; return None if missing/corrupt."""
    p = path or OUTPUT_PATH
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    snap = write_snapshot()
    print(f"source_snapshot_id = {snap['source_snapshot_id']}")
    print(f"merkle_root        = {snap['merkle_root']}")
    print(f"file_count         = {snap['file_count']}")
    for t, h in snap["trees"].items():
        print(f"  tree/{t:20s} = {h[:16]}...")
    print(f"wrote {OUTPUT_PATH}")
