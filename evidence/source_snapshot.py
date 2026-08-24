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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVIDENCE_DIR = Path(__file__).resolve().parents[0]          # evidence/
REPO_ROOT = EVIDENCE_DIR.parent                              # quant_projects/
OUTPUT_PATH = EVIDENCE_DIR / "current" / "source_snapshot.json"

# Directories to scan (relative to REPO_ROOT).
# R46 P0-02: coverage now spans the FULL platform source tree, not just the
# legacy FactorEngine dirs.  quant_evaluator / factor_optimizer / factor_assets /
# factor_preprocess / dataaccess / vectorbt_qs / factor_engine are all regular
# monorepo packages (no submodules) and must be inside the evidence snapshot.
SCAN_DIRS: tuple[str, ...] = (
    "backend",
    "cleaned_operators",
    "mining",
    "planner",
    "runtime",
    "market",
    "fields",
    "factor_engine",
    "quant_evaluator",
    "factor_optimizer",
    "factor_assets",
    "factor_preprocess",
    "dataaccess",
    "vectorbt_qs",
    "alphaprobe",
    "data_access",
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
    mode: int = 0o644        # file mode bits (part of the leaf identity)
    size: int = 0            # byte size (part of the leaf identity)

    def leaf_hash(self) -> str:
        """R46 P0-02: leaf binds relative_path || mode || size || content_sha256.

        The path is part of the leaf, so ``a/foo.py -> b/foo.py`` rename changes
        the Merkle root even when content is byte-identical.  Length-prefixed
        and collision-safe like the other canonical encodings in this repo.
        """
        h = hashlib.sha256()
        h.update(len(self.rel_path.encode("utf-8")).to_bytes(8, "big"))
        h.update(self.rel_path.encode("utf-8"))
        h.update(b"\x00")
        h.update(len(str(self.mode).encode("utf-8")).to_bytes(8, "big"))
        h.update(str(self.mode).encode("utf-8"))
        h.update(b"\x00")
        h.update(len(str(self.size).encode("utf-8")).to_bytes(8, "big"))
        h.update(str(self.size).encode("utf-8"))
        h.update(b"\x00")
        h.update(self.sha256.encode("utf-8"))
        return h.hexdigest()


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
                st = abs_file.stat()
                entries.append(FileEntry(
                    rel_path=rel_file,
                    sha256=file_hash,
                    mode=st.st_mode,
                    size=st.st_size,
                ))

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
            # R46 P0-02: Merkle leaves bind path+mode+size+content so a rename
            # of a byte-identical file still changes the tree hash.
            tree_buckets[top].append(e.leaf_hash())

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
# Platform-wide SourceTreeIdentity (R46 P0-02, P0-C)
# ---------------------------------------------------------------------------

# Regular monorepo packages (all real dirs, NOT submodules).
PLATFORM_PACKAGES: tuple[str, ...] = (
    "factor_engine",
    "quant_evaluator",
    "factor_optimizer",
    "factor_assets",
    "factor_preprocess",
    "dataaccess",
    "vectorbt_qs",
    # P0-AP: AlphaProbe (independent project) is part of the platform source
    # identity.  R26 P0-ALPHA: alphaprobe is scanned for real source only (its
    # pdm.lock / pyproject / env / makefile are excluded from the tree hash —
    # see _ALPHA_EXCLUDE_NAMES).  data_access (the repository-local namespace
    # shim forwarding to dataaccess/) is a platform import boundary and is
    # listed too.
    "alphaprobe",
    "data_access",
)

# P0-ALPHA: alphaprobe is a vendored independent project; its dependency lock /
# packaging / template files are large and pinned to an external env, NOT
# production source.  Excluding them keeps the platform Merkle identity stable
# to REAL Alpha source code.  These file names are excluded from any package
# tree hash (alphaprobe specifically).
_ALPHA_EXCLUDE_NAMES: frozenset[str] = frozenset({
    "pdm.lock",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "makefile",
})

# Dirs that must never be treated as production source even if tracked.
_PLATFORM_EXCLUDE_SUFFIXES: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".a", ".o", ".parquet", ".parq", ".feather",
    ".pickle", ".pkl", ".npy", ".npz",
})


def _package_tree_merkle(pkg_dir: Path) -> str:
    """SHA-256 Merkle root over a package's tracked source files.

    Each leaf = hash(relative_path || mode || size || content_sha256).  Uses the
    platform exclusion set (no evidence/docs/build descent, no generated/binary
    suffixes).  Deterministic for a fixed file set.

    P0-ALPHA: for the vendored alphaprobe project, dependency/packaging/template
    files (pdm.lock, pyproject.toml, .env.example, .gitignore, makefile) are
    EXCLUDED — they are pinned to an external env, not Alpha generation source,
    so editing them must not perturb the platform source identity.  Real Alpha
    source (*.py, configs, prompts, services) is still hashed.
    """
    exclude_names = _ALPHA_EXCLUDE_NAMES if pkg_dir.name == "alphaprobe" else frozenset()
    leaves: list[str] = []
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in EXCLUDE_DIRS and d != ".git"
        )
        rel_dir = os.path.relpath(dirpath, pkg_dir)
        for fname in sorted(filenames):
            if fname in exclude_names:
                continue
            if any(fname.endswith(s) for s in _PLATFORM_EXCLUDE_SUFFIXES):
                continue
            if fname in EXCLUDE_SUFFIXES or fname.endswith(".py.pre_lazy_opt"):
                continue
            abs_file = Path(dirpath) / fname
            if not abs_file.is_file():
                continue
            try:
                st = abs_file.stat()
            except OSError:
                continue
            rel = Path(rel_dir) / fname
            rel_posix = rel.as_posix()
            if rel_posix == ".git" or rel_posix.startswith(".git/"):
                continue
            content = _sha256_file(abs_file)
            leaf = hashlib.sha256()
            leaf.update(len(rel_posix.encode("utf-8")).to_bytes(8, "big"))
            leaf.update(rel_posix.encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(len(str(st.st_mode).encode("utf-8")).to_bytes(8, "big"))
            leaf.update(str(st.st_mode).encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(len(str(st.st_size).encode("utf-8")).to_bytes(8, "big"))
            leaf.update(str(st.st_size).encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(content.encode("utf-8"))
            leaves.append(leaf.hexdigest())
    return _merkle_root(leaves)


def _single_file_hash(rel: str, root: Path) -> str:
    """SHA-256 of one packaging/dependency file ("" if missing)."""
    p = root / rel
    if not p.is_file():
        return ""
    return _sha256_file(p)


def _config_dir_merkle(root: Path) -> str:
    """SHA-256 Merkle over all tracked files under ``config/`` (production conf)."""
    cfg = root / "config"
    if not cfg.is_dir():
        return _merkle_root([])
    leaves: list[str] = []
    for dirpath, dirnames, filenames in os.walk(cfg):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        rel_dir = os.path.relpath(dirpath, cfg)
        for fname in sorted(filenames):
            abs_file = Path(dirpath) / fname
            if not abs_file.is_file():
                continue
            rel = Path(rel_dir) / fname
            rel_posix = rel.as_posix()
            try:
                st = abs_file.stat()
            except OSError:
                continue
            content = _sha256_file(abs_file)
            leaf = hashlib.sha256()
            leaf.update(len(rel_posix.encode("utf-8")).to_bytes(8, "big"))
            leaf.update(rel_posix.encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(str(st.st_mode).encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(str(st.st_size).encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(content.encode("utf-8"))
            leaves.append(leaf.hexdigest())
    return _merkle_root(leaves)


def platform_source_tree_identity(
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Platform-wide SourceTreeIdentity (R46 P0-02, P0-C).

    Merkle-style root over the ENTIRE platform source:
      * per-package tree hash for each regular package (factor_engine,
        quant_evaluator, factor_optimizer, factor_assets, factor_preprocess,
        dataaccess, vectorbt_qs);
      * root orchestration source files (*.py, *.sh at the monorepo root);
      * packaging_identity  = Merkle over all pyproject.toml files;
      * dependency_lock_identity = SHA-256 over the production dependency lock
        (requirements-production.lock at root and factor_engine/);
      * config_identity    = Merkle over the config/ dir;
      * root_merkle        = Merkle over every leaf so a change anywhere is
                             detected even if no single bucket changes alone.

    Excludes generated/evidence/output/reports/backtest dirs (not production
    source).  Each package leaf binds relative_path || mode || size || content,
    so a rename changes the hash.
    """
    root = repo_root or REPO_ROOT
    package_trees: dict[str, str] = {}
    package_packaging: dict[str, str] = {}
    all_leaves: list[str] = []

    for pkg in PLATFORM_PACKAGES:
        pkg_dir = root / pkg
        if not pkg_dir.is_dir():
            continue
        tree_hash = _package_tree_merkle(pkg_dir)
        package_trees[pkg] = tree_hash
        all_leaves.append(
            hashlib.sha256(
                (f"pkg:{pkg}\x00" + tree_hash).encode("utf-8")
            ).hexdigest()
        )
        pkg_pp = _single_file_hash(f"{pkg}/pyproject.toml", root)
        if pkg_pp:
            package_packaging[pkg] = pkg_pp

    # Root orchestration source files (tracked *.py / *.sh at monorepo root).
    root_src: list[str] = []
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", "*.py", "*.sh"],
            cwd=str(root), capture_output=True, text=True, timeout=30,
        )
        for line in sorted(out.stdout.splitlines()):
            rel = line.strip()
            if "/" in rel or not rel:
                continue
            f = root / rel
            if not f.is_file():
                continue
            if _is_non_source(rel):
                continue
            root_src.append(_single_file_hash(rel, root))
    except Exception:
        pass
    if root_src:
        for leaf in root_src:
            all_leaves.append(
                hashlib.sha256(("root-src\x00" + leaf).encode("utf-8")).hexdigest()
            )

    # Packaging config: all pyproject.toml under the platform packages + root.
    pyprojects: list[str] = []
    for rel in ("pyproject.toml",) + tuple(
        f"{p}/pyproject.toml" for p in PLATFORM_PACKAGES
    ):
        h = _single_file_hash(rel, root)
        if h:
            pyprojects.append(h)
    packaging_identity = _merkle_root(pyprojects) if pyprojects else _merkle_root([])
    all_leaves.append(
        hashlib.sha256(("packaging\x00" + packaging_identity).encode("utf-8")).hexdigest()
    )

    # Dependency lock identity (R26 P0-LOCK: SINGLE lock authority — root only).
    dep_leaves: list[str] = []
    for rel in ("requirements-production.lock",):
        h = _single_file_hash(rel, root)
        if h:
            dep_leaves.append(h)
    dependency_lock_identity = _merkle_root(dep_leaves) if dep_leaves else _merkle_root([])
    all_leaves.append(
        hashlib.sha256(("dep-lock\x00" + dependency_lock_identity).encode("utf-8")).hexdigest()
    )

    # Config dir identity.
    config_identity = _config_dir_merkle(root)
    all_leaves.append(
        hashlib.sha256(("config\x00" + config_identity).encode("utf-8")).hexdigest()
    )

    root_merkle = _merkle_root(all_leaves)

    return {
        "schema_version": 1,
        "platform_source_tree_identity": root_merkle,
        "root_merkle": root_merkle,
        "packages": package_trees,
        "package_packaging": package_packaging,
        "root_source_count": len(root_src),
        "packaging_identity": packaging_identity,
        "dependency_lock_identity": dependency_lock_identity,
        "config_identity": config_identity,
    }


def _is_non_source(rel: str) -> bool:
    for excl in ("evidence", "build", "docs", "archives"):
        if rel == excl or rel.startswith(excl + "/"):
            return True
    return False


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
