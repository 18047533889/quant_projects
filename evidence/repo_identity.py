# -*- coding: utf-8 -*-
"""R46 — immutable release identity dataclasses (P0-04 / P0-D, P0-C).

* ``RepoIdentity``: single immutable type replacing the old dual-typed
  ``root_repo_sha()`` return (``dict | str``).  It is returned *always*, so
  callers can never mis-handle a string-vs-dict split.
* ``PackageTreeIdentity``: replaces the fake ``FactorEngineSubmoduleSHA`` /
  ``DataAccessSubmoduleSHA`` "submodule" payloads.  factor_engine/data_access are
  regular monorepo directories now, so the old ``git rev-parse HEAD`` inside
  them returned the PARENT HEAD — a fabricated submodule SHA.  A
  ``PackageTreeIdentity`` is a real Merkle tree hash over the package source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_identity_hash(
    head_sha: str,
    working_tree_hash: str,
    staged_hash: str,
    untracked_source_hash: str,
    dirty: bool,
) -> str:
    """Single stable hash over a RepoIdentity's fields (canonical RootRepoSHA)."""
    blob = "\x1f".join(
        [
            "RepoIdentity:v1",
            head_sha,
            working_tree_hash,
            staged_hash,
            untracked_source_hash,
            "1" if dirty else "0",
        ]
    ).encode("utf-8")
    return _sha256_bytes(blob)


@dataclass(frozen=True)
class RepoIdentity:
    """Immutable identity of the monorepo root working tree (P0-04/P0-D).

    ``head_sha`` is the commit at HEAD.  ``working_tree_hash`` /
    ``staged_hash`` / ``untracked_source_hash`` carry the working-tree delta
    digests; ``dirty`` is True iff any delta is non-empty.  ``identity_hash``
    is a single stable hash over the whole identity (used as RootRepoSHA).
    """

    head_sha: str
    dirty: bool
    working_tree_hash: str
    staged_hash: str
    untracked_source_hash: str
    identity_hash: str

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable dict (the VER-P0-03 DirtyTreeDigest schema)."""
        return {
            "HEAD": self.head_sha,
            "dirty": self.dirty,
            "diff_hash": self.working_tree_hash,
            "cached_diff_hash": self.staged_hash,
            "untracked_source_hash": self.untracked_source_hash,
            "identity_hash": self.identity_hash,
        }

    def __str__(self) -> str:
        # Back-compat: callers that treated the old dual-typed return as a bare
        # HEAD string (e.g. scripts/gen_verification_manifest.py's else branch)
        # still receive a usable git SHA without crashing.
        return self.head_sha

    @classmethod
    def from_parts(
        cls,
        head_sha: str,
        working_tree_hash: str,
        staged_hash: str,
        untracked_source_hash: str,
        dirty: bool,
    ) -> "RepoIdentity":
        return cls(
            head_sha=head_sha,
            dirty=dirty,
            working_tree_hash=working_tree_hash,
            staged_hash=staged_hash,
            untracked_source_hash=untracked_source_hash,
            identity_hash=compute_identity_hash(
                head_sha, working_tree_hash, staged_hash,
                untracked_source_hash, dirty,
            ),
        )


@dataclass(frozen=True)
class PackageTreeIdentity:
    """Merkle-tree identity of one platform package (P0-C, replaces submodule SHA).

    ``tree_hash`` is a SHA-256 Merkle root whose leaves bind
    ``relative_path || mode || size || content_sha256``, so a file rename
    changes the hash (a rename must never go undetected).  ``packaging_hash``
    binds the package's pyproject.toml (if any).  ``version`` is the package
    version if resolvable.
    """

    package_name: str
    source_root: str
    tree_hash: str
    packaging_hash: str
    version: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "package_name": self.package_name,
            "source_root": self.source_root,
            "tree_hash": self.tree_hash,
            "packaging_hash": self.packaging_hash,
            "version": self.version,
        }
