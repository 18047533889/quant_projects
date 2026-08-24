# -*- coding: utf-8 -*-
"""R26 P0-ALPHA — AlphaGeneratorIdentity.

AlphaProbe (``alphaprobe/ashare/alphaprobe-dev``) is an independent generator
project that sits outside the normal platform package tree.  Changing its
generation logic (model, prompts, configs, source) must be detectable so gate
evidence cannot silently match stale Alpha output.

:class:`AlphaGeneratorIdentity` binds the generator's real source and its
declared environment in one stable digest:

    generator_name       — human name ("AlphaProbe")
    source_tree_hash     — SHA-256 Merkle root over Alpha's REAL source tree
                           (same exclusion set as the platform snapshot: no
                           pdm.lock / pyproject / env / makefile)
    environment_lock_hash— SHA-256 over the AlphaProbe pdm.lock (the pinned
                           generator env)
    model_identity       — stable hash of the generator model/runner source
                           (under ``alphaprobe/.../src/alphaprobe/``)
    prompt_policy_hash   — hash of the prompt policy files (configs/prompts/)
    version              — AlphaProbe project version (from pyproject, if any)
    content_hash         — canonical SHA-256 over the fields above

This module is LOCAL-ONLY (no git mutations, no network).  Alpha source is
scanned on the filesystem; a missing/corrupt AlphaProbe tree yields a
deterministic ``"<field> MISSING"`` digest (never a fabricated PASS).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

_EVIDENCE_DIR = Path(__file__).resolve().parents[0]
_REPO_ROOT = _EVIDENCE_DIR.parent

# AlphaProbe source root (relative to repo root).
ALPHA_ROOT_REL = "alphaprobe/ashare/alphaprobe-dev"

# Real generation-source subpaths (model code, configs, prompts, delivery,
# deployment services) that define the generator's behaviour.  Everything else
# under AlphaProbe (figures, lockfiles, env templates, makefile) is NOT
# generator logic.
_ALPHA_MODEL_REL = "src/alphaprobe"
_ALPHA_PROMPT_REL = "configs/prompts"
_ALPHA_ENV_LOCK_REL = "pdm.lock"
_ALPHA_PYPROJECT_REL = "pyproject.toml"

# File names excluded from the Alpha SOURCE tree hash (same set as the
# platform snapshot: dependency lock / packaging / env / template).
_ALPHA_EXCLUDE_NAMES = frozenset({
    "pdm.lock",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "makefile",
})

_EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif"})


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _merkle_root(hashes: list[str]) -> str:
    """Deterministic Merkle root over sorted leaf hashes (empty -> sha256(b""))."""
    import hashlib

    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    leaves = sorted(hashes)

    def _pair(left: str, right: str) -> str:
        return hashlib.sha256((left + right).encode("utf-8")).hexdigest()

    level = leaves
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_pair(level[i], level[i + 1]))
            else:
                nxt.append(level[i])
        level = nxt
    return level[0]


def _tree_hash_merkle(root: Path, sub: str) -> str:
    """SHA-256 Merkle root over the source files under ``root/sub``.

    Leaf = path || mode || size || content, with the Alpha exclusion set and
    binary/generated suffix exclusions.  Empty/missing dir -> sha256("").
    """
    leaves: list[str] = []
    d = root / sub
    if not d.is_dir():
        return _merkle_root([])
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = sorted(
            dn for dn in dirnames
            if dn != ".git" and dn != "__pycache__"
        )
        rel_dir = os.path.relpath(dirpath, d)
        for fname in sorted(filenames):
            if fname in _ALPHA_EXCLUDE_NAMES:
                continue
            if any(fname.endswith(s) for s in _EXCLUDE_SUFFIXES):
                continue
            abs_file = Path(dirpath) / fname
            if not abs_file.is_file():
                continue
            try:
                st = abs_file.stat()
            except OSError:
                continue
            rel = (Path(rel_dir) / fname).as_posix()
            content = _sha256_file(abs_file)
            leaf = hashlib.sha256()
            leaf.update(len(rel.encode("utf-8")).to_bytes(8, "big"))
            leaf.update(rel.encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(str(st.st_mode).encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(str(st.st_size).encode("utf-8"))
            leaf.update(b"\x00")
            leaf.update(content.encode("utf-8"))
            leaves.append(leaf.hexdigest())
    return _merkle_root(leaves)


def _version(alpha_root: Path) -> str:
    pyproject = alpha_root / _ALPHA_PYPROJECT_REL
    if not pyproject.is_file():
        return ""
    try:
        text = pyproject.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("version") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


@dataclass(frozen=True)
class AlphaGeneratorIdentity:
    """Immutable identity of the AlphaProbe generator (R26 P0-ALPHA)."""

    generator_name: str
    source_tree_hash: str
    environment_lock_hash: str
    model_identity: str
    prompt_policy_hash: str
    version: str
    content_hash: str = field(default="")

    @classmethod
    def from_parts(
        cls,
        generator_name: str,
        source_tree_hash: str,
        environment_lock_hash: str,
        model_identity: str,
        prompt_policy_hash: str,
        version: str,
    ) -> "AlphaGeneratorIdentity":
        blob = "\x1f".join([
            "AlphaGeneratorIdentity:v1",
            generator_name,
            source_tree_hash,
            environment_lock_hash,
            model_identity,
            prompt_policy_hash,
            version,
        ]).encode("utf-8")
        return cls(
            generator_name=generator_name,
            source_tree_hash=source_tree_hash,
            environment_lock_hash=environment_lock_hash,
            model_identity=model_identity,
            prompt_policy_hash=prompt_policy_hash,
            version=version,
            content_hash=_sha256_bytes(blob),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "generator_name": self.generator_name,
            "source_tree_hash": self.source_tree_hash,
            "environment_lock_hash": self.environment_lock_hash,
            "model_identity": self.model_identity,
            "prompt_policy_hash": self.prompt_policy_hash,
            "version": self.version,
            "content_hash": self.content_hash,
        }


def alpha_generator_identity(
    repo_root: Path | None = None,
) -> AlphaGeneratorIdentity:
    """Compute the live AlphaGeneratorIdentity for the current tree.

    Scans real Alpha source on the filesystem.  Any sub-identity that cannot
    be resolved is a deterministic ``MISSING:...`` digest (fail-closed) — the
    content_hash never silently collapses on missing source.
    """
    root = repo_root or _REPO_ROOT
    alpha_root = root / ALPHA_ROOT_REL
    if not alpha_root.is_dir():
        source_tree = _merkle_root([])
        env_lock = _missing("pdm.lock")
        model = _missing("src/alphaprobe")
        prompts = _missing("configs/prompts")
        version = ""
    else:
        source_tree = _tree_hash_merkle(alpha_root, ".")
        env_lock_path = alpha_root / _ALPHA_ENV_LOCK_REL
        env_lock = _sha256_file(env_lock_path) if env_lock_path.is_file() else _missing("pdm.lock")
        model = _tree_hash_merkle(alpha_root, _ALPHA_MODEL_REL)
        prompts = _tree_hash_merkle(alpha_root, _ALPHA_PROMPT_REL)
        version = _version(alpha_root)
    return AlphaGeneratorIdentity.from_parts(
        generator_name="AlphaProbe",
        source_tree_hash=source_tree,
        environment_lock_hash=env_lock,
        model_identity=model,
        prompt_policy_hash=prompts,
        version=version,
    )


def _missing(what: str) -> str:
    return "MISSING:" + _sha256_bytes(what.encode("utf-8"))
