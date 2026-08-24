#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REL-P0-01 — Release Verification Manifest generator.

The repo historically accepted "X tests passed" as release truth. This script
produces an automated VerificationManifest that captures the exact source
snapshot + executed commands + results for a release gate review.

Output: evidence/VerificationManifest.json

Captures per run:
  - local exact git SHA (parent repo) + dirty state; submodule SHA where present
  - per-package source-tree SHA256 digests:
        factor_engine, quant_evaluator, factor_optimizer,
        factor_assets, factor_preprocess
  - wheel sha256 if a wheel exists for the package (built dist/*.whl)
  - environment hash (python version, key dependency versions)
  - executed commands (argv, exit code, cwd, runtime, output tail)
  - pytest counts (passed / failed / skipped / xfailed / errors) if a run was
    performed and a junit xml or pytest exit info was provided
  - benchmark summaries
  - the 14 release gates, each defaulting to NOT_RUN unless a check updates it
  - stale-artifact note about .pytest_full.txt if present

Gate statuses are read from config/gates.json.  Checks that prove a gate may
upgrade it to PASS / FAIL / BLOCKED; anything unproven stays NOT_RUN.  This
script is LOCAL-ONLY: no git mutations, no network, no checkout/reset.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVIDENCE_DIR = REPO_ROOT / "evidence"
OUTPUT_PATH = EVIDENCE_DIR / "VerificationManifest.json"
GATES_CONFIG_PATH = REPO_ROOT / "config" / "gates.json"

# Packages that must each carry a source-tree hash in the manifest.
PACKAGES: tuple[str, ...] = (
    "factor_engine",
    "quant_evaluator",
    "factor_optimizer",
    "factor_assets",
    "factor_preprocess",
    "dataaccess",
)

# Optional: JSON file of pre-verified per-gate evidence that a re-run should
# fold into the manifest (see scripts/refresh_gate_evidence.py).  Absent => no
# evidence injection (all gates stay NOT_RUN).
VERIFIED_EVIDENCE_PATH = EVIDENCE_DIR / "verified_gate_evidence.json"

# NOTE: the concept of a "provable set" has been removed.  Every gate with
# honest machine-verified evidence (command_hash + exit_code==0 + failed==0
# + errors==0 + tests>0) is upgraded to PASS.  Hard-coding a subset of gates
# as "provable" was a mistake: it excluded LEAKAGE, PIT, DETERMINISM,
# FRESH_WHEEL, 1K_SCALE, 10K_SCALE, 100K_SCALE from ever being promoted even
# when the evidence file proved them.  The evidence validation below is the
# only gatekeeper — it never fabricates PASS.

# Top-level dirs ignored when hashing a package source tree (never release code).
_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset({
    ".git", "__pycache__", ".pytest_cache", "dist", "build",
    "*.egg-info", "node_modules", ".mypy_cache", ".ruff_cache",
    "logs", "data", ".venv", "venv",
})
# Note: egg-info matching is done by prefix, see _is_excluded_dir.

# Key dependencies captured in the environment hash.
_ENV_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "polars", "pyarrow", "scipy", "pytest",
    "duckdb", "factor_engine", "dataaccess",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> dict:
    """Run a command, capture exit code / duration / tail of stdout+stderr."""
    started = time.time()
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "exit_code": proc.returncode,
        "duration_sec": round(time.time() - started, 3),
        "output_tail": lines[-40:],
        "output_bytes": len(out),
    }


def _git_sha(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(path),
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(path),
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return False


def _is_excluded_dir(name: str) -> bool:
    if name in _EXCLUDE_DIR_NAMES:
        return True
    return name.endswith(".egg-info") or name.endswith(".dist-info")


def _dist_version(name: str) -> str:
    """Installed version of a package, with a fallback for dist-name mismatches.

    ``dataaccess`` is shipped as the dist ``data-access`` (importable as
    ``dataaccess``), so a bare importlib.metadata lookup would report
    NOT_INSTALLED even when installed.  R23: resolve the ``data-access`` dist
    so the manifest pins the installed dataaccess release.
    """
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        pass
    if name == "dataaccess":
        try:
            import importlib.metadata as md
            return md.version("data-access")
        except Exception:
            return "NOT_INSTALLED"
    return "NOT_INSTALLED"


@dataclass(frozen=True)
class ReleaseIdentity:
    """Formal release identity (VER-P0-06).

    Fields mirror ``evidence.current._dirty_tree_digest()``:
      - ``git_sha``                root repo HEAD SHA (or None when not a git checkout)
      - ``dirty``                  working tree has uncommitted changes
      - ``diff_hash``              sha256 of ``git diff`` (unstaged) when dirty
      - ``cached_diff_hash``       sha256 of ``git diff --cached`` when dirty
      - ``untracked_source_hash``  digest of untracked non-ignored source files
      - ``gitlinks``               path -> blob SHA for real git submodules
                                   (git index entries with mode 160000); empty
                                   dict when the repo declares none.

    ``submodule_shas`` in the manifest MUST be populated ONLY from ``gitlinks``
    so the manifest can never fabricate submodule SHAs (this repo has no
    submodules; the old manifest recorded fabricated SHAs for factor_engine /
    dataaccess — that stops here).
    """

    git_sha: str | None = None
    dirty: bool = False
    diff_hash: str | None = None
    cached_diff_hash: str | None = None
    untracked_source_hash: str | None = None
    gitlinks: dict[str, str] = field(default_factory=dict)


def _gitlinks_from_index(root: Path) -> dict[str, str]:
    """Real gitlinks: index entries with mode ``160000`` -> path -> blob SHA.

    ``git ls-files --stage`` emits one line per index entry of the form::

        160000 <blob_sha> 0\t<path>

    Any path registered that way is a REAL submodule pin in this repository.
    No ``160000`` entries => no submodules => empty dict (nothing pinned).
    """
    gitlinks: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["git", "ls-files", "--stage"],
            cwd=str(root), capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                parts = line.split()
                if not parts or parts[0] != "160000":
                    continue
                if len(parts) >= 2:
                    path = parts[-1]
                    gitlinks[path] = parts[1]
    except Exception:
        pass
    return gitlinks


def _release_identity() -> ReleaseIdentity:
    """Single source of truth for the root repo release identity.

    VER-P0-06: delegates to ``evidence.current.root_repo_sha()``.  That API
    returns a plain str for a clean tree, or a dict (HEAD/dirty/diff_hash/
    cached_diff_hash/untracked_source_hash) for a dirty tree.  Both return
    types are handled explicitly — the old code did ``root.split(" ")[0]`` /
    ``root.endswith(...)`` string guessing, which the coordinator flagged as
    fabricated.  Submodule SHAs come ONLY from real git index gitlinks; if
    ``evidence.current`` is unavailable the identity degrades to direct git
    calls with the same schema.
    """
    try:
        from evidence.current import root_repo_sha
        root = root_repo_sha()
        if isinstance(root, dict):
            identity = ReleaseIdentity(
                git_sha=root.get("HEAD") or root.get("git_sha"),
                dirty=bool(root.get("dirty")),
                diff_hash=root.get("diff_hash"),
                cached_diff_hash=root.get("cached_diff_hash"),
                untracked_source_hash=root.get("untracked_source_hash"),
                gitlinks=_gitlinks_from_index(REPO_ROOT),
            )
        else:
            identity = ReleaseIdentity(
                git_sha=str(root) if root else None,
                dirty=False,
                gitlinks=_gitlinks_from_index(REPO_ROOT),
            )
        return identity
    except Exception as exc:  # pragma: no cover - degraded fallback
        print(
            f"warning: evidence.current identity unavailable ({exc}); "
            "falling back to local git calls",
            file=sys.stderr,
        )
        return ReleaseIdentity(
            git_sha=_git_sha(REPO_ROOT),
            dirty=_git_dirty(REPO_ROOT),
            gitlinks=_gitlinks_from_index(REPO_ROOT),
        )


def package_tree_hash(pkg: str) -> str | None:
    """SHA256 over a stable serialization of the package source tree.

    Walks the package dir, skipping exclusions, and hashes relative
    path + byte length + sha256 of each regular file.  Deterministic for a
    fixed file set.  Returns None if the package dir does not exist.
    """
    root = REPO_ROOT / pkg
    if not root.is_dir():
        return None
    parts: list[bytes] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not _is_excluded_dir(d)
        )
        rel_dir = Path(dirpath).relative_to(root)
        for fname in sorted(filenames):
            if fname.endswith((".pyc", ".pyo")):
                continue
            fpath = Path(dirpath) / fname
            if not fpath.is_file():
                continue
            rel = (rel_dir / fname).as_posix()
            parts.append(f"{rel}\0{os.path.getsize(fpath)}\0".encode())
            parts.append(_file_sha256(fpath).encode())
    if not parts:
        return None
    return hashlib.sha256(b"\n".join(parts)).hexdigest()


def package_wheel_sha256(pkg: str) -> dict | None:
    """First dist/*.whl found for a package -> {path, sha256}."""
    dist_dir = REPO_ROOT / pkg / "dist"
    if not dist_dir.is_dir():
        return None
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        return None
    w = wheels[0]
    return {"wheel": w.name, "sha256": _file_sha256(w)}


def env_hash() -> dict:
    py = platform.python_version()
    versions: dict[str, str] = {}
    for name in _ENV_PACKAGES:
        versions[name] = _dist_version(name)
    blob = json.dumps({"python": py, "packages": versions}, sort_keys=True)
    return {
        "python_version": py,
        "platform": platform.platform(),
        "dependencies": versions,
        "hash": _sha256_bytes(blob.encode()),
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def load_gates() -> list[dict]:
    cfg: dict = {}
    if GATES_CONFIG_PATH.is_file():
        try:
            cfg = json.loads(GATES_CONFIG_PATH.read_text())
        except Exception as exc:  # pragma: no cover - config repair path
            print(f"warning: cannot parse {GATES_CONFIG_PATH}: {exc}", file=sys.stderr)
    raw = cfg.get("gates", [])
    gates: list[dict] = []
    for g in raw:
        name = g.get("name")
        if not name:
            continue
        gates.append({
            "name": name,
            "description": g.get("description", ""),
            "status": g.get("default_status", "NOT_RUN"),
            "evidence": [],
        })
    # Guard: every gate must exist even if config is missing/incomplete.
    known = {g["name"] for g in gates}
    all_gates = (
        "UNIT", "NUMERICAL_ORACLE", "PROPERTY", "CROSS_PACKAGE",
        "LEAKAGE", "PIT", "DETERMINISM", "SERIALIZATION",
        "CHECKPOINT_RESUME", "FRESH_WHEEL", "1K_SCALE", "10K_SCALE",
        "100K_SCALE", "ASHARE_SEMANTIC_CONTRACT_GOLDEN",
        "ASHARE_REAL_DATA_SHADOW",
        # VER-P0-05 hard/structural gates (default NOT_RUN).
        "SOURCE_AUTHORITY", "SUPPLY_CHAIN", "SUBMODULE_REACHABILITY",
        "FRESH_WHEEL_MATRIX", "EVIDENCE_CURRENT",
    )
    for name in all_gates:
        if name not in known:
            gates.append({
                "name": name,
                "description": "",
                "status": "NOT_RUN",
                "evidence": [],
            })
    return gates


def _gate(gates: list[dict], name: str) -> dict:
    for g in gates:
        if g["name"] == name:
            return g
    raise KeyError(name)


def _set_gate(gates: list[dict], name: str, status: str, evidence: str) -> None:
    g = _gate(gates, name)
    # BLOCKED is sticky; PASS/FAIL downgrade from NOT_RUN is allowed.
    if g["status"] == "BLOCKED":
        return
    if status in ("PASS", "FAIL", "BLOCKED"):
        g["status"] = status
    g["evidence"].append({"ts": datetime.datetime.now().isoformat(), "note": evidence})


def _load_verified_evidence() -> list[dict] | None:
    """Read scripts/refresh_gate_evidence.py output (per-gate evidence).

    Expected shape (see scripts/gate_runner.py + scripts/refresh_gate_evidence.py):
        {
          "schema_version": 2,
          "generated_by": "refresh_gate_evidence",
          "git_sha": "...",
          "timestamp": "...",
          "gates": [
            {
              "name": "CROSS_PACKAGE",
              "status": "PASS",
              "evidence": [{
                "test": "...",
                "command_hash": "<sha256 of argv>",
                "exit_code": 0,
                "passed": 6, "failed": 0, "errors": 0, "skipped": 0,
                "tests": 6, "duration_sec": 1.2, "executed_at": "...",
                "run_at": "...",
              }]
            }
          ]
        }

    NOTE (VER-P0-01): the per-gate ``status`` field in this file is NOT trusted.
    A gate is upgraded to PASS here only when every evidence item carries real
    machine fields (command_hash + exit_code == 0 + failed == 0 + errors == 0 +
    tests > 0).  Any entry missing those fields is treated as NOT_RUN — never
    PASS.

    Returns a list of gate entries, or None if the file is absent/unparsable.
    """
    if not VERIFIED_EVIDENCE_PATH.is_file():
        return None
    try:
        data = json.loads(VERIFIED_EVIDENCE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - evidence is non-blocking
        print(f"warning: cannot parse {VERIFIED_EVIDENCE_PATH}: {exc}", file=sys.stderr)
        return None
    gates = data.get("gates", []) if isinstance(data, dict) else []
    return [g for g in gates if isinstance(g, dict) and g.get("name")]


# ---------------------------------------------------------------------------
# VER-P0-02 — exact-identity gate aggregation
# ---------------------------------------------------------------------------
def _build_gate_expectations() -> tuple[dict[str, set[str]], dict[str, str], dict[str, set[str]]]:
    """Expected test sets per gate, from scripts/gate_runner.GATE_SPECS.

    Returns (expected_tests, skip_policies, allowed_skips):
      - expected_tests: gate_id -> set(spec.tests)
      - skip_policies:  gate_id -> spec.skip_policy ("allowed" | "fail_on_skip")
      - allowed_skips:  gate_id -> set of test paths that may skip (from
                        ``allowed_skip_inventory`` if the GateSpec exposes it;
                        tolerated as absent — the other agent is adding it).

    Falls back to a local constant (the 15 gate test-file lists from config)
    when ``scripts.gate_runner`` cannot be imported (in-flight edit).  Gates
    with no known spec simply have an empty expected set — their evidence is
    validated by machine fields but never promoted to PASS by the exact-match
    check below.
    """
    try:
        from scripts.gate_runner import GateSpec, GATE_SPECS  # noqa: F401
        specs = list(GATE_SPECS)
        source = "gate_runner"
    except Exception:
        try:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from gate_runner import GateSpec, GATE_SPECS  # noqa: F401
            specs = list(GATE_SPECS)
            source = "gate_runner"
        except Exception as exc:
            print(
                f"warning: scripts.gate_runner unavailable ({exc}); using the "
                "local 15-gate fallback expectation map",
                file=sys.stderr,
            )
            specs = _FALLBACK_GATE_SPECS
            source = "local-fallback"
    expected: dict[str, set[str]] = {}
    policies: dict[str, str] = {}
    allowed: dict[str, set[str]] = {}
    for spec in specs:
        gid = getattr(spec, "gate_id", None)
        if not gid:
            continue
        tests = tuple(getattr(spec, "tests", ()) or ())
        expected[gid] = {str(t) for t in tests}
        pol = getattr(spec, "skip_policy", "allowed")
        policies[gid] = str(pol) if pol else "allowed"
        # VER-P0-02: the sibling agent's allowed_skip_inventory is a dict
        # {test_path: {reason_regex: allowance}} (or a plain dict of test
        # paths).  A path registered with a non-empty rule set counts as
        # covered: a skip on that file with a registered reason is a
        # legitimate expected skip.  Unregistered test paths (or an empty rule
        # set) are NOT covered.
        inv = getattr(spec, "allowed_skip_inventory", None)
        covered: set[str] = set()
        if isinstance(inv, dict):
            for path, rules in inv.items():
                if isinstance(rules, dict) and rules:
                    covered.add(str(path))
                elif isinstance(rules, (list, tuple)) and rules:
                    covered.add(str(path))
                elif rules:  # truthy non-container (legacy bool/1)
                    covered.add(str(path))
        allowed[gid] = covered
    if verbose_debug():
        print(f"  gate expectations source: {source} ({len(expected)} gates)")
    return expected, policies, allowed


@dataclass(frozen=True)
class _FallbackSpec:
    gate_id: str
    tests: tuple[str, ...]
    skip_policy: str = "allowed"
    commands: tuple[tuple[str, ...], ...] = ()


# Fallback expectation map used ONLY when scripts.gate_runner cannot be
# imported (its schema is mid-refactor under a sibling agent).  Values mirror
# config/gates.json + the 15 gate test-file lists from gate_runner.GATE_SPECS.
_FALLBACK_GATE_SPECS: list[Any] = [
    _FallbackSpec("NUMERICAL_ORACLE", ("quant_evaluator/tests/test_numerical_oracle.py",)),
    _FallbackSpec("PROPERTY", ("quant_evaluator/tests/test_metamorphic.py", "quant_evaluator/tests/test_consistency.py")),
    _FallbackSpec("CROSS_PACKAGE", ("integration_tests/test_cross_package_contracts.py",)),
    _FallbackSpec("ASHARE_SEMANTIC_CONTRACT_GOLDEN", ("integration_tests/test_ashare_semantic_golden.py",)),
    _FallbackSpec("ASHARE_REAL_DATA_SHADOW", ("integration_tests/test_real_ashare_shadow.py",), skip_policy="fail_on_skip"),
    _FallbackSpec("SERIALIZATION", ("quant_evaluator/tests/test_qe_serialization.py", "factor_assets/tests/registry/test_serialization_codec.py")),
    _FallbackSpec("CHECKPOINT_RESUME", ("factor_engine/tests/runtime/test_r10_stateful_checkpoint_2026_08.py",)),
    _FallbackSpec("UNIT", ("factor_engine/tests/test_capability_registry.py", "factor_engine/tests/test_concurrency_safety.py")),
    _FallbackSpec("LEAKAGE", ("factor_engine/tests/modeling/test_evaluation_leakage_evidence.py", "factor_preprocess/tests/contracts/test_leakage_properties.py")),
    _FallbackSpec("PIT", ("factor_engine/tests/runtime/test_pit_audit.py", "factor_engine/tests/runtime/test_label_pit.py", "factor_engine/tests/runtime/test_r10_pit_tristate.py")),
    _FallbackSpec("DETERMINISM", ("factor_engine/tests/test_comprehensive_data_consistency.py", "factor_engine/tests/backend/test_r21_def5_payload_hash.py")),
    _FallbackSpec("FRESH_WHEEL", ("factor_engine/scripts/wheel_clean_install_smoke.py",)),
    _FallbackSpec("1K_SCALE", ("quant_evaluator/tests/test_scale_gates.py",)),
    _FallbackSpec("10K_SCALE", ("quant_evaluator/tests/test_scale_gates.py",)),
    _FallbackSpec("100K_SCALE", ("quant_evaluator/tests/test_scale_gates.py",)),
]

_verbose_flag: bool | None = None


def verbose_debug() -> bool:
    global _verbose_flag
    if _verbose_flag is None:
        _verbose_flag = "--verbose" in sys.argv or "-v" in sys.argv
    return _verbose_flag


def apply_verified_evidence(gates: list[dict], verbose: bool = True) -> list[dict]:
    """Fold current-tree verified evidence (from verified_gate_evidence.json) in.

    VER-P0-01 (honesty): a gate is moved to PASS ONLY when every evidence item
    carries real machine fields proving an actual executed run:

        * ``command_hash``  — sha256 of the exact pytest argv (non-empty)
        * ``exit_code``     — 0
        * ``failed``        — 0
        * ``errors``        — 0
        * ``tests``         — > 0 (a gate that ran nothing is not a pass)

    VER-P0-02 (exact identity): the set of machine-verified ``test`` paths for
    a gate must EXACTLY equal the expected test file set from the gate spec —
    no missing, no extra, no duplicate.  A partial run (e.g. only 1 of 2
    expected files) leaves the gate NOT_RUN.  The per-gate ``status`` string in
    the evidence file is NOT trusted; it is advisory only.

    The ``result: "passed"`` string and per-gate ``status`` in the evidence file
    are NOT trusted; they are advisory only.  An entry missing those machine
    fields is treated as NOT_RUN — never PASS.  This is deliberate: the old
    system could fabricate PASS from config alone; this one cannot.

    Any gate NOT covered by honest evidence stays NOT_RUN -- the manifest must
    not overclaim.

    Note: historically only a hard-coded _PROVABLE_GATE_NAMES subset could be
    upgraded to PASS; LEAKAGE, PIT, DETERMINISM, FRESH_WHEEL, 1K_SCALE,
    10K_SCALE, 100K_SCALE were excluded even when the evidence proved them.
    That exclusion has been removed: the machine-field validation + exact
    test-set match below are the only gatekeepers, applied uniformly to ALL
    gates.
    """
    entries = _load_verified_evidence()
    if entries is None:
        return gates
    expected_tests, skip_policies, allowed_skips = _build_gate_expectations()
    applied: list[str] = []
    for entry in entries:
        name = entry["name"]
        try:
            g = _gate(gates, name)
        except KeyError:
            continue
        if g["status"] != "NOT_RUN":
            if verbose:
                print(f"  [skip] gate {name}: status already {g['status']}", file=sys.stderr)
            continue
        ev_items = entry.get("evidence") or []
        if not ev_items:
            if verbose:
                print(f"  [skip] gate {name}: no evidence items", file=sys.stderr)
            continue
        expected: set[str] = expected_tests.get(name, set())
        # VER-P0-02: machine-verified evidence items ONLY; a ``result:
        # "passed"`` string or a bare ``count`` does NOT count.
        ok_items: list[dict] = []
        has_extra = False
        has_duplicate = False
        for it in ev_items:
            if not isinstance(it, dict):
                continue
            if not it.get("test"):
                continue
            test_path = str(it["test"])
            # VER-P0-02: evidence for a test file that is NOT part of this
            # gate's expected set is EXTRANEOUS.  It is excluded from the ok
            # set AND its presence keeps the gate NOT_RUN (exact identity: no
            # extra evidence allowed).
            if expected and test_path not in expected:
                has_extra = True
                if verbose:
                    print(
                        f"  [drop-extra] gate {name}: evidence test {test_path!r} "
                        f"not in expected set {sorted(expected)} -> EXTRANEous "
                        "evidence; gate stays NOT_RUN",
                        file=sys.stderr,
                    )
                continue
            cmd_hash = it.get("command_hash")
            exit_code = it.get("exit_code")
            if not cmd_hash or not isinstance(exit_code, int):
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {test_path} "
                        "missing command_hash/exit_code -> treated NOT_RUN",
                        file=sys.stderr,
                    )
                continue
            failed = it.get("failed")
            errors = it.get("errors")
            tests = it.get("tests")
            if failed != 0 or errors != 0:
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {test_path} "
                        f"has failed={failed} errors={errors} -> NOT a pass",
                        file=sys.stderr,
                    )
                continue
            if exit_code != 0:
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {test_path} "
                        f"has exit_code={exit_code} != 0 -> NOT a pass",
                        file=sys.stderr,
                    )
                continue
            if not isinstance(tests, int) or tests <= 0:
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {test_path} "
                        f"has tests={tests!r} (<=0) -> NOT a pass",
                        file=sys.stderr,
                    )
                continue
            # VER-P0-02: a duplicate test path is never acceptable — one test
            # file must produce exactly one honest evidence entry, and a
            # duplicate keeps the gate NOT_RUN (exact identity).
            if any(o["test"] == test_path for o in ok_items):
                has_duplicate = True
                if verbose:
                    print(
                        f"  [drop-dupe] gate {name}: duplicate evidence for "
                        f"{test_path} -> DUPLICATE evidence; gate stays NOT_RUN",
                        file=sys.stderr,
                    )
                continue
            ok_items.append({
                "test": test_path,
                "command_hash": str(cmd_hash),
                "exit_code": int(exit_code),
                "passed": int(it["passed"]) if isinstance(it.get("passed"), int) else None,
                "failed": int(failed),
                "errors": int(errors),
                "skipped": int(it["skipped"]) if isinstance(it.get("skipped"), int) else None,
                "tests": int(tests),
                "duration_sec": it.get("duration_sec"),
                "executed_at": str(it.get("executed_at") or ""),
                "run_at": str(it.get("run_at") or ""),
            })
        # VER-P0-02: extra or duplicate evidence keeps the gate NOT_RUN even
        # when the expected set would otherwise be covered.
        if has_extra:
            if verbose:
                print(
                    f"  [skip] gate {name}: EXTRANEOUS evidence present -> "
                    "NOT_RUN (exact identity forbids extra tests)",
                    file=sys.stderr,
                )
            continue
        if has_duplicate:
            if verbose:
                print(
                    f"  [skip] gate {name}: DUPLICATE evidence present -> "
                    "NOT_RUN (exact identity forbids duplicates)",
                    file=sys.stderr,
                )
            continue
        if not ok_items:
            if verbose:
                print(f"  [skip] gate {name}: no machine-verified evidence items", file=sys.stderr)
            continue
        ok_paths: set[str] = {o["test"] for o in ok_items}

        # VER-P0-02: exact identity — the verified set MUST equal the expected
        # set.  Missing / extra / duplicate test paths all keep the gate NOT_RUN.
        if not expected:
            if verbose:
                print(
                    f"  [skip] gate {name}: no expectation map entry (unknown spec); "
                    "exact-identity gate cannot be proven -> NOT_RUN",
                    file=sys.stderr,
                )
            continue
        missing = expected - ok_paths
        extra = ok_paths - expected
        if missing:
            if verbose:
                print(
                    f"  [skip] gate {name}: MISSING verified evidence for expected "
                    f"tests {sorted(missing)} -> NOT_RUN (exact identity requires all)",
                    file=sys.stderr,
                )
            continue
        if extra:
            if verbose:
                print(
                    f"  [skip] gate {name}: EXTRA verified evidence {sorted(extra)} "
                    f"beyond expected {sorted(expected)} -> NOT_RUN (exact identity)",
                    file=sys.stderr,
                )
            continue

        # VER-P0-02: skip policy — when the gate fails on skips (or has an
        # allowed-skip inventory) any ok item with skipped > 0 not covered by
        # the allowed inventory keeps the gate NOT_RUN.
        policy = skip_policies.get(name, "allowed")
        allowed = allowed_skips.get(name, set())
        skip_fail = False
        if policy == "fail_on_skip" or allowed:
            for o in ok_items:
                skipped = o.get("skipped") or 0
                if skipped > 0 and o["test"] not in allowed:
                    if verbose:
                        print(
                            f"  [skip] gate {name}: evidence for {o['test']} has "
                            f"skipped={skipped} (not covered by allowed-skip "
                            f"inventory {sorted(allowed) or 'none'}) -> NOT a pass",
                            file=sys.stderr,
                        )
                    skip_fail = True
                    break
        if skip_fail:
            continue

        g["status"] = "PASS"
        g["evidence"] = ok_items
        applied.append(f"{name}({len(ok_items)} evidence items)")
    if applied:
        print(f"  gates moved PASS by machine-verified evidence: {', '.join(applied)}")
    return gates

def run_pytest_gate(gates: list[dict], command: str | None = None) -> dict | None:
    """Run a lightweight UNIT gate check.

    Runs the provided command (default: a focused subset under tests/) and
    records counts from the pytest exit line.  The manifest does NOT claim
    full-suite PASS from this; it is evidence for the UNIT gate.
    """
    if not command:
        command = f"{sys.executable} -m pytest tests/test_evidence_registry.py -q"
    cmd = command.split()
    rec = _run(cmd, REPO_ROOT, timeout=900)
    tail = "\n".join(rec["output_tail"])
    counts = {
        "passed": None, "failed": None, "skipped": None,
        "xfailed": None, "errors": None,
    }
    for ln in rec["output_tail"]:
        if "passed" in ln and ("failed" in ln or "skipped" in ln or "error" in ln):
            for key in counts:
                if key in ln:
                    pass
        if ln.startswith("=") and "passed" in ln and "failed" in ln and "in " in ln:
            for key, pattern in (
                ("passed", " passed"), ("failed", " failed"),
                ("skipped", " skipped"), ("xfailed", " xfailed"),
                ("errors", " error"),
            ):
                if pattern in ln:
                    counts[key] = int(ln.split(pattern)[0].split()[-1])
    outcome = (
        "PASS" if rec["exit_code"] == 0 else
        "BLOCKED" if "error" in tail and "Interrupted" in tail else "FAIL"
    )
    if outcome == "PASS":
        _set_gate(
            gates, "UNIT", "PASS",
            f"pytest '{' '.join(cmd)}' exit={rec['exit_code']} in {rec['duration_sec']}s",
        )
    return {"run": rec, "counts": counts}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_manifest() -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    gates = load_gates()
    gates = apply_verified_evidence(gates)

    # --- source snapshot (ReleaseIdentity from evidence.current) ------------
    identity = _release_identity()
    git_sha = identity.git_sha
    git_dirty = identity.dirty

    packages: dict[str, dict] = {}
    for pkg in PACKAGES:
        pkg_root = REPO_ROOT / pkg
        entry: dict = {
            "tree_sha256": package_tree_hash(pkg),
            "wheel_sha256": package_wheel_sha256(pkg),
        }
        # VER-P0-06: only REAL git index gitlinks count as submodule pins.
        # factor_engine / dataaccess are plain tracked directories here (no
        # mode-160000 index entries), so they get NO git_sha / git_dirty / no
        # submodule fabrication.
        if pkg in identity.gitlinks:
            entry["git_sha"] = identity.gitlinks[pkg]
            entry["is_git_submodule"] = True
            entry["git_dirty"] = _git_dirty(pkg_root)
        elif (pkg_root / ".git").exists():
            entry["git_sha"] = _git_sha(pkg_root)
            entry["git_dirty"] = _git_dirty(pkg_root)
        if (pkg_root / ".gitmodules").exists():
            entry["is_git_submodule"] = True
        packages[pkg] = entry

    # stale release-truth artifact (REL-P0-02)
    stale_notes: list[dict] = []
    for path in (REPO_ROOT / ".pytest_full.txt",
                 REPO_ROOT / "factor_engine" / ".pytest_full.txt"):
        if path.is_file():
            stale_notes.append({
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "mtime": datetime.datetime.fromtimestamp(
                    path.stat().st_mtime, tz=datetime.timezone.utc
                ).isoformat(),
            })

    # --- executed commands ---------------------------------------------------
    executed = [{
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "exit_code": 0,
        "note": "manifest generator",
    }]

    # --- benchmarks (best-effort) -------------------------------------------
    benchmark_summaries: list[dict] = []
    bench_json = REPO_ROOT / "factor_engine" / "benchmarks" / "backend_cost_baseline.json"
    if bench_json.is_file():
        try:
            b = json.loads(bench_json.read_text())
            benchmark_summaries.append({
                "source": "factor_engine/benchmarks/backend_cost_baseline.json",
                "schema_version": b.get("schema_version"),
                "generated_by": b.get("generated_by"),
                "sizes": b.get("sizes"),
            })
        except Exception as exc:
            benchmark_summaries.append({"source": "backend_cost_baseline.json", "parse_error": str(exc)})

    manifest = {
        "schema_version": 1,
        "manifest_id": "rel-v1:" + now,
        "timestamp": now,
        "host": socket.gethostname(),
        "source_snapshot": {
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "diff_hash": identity.diff_hash,
            "cached_diff_hash": identity.cached_diff_hash,
            "untracked_source_hash": identity.untracked_source_hash,
            # VER-P0-06: submodule_shas is populated ONLY from real git index
            # gitlinks (mode 160000).  This repo declares none, so the field is
            # absent here and empty below — no fabricated SHAs.
            "submodule_shas": dict(identity.gitlinks),
            "git_repo": git_sha is not None,
        },
        "packages": packages,
        "environment": env_hash(),
        "executed_commands": executed,
        "pytest_counts": None,
        "benchmarks": benchmark_summaries,
        "gates": gates,
        "release_notes": [],
    }

    # REL-P0-02 stale artifact note
    if stale_notes:
        note_lines = [
            ".pytest_full.txt artifacts in the pushed snapshot are stale. "
            "They reference tests/test_operators_extension.py:148 "
            "test_neutralize_ols_residual, which no longer exists in the "
            "current tree (grep finds the string only inside the stale "
            "artifact itself, not in any source file). Recommendation: "
            "stop treating .pytest_full.txt as release truth.",
        ]
        # Keep the note truthful even if the stale test were ever resurrected.
        if not any(
            path.is_file()
            for path in (REPO_ROOT / "tests" / "test_operators_extension.py",)
        ):
            note_lines.append(
                "Checked at manifest generation: tests/test_operators_extension.py "
                "does not exist in the current tree."
            )
        manifest["release_notes"].append({
            "id": "REL-P0-02",
            "note": " ".join(note_lines),
            "artifacts": stale_notes,
        })

    return manifest


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print(f"git_sha={manifest['source_snapshot']['git_sha']} "
          f"dirty={manifest['source_snapshot']['git_dirty']}")
    for g in manifest["gates"]:
        print(f"  gate {g['name']:22s} -> {g['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
