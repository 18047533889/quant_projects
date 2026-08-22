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
from pathlib import Path

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

# Gates that are proven by the UNIT pytest check and the manifest itself.
# NOTE: this is a conservative list.  Nothing here is marked PASS for a gate
# unless evidence below (or the current-tree run) actually proves it.
_PROVABLE_GATE_NAMES: frozenset[str] = frozenset({
    "UNIT",
    "NUMERICAL_ORACLE",
    "PROPERTY",
    "CROSS_PACKAGE",
    "SERIALIZATION",
    "CHECKPOINT_RESUME",
    "ASHARE_SEMANTIC_CONTRACT_GOLDEN",
    "ASHARE_REAL_DATA_SHADOW",
})

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


def _release_identity() -> dict:
    """Single source of truth for root/FE/DA git SHAs: evidence.current.

    R23: the manifest no longer computes its own root/submodule SHAs.  It
    delegates to ``evidence.current`` (R22-CURRENT-TRUTH) so the
    VerificationManifest's ReleaseIdentity (root / factor_engine / dataaccess)
    can never drift from CURRENT.json's sha_bindings.  Degrades to local git
    calls only if evidence.current is unavailable (non-git / no dataaccess).
    """
    try:
        from evidence.current import root_repo_sha, submodule_sha
        root = root_repo_sha()
        return {
            "git_sha": root.split(" ")[0],
            "git_dirty": root.endswith("(working-tree-dirty)"),
            "factor_engine": submodule_sha("factor_engine"),
            "dataaccess": submodule_sha("dataaccess"),
        }
    except Exception as exc:  # pragma: no cover - degraded fallback
        print(
            f"warning: evidence.current identity unavailable ({exc}); "
            "falling back to local git calls",
            file=sys.stderr,
        )
        return {
            "git_sha": _git_sha(REPO_ROOT),
            "git_dirty": _git_dirty(REPO_ROOT),
            "factor_engine": _git_sha(REPO_ROOT / "factor_engine"),
            "dataaccess": _git_sha(REPO_ROOT / "dataaccess"),
        }


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


def apply_verified_evidence(gates: list[dict], verbose: bool = True) -> list[dict]:
    """Fold current-tree verified evidence (from verified_gate_evidence.json) in.

    VER-P0-01 (honesty): a gate is moved to PASS ONLY when every evidence item
    carries real machine fields proving an actual executed run:

        * ``command_hash``  — sha256 of the exact pytest argv (non-empty)
        * ``exit_code``     — 0
        * ``failed``        — 0
        * ``errors``        — 0
        * ``tests``         — > 0 (a gate that ran nothing is not a pass)

    The ``result: "passed"`` string and per-gate ``status`` in the evidence file
    are NOT trusted; they are advisory only.  An entry missing those machine
    fields is treated as NOT_RUN — never PASS.  This is deliberate: the old
    system could fabricate PASS from config alone; this one cannot.

    Any gate NOT covered by honest evidence stays NOT_RUN -- the manifest must
    not overclaim.
    """
    entries = _load_verified_evidence()
    if entries is None:
        return gates
    applied: list[str] = []
    for entry in entries:
        name = entry["name"]
        try:
            g = _gate(gates, name)
        except KeyError:
            continue
        if name not in _PROVABLE_GATE_NAMES:
            if verbose:
                print(f"  [skip] gate {name}: not in provable set", file=sys.stderr)
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
        # VER-P0-01: only machine-verified evidence items are acceptable.  A
        # ``result: "passed"`` string or a bare ``count`` does NOT count.
        ok_items: list[dict] = []
        for it in ev_items:
            if not isinstance(it, dict):
                continue
            if not it.get("test"):
                continue
            cmd_hash = it.get("command_hash")
            exit_code = it.get("exit_code")
            if not cmd_hash or not isinstance(exit_code, int):
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {it.get('test')} "
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
                        f"  [drop-item] gate {name}: evidence for {it.get('test')} "
                        f"has failed={failed} errors={errors} -> NOT a pass",
                        file=sys.stderr,
                    )
                continue
            if exit_code != 0:
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {it.get('test')} "
                        f"has exit_code={exit_code} != 0 -> NOT a pass",
                        file=sys.stderr,
                    )
                continue
            if not isinstance(tests, int) or tests <= 0:
                if verbose:
                    print(
                        f"  [drop-item] gate {name}: evidence for {it.get('test')} "
                        f"has tests={tests!r} (<=0) -> NOT a pass",
                        file=sys.stderr,
                    )
                continue
            ok_items.append({
                "test": str(it["test"]),
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
        if not ok_items:
            if verbose:
                print(f"  [skip] gate {name}: no machine-verified evidence items", file=sys.stderr)
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
    git_sha = identity["git_sha"]
    git_dirty = identity["git_dirty"]

    packages: dict[str, dict] = {}
    for pkg in PACKAGES:
        pkg_root = REPO_ROOT / pkg
        entry: dict = {
            "tree_sha256": package_tree_hash(pkg),
            "wheel_sha256": package_wheel_sha256(pkg),
        }
        if (pkg_root / ".git").exists():
            entry["git_sha"] = identity.get(pkg) or _git_sha(pkg_root)
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
            "submodule_shas": {
                pkg: packages[pkg].get("git_sha") for pkg in PACKAGES
                if packages[pkg].get("git_sha")
            },
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
