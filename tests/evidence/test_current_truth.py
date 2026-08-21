# -*- coding: utf-8 -*-
"""R22-CURRENT-TRUTH — regression tests for the SHA-bound CURRENT.json truth.

These tests are LOCAL ONLY (no git mutations, no network), serial pytest
(no xdist), and honor thread env vars (OPENBLAS_NUM_THREADS=1, ...).

Coverage:
  1. current_json_binds_full_sha_set      — the committed CURRENT.json carries
     all eight sha_bindings fields (the spec's mandatory binding set).
  2. bound_shas_match_live_tree           — recomputing the bindings from the
     current tree yields the exact values stored in CURRENT.json, i.e. the
     file is CURRENT (not STALE) against the live tree.
  3. stale_auto_detects_on_sha_change     — changing any single binding flips
     the validator's evaluation to STALE with that field named (no manual
     edit ever "fixes" a stale file).
  4. generator_dry_run_reproduces_file    — ``python -m evidence.current``
     dry-run output reconstructs the exact sha_bindings of the committed file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CURRENT_JSON = _REPO_ROOT / "evidence" / "CURRENT.json"

# Every entry CURRENT.json must bind (the spec's mandatory SHA set).
REQUIRED_BINDINGS = (
    "RootRepoSHA",
    "FactorEngineSubmoduleSHA",
    "DataAccessSubmoduleSHA",
    "OperatorCatalogSHA",
    "SemanticCatalogIdentity",
    "ImplementationClosureHashSet",
    "DependencyLockHash",
    "TestEnvironmentIdentity",
)

# Exact SHA set the working tree must currently bind (bound to the pinned
# submodule/HEAD snapshots at R22: root d05455c1, factor_engine 26ed6647,
# dataaccess ba072d23).
EXPECTED_BINDINGS = {
    "FactorEngineSubmoduleSHA": "26ed66472dccc33b02f2919d2685199491fba035",
    "DataAccessSubmoduleSHA": "ba072d23b7a48440d05977ef0a08fe1fce45d629",
    "DependencyLockHash": "b982526eea6469422787b2fe4031af472e1ceb49afaff11f2c0fe260bbc3ae1b",
    "TestEnvironmentIdentity": "dcab69f70cc819b6b2e2f93759faf1f5ebeab3872d0d98c68e3e7970aff7bd68",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_current() -> dict:
    assert _CURRENT_JSON.is_file(), (
        f"missing {_CURRENT_JSON} — run: python -m evidence.current --write"
    )
    return json.loads(_CURRENT_JSON.read_text(encoding="utf-8"))


def _fresh_bindings() -> dict:
    from evidence.current import build_current
    return build_current()["sha_bindings"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_current_json_binds_full_sha_set():
    """The committed CURRENT.json carries all eight mandatory sha_bindings."""
    data = _load_current()
    assert "sha_bindings" in data
    bindings = data["sha_bindings"]
    missing = [k for k in REQUIRED_BINDINGS if k not in bindings]
    assert not missing, f"missing sha_bindings: {missing}"
    for key in REQUIRED_BINDINGS:
        value = bindings[key]
        assert value, f"sha_binding {key!r} is empty in CURRENT.json"


def test_bound_shas_match_live_tree():
    """Recomputing the bindings from the current tree equals the stored file.

    This is the auto-STALE gate: if anything (git HEAD, submodule pin,
    operator catalog, semantic catalog, dependency lock, env) changed, the
    recomputed binding differs and the file is STALE until regenerated.
    """
    data = _load_current()
    stored = data.get("sha_bindings", {})
    fresh = _fresh_bindings()

    mismatch = {
        k: {"stored": stored.get(k), "fresh": fresh.get(k)}
        for k in REQUIRED_BINDINGS
        if stored.get(k) != fresh.get(k)
    }
    assert not mismatch, (
        f"CURRENT.json is STALE — run `python -m evidence.current --write`. "
        f"Changed bindings: {json.dumps(mismatch, indent=2)}"
    )


def test_stale_auto_detects_on_sha_change():
    """Perturbing any single binding makes the validator report STALE."""
    from evidence.current import check_stale

    data = _load_current()
    fresh = _fresh_bindings()

    for key in REQUIRED_BINDINGS:
        tampered = dict(fresh)
        good = tampered[key]
        tampered[key] = good[:-1] + ("0" if good[-1] != "0" else "1")
        is_current, changed_names = check_stale(
            {"sha_bindings": fresh}, {"sha_bindings": tampered}
        )
        assert not is_current, f"tampered {key} still reported CURRENT"
        assert key in changed_names, f"changed field {key} not reported"


def test_stale_detected_when_file_generated_at_old_sha():
    """A CURRENT.json bound to an OLD root/factor_engine/dataaccess SHA set
    must evaluate STALE with those exact fields named — the concrete failure
    the previous R21 CURRENT.json exhibited."""
    from evidence.current import check_stale

    fresh = _fresh_bindings()
    data = _load_current()
    stored = data.get("sha_bindings", {})

    stale_file = {
        "RootRepoSHA": "0000000000000000000000000000000000000000 (working-tree-dirty)",
        "FactorEngineSubmoduleSHA": "ffffffffffffffffffffffffffffffffffffffff",
        "DataAccessSubmoduleSHA": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    }
    stale_file.update(
        {
            k: stored.get(k)
            for k in REQUIRED_BINDINGS
            if k not in stale_file
        }
    )
    is_current, changed_names = check_stale(
        {"sha_bindings": fresh}, {"sha_bindings": stale_file}
    )
    assert not is_current
    assert "RootRepoSHA" in changed_names
    assert "FactorEngineSubmoduleSHA" in changed_names
    assert "DataAccessSubmoduleSHA" in changed_names


def test_generator_dry_run_reproduces_file():
    """`python -m evidence.current` (dry-run) reproduces the file's bindings."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "evidence.current", "--skip-snapshot"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-500:]

    payload = json.loads(result.stdout.split("evaluation:")[0].strip())
    fresh = payload["sha_bindings"]

    data = _load_current()
    stored = data["sha_bindings"]
    for key in REQUIRED_BINDINGS:
        assert fresh[key] == stored[key], (
            f"dry-run binding {key} diverges from stored CURRENT.json"
        )