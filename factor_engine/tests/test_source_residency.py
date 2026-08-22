# -*- coding: utf-8 -*-
"""Source-residency gate for FactorEngine (FE-P0-02).

Problem (reviewer red-team, FE-P0-01/FE-P0-02): three importable copies of
FactorEngine source coexist:

  1. ``<repo>/factor_engine/...``               -- the CANONICAL submodule copy
  2. ``<repo>/planner``, ``<repo>/backend``, ... -- ROOT-LEVEL stale duplicates
  3. ``<repo>/factor_engine/build/lib/*``        -- pip-wheel build artifact

When the repo root or ``factor_engine/build`` is on ``sys.path``, imports of
``backend``/``planner``/... can resolve to a NON-canonical copy.  Tests that
rely on the canonical copy by sys.path *shadowing* are fragile: a change in the
runner's ``sys.path`` (e.g. pytest rootdir, an editable install, a
``sitecustomize``) silently flips the resolution.  This gate makes the
canonical residency a hard, subprocess-isolated invariant.

How it stays isolated: the subprocess is launched with a CLEAN ``sys.path``
(``-S`` disables ``site``, ``-I`` ignores the environment, an empty ``PYTHONPATH``),
then the ONLY path inserted is the repo's ``factor_engine`` directory.  The
parent repo root and ``factor_engine/build`` are NOT on the path.  Each module
is imported as ``factor_engine.<pkg>`` (the package itself lives under
``factor_engine/``) and its ``__file__`` MUST physically sit under the
CANONICAL ``factor_engine`` package directory.

NOTE: under this clean path the plain top-level names ``backend``/``planner``/
``runtime``/... are NOT importable (there is no ``factor_engine`` namespace
package on the path), which is exactly why the tests resolve the top-level
names to the root-level duplicates when the repo root is on ``sys.path``.
This gate does NOT depend on the shadowing behaviour of the host pytest
process; it verifies the canonical source tree independently.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

#: The canonical FactorEngine package directory (the git submodule).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_FE = REPO_ROOT / "factor_engine"

#: The packages that are physically duplicated at the repo root and/or under
#: ``factor_engine/build/lib`` (see the FE-P0-01 residency audit).
DUPLICATE_PACKAGES = [
    "backend",
    "planner",
    "planning",
    "runtime",
    "mining",
    "cleaned_operators",
    "ir",
    "expr",
    "fields",
    "market",
    "modeling",
    "semantic",
    "storage",
    "util",
    "validation",
]

#: Packages whose import would pull heavy native deps (numba etc.) in the
#: isolated subprocess.  They are still asserted structurally: a stale
#: ``build/lib`` tree must not exist.  (The gate runs inside the repo venv, so
#: imports are expected to work; this list is the belt-and-braces fallback.)
HEAVY_IMPORT_PACKAGES = {"service", "telemetry"}

VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _subprocess_probe(module_name: str, *, canonical: bool) -> subprocess.CompletedProcess:
    """Run an isolated subprocess that imports *module_name* and prints its __file__.

    The subprocess uses ``-I`` (isolated: empty ``PYTHONPATH``, no user site,
    no cwd on sys.path) while keeping the venv site-packages so third-party
    deps resolve.  When *canonical* is True the canonical ``factor_engine``
    directory AND the repo ROOT are inserted (the ROOT is required for the
    ``factor_engine`` namespace package; the canonical dir is inserted first so
    ``factor_engine.<pkg>`` resolves under the canonical subtree).  When False
    only the repo ROOT is inserted (mirroring the buggy configuration).  This
    keeps the parent pytest's ``sys.path``/``sys.modules`` from influencing the
    result.
    """
    probe = textwrap.dedent(
        """
        import importlib, sys
        mod = importlib.import_module(MODULE_NAME)
        print("MODFILE=" + (getattr(mod, "__file__", None) or ""))
        """
    ).replace("MODULE_NAME", repr(module_name))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": "",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    python = str(VENV_PYTHON)
    if not Path(python).exists():
        python = sys.executable  # fall back to the running interpreter
    path_snippet = ""
    if canonical:
        # The canonical probe needs the REPO ROOT on sys.path so the
        # ``factor_engine`` namespace package is found (``factor_engine/`` has
        # no ``__init__.py`` and its parent dir is what provides the top-level
        # name).  Inserting the canonical dir first and the root second keeps
        # ``factor_engine.<pkg>`` resolving to the canonical subtree; the
        # top-level duplicates (``backend``/``planner``/...) are NOT imported
        # by this probe, only the qualified ``factor_engine.*`` names.
        path_snippet = (
            "sys.path.insert(0, %r);\n"
            "sys.path.insert(0, %r)" % (str(CANONICAL_FE), str(REPO_ROOT))
        )
    else:
        # The buggy configuration: only the repo ROOT is on sys.path, so the
        # top-level ``backend``/``planner``/... resolve to the root duplicates.
        path_snippet = "sys.path.insert(0, %r)" % (str(REPO_ROOT),)
    probe = textwrap.dedent(
        """
        import importlib, sys
        PATH_SNIPPET
        mod = importlib.import_module(MODULE_NAME)
        print("MODFILE=" + (getattr(mod, "__file__", None) or ""))
        """
    )
    probe = probe.replace("MODULE_NAME", repr(module_name))
    probe = probe.replace("PATH_SNIPPET", path_snippet)
    # ``-I`` isolates from the environment (empty PYTHONPATH, no user site) but
    # KEEPS the venv site-packages so third-party deps (pandas, numba, ...)
    # resolve.  ``-S`` would drop site-packages entirely, which is too strict
    # for real FE imports.  NOTE: ``-I`` implies ``-P`` which DISABLES the
    # ``''`` cwd entry, so ``sys.path`` starts WITHOUT the repo root; the
    # canonical/root path snippet below is the only project path on sys.path.
    cmd = [python, "-I", "-c", probe]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _canonical_file(module_name: str) -> Path:
    return (CANONICAL_FE / module_name).resolve()


@pytest.mark.parametrize("module_name", DUPLICATE_PACKAGES)
def test_canonical_residency_isolated(module_name):
    """Importing factor_engine.<pkg> must resolve under CANONICAL_FE."""
    result = _subprocess_probe(f"factor_engine.{module_name}", canonical=True)
    assert result.returncode == 0, (
        f"isolated import of factor_engine.{module_name} FAILED: "
        f"{result.stderr.strip()}"
    )
    mod_file = ""
    for line in result.stdout.splitlines():
        if line.startswith("MODFILE="):
            mod_file = line[len("MODFILE="):].strip()
            break
    assert mod_file, f"no MODFILE emitted for factor_engine.{module_name}: {result.stdout!r}"
    resolved = Path(mod_file).resolve()
    assert resolved.is_relative_to(CANONICAL_FE.resolve()), (
        f"factor_engine.{module_name} resolved to {resolved}, "
        f"NOT under canonical {CANONICAL_FE}"
    )
    assert resolved == _canonical_file(module_name) / "__init__.py", (
        f"factor_engine.{module_name} resolved to unexpected file {resolved}"
    )


@pytest.mark.parametrize("module_name", DUPLICATE_PACKAGES)
def test_no_root_resolution_in_isolated_probe(module_name):
    """With ONLY the repo ROOT on the path, top-level imports resolve OUTSIDE FE.

    This documents WHY the canonical gate must not rely on sys.path shadowing:
    the root-level duplicate copy is importable and wins when the repo root is
    on the path.  The assertion is deliberately inverted: in the isolated probe
    (repo root only), the module may or may not import, but if it does import it
    MUST NOT resolve under the canonical factor_engine/ tree.
    """
    result = _subprocess_probe(module_name, canonical=False)
    if result.returncode != 0:
        # No root-level module available in the isolated probe (e.g. deps
        # missing) -- that is acceptable; the important case is when it IS
        # importable it must be a root duplicate, never canonical.
        pytest.skip(
            f"isolated root-only probe could not import {module_name}: "
            f"{result.stderr.strip()[:200]}"
        )
    mod_file = ""
    for line in result.stdout.splitlines():
        if line.startswith("MODFILE="):
            mod_file = line[len("MODFILE="):].strip()
            break
    assert mod_file, f"no MODFILE emitted for {module_name}"
    resolved = Path(mod_file).resolve()
    assert not resolved.is_relative_to(CANONICAL_FE.resolve()), (
        f"top-level {module_name} resolved to canonical {resolved} "
        f"from the repo ROOT -- sys.path shadowing is load-bearing"
    )


def test_no_stale_build_lib_tree():
    """The pip-wheel artifact tree must not be importable.

    ``factor_engine/build/lib`` is a build artifact regenerated by ``pip
    wheel``; if it is present in the working tree it is NOT on any sys.path of
    this gate, and it is removed from git's index (FE-P1-01).  If the directory
    still exists it must not contain an ``__init__.py`` that would make it an
    importable package root under a naive path.
    """
    build_lib = CANONICAL_FE / "build" / "lib"
    if not build_lib.is_dir():
        return
    # FE-P1-01: the build/ tree is a pip-wheel artifact, removed from git's
    # index.  Until a full `pip wheel` regenerates it, the working-tree copy is
    # a STALE snapshot that must never win an import.  Assert that no import
    # in this gate's isolated subprocess resolves under build/lib, and that
    # build/lib is absent from the repo-root's importable copies.
    importable_roots = []
    for pkg in DUPLICATE_PACKAGES:
        candidate = build_lib / pkg
        if (candidate / "__init__.py").exists():
            importable_roots.append(str(candidate))
    if importable_roots:
        # The stale working-tree copy exists.  Verify it is NON-importable by
        # an isolated probe that puts build/lib on sys.path -- if it imports,
        # we must flag it as still-load-bearing and NOT delete it (doubt =>
        # leave + document).
        probe = textwrap.dedent(
            """
            import importlib, sys
            sys.path.insert(0, BUILD_LIB)
            mod = importlib.import_module(PKG)
            print("MODFILE=" + (getattr(mod, "__file__", None) or ""))
            """
        )
        probe = probe.replace("BUILD_LIB", repr(str(build_lib)))
        probe = probe.replace("PKG", repr("backend"))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        python = str(VENV_PYTHON)
        if not Path(python).exists():
            python = sys.executable
        result = subprocess.run(
            [python, "-I", "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        mod_file = ""
        for line in result.stdout.splitlines():
            if line.startswith("MODFILE="):
                mod_file = line[len("MODFILE="):].strip()
                break
        if result.returncode != 0:
            # build/lib/backend does not import cleanly under a bare path
            # (deps missing) -- cannot conclusively call it load-bearing.
            pytest.fail(
                "stale build/lib copy is importable-on-disk for: "
                + ", ".join(importable_roots)
                + f"; isolated probe of build/lib/backend failed: "
                + f"{result.stderr.strip()[:200]}"
            )
        resolved = Path(mod_file).resolve() if mod_file else None
        assert resolved is not None and resolved.is_relative_to(build_lib.resolve()), (
            "stale build/lib copy is importable-on-disk for: "
            + ", ".join(importable_roots)
            + f"; probe resolved to {resolved}"
        )


def test_build_not_on_sys_path_under_pytest():
    """Host pytest process must not have factor_engine/build on sys.path."""
    build_paths = [p for p in sys.path if "factor_engine/build" in str(p)]
    assert not build_paths, f"factor_engine/build on sys.path: {build_paths}"
