#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiered test-suite dispatcher for factor_engine.

Three additive tiers keep production regressions visible even while the
pre-existing operator/evidence drift (4961 failures at the 2026-09-04 baseline)
is being worked down:

  * production_critical — MUST be 100% green. Exits non-zero on ANY failure.
    A missing file in the list is also an error (fail-closed).
  * research_extended  — tracked/known-fail (xfail-able). Exits 0 but prints the
    exact failing list so it can be audited.
  * legacy_quarantine  — legacy/deprecated/auto-harness/collection-error drift,
    excluded from production gating but still collected for reference. Exits 0
    and prints the failing list.

Lists live in ``factor_engine/tests/<tier>.txt`` (machine-generated, one file
path or pytest node id per line; ``#`` starts an inline comment/reason).

Usage
-----
    python factor_engine/scripts/run_test_tiers.py --tier production_critical
    python factor_engine/scripts/run_test_tiers.py --tier all
    python factor_engine/scripts/run_test_tiers.py --gate production_critical   # CI gate

``--tier all`` runs every tier in order and fails only if production_critical
is red.  ``--gate`` is the fail-closed GO-process entrypoint: it runs only the
given tier (default production_critical), fails on any red AND on any missing
file in the list, and writes a JUnit XML summary under ``--junitxml-out``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "factor_engine" / "tests"
DEFAULT_PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
TIERS = ("production_critical", "research_extended", "legacy_quarantine")
TIER_FILES = {t: TESTS_DIR / f"{t}.txt" for t in TIERS}


def _env_with_omp(workers: int) -> dict[str, str]:
    env = dict(os.environ)
    # 31-core hard cap shared machine rule; keep test processes <= OMP limit.
    env["OMP_NUM_THREADS"] = str(max(1, min(31, workers)))
    env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    env["POLARS_MAX_THREADS"] = env["OMP_NUM_THREADS"]
    return env


def read_list(tier: str) -> list[str]:
    path = TIER_FILES[tier]
    if not path.is_file():
        raise SystemExit(f"{tier}: list file missing: {path} (run the tier builder)")
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entry = line.split("  #", 1)[0].strip()
        if entry:
            entries.append(entry)
    return entries


def verify_list_paths(tier: str, entries: list[str]) -> list[str]:
    """Fail-closed: every file in the tier list must exist on disk."""
    missing: list[str] = []
    for entry in entries:
        file_part = entry.split("::", 1)[0]
        if not Path(file_part).is_file():
            missing.append(file_part)
    return missing


def run_pytest(
    tier: str,
    entries: list[str],
    *,
    fail_fast: bool,
    junitxml_out: Path | None,
    workers: int,
    python: str,
    timeout: int = 300,
) -> int:
    args = [
        python,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--timeout={timeout}",
    ]
    if workers and workers > 1:
        if shutil.which("pytest-xdist"):  # pragma: no cover - env dependent
            args.append("-n")
            args.append(str(workers))
    if fail_fast:
        args.append("-x")
    if junitxml_out is not None:
        junitxml_out.mkdir(parents=True, exist_ok=True)
        args.append(f"--junitxml={junitxml_out / f'{tier}.xml'}")
    args.extend(entries)
    env = _env_with_omp(workers or 1)
    proc = subprocess.run(args, cwd=str(REPO_ROOT), env=env)
    return proc.returncode


def _print_failing_from_xml(junit: Path) -> None:
    """Best-effort exact failing-list printout from the junit XML."""
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(str(junit)).getroot()
        for case in root.iter("testcase"):
            for fail in case.iter("failure"):
                print(f"FAILED {case.get('classname')}::{case.get('name')}")
            for err in case.iter("error"):
                print(f"ERROR  {case.get('classname')}::{case.get('name')}")
    except Exception as exc:  # pragma: no cover
        print(f"(could not parse junit for failing list: {exc})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=("all",) + TIERS, default="all")
    ap.add_argument("--gate", metavar="TIER", nargs="?", const="production_critical",
                    help="fail-closed CI gate: run one tier, fail on any red or missing list file")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--junitxml-out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="xdist workers (default 1; capped at 31 via OMP)")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    gate_tier = args.gate
    if gate_tier:
        if gate_tier not in TIERS:
            raise SystemExit(f"--gate expects one of {TIERS}; got {gate_tier!r}")
        entries = read_list(gate_tier)
        missing = verify_list_paths(gate_tier, entries)
        if missing:
            print(f"[gate:{gate_tier}] FAIL-CLOSED: missing files in list ({len(missing)}):")
            for m in missing:
                print("  MISSING", m)
            return 2
        print(f"[gate:{gate_tier}] running {len(entries)} entries ...")
        rc = run_pytest(
            gate_tier, entries,
            fail_fast=args.fail_fast,
            junitxml_out=args.junitxml_out,
            workers=args.workers,
            python=args.python,
            timeout=args.timeout,
        )
        if rc != 0:
            print(f"[gate:{gate_tier}] FAILED (exit {rc}) — production gate is red")
        else:
            print(f"[gate:{gate_tier}] PASSED — all {len(entries)} entries green")
        return rc

    tiers = TIERS if args.tier == "all" else (args.tier,)
    overall = 0
    for tier in tiers:
        entries = read_list(tier)
        missing = verify_list_paths(tier, entries)
        if missing:
            print(f"[{tier}] list references missing files ({len(missing)}); fix the list first")
            overall = overall or 2
            continue
        print(f"[{tier}] running {len(entries)} entries ...")
        junit = args.junitxml_out
        rc = run_pytest(
            tier, entries,
            fail_fast=args.fail_fast,
            junitxml_out=junit,
            workers=args.workers,
            python=args.python,
            timeout=args.timeout,
        )
        if tier == "production_critical":
            if rc != 0:
                print(f"[production_critical] RED (exit {rc})")
                overall = overall or rc
            else:
                print("[production_critical] GREEN")
        else:
            # research/legacy: exit 0, but print the exact failing list.
            if rc != 0 and junit is not None:
                print(f"[{tier}] tracked failures (non-gating):")
                _print_failing_from_xml(junit / f"{tier}.xml")
            elif rc != 0:
                print(f"[{tier}] non-gating tier exited {rc}; run with --junitxml-out to see the failing list")
            else:
                print(f"[{tier}] green")
    return overall


if __name__ == "__main__":
    sys.exit(main())
