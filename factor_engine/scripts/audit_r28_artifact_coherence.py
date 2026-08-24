#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R28 §八十六: artifact coherence — the per-canonical evidence must cover the
same canonical set as the current registry/catalog.

Checks:
- set(inventory canonicals) == set(registry canonicals) == set(catalog canonicals)
- every canonical has a disposition row in R28_CANONICAL_TEST_COVERAGE
- no evidence row for a canonical that no longer exists

Run:  python3 scripts/audit_r28_artifact_coherence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evidence" / "factor_engine" / "r28"


def _load() -> None:
    sys.path.insert(0, str(REPO))


def main() -> int:
    _load()
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    registry = set(OperatorRegistry.list_canonical())
    catalog = set(OperatorRegistry.catalog().keys())

    inventory = set()
    inv_path = OUT / "R28_OPERATOR_INVENTORY.json"
    if inv_path.exists():
        inventory = {row["canonical"] for row in json.loads(inv_path.read_text())["operators"]}

    coverage = set()
    cov_path = OUT / "R28_CANONICAL_TEST_COVERAGE.json"
    if cov_path.exists():
        coverage = {row["canonical"] for row in json.loads(cov_path.read_text())["rows"]}

    problems = []
    if inventory != registry:
        problems.append(
            f"inventory({len(inventory)}) != registry({len(registry)}) "
            f"extra_inv={sorted(inventory - registry)[:10]} missing_inv={sorted(registry - inventory)[:10]}"
        )
    if coverage != registry:
        problems.append(
            f"coverage({len(coverage)}) != registry({len(registry)}) "
            f"extra_cov={sorted(coverage - registry)[:10]} missing_cov={sorted(registry - coverage)[:10]}"
        )
    if catalog != registry:
        problems.append(
            f"catalog({len(catalog)}) != registry({len(registry)}) "
            f"diff={sorted(catalog ^ registry)[:10]}"
        )

    ok = not problems
    print(f"R28 artifact coherence: registry={len(registry)} inventory={len(inventory)} "
          f"coverage={len(coverage)} catalog={len(catalog)}")
    for p in problems:
        print(f"  !! {p}")
    if ok:
        print("  R28_CANONICAL_SET_COHERENT == TRUE")
        return 0
    print("  R28_CANONICAL_SET_COHERENT == FALSE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
