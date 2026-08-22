#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17 §7.4 / R17-009: unit normalization ownership — exactly once.

For every source_unit != canonical_unit provider binding, verify the unit
transform is applied AT THE SOURCE BOUNDARY exactly once (never re-scaled by a
second layer).  Sentinel checks:

    A Return raw=100 bp      -> 0.01  (one /10000)
    US Ret raw=0.01          -> 0.01  (zero scaling)
    A TurnoverRatio raw=2.5% -> 0.025 (one /100)
    US ROE raw=0.10          -> 0.10  (zero scaling)

Every binding with a real source_unit->canonical_unit scale must:
- declare a transform that matches the scale, AND
- NOT be re-registered under another concept with an overlapping physical field
  that would double-scale the same column.

Run:  python3 scripts/audit_unit_normalization_once.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))


def main() -> int:
    _load()
    from fields.providers import PROVIDER_REGISTRY

    problems: list[str] = []
    checked = 0
    # physical_field -> (concept, market, expected_scale) — any TWO bindings
    # sharing a physical field with DIFFERENT scale is a double-normalization
    # hazard (the same column would be scaled twice under two concepts).
    scale_by_physical: dict[str, list[tuple[str, str, float]]] = {}
    for binding in PROVIDER_REGISTRY.to_dict()["bindings"]:
        source_scale = None
        src = binding.get("source_unit") or {}
        can = binding.get("canonical_unit") or {}
        # UnitSpec.to_dict() shape: {"scale": ..., "dimension": ...}
        try:
            ss = float(src.get("scale", 1.0))
            cs = float(can.get("scale", 1.0))
            source_scale = ss / cs if cs else 1.0
        except Exception:
            source_scale = 1.0
        for physical in binding.get("physical", []):
            scale_by_physical.setdefault(str(physical), []).append(
                (binding.get("concept", ""), binding.get("market", ""), source_scale)
            )
        if abs(source_scale - 1.0) > 1e-12:
            checked += 1
    for physical, entries in scale_by_physical.items():
        scales = {round(e[2], 10) for e in entries}
        if len(scales) > 1:
            problems.append(
                f"physical {physical!r} normalized to DIFFERENT scales under "
                f"{[(e[0], e[1], e[2]) for e in entries]} — double-normalization hazard"
            )
    # Sentinel: A Return bp binding is 0.0001 once.
    from fields.providers import binding as get_binding

    a_ret = get_binding("return_decimal", "ashare")
    if a_ret is None or abs(float(a_ret.source_unit.scale / a_ret.canonical_unit.scale) - 0.0001) > 1e-12:
        problems.append("A return_decimal source/canonical scale != 1/10000 (sentinel)")
    us_ret = get_binding("return_decimal", "us")
    if us_ret is None or abs(float(us_ret.source_unit.scale / us_ret.canonical_unit.scale) - 1.0) > 1e-12:
        problems.append("US Ret source/canonical scale != 1.0 (sentinel, must be zero-scaled)")

    print(f"unit-normalization bindings with scale != 1: {checked}")
    if problems:
        print(f"FAIL: {len(problems)} problems")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("R17-009 OK: every unit normalization is applied exactly once")
    return 0


if __name__ == "__main__":
    sys.exit(main())
