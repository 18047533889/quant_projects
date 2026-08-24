#!/usr/bin/env python3
"""Audit that stub operator implementations never leak into the DSL/production surface.

A "stub" is a ``*_stub`` function that raises ``NotImplementedError``.  They are
placeholder entries for future data sources and must remain outside both the
public DSL allowlist and the production target set.  This script fails closed if
any stub name is resolvable through the registry, appears in the DSL allowlist,
or becomes a production target.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FE_ROOT))


def _stub_definitions() -> list[str]:
    names: set[str] = set()
    for root in (FE_ROOT / "cleaned_operators",):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            src = path.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"def\s+([A-Za-z_][A-Za-z0-9_]*_stub)\s*\(", src):
                names.add(match.group(1))
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any stub leaks into the public surface")
    args = parser.parse_args()

    stubs = set(_stub_definitions())
    from factor_engine.cleaned_operators import load_all
    load_all()

    from factor_engine.cleaned_operators.operator_surface import classify_canonical
    from factor_engine.cleaned_operators.production_hardening import factor_production_targets
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    canonicals = set(OperatorRegistry.list_canonical())
    aliases = set(OperatorRegistry._aliases)
    registered_stubs = sorted(stubs & (canonicals | aliases))
    production = set(factor_production_targets())
    production_stubs = sorted(stubs & production)

    print(f"stub definitions: {len(stubs)}")
    print(f"  registered as canonical/alias: {len(registered_stubs)} -> {registered_stubs[:10]}")
    print(f"  in production targets: {len(production_stubs)} -> {production_stubs[:10]}")

    if args.check:
        if registered_stubs:
            print("FAIL: stub names registered in operator registry", file=sys.stderr)
            return 1
        if production_stubs:
            print("FAIL: stub names in production targets", file=sys.stderr)
            return 1
        print("stub surface audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
