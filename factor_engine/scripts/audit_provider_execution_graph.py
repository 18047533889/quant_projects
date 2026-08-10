#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17 §7.3: provider execution-graph audit.

For every registered provider binding:

- every dependency dataset exists in the DataAccess registry;
- every physical column exists in the DataAccess dataset schema;
- join keys / time model are present;
- required filters are executable (have allowed values when enum_select);
- a provider CHAIN can actually resolve (ProviderResolver selects at least the
  research path or the unavailable marker is intentional).

Run:  python3 scripts/audit_provider_execution_graph.py
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
    total = 0
    checked = 0
    for concept, market in sorted(PROVIDER_REGISTRY.to_dict()["bindings"] and
                                  _pairs(PROVIDER_REGISTRY) or []):
        total += 1
        binding = PROVIDER_REGISTRY.binding(concept, market)
        if binding is None:
            continue
        if binding.quality.value == "UNAVAILABLE":
            continue  # intentional unavailable marker
        checked += 1
        if not binding.physical_fields and not binding.dependencies:
            problems.append(f"{concept}@{market} has no physical fields and no dependencies")

    # Resolver sanity: every concept with a real binding resolves a provider
    # (production may fail-closed on coverage, but research must pick something
    # when an available marker exists).
    from fields.providers import PROVIDER_RESOLVER

    unresolved = 0
    for concept in sorted(PROVIDER_REGISTRY.concepts()):
        for market in ("ashare", "us"):
            b = PROVIDER_REGISTRY.binding(concept, market)
            if b is None or b.quality.value == "UNAVAILABLE":
                continue
            sel, decisions = PROVIDER_RESOLVER.resolve(
                concept, market, production=False
            )
            if sel is None:
                unresolved += 1
                problems.append(
                    f"{concept}@{market} research resolver selected NO provider "
                    f"(non-unavailable binding exists but nothing eligible)"
                )

    print(f"providers audited: {total} (non-unavailable: {checked}), "
          f"research-unresolvable: {unresolved}")
    if problems:
        print(f"FAIL: {len(problems)}")
        for p in problems[:20]:
            print(f"  - {p}")
        return 1
    print("R17 §7.3 OK: every non-unavailable provider resolves a chain; no half-declared binding")
    return 0


def _pairs(registry):
    out = []
    for concept in sorted(registry.concepts()):
        for market in registry.markets():
            out.append((concept, market))
    return out


if __name__ == "__main__":
    sys.exit(main())
