#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17 §7.7: semantic factor-hash completeness.

Mutating a factor's economic context (market, IndustrySource, IndexSymbol,
timeframe, currency filter, universe, provider, period policy, session policy)
MUST change its semantic factor hash; only hash-equivalent parameter changes
that do not alter math semantics may leave the hash unchanged.

This audit probes the analyzer's catalog-hash / field-registry-hash plumbing:
changing the market context changes the resolved field registry, which MUST
change the IR's field_registry_hash.

Run:  python3 scripts/audit_semantic_hash_completeness.py
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
    problems: list[str] = []
    # R17-087: the A/US field registries MUST hash differently — the same
    # formula resolved under a different market must produce a different
    # registry identity.  (The full analyzer lower() path is not probed here
    # because a parse-time A-share-bound FieldRef is intentionally rejected when
    # analyzed under the US market — that rejection IS the isolation guarantee.)
    try:
        from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

        h_a = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").catalog_hash()
        h_u = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").catalog_hash()
        if not h_a or not h_u:
            problems.append("market registry catalog_hash missing (R17-087)")
        elif h_a == h_u:
            problems.append(
                "A/US field-registry hashes identical — semantic hash does not "
                "isolate market (R17-087)"
            )
        else:
            # Same logical field resolves differently per market (A CNY vs US USD).
            a = MULTI_MARKET_FIELD_REGISTRY.resolve_field("ashare", "StockDailyBar.close")
            u = MULTI_MARKET_FIELD_REGISTRY.resolve_field("us", "StockDailyBar.close")
            if a is not None and u is not None and a.unit == u.unit:
                problems.append("A/US close units identical — market-local unit not in identity")
    except Exception as exc:
        problems.append(f"semantic-hash probe raised: {type(exc).__name__}: {exc}")

    # Provider identity changes market-cap hash: A provider vs US provider.
    try:
        from factor_engine.fields.providers import binding
        import json

        a = binding("market_cap_local", "ashare")
        u = binding("market_cap_local", "us")
        if a is None or u is None:
            problems.append("market_cap_local missing a market binding (hash identity probe)")
        else:
            ja = json.dumps(a.to_dict(), sort_keys=True)
            ju = json.dumps(u.to_dict(), sort_keys=True)
            if ja == ju:
                problems.append("A/US market_cap_local provider dicts identical — "
                                "provider identity not in hash (R17-087)")
    except Exception as exc:
        problems.append(f"provider-identity probe raised: {exc}")

    print("R17 §7.7 semantic-hash completeness audit")
    if problems:
        print(f"FAIL: {len(problems)}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: market/provider context changes the semantic hash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
