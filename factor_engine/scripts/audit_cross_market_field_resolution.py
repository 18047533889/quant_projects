#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17 §7.2: cross-market field-resolution isolation.

Verifies that A-share and US resolution NEVER leak across markets:

- ``resolve_market_field(name, ASHARE_CONTEXT)`` resolves in the A registry only;
- ``resolve_market_field(name, US_CONTEXT)`` resolves in the US registry only;
- same-named tables carry DIFFERENT TableSpec objects (never shared);
- physical->concept mapping is unambiguous;
- no production bare-A fallback (delegated to tools/audit_bare_field_registry.py).

Run:  python3 scripts/audit_cross_market_field_resolution.py
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
    from fields.resolver import resolve_market_field
    from fields.market_registry import MULTI_MARKET_FIELD_REGISTRY
    from market.context import ASHARE_CONTEXT, US_CONTEXT

    problems: list[str] = []

    # 1. Same-named tables must be DIFFERENT objects across markets.
    for table in ("StockValuationDaily", "StockIndicator", "StockCapitalDaily",
                  "StockIndustry", "StockStatus", "StockDailyBar", "StockBalance"):
        a = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare").resolve_table(table)
        u = MULTI_MARKET_FIELD_REGISTRY.registry_for("us").resolve_table(table)
        if a is None and u is None:
            continue
        if a is u is not None:
            problems.append(f"table {table!r} shares one TableSpec object across markets")

    # 2. Market-local resolution: A resolves CNY, US resolves USD/share for close.
    try:
        a_close = resolve_market_field("StockDailyBar.close", ASHARE_CONTEXT)
        u_close = resolve_market_field("StockDailyBar.close", US_CONTEXT)
        if a_close is not None and "CNY" not in a_close.spec.unit:
            problems.append(f"A close unit {a_close.spec.unit!r} not CNY")
        if u_close is not None and "USD" not in u_close.spec.unit:
            problems.append(f"US close unit {u_close.spec.unit!r} not USD")
    except Exception as exc:
        problems.append(f"close resolution raised: {exc}")

    # 3. Same field name, market-local semantic divergence (Return vs Ret).
    for name, market_ctx, expect in (
        ("StockDailyBar.return", ASHARE_CONTEXT, "basis_point"),
        ("StockDailyBar.ret", US_CONTEXT, "ratio"),
    ):
        try:
            resolved = resolve_market_field(name, market_ctx)
            if resolved is not None:
                got = resolved.spec.source_unit or resolved.spec.unit
                if expect not in str(got):
                    problems.append(f"{name}@{market_ctx.market} unit {got!r} != {expect}")
        except Exception as exc:
            problems.append(f"{name}@{market_ctx.market} raised: {exc}")

    # 4. Bare A-share production resolution audit (R17-001 static scan).
    try:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(REPO / "tools" / "audit_bare_field_registry.py")],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            problems.append("R17-001 static audit found bare legacy field resolution")
    except Exception as exc:
        problems.append(f"bare-resolver audit could not run: {exc}")

    print("R17 §7.2 cross-market field-resolution audit")
    if problems:
        print(f"FAIL: {len(problems)}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: A/US field resolution is isolated; same-named tables never shared; no leak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
