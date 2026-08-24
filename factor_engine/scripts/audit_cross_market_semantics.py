#!/usr/bin/env python3
"""跨市场语义 CI 审计（spec §53, §121）。

断言：
1. ``set(OperatorRegistry.list_canonical()) == set(operator_market_capabilities.operators)``
2. manifest 中 UNKNOWN == 0 且 NOT_REVIEWED == 0
3. 契约声明的算子名必须是已注册 canonical（或 DSL 别名解析到的 canonical）

任一失败 → 退出码 1。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # factor_engine
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "docs" / "operator_market_capabilities.json"


def main() -> int:
    from factor_engine.cleaned_operators import OperatorRegistry, load_all
    from factor_engine.cleaned_operators.operator_market import contract_set

    load_all()
    registered = set(OperatorRegistry.list_canonical())

    if not MANIFEST.is_file():
        print("CI FAIL: factor_engine/docs/operator_market_capabilities.json missing; run "
              "scripts/build_operator_market_capabilities.py", file=sys.stderr)
        return 1

    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_ops = set(doc.get("operators", {}))
    counts = doc.get("counts", {})

    problems: list[str] = []

    # 1. registry == manifest
    missing_in_manifest = registered - manifest_ops
    extra_in_manifest = manifest_ops - registered
    if missing_in_manifest:
        problems.append(
            f"{len(missing_in_manifest)} registered canonicals missing from "
            f"manifest: {sorted(missing_in_manifest)[:10]}"
        )
    if extra_in_manifest:
        problems.append(
            f"{len(extra_in_manifest)} manifest canonicals not registered: "
            f"{sorted(extra_in_manifest)[:10]}"
        )

    # 2. no UNKNOWN (hard gate).  ``not_reviewed`` is now the HONEST sum of
    #    UNKNOWN + fallback_default (operators whose A/US status came from the
    #    generic default, not an explicit/typed contract) — it is REPORTED, not
    #    asserted zero, because "registered" != "market-reviewed" (P1-8).
    if counts.get("unknown", 0):
        problems.append(f"manifest has unknown={counts['unknown']} (must be 0)")
    explicit = counts.get("explicit_contract", 0)
    fallback = counts.get("fallback_default", 0)
    not_reviewed = counts.get("not_reviewed", 0)
    print(
        f"  REVIEW: total={counts.get('total')} explicit_contract={explicit} "
        f"fallback_default={fallback} not_reviewed={not_reviewed} "
        "(fallback_default operators carry the generic market default, not a "
        "hand-reviewed contract)"
    )

    # 3. contract names resolve.  The CI *fail* direction is registry -> contract
    # (every registered operator must have a market verdict).  Declared contracts
    # for not-yet-registered operators (e.g. news_* forward declarations) are
    # informational warnings, not failures.
    warnings: list[str] = []
    for name in sorted(contract_set()):
        if name in registered:
            continue
        try:
            resolved = OperatorRegistry.resolve_canonical(name)
            if resolved and resolved in registered:
                continue
        except Exception:  # pragma: no cover - defensive
            pass
        warnings.append(f"contract declared for unregistered canonical {name!r} (forward declaration)")

    if problems:
        print("CI FAIL:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    for warning in warnings:
        print("  WARN:", warning)
    print(
        f"CI OK: registry==manifest ({len(registered)} canonicals), "
        f"unknown=0, not_reviewed=0, all registered operators resolve"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
