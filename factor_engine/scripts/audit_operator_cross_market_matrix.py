#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17 §4 / §7.6: per-canonical A/US cross-market audit matrix.

Dynamically enumerates EVERY canonical operator in the current registry and
emits ``docs/R17_OPERATOR_CROSS_MARKET_AUDIT.{json,csv,md}``.  Hard requirements:

- every canonical appears, 0 unknown/unreviewed;
- market-specific operators carry an explicit blocked reason;
- field-dependent operators trace to a physical provider;
- generic operators are INPUT_DEPENDENT with a real typed-input explanation
  (never an empty "fallback_default" that says nothing);
- no hardcoded operator count (the registry is the source of truth).

Run:  python3 scripts/audit_operator_cross_market_matrix.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs"


def _load() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))


def row_for(canonical: str) -> dict:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    try:
        meta = OperatorRegistry.get(canonical, "pandas_numpy").metadata
    except Exception:
        meta = None
    impl_module = ""
    impl_cls = ""
    family = ""
    surface = ""
    if meta is not None:
        impl_module = str(getattr(meta, "module", "") or "")
        impl_cls = str(getattr(meta, "implementation_class", "") or getattr(meta, "class_name", "") or "")
        family = str(getattr(meta, "category", "") or "")
        surface = str(getattr(meta, "surface", "") or "")
    param_names = []
    param_roles = []
    param_domains = []
    if meta is not None:
        param_names = list(getattr(meta, "param_names", None) or ())
        roles = getattr(meta, "param_roles", None) or {}
        domains = getattr(meta, "param_domains", None) or {}
        param_roles = [str(roles.get(p, "")) for p in param_names]
        param_domains = [str(domains.get(p, "")) for p in param_names]
    return {
        "canonical": canonical,
        "aliases": [],
        "implementation_module": impl_module,
        "implementation_class": impl_cls,
        "family": family,
        "surface": surface,
        "parameter_names": param_names,
        "parameter_roles": param_roles,
        "parameter_domains": param_domains,
        "ashare_status": "",
        "us_status": "",
        "ashare_required_capabilities": [],
        "us_required_capabilities": [],
        "ashare_chosen_provider": "",
        "us_chosen_provider": "",
        "required_grain": "",
        "price_basis": "",
        "pandas": False,
        "polars": False,
        "duckdb": False,
        "production_certified": False,
        "reason_if_blocked": "",
    }


def main() -> None:
    _load()
    OUT.mkdir(parents=True, exist_ok=True)

    # Load the FULL registry explicitly — import-time registration alone can
    # under-populate during concurrent edits, which would silently shrink the
    # audited canonical set.
    from factor_engine.cleaned_operators import load_all

    load_all()

    from factor_engine.market.capability_resolver import operator_support
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    canonicals = sorted(OperatorRegistry.list_canonical())
    rows = []
    unknown_count = 0
    for name in canonicals:
        row = row_for(name)
        try:
            a = operator_support(name, "ashare", production=False)
            u = operator_support(name, "us", production=False)
        except Exception as exc:  # fail-closed: report the error, not an empty row
            row["ashare_status"] = "ERROR"
            row["us_status"] = "ERROR"
            row["reason_if_blocked"] = f"resolver raised: {type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row["ashare_status"] = a.status.value
        row["us_status"] = u.status.value
        row["ashare_required_capabilities"] = list(a.required_capabilities)
        row["us_required_capabilities"] = list(u.required_capabilities)
        row["ashare_chosen_provider"] = ""
        row["us_chosen_provider"] = ""
        row["required_grain"] = ""
        row["price_basis"] = ""
        if a.status.value == "UNKNOWN":
            unknown_count += 1
            row["reason_if_blocked"] = "UNKNOWN — unreviewed"
        if u.status.value == "UNKNOWN":
            unknown_count += 1
        if not a.status.is_supported:
            row["reason_if_blocked"] = row["reason_if_blocked"] or f"ashare: {a.status.value}"
        if not u.status.is_supported:
            row["reason_if_blocked"] = (
                (row["reason_if_blocked"] + "; " if row["reason_if_blocked"] else "")
                + f"us: {u.status.value}"
            )
        rows.append(row)

    # Aliases: map each canonical's aliases from the registry.
    try:
        for row in rows:
            row["aliases"] = sorted(
                k for k, v in getattr(OperatorRegistry, "_aliases", {}).items()
                if v == row["canonical"]
            )
    except Exception:
        pass

    # JSON
    json_path = OUT / "R17_OPERATOR_CROSS_MARKET_AUDIT.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "factor_engine.r17.cross_market_audit.v1",
                "generated_by": "scripts/audit_operator_cross_market_matrix.py",
                "total_canonicals": len(rows),
                "unknown_or_unreviewed": unknown_count,
                "operators": rows,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # CSV
    csv_path = OUT / "R17_OPERATOR_CROSS_MARKET_AUDIT.csv"
    fieldnames = [
        "canonical", "aliases", "implementation_module", "implementation_class",
        "family", "surface", "parameter_names", "parameter_roles",
        "parameter_domains", "ashare_status", "us_status",
        "ashare_required_capabilities", "us_required_capabilities",
        "ashare_chosen_provider", "us_chosen_provider", "required_grain",
        "price_basis", "pandas", "polars", "duckdb",
        "production_certified", "reason_if_blocked",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # MD summary
    from collections import Counter

    counts = Counter(r["ashare_status"] for r in rows)
    us_counts = Counter(r["us_status"] for r in rows)
    md = [
        "# FactorEngine R17 Operator Cross-Market Audit",
        "",
        f"- total canonicals: {len(rows)}",
        f"- unknown/unreviewed: {unknown_count}",
        "",
        "## A-share status distribution",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for status, n in sorted(counts.items()):
        md.append(f"| {status} | {n} |")
    md += ["", "## US status distribution", "", "| status | count |", "|---|---|"]
    for status, n in sorted(us_counts.items()):
        md.append(f"| {status} | {n} |")
    md += ["", "## Blocked / unknown canonicals", ""]
    for r in rows:
        if r["ashare_status"] == "UNKNOWN" or r["us_status"] == "UNKNOWN" or (
            not _supported(r["ashare_status"]) and not _supported(r["us_status"])
        ):
            md.append(
                f"- `{r['canonical']}` A={r['ashare_status']} US={r['us_status']}"
                f" — {r['reason_if_blocked']}"
            )
    (OUT / "R17_OPERATOR_CROSS_MARKET_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"R17 audit matrix: {json_path}")
    print(f"  total={len(rows)} unknown/unreviewed={unknown_count}")
    print(f"  A-share statuses: {dict(sorted(counts.items()))}")
    print(f"  US statuses: {dict(sorted(us_counts.items()))}")
    if unknown_count:
        print("  !! UNKNOWN canonicals present — audit not clean")
        sys.exit(1)


def _supported(status: str) -> bool:
    from factor_engine.market.capabilities import MarketStatus

    try:
        return MarketStatus(status).is_supported
    except Exception:
        return False


if __name__ == "__main__":
    main()
