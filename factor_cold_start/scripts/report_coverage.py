#!/usr/bin/env python3
"""Generate field/operator/family/horizon coverage for the cold-start library."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from _bootstrap import REPO_ROOT
except ImportError:  # module execution
    REPO_ROOT = Path(__file__).resolve().parents[2]

from factor_cold_start.catalog import load_all_catalogs, load_catalog
from factor_cold_start.generator import EXCLUDED_OPERATOR_REASONS, existing_formula_hashes

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _surface_targets() -> dict[str, set[str]]:
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
    )

    return {
        "daily": set(DAILY_CANONICALS),
        "extended": set(EXTENDED_ONLY_CANONICALS),
        "research": set(RESEARCH_ONLY_CANONICALS),
    }


def build_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    targets = _surface_targets()
    catalogs: dict[str, Any] = {}
    total = 0
    all_ids: set[str] = set()
    global_operators: Counter[str] = Counter()
    for market in ("ashare", "us"):
        for surface in ("daily", "extended", "research"):
            rows = load_catalog(market, surface)
            key = f"{market}_{surface}"
            total += len(rows)
            all_ids.update(row.factor_id for row in rows)
            operators = Counter(op for row in rows for op in row.operators)
            global_operators.update(operators)
            fields = Counter(field for row in rows for field in row.required_fields)
            families = Counter(row.family for row in rows)
            tiers = Counter(row.availability_tier for row in rows)
            complexities = Counter(row.complexity for row in rows)
            horizon_buckets = Counter(
                "none"
                if row.horizon is None
                else "1-10"
                if row.horizon <= 10
                else "11-60"
                if row.horizon <= 60
                else "61-252"
                for row in rows
            )
            target = targets[surface]
            covered = target & set(operators)
            catalogs[key] = {
                "factor_count": len(rows),
                "family_count": len(families),
                "families": dict(sorted(families.items())),
                "availability_tiers": dict(sorted(tiers.items())),
                "complexities": dict(sorted(complexities.items())),
                "horizon_buckets": dict(sorted(horizon_buckets.items())),
                "field_count": len(fields),
                "fields": dict(sorted(fields.items())),
                "operator_count_all": len(operators),
                "operators": dict(sorted(operators.items())),
                "surface_operator_count": len(target),
                "surface_operator_covered": len(covered),
                "surface_operator_coverage_pct": round(100.0 * len(covered) / max(1, len(target)), 2),
                "missing_surface_operators": sorted(target - covered),
            }

    active = set().union(*targets.values())
    active_covered = active & set(global_operators)
    return {
        "schema_version": "factor_cold_start.coverage.v2",
        "summary": {
            "factor_count": total,
            "catalog_count": len(catalogs),
            "unique_factor_ids": len(all_ids),
            "existing_gtja_week2_formula_keys": len(existing_formula_hashes(repo_root)),
            "active_operator_count": len(active),
            "active_operator_covered": len(active_covered),
            "active_operator_coverage_pct": round(100.0 * len(active_covered) / max(1, len(active)), 2),
            "missing_active_operators": sorted(active - active_covered),
            "non_authoring_exclusions": dict(sorted(EXCLUDED_OPERATOR_REASONS.items())),
        },
        "surface_targets": {key: sorted(value) for key, value in targets.items()},
        "global_operators": dict(sorted(global_operators.items())),
        "catalogs": catalogs,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Factor cold-start coverage",
        "",
        f"- Total factors: **{summary['factor_count']}**",
        f"- Catalogs: **{summary['catalog_count']}**",
        f"- Active daily/extended/research operators: **{summary['active_operator_covered']}/{summary['active_operator_count']} ({summary['active_operator_coverage_pct']}%)**",
        f"- Existing GTJA/Week2 formulas excluded structurally: **{summary['existing_gtja_week2_formula_keys']}**",
        "",
        "## Catalog summary",
        "",
        "| Catalog | Factors | Families | Fields | Surface operators | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, data in report["catalogs"].items():
        lines.append(
            f"| `{key}` | {data['factor_count']} | {data['family_count']} | {data['field_count']} | "
            f"{data['surface_operator_covered']}/{data['surface_operator_count']} | {data['surface_operator_coverage_pct']}% |"
        )
    lines.extend([
        "",
        "## Coverage policy",
        "",
        "- Every active canonical on `daily`, `extended`, and `research` appears in each market's matching catalog.",
        "- Fiscal operators are isolated behind the `fundamental` availability tier and require PIT-aligned disclosure data.",
        "- Intraday-only operators are isolated behind the `intraday` tier.",
        "- Singular or explosive transforms are used only on explicitly bounded, singularity-free inputs and remain research-default-off.",
        "- Internal implementation primitives and deprecated legacy aliases are not cold-start authoring targets.",
        "",
        "### Non-authoring exclusions",
        "",
    ])
    for op, reason in summary["non_authoring_exclusions"].items():
        lines.append(f"- `{op}`: {reason}")

    for key, data in report["catalogs"].items():
        lines.extend(["", f"## `{key}`", "", "### Families", ""])
        lines.append("| Family | Count |")
        lines.append("|---|---:|")
        for family, count in sorted(data["families"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{family}` | {count} |")
        lines.extend(["", "### Availability tiers", ""])
        lines.append("| Tier | Count |")
        lines.append("|---|---:|")
        for tier, count in data["availability_tiers"].items():
            lines.append(f"| `{tier}` | {count} |")
        if data["missing_surface_operators"]:
            lines.extend([
                "",
                "### Missing active surface operators",
                "",
                ", ".join(f"`{x}`" for x in data["missing_surface_operators"]),
            ])
    lines.append("")
    return "\n".join(lines)


def write_reports(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    report = build_report(repo_root)
    out = PACKAGE_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "coverage.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    report = write_reports(REPO_ROOT)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
