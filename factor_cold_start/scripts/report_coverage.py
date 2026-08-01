#!/usr/bin/env python3
"""Generate production-admission and research-candidate coverage reports."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from _bootstrap import REPO_ROOT
except ImportError:  # module execution
    REPO_ROOT = Path(__file__).resolve().parents[2]

from factor_cold_start.catalog import load_candidate_catalog, load_catalog
from factor_cold_start.generator import existing_formula_hashes
from factor_cold_start.production_admission import admit_factor, rejected_production_factors

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _production_operator_targets(market: str) -> tuple[set[str], dict[str, str]]:
    """Derive public production targets from live policy and evidence."""
    from backend.operator_capability import production_eligible_backends
    from cleaned_operators import load_all
    from cleaned_operators.operator_spec import infer_production_policy
    from cleaned_operators.operator_surface import DAILY_CANONICALS, EXTENDED_ONLY_CANONICALS

    load_all()
    public = set(DAILY_CANONICALS) | set(EXTENDED_ONLY_CANONICALS)
    targets = {
        name
        for name in public
        if infer_production_policy(name) == "allowed"
        and bool(production_eligible_backends(name))
    }
    excluded: dict[str, str] = {}
    if market == "us" and "real_turnover_rate" in targets:
        targets.remove("real_turnover_rate")
        excluded["real_turnover_rate"] = (
            "requires a verified free-float share-count source; the current US source contract does not guarantee one"
        )
    return targets, excluded


def _row_statistics(rows) -> dict[str, Any]:
    raw_operators = Counter(op for row in rows for op in row.operators)
    canonical_operators: Counter[str] = Counter()
    backend_coverage: Counter[str] = Counter()
    for row in rows:
        admission = admit_factor(row)
        for canonical in admission.canonical_operators:
            canonical_operators[canonical] += 1
        for _, backends in admission.certified_backends:
            for backend in backends:
                backend_coverage[backend] += 1

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
    return {
        "factor_count": len(rows),
        "family_count": len(families),
        "families": dict(sorted(families.items())),
        "availability_tiers": dict(sorted(tiers.items())),
        "complexities": dict(sorted(complexities.items())),
        "horizon_buckets": dict(sorted(horizon_buckets.items())),
        "field_count": len(fields),
        "fields": dict(sorted(fields.items())),
        "operator_count_raw": len(raw_operators),
        "raw_operators": dict(sorted(raw_operators.items())),
        "operator_count_canonical": len(canonical_operators),
        "canonical_operators": dict(sorted(canonical_operators.items())),
        "certified_backend_occurrences": dict(sorted(backend_coverage.items())),
    }


def build_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    catalogs: dict[str, Any] = {}
    total = 0
    all_ids: set[str] = set()

    for market in ("ashare", "us"):
        production_rows = load_catalog(market, "daily")
        key = f"{market}_daily"
        total += len(production_rows)
        all_ids.update(row.factor_id for row in production_rows)
        stats = _row_statistics(production_rows)
        targets, excluded = _production_operator_targets(market)
        covered = targets & set(stats["canonical_operators"])

        candidate_rows: list = []
        seen_hashes: set[str] = set()
        for surface in ("daily", "extended"):
            for row in load_candidate_catalog(market, surface):
                if row.formula_hash in seen_hashes:
                    continue
                seen_hashes.add(row.formula_hash)
                candidate_rows.append(row)
        rejected = rejected_production_factors(candidate_rows)
        rejection_reasons = Counter(
            violation
            for _, result in rejected
            for violation in result.violations
        )

        catalogs[key] = {
            **stats,
            "readiness": "production",
            "output_frequency": "daily",
            "candidate_factor_count": len(candidate_rows),
            "rejected_candidate_count": len(rejected),
            "production_operator_target_count": len(targets),
            "production_operator_covered": len(covered),
            "production_operator_coverage_pct": round(
                100.0 * len(covered) / max(1, len(targets)), 2
            ),
            "missing_production_operators": sorted(targets - covered),
            "source_contract_exclusions": excluded,
            "top_rejection_reasons": dict(rejection_reasons.most_common(100)),
        }

        research_rows = load_catalog(market, "extended")
        research_key = f"{market}_research_archive"
        research_stats = _row_statistics(research_rows)
        catalogs[research_key] = {
            **research_stats,
            "readiness": "research",
            "output_frequency": "daily",
            "production_coverage_claim": False,
        }

    return {
        "schema_version": "factor_cold_start.coverage.v2",
        "summary": {
            "production_factor_count": total,
            "catalog_count": len(catalogs),
            "unique_production_factor_ids": len(all_ids),
            "existing_gtja_week2_formula_keys": len(existing_formula_hashes(repo_root)),
        },
        "catalogs": catalogs,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Factor cold-start production coverage",
        "",
        f"- Production-admitted daily-output factors: **{report['summary']['production_factor_count']}**",
        f"- Catalogs: **{report['summary']['catalog_count']}**",
        f"- Existing GTJA/Week2 formulas excluded structurally: **{report['summary']['existing_gtja_week2_formula_keys']}**",
        "",
        "`daily` now means the default production-admitted daily-output pack. "
        "The research archive is opt-in and makes no production claim.",
        "",
        "## Catalog summary",
        "",
        "| Catalog | Readiness | Factors | Families | Fields | Production operators | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, data in report["catalogs"].items():
        if data["readiness"] == "production":
            coverage = (
                f"{data['production_operator_covered']}/"
                f"{data['production_operator_target_count']}"
            )
            pct = f"{data['production_operator_coverage_pct']}%"
        else:
            coverage = "n/a"
            pct = "n/a"
        lines.append(
            f"| `{key}` | {data['readiness']} | {data['factor_count']} | "
            f"{data['family_count']} | {data['field_count']} | {coverage} | {pct} |"
        )

    for key, data in report["catalogs"].items():
        lines.extend(["", f"## `{key}`", "", "### Families", ""])
        lines.append("| Family | Count |")
        lines.append("|---|---:|")
        for family, count in sorted(
            data["families"].items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"| `{family}` | {count} |")
        lines.extend(["", "### Availability tiers", ""])
        lines.append("| Tier | Count |")
        lines.append("|---|---:|")
        for tier, count in data["availability_tiers"].items():
            lines.append(f"| `{tier}` | {count} |")

        if data["readiness"] == "production":
            if data["missing_production_operators"]:
                lines.extend([
                    "",
                    "### Missing production operators",
                    "",
                    ", ".join(
                        f"`{name}`" for name in data["missing_production_operators"]
                    ),
                ])
            lines.extend(["", "### Source-contract exclusions", ""])
            for op, reason in sorted(data["source_contract_exclusions"].items()):
                lines.append(f"- `{op}`: {reason}")
            lines.extend(["", "### Top production rejection reasons", ""])
            for reason, count in data["top_rejection_reasons"].items():
                lines.append(f"- {count} × `{reason}`")
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
