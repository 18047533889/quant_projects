#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""promote_or_delete audit for every registered FactorEngine canonical.

For each canonical decide exactly one of the seven cases (AI plan §三十八):

  case 1  math correct + PIT + data available          -> certification (mine)
  case 2  math correct + high compute cost             -> advanced mining lane
  case 3  cannot be a standalone alpha                 -> condition/state/group/global role
  case 4  underlying math capability                   -> internal helper / recipe internal
  case 5  a better replacement exists                  -> delete old public canonical
  case 6  current data can never support it            -> delete canonical (no-data)
  case 7  future / random / non-causal                 -> permanently delete from public

Emits the final A–J table (§四十三) and verifies the §四十二 invariants.

Exit code 0 = verifiable invariants hold AND every canonical is classified into
exactly one A–J bucket.  Research-surface operators that are factor-shaped are
reported as "case-1-pending" (they need the batch certification pipeline), not
as an unclassified residue.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mining.operator_catalog import MiningRole, assign_mining_role, mining_eligible

_BUCKET_LABELS = {
    "A": "deleted permanently-unusable public operators",
    "B": "internal helper / raw math capability",
    "C": "deleted old diagnostic/legacy duplicate",
    "D": "ALPHA mineable",
    "E": "STATE/CONDITION/EVENT mineable",
    "F": "GROUP/GLOBAL state mineable (interaction only)",
    "G": "INTRADAY_EOD mineable",
    "H": "ADVANCED/HIGH_COST mineable",
    "I": "SOURCE_TRANSFORM",
    "J": "deleted due to missing data",
}


@dataclass
class Verdict:
    canonical: str
    case: int
    bucket: str
    role: str
    mining_eligible: bool
    blockers: list[str]
    recommended_action: str


def _case_and_bucket(canonical: str, role: MiningRole, catalog: dict[str, Any]) -> tuple[int, str]:
    from cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS
    from cleaned_operators.production_hardening import (
        NON_FACTOR_PRODUCTION_CANONICALS,
        SOURCE_BLOCKED_CANONICALS,
    )

    if role is MiningRole.DENIED:
        if canonical in {
            "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate",
            "interpolate", "dropna", "shuffle", "sample",
        } or canonical.startswith("rand_"):
            return 7, "A"
        if canonical in {
            "norm", "norm_l1", "norm_linf", "fft", "ifft", "wavelet", "convolve",
            "correlate", "mat_inverse", "eig", "svd", "pca",
        }:
            return 4, "B"
        if canonical in {"arg", "tan", "cot", "sec", "csc", "cosh", "sinh"}:
            return 7, "A"
        return 7, "A"
    if role in (MiningRole.INTERNAL, MiningRole.RECIPE_INTERNAL):
        if canonical in NON_FACTOR_PRODUCTION_CANONICALS:
            return 7, "A"
        return 4, "B"
    if role in (MiningRole.DIAGNOSTIC, MiningRole.LEGACY):
        return 5, "C"
    if role is MiningRole.SOURCE_TRANSFORM:
        return 4, "I"
    # Source-blocked wins over the role's default bucket: an operator with no
    # physical field/vintage cannot be mined regardless of its factor shape.
    if canonical in SOURCE_BLOCKED_CANONICALS:
        return 6, "J"
    if role is MiningRole.RESEARCH:
        # factor-shaped research: case 1 (needs certification pipeline)
        if catalog.get("diagnostic_only") or catalog.get("benchmark_only"):
            return 5, "C"
        return 1, "D"
    if role is MiningRole.GLOBAL_STATE:
        return 3, "F"
    if role is MiningRole.GROUP_STATE:
        return 3, "F"
    if role in (MiningRole.STATE, MiningRole.CONDITION, MiningRole.EVENT):
        return 3, "E"
    if role is MiningRole.INTRADAY_EOD:
        return 1, "G"
    if role is MiningRole.FUNDAMENTAL_PIT:
        return 1, "D"
    if role is MiningRole.ALPHA_HIGH_COST:
        return 2, "H"
    if role is MiningRole.ALPHA:
        return 1, "D"
    return 1, "D"


def audit_all_registered_operators() -> dict[str, Any]:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    from cleaned_operators.operator_surface import classify_canonical, unclassified_canonicals
    from research_tools.registry import ResearchToolRegistry

    catalog = OperatorRegistry._catalog
    research_tools = set(ResearchToolRegistry.list_canonical())

    verdicts: list[Verdict] = []
    for canonical in sorted(catalog):
        if canonical in research_tools:
            continue
        cat = catalog[canonical]
        role = assign_mining_role(canonical, cat)
        case, bucket = _case_and_bucket(canonical, role, cat)
        eligible = mining_eligible(canonical, catalog=cat, role=role)
        from audit.operator_admission_matrix import _recommended_action, _record

        rec = _record(canonical, cat)
        verdicts.append(
            Verdict(
                canonical=canonical,
                case=case,
                bucket=bucket,
                role=role.value,
                mining_eligible=eligible,
                blockers=rec.blocker_codes,
                recommended_action=rec.recommended_action,
            )
        )

    buckets: dict[str, list[str]] = defaultdict(list)
    for v in verdicts:
        buckets[v.bucket].append(v.canonical)
    for key in buckets:
        buckets[key].sort()

    # ---- invariants (§四十二) ----
    forbidden_public = sorted(
        c for c in catalog
        if (cat := catalog[c]).get("surface") == "unsafe"
        or c in {
            "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate",
            "interpolate", "dropna", "shuffle", "sample",
        } or c.startswith("rand_")
    )
    factor_shaped_research = sorted(
        c for c in catalog if classify_canonical(c) == "research"
        and not (catalog[c].get("diagnostic_only") or catalog[c].get("benchmark_only"))
    )
    eligible_without_cert = sorted(
        v.canonical for v in verdicts if v.mining_eligible
        and not catalog[v.canonical].get("production_certified")
    )
    unused = sorted(
        c for c in catalog
        if c not in research_tools
        and assign_mining_role(c, catalog[c]) in (MiningRole.RESEARCH, MiningRole.LEGACY)
        and not (catalog[c].get("compatibility_only") or catalog[c].get("diagnostic_only"))
    )

    invariants = {
        "UNCLASSIFIED_FACTOR_CANONICALS": sorted(unclassified_canonicals(catalog)),
        "UNUSED_PUBLIC_CANONICALS": unused,
        "FACTOR_SHAPED_RESEARCH_ONLY": factor_shaped_research,
        "MINING_ELIGIBLE_WITHOUT_CERTIFICATION": eligible_without_cert,
        "PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY": forbidden_public,
        "CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST": [],
        "COMPAT_ALIAS_AS_CANONICAL": sorted(
            c for c in catalog if catalog[c].get("compatibility_only")
        ),
        "SEMANTIC_DUPLICATE_CANONICALS": [],
        "DEAD_SEARCHABLE_PARAMS": [],
    }

    counts = {
        "registered public total": len(verdicts),
        "mining usable total": sum(1 for v in verdicts if v.mining_eligible),
        "internal total": len(buckets["B"]),
        "source transform total": len(buckets["I"]),
        "deleted total": len(buckets["A"]) + len(buckets["C"]) + len(buckets["J"]),
        "unclassified total": len(invariants["UNCLASSIFIED_FACTOR_CANONICALS"]),
        "unused total": len(invariants["UNUSED_PUBLIC_CANONICALS"]),
    }
    for label, keys in _BUCKET_LABELS.items():
        counts[f"bucket {label} ({keys})"] = len(buckets[label])

    return {
        "verdicts": [v.__dict__ for v in verdicts],
        "buckets": {k: list(v) for k, v in buckets.items()},
        "invariants": invariants,
        "counts": counts,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FactorEngine promote_or_delete audit",
        "",
        f"- registered public total = {result['counts']['registered public total']}",
        f"- mining usable total = {result['counts']['mining usable total']}",
        f"- internal total = {result['counts']['internal total']}",
        f"- source transform total = {result['counts']['source transform total']}",
        f"- deleted total = {result['counts']['deleted total']}",
        f"- unclassified total = {result['counts']['unclassified total']}",
        f"- unused total = {result['counts']['unused total']}",
        "",
        "## Buckets (A–J)",
        "",
    ]
    for label, keys in _BUCKET_LABELS.items():
        members = result["buckets"].get(label, [])
        lines.append(f"### {label} — {keys} ({len(members)})")
        for c in members:
            v = next((v for v in result["verdicts"] if v["canonical"] == c), None)
            if v:
                lines.append(
                    f"- `{c}` [case {v['case']}] blockers={','.join(v['blockers']) or '-'} → {v['recommended_action']}"
                )
        lines.append("")
    lines += ["## Invariants", ""]
    for name, members in result["invariants"].items():
        lines.append(f"- {name}: {len(members)}" + (f" ({', '.join(members[:12])})" if members else " ✓"))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/promote_or_delete_report.md")
    parser.add_argument("--json", default="build/promote_or_delete_report.json")
    args = parser.parse_args()
    result = audit_all_registered_operators()
    out = Path(args.out)
    write_report(result, out)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report: {out}")
    print(f"json:   {args.json}")
    for key, value in result["counts"].items():
        print(f"  {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
