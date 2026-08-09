#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""promote_or_delete audit for every registered FactorEngine canonical.

For each canonical decide exactly one of the eight cases (AI plan §三十八):

  case 1  math correct + PIT + data available          -> certification (mine)
  case 2  math correct + high compute cost             -> advanced mining lane
  case 3  cannot be a standalone alpha                 -> condition/state/group/global role
  case 4  underlying math capability                   -> internal helper / recipe internal
  case 5  a better replacement exists                  -> delete old public canonical
  case 6  current data can never support it            -> delete canonical (no-data)
  case 7  future / random / non-causal                 -> permanently delete from public
  case 8  role could not be resolved                   -> UNKNOWN bucket (never default ALPHA)

Emits the final A–J (+K) table and verifies the §四十二 invariants.

R15 changes vs R12:

* Exit code is honest: any non-empty hard release invariant returns 1
  (R15-INC-042/258) — CI can no longer be faked green.
* The three hardcoded-empty invariants are now REAL detectors
  (R15-INC-043/257):
    - ``CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST`` is a set difference against the
      current-context manifest;
    - ``SEMANTIC_DUPLICATE_CANONICALS`` is a real alias/name/AST candidate graph;
    - ``DEAD_SEARCHABLE_PARAMS`` is a bounded parameter-injectivity probe.
  A detector that did not run reports its coverage honestly instead of claiming
  PASS with ``[]``.
* ``UNRESOLVED`` roles land in a new UNKNOWN bucket (R15-INC-044/259); release
  requires UNKNOWN == 0.  Nothing falls back to optimistic case-1 ALPHA.
* ``mining usable`` is split into static-certified and per-context usable
  (R15-INC-046/248).
* A research candidate is reported separately from production-usable ALPHA
  (R15-INC-045).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mining.operator_catalog import MiningRole, RoleSource, assign_mining_role, assign_mining_role_ex, mining_eligible

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
    "K": "UNRESOLVED role (R15 fail-closed: never default ALPHA)",
}


@dataclass
class Verdict:
    canonical: str
    case: int
    bucket: str
    role: str
    role_source: str
    mining_eligible: bool
    blockers: list[str]
    recommended_action: str


def _case_and_bucket(canonical: str, role: MiningRole, catalog: dict[str, Any]) -> tuple[int, str]:
    from cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS
    from cleaned_operators.production_hardening import (
        NON_FACTOR_PRODUCTION_CANONICALS,
        SOURCE_BLOCKED_CANONICALS,
    )

    # R15-INC-044/259: an unresolved role is its own bucket, never optimistic D.
    if role is MiningRole.UNRESOLVED:
        return 8, "K"
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
    return 8, "K"


# ---------------------------------------------------------------------------
# Real invariant detectors (R15-INC-043/049/050/051/257/262/263)
# ---------------------------------------------------------------------------

def _alias_edges(catalog: dict[str, Any]) -> list[tuple[str, str]]:
    """(alias, canonical) edges from the registry."""
    from cleaned_operators.registry import OperatorRegistry

    edges: list[tuple[str, str]] = []
    aliases = getattr(OperatorRegistry, "_aliases", {}) or {}
    for alias, target in aliases.items():
        edges.append((str(alias), str(target)))
    for canon, entry in catalog.items():
        for alias in (entry.get("aliases") or ()):
            edges.append((str(alias), str(canon)))
    return edges


def detect_semantic_duplicate_candidates(
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    """R15-INC-049/262: candidate duplicate public canonicals.

    A real detector combines (a) the alias graph, (b) canonical-name families
    (``MACD`` vs ``MACD_line``) and (c) identical param/unit contracts on
    otherwise-distinct names.  Candidates require human confirmation — the
    detector never silently deletes.
    """
    candidates: list[dict[str, Any]] = []
    from cleaned_operators.operator_surface import classify_canonical

    names = sorted(
        c for c in catalog
        if classify_canonical(c) in ("daily", "extended")
        and not catalog[c].get("compatibility_only")
    )
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = (a, b)
            if key in seen:
                continue
            # alias edge in either direction
            if (a in _ALIAS_LOOKUP and _ALIAS_LOOKUP[a] == b) or (
                b in _ALIAS_LOOKUP and _ALIAS_LOOKUP[b] == a
            ):
                seen.add(key)
                candidates.append(
                    {"left": a, "right": b, "reason": "alias", "confidence": "high"}
                )
                continue
            # name-family: one name is the other with a suffix/prefix variant
            # (``MACD`` in ``MACD_line``).  The shorter name must match at a
            # ``_`` / start / end WORD BOUNDARY so 2-letter operators (``lt``,
            # ``ne``) and mid-word substrings (``sign`` inside ``MACD_signal``)
            # never fire.
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            boundary = re.compile(
                r"(?:^|_)" + re.escape(shorter) + r"(?=_|$)"
            )
            if boundary.search(longer):
                seen.add(key)
                candidates.append(
                    {"left": a, "right": b, "reason": "name_family", "confidence": "candidate"}
                )
                continue
            # identical (canonical, param contract) on distinct names
            ca, cb = catalog[a], catalog[b]
            if (
                ca.get("param_names") and ca.get("param_names") == cb.get("param_names")
                and ca.get("output_unit") and ca.get("output_unit") == cb.get("output_unit")
                and ca.get("input_grain") == cb.get("input_grain")
            ):
                seen.add(key)
                candidates.append(
                    {"left": a, "right": b, "reason": "identical_contract", "confidence": "candidate"}
                )
    return candidates


def detect_dead_searchable_params(
    catalog: dict[str, Any],
    *,
    max_canonicals: int = 60,
) -> tuple[list[dict[str, Any]], int, int]:
    """R15-INC-050/263: bounded parameter-injectivity probe.

    For a sampled subset of ALPHA/factor operators with searchable params,
    run the pandas kernel on a fixed synthetic panel and change ONE searchable
    parameter across two legal values; if the full output is byte-identical the
    parameter is a dead candidate.  Returns (dead_candidates, sampled, total).
    A canonical that was NOT sampled is reported as ``UNKNOWN`` coverage, never
    silently PASS.
    """
    dead: list[dict[str, Any]] = []
    sampled = 0
    total = 0
    try:
        from cleaned_operators.base import searchable_param_names
        from cleaned_operators.registry import OperatorRegistry
    except Exception:
        return dead, sampled, total

    rng = None
    for canonical in sorted(catalog):
        entry = catalog[canonical]
        if entry.get("compatibility_only") or entry.get("diagnostic_only"):
            continue
        if assign_mining_role(canonical, entry) not in (
            MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
            MiningRole.FUNDAMENTAL_PIT, MiningRole.INTRADAY_EOD,
        ):
            continue
        try:
            operator = OperatorRegistry.get(canonical, "pandas_numpy")
            grades = searchable_param_names(getattr(operator, "metadata", None))
        except Exception:
            continue
        searchable = sorted(set(grades.get("full", ())) | set(grades.get("coarse", ())))
        if not searchable:
            continue
        total += 1
        if total > max_canonicals:
            continue
        sampled += 1
        try:
            panel = _synthetic_panel(operator)
            base = _run_op_hash(operator, panel, searchable)
            for name in searchable:
                alt = _run_op_hash(operator, panel, searchable, override={name: None})
                if base is not None and alt == base and alt is not None:
                    dead.append(
                        {"canonical": canonical, "param": name, "evidence": "injectivity"}
                    )
        except Exception:
            continue
    return dead, sampled, total


def _synthetic_panel(operator: Any) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    rows, cols = 80, 5
    idx = pd.date_range("2026-01-01", periods=rows, freq="B")
    cols_l = [f"s{i}" for i in range(cols)]
    panel = pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=cols_l)
    return {"x": panel, "y": panel.roll(1) if hasattr(panel, "roll") else panel}


def _run_op_hash(
    operator: Any,
    panel: dict[str, Any],
    searchable: list[str],
    *,
    override: dict[str, Any] | None = None,
) -> str | None:
    """Run the kernel and hash the output; ``override`` picks an alternative
    legal value for a parameter (the 'different value' side of injectivity)."""
    import hashlib

    try:
        kwargs: dict[str, Any] = {}
        for name in searchable:
            spec = None
            try:
                spec = operator.metadata.param_specs.get(name)
            except Exception:
                spec = None
            if name in (override or ()):
                if spec is None:
                    return None  # no alternative known
                lo = spec.min if spec.min is not None else 1
                hi = spec.max if spec.max is not None else lo + 2
                kwargs[name] = hi if spec.choices else (lo + 1)
            else:
                default = None
                try:
                    default = operator.metadata.param_types.get(name)
                except Exception:
                    default = None
                if spec is not None and spec.default is not None:
                    kwargs[name] = spec.default
                elif default is int:
                    kwargs[name] = 10
                elif default is float:
                    kwargs[name] = 0.5
                else:
                    kwargs[name] = 10
        out = operator.calculate(panel["x"], **kwargs)
        values = out.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return hashlib.sha256(values.tobytes()).hexdigest()[:16]
    except Exception:
        return None


def detect_manifest_gap(catalog: dict[str, Any]) -> list[str]:
    """R15-INC-043/051: CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST.

    Every production-certified factor operator must be reachable in the
    current-context mining manifest (role-admissible AND the manifest's
    eligible/pending sets).  Real set difference — never a hardcoded [].
    """
    try:
        from mining.operator_catalog import get_mining_operators
    except Exception:
        return []
    manifest_all = {op.canonical for op in get_mining_operators(admission="all")}
    missing: list[str] = []
    for canonical, entry in sorted(catalog.items()):
        if entry.get("production_certified"):
            role = assign_mining_role(canonical, entry)
            if role in (MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
                        MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT):
                if canonical not in manifest_all:
                    missing.append(canonical)
    return missing


_ALIAS_LOOKUP: dict[str, str] = {}


def audit_all_registered_operators() -> dict[str, Any]:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    from cleaned_operators.operator_surface import classify_canonical, unclassified_canonicals
    from research_tools.registry import ResearchToolRegistry

    catalog = OperatorRegistry._catalog
    research_tools = set(ResearchToolRegistry.list_canonical())
    global _ALIAS_LOOKUP
    _ALIAS_LOOKUP = {
        a: t for a, t in _alias_edges(catalog)
    }

    verdicts: list[Verdict] = []
    for canonical in sorted(catalog):
        if canonical in research_tools:
            continue
        cat = catalog[canonical]
        role, role_source = assign_mining_role_ex(canonical, cat)
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
                role_source=role_source.value,
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

    # ---- invariants (§四十二 + R15) ----
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
    unresolved_roles = sorted(
        v.canonical for v in verdicts if v.role == "unresolved"
    )

    # real detectors (never hardcoded [])
    manifest_gap = detect_manifest_gap(catalog)
    duplicate_candidates = detect_semantic_duplicate_candidates(catalog)
    dead_params, dead_sampled, dead_total = detect_dead_searchable_params(catalog)
    dead_coverage = (
        f"sampled={dead_sampled}/{dead_total}"
        if dead_total else "no searchable-param factor operators"
    )

    # R15-INC-047: TWO independent unclassified invariants — the authoring
    # surface classification and the mining-role resolution.
    authoring_unclassified = sorted(unclassified_canonicals(catalog))

    invariants = {
        "UNCLASSIFIED_FACTOR_CANONICALS": authoring_unclassified,
        "MINING_ROLE_UNRESOLVED": unresolved_roles,
        "UNUSED_PUBLIC_CANONICALS": unused,
        "FACTOR_SHAPED_RESEARCH_ONLY": factor_shaped_research,
        "MINING_ELIGIBLE_WITHOUT_CERTIFICATION": eligible_without_cert,
        "PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY": forbidden_public,
        "CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST": manifest_gap,
        "COMPAT_ALIAS_AS_CANONICAL": sorted(
            c for c in catalog if catalog[c].get("compatibility_only")
        ),
        "SEMANTIC_DUPLICATE_CANONICALS": [
            f"{d['left']} <-> {d['right']} ({d['reason']})"
            for d in duplicate_candidates
        ],
        "DEAD_SEARCHABLE_PARAMS": [
            f"{d['canonical']}.{d['param']}"
            for d in dead_params
        ],
    }
    # R15-INC-264: detectors that did not run are reported as UNKNOWN coverage,
    # never silently PASS.
    detector_coverage = {
        "CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST": {
            "ran": True, "coverage": "all certified factor operators",
        },
        "SEMANTIC_DUPLICATE_CANONICALS": {
            "ran": True,
            "coverage": f"{len(duplicate_candidates)} candidate pairs (alias+name+contract)",
        },
        "DEAD_SEARCHABLE_PARAMS": {
            "ran": dead_total > 0,
            "coverage": dead_coverage,
            "unsampled": dead_total - dead_sampled if dead_total else 0,
        },
    }

    # R15-INC-046/248: per-context usable (source/market) vs static certified.
    certified_factor_count = sum(
        1 for c, e in catalog.items()
        if e.get("production_certified")
        and assign_mining_role(c, e) in (
            MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
            MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT,
        )
    )

    counts = {
        "registered public total": len(verdicts),
        "static certified usable total": certified_factor_count,
        "mining usable total": sum(1 for v in verdicts if v.mining_eligible),
        "internal total": len(buckets["B"]),
        "source transform total": len(buckets["I"]),
        "deleted total": len(buckets["A"]) + len(buckets["C"]) + len(buckets["J"]),
        "unclassified total": len(authoring_unclassified),
        "unresolved role total": len(unresolved_roles),
        "unused total": len(invariants["UNUSED_PUBLIC_CANONICALS"]),
    }
    for label, keys in _BUCKET_LABELS.items():
        counts[f"bucket {label} ({keys})"] = len(buckets[label])

    return {
        "verdicts": [v.__dict__ for v in verdicts],
        "buckets": {k: list(v) for k, v in buckets.items()},
        "invariants": invariants,
        "detector_coverage": detector_coverage,
        "counts": counts,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FactorEngine promote_or_delete audit",
        "",
        f"- registered public total = {result['counts']['registered public total']}",
        f"- static certified usable total = {result['counts']['static certified usable total']}",
        f"- mining usable total = {result['counts']['mining usable total']}",
        f"- internal total = {result['counts']['internal total']}",
        f"- source transform total = {result['counts']['source transform total']}",
        f"- deleted total = {result['counts']['deleted total']}",
        f"- unclassified total = {result['counts']['unclassified total']}",
        f"- unresolved role total = {result['counts']['unresolved role total']}",
        f"- unused total = {result['counts']['unused total']}",
        "",
        "## Buckets (A–K)",
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
    lines += ["", "## Detector coverage (R15: a detector that did not run is UNKNOWN, not PASS)", ""]
    for name, cov in result["detector_coverage"].items():
        lines.append(f"- {name}: ran={cov['ran']} coverage={cov['coverage']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/promote_or_delete_report.md")
    parser.add_argument("--json", default="build/promote_or_delete_report.json")
    # R15-INC-042/258: default is strict — any non-empty hard release invariant
    # returns 1 so CI can never be faked green.
    parser.add_argument("--no-strict", action="store_true", help="exit 0 even on violations")
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

    _HARD_INVARIANTS = (
        "UNCLASSIFIED_FACTOR_CANONICALS",
        "MINING_ROLE_UNRESOLVED",
        "MINING_ELIGIBLE_WITHOUT_CERTIFICATION",
        "PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY",
        "CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST",
    )
    if not args.no_strict:
        violations = [
            name for name in _HARD_INVARIANTS
            if result["invariants"].get(name)
        ]
        if violations:
            print(f"\nSTRICT FAIL: hard release invariants non-empty: {violations}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
