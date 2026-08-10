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

    A real detector combines (a) the alias graph and (b) canonical-name
    families (``MACD`` vs ``MACD_line``).  R16-010: an identical
    ``(param_names, output_unit, input_grain)`` contract is NO LONGER treated as
    a duplicate — ``ts_mean`` and ``ts_std`` share the exact same interface yet
    are mathematically different.  Identical interface is only candidate material;
    a true duplicate requires implementation/AST fingerprint + golden behavioural
    + metamorphic equivalence (the machine-audit layer, R16-225).  Candidates
    require human confirmation — the detector never silently deletes.
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
    return candidates


def _panel_arg_names(operator: Any) -> list[str]:
    """Typed panel-parameter names for a kernel (R16-005).

    The dead-param runner must bind the FULL declared panel inputs through the
    typed contract — never ``calculate(panel['x'], ...)`` for a binary/ternary/
    variadic operator (which throws, gets swallowed, and the operator silently
    leaves the audit).  ``metadata.panel_params`` is authoritative; a signature
    heuristic (leading no-default positional params, excluding declared scalars)
    covers operators that declare panels only in the kernel.
    """
    md = getattr(operator, "metadata", None)
    pp = getattr(md, "panel_params", None)
    if pp:
        return [p for p in pp if p not in ("", "_fn")]
    scalar_names = set(getattr(md, "scalar_params", None) or ())
    fn = getattr(operator, "_calculate_series", None) or getattr(operator, "calculate", None)
    if fn is not None:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            panels: list[str] = []
            for name, param in sig.parameters.items():
                if name in ("self", "_fn", "kwargs") or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if name in scalar_names:
                    continue
                if param.default is not inspect.Parameter.empty:
                    # first defaulted param ends the positional panel prefix
                    break
                panels.append(name)
            if panels:
                return panels
    return ["x"]


def _synthetic_panels(operator: Any) -> dict[str, Any]:
    """Deterministic synthetic panels keyed by panel-param name (R16-005).

    Every declared panel slot gets its OWN data so a parameter that changes
    behavior only through a second/third input is not masked by feeding the
    same array everywhere.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    rows, cols = 80, 5
    idx = pd.date_range("2026-01-01", periods=rows, freq="B")
    columns = [f"s{i}" for i in range(cols)]
    panels: dict[str, Any] = {}
    for name in _panel_arg_names(operator):
        panels[name] = pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=columns)
    if not panels:
        panels["x"] = pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=columns)
    return panels


def _behavior_fingerprint(out: Any) -> str:
    """Full behavioral fingerprint (R16-006).

    Comparing only finite values let two different missing-masks hash equal — a
    parameter whose change only altered missingness was misjudged equivalent.
    The fingerprint hashes shape / index / columns / raw values / NaN mask /
    ±Inf mask, so ANY behavioral difference (including NaN topology) changes it.
    """
    import hashlib
    import json

    import numpy as np

    h = hashlib.sha256()
    h.update(f"{out.shape[0]}x{out.shape[1]}".encode("utf-8"))
    idx_head = [str(i) for i in (out.index[:40] if len(out.index) > 40 else out.index)]
    h.update(json.dumps(idx_head, ensure_ascii=False).encode("utf-8"))
    col_head = [str(c) for c in (out.columns[:40] if len(out.columns) > 40 else out.columns)]
    h.update(json.dumps(col_head, ensure_ascii=False).encode("utf-8"))
    arr = np.asarray(out, dtype=np.float64)
    h.update(arr.tobytes())
    h.update(np.isnan(arr).tobytes())
    h.update(np.isposinf(arr).tobytes())
    h.update(np.isneginf(arr).tobytes())
    return h.hexdigest()[:16]


def _run_op_hash(
    operator: Any,
    panels: dict[str, Any],
    searchable: list[str],
    *,
    override: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Run the kernel on the full panel binding and fingerprint the output.

    Returns ``(fingerprint, error_summary)``.  ``override`` picks an alternative
    legal value for a parameter (the 'different value' side of injectivity).
    R16-007: an exception is NEVER silently skipped — it is returned as the
    ``error_summary`` so the caller records an AUDIT_ERROR for the canonical.
    """
    import numpy as np

    error: str | None = None
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
                    return None, "no ParamSpec for overridden searchable param"
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
        # R16-005: bind the FULL declared panel inputs, in declared order.
        panel_names = _panel_arg_names(operator)
        missing = [p for p in panel_names if p not in panels]
        if missing:
            return None, f"missing synthetic panels for {missing}"
        args = [panels[p] for p in panel_names]
        out = operator.calculate(*args, **kwargs)
        if out is None:
            return None, "kernel returned None"
        return _behavior_fingerprint(out), None
    except Exception as exc:  # noqa: BLE001 — the error becomes an AUDIT_ERROR
        error = f"{type(exc).__name__}: {exc}"
    return None, error


def detect_dead_searchable_params(
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    """R16-004: FULL parameter-injectivity probe — no sampling.

    EVERY ALPHA/factor operator with a searchable param gets a machine outcome:
    run the kernel on a fixed synthetic panel (full typed panel binding, R16-005)
    and change ONE searchable parameter across two legal values; identical full
    behavior fingerprint (incl. NaN topology, R16-006) => dead candidate.
    Returns ``(dead, audited, total, audit_errors)``.  An operator that could
    not be constructed/executed is reported as AUDIT_ERROR (R16-007), never
    silently dropped.
    """
    dead: list[dict[str, Any]] = []
    audit_errors: list[dict[str, Any]] = []
    audited = 0
    total = 0
    try:
        from cleaned_operators.base import searchable_param_names
        from cleaned_operators.registry import OperatorRegistry
    except Exception as exc:
        audit_errors.append({"canonical": "<import>", "param": "*", "error": str(exc)})
        return dead, audited, total, audit_errors

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
        except Exception as exc:
            audit_errors.append({"canonical": canonical, "param": "*", "error": str(exc)})
            continue
        searchable = sorted(set(grades.get("full", ())) | set(grades.get("coarse", ())))
        if not searchable:
            continue
        total += 1
        panels = _synthetic_panels(operator)
        base, base_err = _run_op_hash(operator, panels, searchable)
        if base_err is not None:
            audit_errors.append({"canonical": canonical, "param": "*", "error": base_err})
            continue
        audited += 1
        for name in searchable:
            alt, alt_err = _run_op_hash(operator, panels, searchable, override={name: None})
            if alt_err is not None:
                audit_errors.append({"canonical": canonical, "param": name, "error": alt_err})
                continue
            if base is not None and alt is not None and alt == base:
                dead.append({"canonical": canonical, "param": name, "evidence": "injectivity"})
    return dead, audited, total, audit_errors


def detect_manifest_gap(catalog: dict[str, Any]) -> list[str]:
    """R15-INC-043/051: CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST.

    Every production-certified factor operator must be reachable in the
    current-context ELIGIBLE mining manifest (role-admissible AND runtime-
    eligible).  R16-009: comparing against ``admission="all"`` was self-
    validating — ``all`` is nearly the whole registry, so it could never prove
    the ELIGIBLE runtime manifest actually contains every certified mineable
    factor.  The eligible-only set is the real runtime contract.
    """
    try:
        from mining.operator_catalog import get_mining_operators
    except Exception as exc:
        # A detector that cannot run must not claim PASS with [] — the caller
        # records this as AUDIT_ERROR coverage.
        raise RuntimeError(f"detect_manifest_gap: mining catalog unavailable: {exc}") from exc
    manifest_eligible = {op.canonical for op in get_mining_operators(admission="eligible")}
    missing: list[str] = []
    for canonical, entry in sorted(catalog.items()):
        if entry.get("production_certified"):
            role = assign_mining_role(canonical, entry)
            if role in (MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
                        MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT):
                if canonical not in manifest_eligible:
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
    dead_params, dead_audited, dead_total, dead_audit_errors = detect_dead_searchable_params(catalog)
    dead_coverage = (
        f"audited={dead_audited}/{dead_total}"
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
    # R16-008: release-critical invariants are EXPLICITLY marked
    # ``release_blocking``.  ``COMPAT_ALIAS_AS_CANONICAL`` is advisory (an
    # alias-only compat surface is informational, not a release gate).
    release_blocking = {
        "UNCLASSIFIED_FACTOR_CANONICALS",
        "MINING_ROLE_UNRESOLVED",
        "UNUSED_PUBLIC_CANONICALS",
        "FACTOR_SHAPED_RESEARCH_ONLY",
        "MINING_ELIGIBLE_WITHOUT_CERTIFICATION",
        "PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY",
        "CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST",
        "SEMANTIC_DUPLICATE_CANONICALS",
        "DEAD_SEARCHABLE_PARAMS",
    }
    # R15-INC-264: detectors that did not run are reported as UNKNOWN coverage,
    # never silently PASS.  R16-004: dead-param coverage is FULL (no sampling);
    # any AUDIT_ERROR on a dead-param probe is itself a release blocker.
    detector_coverage = {
        "CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST": {
            "ran": True, "coverage": "all certified factor operators (eligible-only manifest)",
        },
        "SEMANTIC_DUPLICATE_CANONICALS": {
            "ran": True,
            "coverage": f"{len(duplicate_candidates)} candidate pairs (alias+name-family)",
        },
        "DEAD_SEARCHABLE_PARAMS": {
            "ran": dead_total > 0,
            "coverage": dead_coverage,
            "audit_errors": dead_audit_errors,
            "unrun": dead_total - dead_audited if dead_total else 0,
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
        "release_blocking": sorted(release_blocking),
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

    # R16-008: release-blocking invariants are marked in the result, and strict
    # mode fails on ANY of them — not just the five legacy hard gates.
    release_blocking = set(result.get("release_blocking") or ())
    # R16-007: a mineable/production canonical whose dead-param probe raised an
    # AUDIT_ERROR is itself a release blocker (it was never audited).
    dead_audit_errors = (
        result.get("detector_coverage", {}).get("DEAD_SEARCHABLE_PARAMS", {}).get("audit_errors") or []
    )
    if not args.no_strict:
        violations = [
            name for name in sorted(release_blocking)
            if result["invariants"].get(name)
        ]
        if violations:
            print(f"\nSTRICT FAIL: release-blocking invariants non-empty: {violations}")
            return 1
        if dead_audit_errors:
            print(
                f"\nSTRICT FAIL: {len(dead_audit_errors)} dead-param probe AUDIT_ERRORs "
                f"(first: {dead_audit_errors[0]})"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
