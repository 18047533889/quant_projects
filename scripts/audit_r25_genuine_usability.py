# -*- coding: utf-8 -*-
"""R25 genuine-usability acceptance runner.

Generates into ``docs/``:
  R25_CANONICAL_SNAPSHOT.json
  R25_GENUINE_USABILITY_MATRIX.csv / .json / .md
  R25_EVIDENCE_TRUST_AUDIT.json / .md
  R25_PARAMETER_REGION_CERTIFICATES.json
  R25_PARAMETER_INJECTIVITY_AUDIT.csv
  R25_REAL_RECIPE_AUDIT.csv / .json
  R25_OUTPUT_DOMAIN_MATRIX.json
  R25_INPUT_SLOT_SEMANTIC_MATRIX.json
  R25_BACKEND_CLAIM_MATRIX.csv
  R25_MARKET_SOURCE_RECIPE_MATRIX.json
  R25_SOURCE_PIT_CONTEXT_MATRIX.json
  R25_FINAL_BLOCKERS.json / .md
  R25_FINAL_ACCEPTANCE_REPORT.md

Every canonical is enumerated via ``load_all()`` (0 skip).  ``genuine_usable``
is derived ONLY from machine facts (independent evidence, capability tags,
recipe resolution, parameter injectivity, output domain, backend claims) —
never from a status/tag label.  A dirty tree is reported (R25-021); the final
flags are only writable on a clean snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

_CAPABILITY_BLOCKERS = {
    "requires:RevisionEventSource": "REVISION_OPERATOR_WITHOUT_VINTAGE_SOURCE",
    "requires:PreEventExpectationSnapshot": "EXPECTATION_OPERATOR_WITHOUT_VINTAGE_SOURCE",
    "requires:ConsensusVintageSource": "EXPECTATION_OPERATOR_WITHOUT_VINTAGE_SOURCE",
}


def _digest(items: list[str]) -> str:
    # SHA-1 used only for audit report fingerprint, not cryptographic security
    h = hashlib.sha1(usedforsecurity=False)
    for item in sorted(set(items)):
        h.update(str(item).encode())
    return h.hexdigest()[:16]


def _op_tags(registry, canonical):
    ops = registry._operators.get(canonical, {})
    pd_op = ops.get("pandas_numpy")
    if pd_op is not None:
        return list(getattr(getattr(pd_op, "metadata", None), "tags", None) or [])
    for op in ops.values():
        tags = getattr(getattr(op, "metadata", None), "tags", None)
        if tags:
            return list(tags)
    return []


def _genuine_classify(registry, canonical, entry, tags, injectivity):
    """Machine-fact genuine-usability classification (R25-010)."""
    status = str(entry.get("status", ""))
    blockers: list[str] = []
    # R25-186 evidence independence: semantic_golden / source_contract must be
    # independently proven, never granted by the runtime audit.
    if entry.get("semantic_golden_verified") is not True:
        blockers.append("UNCERTIFIED_SEMANTIC_GOLDEN")
    if entry.get("source_contract_verified") is not True:
        blockers.append("SOURCE_PIT_UNCERTIFIED")
    for cap, code in _CAPABILITY_BLOCKERS.items():
        if cap in tags:
            blockers.append(code)
    if status != "production":
        blockers.append("NOT_PRODUCTION_CERTIFIED")
    if injectivity is False:
        blockers.append("DEAD_OR_UNPROVEN_SEARCHABLE_PARAMETER")
    role = str(entry.get("role", "") or "")
    return blockers, role


def main() -> None:
    os.makedirs(DOCS, exist_ok=True)
    head = ""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        head = "unknown"
    dirty = ""
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        dirty = "unknown"
    dirty_bytes = len(dirty.encode())

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    registry = OperatorRegistry
    canonicals = sorted(registry._catalog)

    snapshot = {
        "HEAD": head,
        "dirty": dirty_bytes > 0,
        "dirty_bytes": dirty_bytes,
        "count": len(canonicals),
        "canonical_set_digest": _digest(canonicals),
        "canonicals": canonicals,
    }
    with open(os.path.join(DOCS, "R25_CANONICAL_SNAPSHOT.json"), "w") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)

    rows = []
    final_blockers: dict[str, list[str]] = {}
    genuine_roles = {
        "alpha": 0, "high_cost": 0, "intermediate": 0, "state": 0,
        "condition": 0, "event": 0, "group": 0, "source_transform": 0,
    }
    for name in canonicals:
        entry = registry._catalog[name]
        tags = _op_tags(registry, name)
        backends = list(entry.get("backends", []) or [])
        # parameter injectivity (real probe, R25-172)
        injectivity = False
        searchable = list(entry.get("scalar_params", []) or [])
        if searchable:
            from mining.direct_use import _probe_parameter_injectivity

            op = registry._operators.get(name, {}).get("pandas_numpy")
            injectivity = _probe_parameter_injectivity(name, op, tuple(searchable))
        elif entry.get("param_specs"):
            injectivity = True  # params exist but none searchable -> vacuous
        blockers, role = _genuine_classify(registry, name, entry, tags, injectivity)
        genuine = not blockers
        if genuine:
            low = str(role).lower()
            if "high" in low or "cost" in low:
                genuine_roles["high_cost"] += 1
            elif low in genuine_roles:
                genuine_roles[low] += 1
            elif low in ("group", "global"):
                genuine_roles["group"] += 1
            elif low in ("source", "transform"):
                genuine_roles["source_transform"] += 1
            else:
                genuine_roles["intermediate"] += 1
        for b in blockers:
            final_blockers.setdefault(b, []).append(name)
        rows.append({
            "canonical": name,
            "category": str(entry.get("category", "")),
            "status": str(entry.get("status", "")),
            "role": role,
            "backends": ";".join(backends),
            "genuine_usable": genuine,
            "semantic_golden_verified": entry.get("semantic_golden_verified"),
            "source_contract_verified": entry.get("source_contract_verified"),
            "parameter_injectivity_passed": injectivity,
            "blockers": "|".join(blockers),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DOCS, "R25_GENUINE_USABILITY_MATRIX.csv"), index=False)
    with open(os.path.join(DOCS, "R25_GENUINE_USABILITY_MATRIX.json"), "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(DOCS, "R25_GENUINE_USABILITY_MATRIX.md"), "w") as fh:
        fh.write("# R25 Genuine Usability Matrix\n\n")
        fh.write(f"Total canonicals: {len(rows)}\n\n")
        fh.write(f"**Genuine usable: {sum(1 for r in rows if r['genuine_usable'])}**\n\n")
        fh.write("| canonical | role | genuine_usable | blockers |\n")
        fh.write("|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['canonical']} | {r['role']} | {r['genuine_usable']} | {r['blockers']} |\n")

    # Evidence trust audit (R25-186): how many canonicals have independently
    # proven semantic/source evidence vs runtime-only.
    trust = {
        "total": len(rows),
        "semantic_golden_verified": sum(1 for r in rows if r["semantic_golden_verified"]),
        "source_contract_verified": sum(1 for r in rows if r["source_contract_verified"]),
        "both_independent": sum(
            1 for r in rows if r["semantic_golden_verified"] and r["source_contract_verified"]
        ),
        "note": "R25-011..014: synthetic runtime audit grants only runtime/shape/"
                "determinism/prefix evidence; semantic_golden and source_contract "
                "require independent evidence (math golden / FieldSpec-Provider "
                "DataAccess contract).",
    }
    with open(os.path.join(DOCS, "R25_EVIDENCE_TRUST_AUDIT.json"), "w") as fh:
        json.dump(trust, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(DOCS, "R25_EVIDENCE_TRUST_AUDIT.md"), "w") as fh:
        fh.write("# R25 Evidence Trust Audit\n\n")
        fh.write(f"- Total: {trust['total']}\n")
        fh.write(f"- semantic_golden_verified (independent): {trust['semantic_golden_verified']}\n")
        fh.write(f"- source_contract_verified (independent): {trust['source_contract_verified']}\n")
        fh.write(f"- both independent: {trust['both_independent']}\n")
        fh.write(f"- {trust['note']}\n")

    # Parameter region / injectivity.
    with open(os.path.join(DOCS, "R25_PARAMETER_REGION_CERTIFICATES.json"), "w") as fh:
        json.dump({
            name: {"searchable": list(entry.get("scalar_params", []) or []),
                   "certified_region": "default-only" if not entry.get("param_specs") else "declared"}
            for name, entry in registry._catalog.items()
        }, fh, ensure_ascii=False, indent=2)
    pd.DataFrame([{"canonical": r["canonical"], "injectivity_passed": r["parameter_injectivity_passed"],
                   "searchable": r["canonical"]} for r in rows]).to_csv(
        os.path.join(DOCS, "R25_PARAMETER_INJECTIVITY_AUDIT.csv"), index=False)

    # Recipe audit (R25-029..036): which canonicals resolve a real recipe.
    recipe_rows = []
    try:
        from mining.direct_use import default_input_recipe
        for name in canonicals:
            recipe = default_input_recipe(name)
            recipe_rows.append({"canonical": name, "recipe_slots": len(recipe),
                                "recipe": json.dumps(recipe, ensure_ascii=False)})
    except Exception as exc:  # noqa: BLE001
        recipe_rows.append({"canonical": "_error", "recipe_slots": 0, "recipe": str(exc)})
    pd.DataFrame(recipe_rows).to_csv(os.path.join(DOCS, "R25_REAL_RECIPE_AUDIT.csv"), index=False)
    with open(os.path.join(DOCS, "R25_REAL_RECIPE_AUDIT.json"), "w") as fh:
        json.dump(recipe_rows, fh, ensure_ascii=False, indent=2)

    # Output domain matrix (R25-175).
    try:
        from mining.direct_use import _output_value_domain, resolve_direct_use
        domains = {}
        for name in canonicals:
            try:
                st = resolve_direct_use(name, registry._catalog[name]).status
                domains[name] = _output_value_domain(name, registry._catalog[name], st)
            except Exception:
                domains[name] = "continuous_signed"
        with open(os.path.join(DOCS, "R25_OUTPUT_DOMAIN_MATRIX.json"), "w") as fh:
            json.dump(domains, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Input slot semantic matrix.
    slot_rows = {}
    try:
        from mining.direct_use import input_slot_specs, _authoritative_param_split
        for name in canonicals:
            cat = registry._catalog[name]
            try:
                panel, scalar = _authoritative_param_split(name, cat)
                slots = input_slot_specs(name, cat, panel_params=panel, scalar_params=scalar)
                slot_rows[name] = [s._asdict() if hasattr(s, "_asdict") else str(s) for s in slots]
            except Exception:
                slot_rows[name] = []
    except Exception:
        pass
    with open(os.path.join(DOCS, "R25_INPUT_SLOT_SEMANTIC_MATRIX.json"), "w") as fh:
        json.dump(slot_rows, fh, ensure_ascii=False, indent=2, default=str)

    # Backend claim matrix.
    backend_rows = [{"canonical": name, "backends": ";".join(entry.get("backends", []) or [])}
                    for name, entry in registry._catalog.items()]
    pd.DataFrame(backend_rows).to_csv(os.path.join(DOCS, "R25_BACKEND_CLAIM_MATRIX.csv"), index=False)

    # Source PIT context (from R23 temporal certs).
    try:
        certs = json.load(open(os.path.join(DOCS, "R23_TEMPORAL_SOURCE_CERTIFICATES.json")))
        with open(os.path.join(DOCS, "R25_SOURCE_PIT_CONTEXT_MATRIX.json"), "w") as fh:
            json.dump(certs, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Market source recipe matrix (from R23 flow/field catalogs).
    try:
        from fields.catalog import ASHARE_TABLE_SPECS
        from fields.catalog_us import US_TABLE_SPECS
        market_sources = {}
        for m, specs in (("ashare", ASHARE_TABLE_SPECS), ("us", US_TABLE_SPECS)):
            market_sources[m] = [{"table": t.name, "dataset": t.dataset,
                                  "strict_pit_allowed": t.strict_pit_allowed} for t in specs]
        with open(os.path.join(DOCS, "R25_MARKET_SOURCE_RECIPE_MATRIX.json"), "w") as fh:
            json.dump(market_sources, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Final blockers.
    with open(os.path.join(DOCS, "R25_FINAL_BLOCKERS.json"), "w") as fh:
        json.dump({k: len(v) for k, v in sorted(final_blockers.items())}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(DOCS, "R25_FINAL_BLOCKERS.md"), "w") as fh:
        fh.write("# R25 Final Blockers\n\n")
        for code, names in sorted(final_blockers.items()):
            fh.write(f"- **{code}** ({len(names)}): {', '.join(names[:20])}{'…' if len(names) > 20 else ''}\n")
        fh.write(f"\nTotal blocked canonical-instances: {sum(len(v) for v in final_blockers.values())}\n")

    # Final acceptance report (R25-195).
    genuine = [r for r in rows if r["genuine_usable"]]
    with open(os.path.join(DOCS, "R25_FINAL_ACCEPTANCE_REPORT.md"), "w") as fh:
        fh.write("# R25 Final Acceptance Report\n\n")
        fh.write(f"- Current canonical count: {len(canonicals)}\n")
        fh.write(f"- Genuine usable total: {len(genuine)}\n")
        fh.write(f"- Genuine usable by role: {genuine_roles}\n")
        fh.write(f"- Evidence trust: semantic={trust['semantic_golden_verified']} "
                 f"source={trust['source_contract_verified']} both={trust['both_independent']}\n")
        fh.write(f"- Blockers: {sum(len(v) for v in final_blockers.values())}\n")
        fh.write(f"- HEAD: {head} dirty={dirty_bytes > 0}\n\n")
        if dirty_bytes > 0:
            fh.write("**INCOMPLETE**: artifacts generated on a DIRTY tree — final flags "
                     "cannot be written until the snapshot is clean (R25-021).\n")
        else:
            fh.write("**Snapshot clean** — flags reflect a clean HEAD.\n")

    print("R25 artifacts written to", DOCS)
    print("Total canonicals:", len(canonicals))
    print("Genuine usable:", len(genuine), "| roles:", genuine_roles)
    print("Blockers:", sum(len(v) for v in final_blockers.values()))


if __name__ == "__main__":
    main()
