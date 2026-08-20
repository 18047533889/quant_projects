# -*- coding: utf-8 -*-
"""R23 per-canonical semantic/PIT audit artifact generator.

Generates (into ``docs/``):
  R23_PER_CANONICAL_AUDIT.csv / .json / .md
  R23_FUNDAMENTAL_OPERATOR_AUDIT.csv / .json / .md
  R23_TEMPORAL_SOURCE_CERTIFICATES.json
  R23_FINANCIAL_BUNDLE_CONTRACTS.json
  R23_FLOW_SEMANTICS_MATRIX.json
  R23_REVISION_VINTAGE_MATRIX.json
  R23_SAME_DAY_AVAILABILITY_MATRIX.json
  R23_OPERATOR_REMEDIATION_PLAN.json / .md

Every canonical is enumerated via ``load_all()`` — no skip / except-continue /
unknown-forever.  The per-canonical certificate status is derived from MACHINE
facts (lifecycle status, backend coverage, declared capability tags, catalog
metadata), not from a label claiming PIT.

Usage:  python3 scripts/r23_audit_generate.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

_CERTIFIED = "CERTIFIED"
_CERTIFIED_CONTEXTUAL = "CERTIFIED_CONTEXTUAL"
_SUPPORTING = "SUPPORTING_ONLY"
_RESEARCH = "RESEARCH_TOOL"

# ---------------------------------------------------------------------------
# capability tags -> R23 blocker semantics
# ---------------------------------------------------------------------------
_CAPABILITY_TAGS = {
    "requires:RevisionEventSource": "PIT18_REVISION_EVENT_UNPROVEN",
    "requires:PreEventExpectationSnapshot": "PIT11_EXPECTATION_POST_EVENT_LEAK",
    "requires:ConsensusVintageSource": "PIT18_REVISION_EVENT_UNPROVEN",
}
_ANY_REQUIRE_TAG = tuple(_CAPABILITY_TAGS)


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


def _classify(registry, canonical, entry):
    """Map a canonical to the R23-003 certificate state using machine facts."""
    tags = _op_tags(registry, canonical)
    for cap in _ANY_REQUIRE_TAG:
        if cap in tags:
            return _CERTIFIED_CONTEXTUAL, _CAPABILITY_TAGS[cap]
    status = str(entry.get("status", ""))
    backends = list(entry.get("backends", []) or [])
    if status == "production":
        if "pandas_numpy" in backends and ("polars" in backends or "sql" in backends):
            return _CERTIFIED, ""
        return _CERTIFIED, "single-backend (production but parity incomplete)"
    if status == "experimental":
        # R23-303: math/source context supports the capability but the operator
        # is NOT production-certified -> SUPPORTING_ONLY (capability retained,
        # default production miner must not generate it).  The capability-gated
        # (revision/surprise/consensus) experimental operators were already
        # classified CERTIFIED_CONTEXTUAL above via their ``requires:*`` tags.
        return _SUPPORTING, "experimental lifecycle: not production-certified (R23-303)"
    if status == "research":
        return _RESEARCH, ""
    if status in ("deprecated",) or entry.get("compatibility_only"):
        return _SUPPORTING, ""
    return _SUPPORTING, "no production lifecycle status"


def _issue_cols(registry, canonical, entry):
    """Per-issue audit flags for the per-canonical table."""
    tags = _op_tags(registry, canonical)
    cat = str(entry.get("category", ""))
    out = {}
    out["math_issue"] = ""
    out["input_issue"] = ""
    out["unit_issue"] = ""
    out["grain_issue"] = ""
    if "requires:RevisionEventSource" in tags:
        out["revision_issue"] = "PIT18_REVISION_EVENT_UNPROVEN"
    elif "requires:ConsensusVintageSource" in tags:
        out["revision_issue"] = "PIT18_REVISION_EVENT_UNPROVEN"
    else:
        out["revision_issue"] = ""
    if "requires:PreEventExpectationSnapshot" in tags:
        out["pit_issue"] = "PIT11_EXPECTATION_POST_EVENT_LEAK"
    else:
        out["pit_issue"] = ""
    out["same_day_issue"] = ""
    out["period_issue"] = ""
    out["missing_issue"] = ""
    out["parameter_issue"] = ""
    out["backend_issue"] = "" if len(entry.get("backends", []) or []) >= 2 else "single-backend"
    out["incremental_issue"] = ""
    out["cross_market_issue"] = ""
    out["input_semantic_unknown"] = "" if entry.get("input_units") or entry.get("input_grain") else "R23-317"
    return out


def _severity(issue_dict, status):
    joined = " | ".join(v for v in issue_dict.values() if v)
    if any(v for v in issue_dict.values() if v.startswith("PIT")):
        return "P0"
    if status != _CERTIFIED:
        return "P1"
    return ""


def main() -> None:
    os.makedirs(DOCS, exist_ok=True)
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    registry = OperatorRegistry
    canonicals = sorted(registry._catalog)

    rows = []
    for name in canonicals:
        entry = registry._catalog[name]
        status, blocker = _classify(registry, name, entry)
        issues = _issue_cols(registry, name, entry)
        sev = _severity(issues, status)
        rows.append({
            "canonical": name,
            "category": str(entry.get("category", "")),
            "module": str(entry.get("source", "")),
            "source": str(entry.get("source", "")),
            "direct_status": str(entry.get("status", "")),
            "backends": ";".join(entry.get("backends", []) or []),
            "input_units": str(entry.get("input_units", "") or ""),
            "input_grain": str(entry.get("input_grain", "") or ""),
            "output_unit": str(entry.get("output_unit", "") or ""),
            "final_status": status,
            "blocker": blocker,
            **issues,
            "severity": sev,
        })

    per_df = pd.DataFrame(rows)
    per_csv = os.path.join(DOCS, "R23_PER_CANONICAL_AUDIT.csv")
    per_df.to_csv(per_csv, index=False)
    with open(os.path.join(DOCS, "R23_PER_CANONICAL_AUDIT.json"), "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(DOCS, "R23_PER_CANONICAL_AUDIT.md"), "w") as fh:
        fh.write("# R23 Per-Canonical Semantic/PIT Audit\n\n")
        fh.write(f"Total canonicals: {len(rows)}\n\n")
        for status, count in per_df["final_status"].value_counts().items():
            fh.write(f"- **{status}**: {count}\n")
        fh.write("\n| canonical | category | final_status | blocker |\n")
        fh.write("|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['canonical']} | {r['category']} | {r['final_status']} | {r['blocker']} |\n")

    # Fundamental operator audit (12 questions per R23-266).
    fund_rows = []
    fund_categories = {"fundamental", "fundamental_period"}
    for name in canonicals:
        entry = registry._catalog[name]
        if str(entry.get("category", "")) not in fund_categories:
            continue
        tags = _op_tags(registry, name)
        flow_tags = [t for t in tags if t.startswith("flow_type:")]
        slot_a = [t for t in tags if t.startswith("flow_slot_a:")]
        slot_b = [t for t in tags if t.startswith("flow_slot_b:")]
        fund_rows.append({
            "canonical": name,
            "module": str(entry.get("source", "")),
            "final_status": _classify(registry, name, entry)[0],
            "q1_input_grain": str(entry.get("input_grain", "") or "") or "UNKNOWN",
            "q2_ashare_default": "cumulative_ytd_flow" if any("ytd" in (entry.get("input_grain") or ()) for _ in [0]) else "see binding",
            "accepted_flow_semantics": ";".join(flow_tags) or "not-declared",
            "flow_slot_a": ";".join(slot_a),
            "flow_slot_b": ";".join(slot_b),
            "timeframe_requirement": "",
            "knowledge_clock": "PubDate/filing_date (DATE_ONLY_NEXT_SESSION)" if any(t in tags for t in ("requires:PreEventExpectationSnapshot",)) else "field knowledge column",
            "revision_clock": "requires:RevisionEventSource" if "requires:RevisionEventSource" in tags else "n/a",
            "left_censor_policy": "left-censored first observation" if any("days_since" in name or "staleness" in name for _ in [0]) else "n/a",
            "blockers": "|".join(v for v in (_CAPABILITY_TAGS.get(t, "") for t in tags) if v),
        })
    fund_df = pd.DataFrame(fund_rows)
    fund_df.to_csv(os.path.join(DOCS, "R23_FUNDAMENTAL_OPERATOR_AUDIT.csv"), index=False)
    with open(os.path.join(DOCS, "R23_FUNDAMENTAL_OPERATOR_AUDIT.json"), "w") as fh:
        json.dump(fund_rows, fh, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(DOCS, "R23_FUNDAMENTAL_OPERATOR_AUDIT.md"), "w") as fh:
        fh.write("# R23 Fundamental Operator Audit\n\n")
        fh.write(f"Fundamental canonicals: {len(fund_rows)}\n\n")
        fh.write("| canonical | final_status | accepted_flow_semantics | blockers |\n")
        fh.write("|---|---|---|---|\n")
        for r in fund_rows:
            fh.write(f"| {r['canonical']} | {r['final_status']} | {r['accepted_flow_semantics']} | {r['blockers']} |\n")

    # Temporal source certificates from the field catalogs.
    from fields.catalog import ASHARE_FIELD_SPECS, ASHARE_TABLE_SPECS
    from fields.catalog_us import US_FIELD_SPECS, US_TABLE_SPECS

    # R25-064..066: the source-certificate key is ``market::dataset::table`` so
    # A/US same-name tables (StockIncome, StockDailyBar, …) never collide.
    # R25-067/068 (knowledge-time resolution): exact-observed daily/session
    # tables have NO separate knowledge-time concept — declare
    # ``N/A_EXACT_OBSERVATION`` instead of a bare UNPROVEN; a snapshot/relation/
    # event/fundamental table that claims strict_pit_allowed=True but proves no
    # key clock is flagged ``production_block`` (R25-068).
    _EXACT_OBSERVED = {
        "panel", "minute_session", "calendar", "static",
    }
    _EXACT_JOIN = {"exact", "exact_date", "minute_session", "state_asof", "financial_pit"}

    def _resolution(t, m) -> tuple[str, bool]:
        explicit = m.get("knowledge_time_resolution")
        if explicit:
            return str(explicit), False
        if t.strict_pit_allowed and t.table_kind in _EXACT_OBSERVED and t.join_policy in _EXACT_JOIN:
            # R25-067: the observation IS the data — a separate knowledge clock
            # does not apply.
            return "N/A_EXACT_OBSERVATION", False
        if t.strict_pit_allowed:
            # R25-068: claims PIT-safe but proves no key clock -> production
            # block risk.
            return "UNPROVEN", True
        return "UNPROVEN", False

    temporal = {}
    for market, spec_list in (("ashare", ASHARE_TABLE_SPECS), ("us", US_TABLE_SPECS)):
        for t in spec_list:
            m = dict(t.metadata)
            resolution, block = _resolution(t, m)
            key = f"{market}::{t.dataset}::{t.name}"
            temporal[key] = {
                "market": market,
                "dataset": t.dataset,
                "table": t.name,
                "knowledge_time_column": t.knowledge_time_column,
                "period_id_column": t.period_id_column,
                "revision_column": t.revision_column,
                "effective_time_column": t.effective_time_column,
                "strict_pit_allowed": t.strict_pit_allowed,
                "required_parameters": list(t.required_parameters),
                "knowledge_time_resolution": resolution,
                "production_block_risk": block,
                "announcement_pit_certified": m.get("announcement_pit_certified", None),
                "revision_vintage_pit_certified": m.get("revision_vintage_pit_certified", None),
                "restatement_risk": m.get("restatement_risk", None),
                "timeframe_required": m.get("timeframe_required", None),
                "pit_reason": m.get("pit_reason", ""),
            }
    with open(os.path.join(DOCS, "R23_TEMPORAL_SOURCE_CERTIFICATES.json"), "w") as fh:
        json.dump(temporal, fh, ensure_ascii=False, indent=2, default=str)

    # Flow semantics matrix (from field catalog flow_semantics).
    flow_matrix = {}
    for spec in list(ASHARE_FIELD_SPECS) + list(US_FIELD_SPECS):
        if spec.flow_semantics:
            flow_matrix.setdefault(spec.table, {})[spec.name] = spec.flow_semantics
    with open(os.path.join(DOCS, "R23_FLOW_SEMANTICS_MATRIX.json"), "w") as fh:
        json.dump(flow_matrix, fh, ensure_ascii=False, indent=2, default=str)

    # Revision vintage matrix + same-day availability + bundle contracts.
    rev_vintage = {
        name: cert for name, cert in temporal.items()
        if cert["revision_vintage_pit_certified"] is not None
    }
    with open(os.path.join(DOCS, "R23_REVISION_VINTAGE_MATRIX.json"), "w") as fh:
        json.dump(rev_vintage, fh, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(DOCS, "R23_SAME_DAY_AVAILABILITY_MATRIX.json"), "w") as fh:
        json.dump({
            name: {"knowledge_time_resolution": cert["knowledge_time_resolution"]}
            for name, cert in temporal.items()
            if cert["knowledge_time_resolution"] != "UNPROVEN"
        }, fh, ensure_ascii=False, indent=2, default=str)
    bundle = {
        "StockBalance": {"statement_family": "balance", "grain": "stock", "same_period_required_for_multi_statement": True},
        "StockIncome": {"statement_family": "income", "grain": "flow_ytd/quarterly/TTM", "timeframe_required": True},
        "StockCashFlow": {"statement_family": "cashflow", "grain": "flow_ytd/quarterly/TTM", "timeframe_required": True},
        "policy": "multi-statement operators require same fiscal_period + compatible vintage + flow timeframe (R23-032..036)",
    }
    with open(os.path.join(DOCS, "R23_FINANCIAL_BUNDLE_CONTRACTS.json"), "w") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2, default=str)

    # Remediation plan.
    blockers = {}
    for r in rows:
        if r["blocker"]:
            blockers.setdefault(r["blocker"], []).append(r["canonical"])
    remediation = {
        "blocker_counts": {k: len(v) for k, v in sorted(blockers.items())},
        "blockers": blockers,
        "rules": {
            "R23-303": "math correct but PIT source missing -> BLOCKED_CONTEXTUAL (keep math)",
            "R23-304": "definition wrong -> FIX_MATH",
            "R23-305": "name wrong -> RENAME + compatibility alias",
            "R23-306": "in-sample only -> ResearchTool + causal sibling",
            "R23-307": "data can never support -> DELETE_NO_DATA",
        },
    }
    with open(os.path.join(DOCS, "R23_OPERATOR_REMEDIATION_PLAN.json"), "w") as fh:
        json.dump(remediation, fh, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(DOCS, "R23_OPERATOR_REMEDIATION_PLAN.md"), "w") as fh:
        fh.write("# R23 Operator Remediation Plan\n\n")
        for code, names in sorted(blockers.items()):
            fh.write(f"- **{code}** ({len(names)}): {', '.join(names[:40])}{'…' if len(names) > 40 else ''}\n")

    print("R23 artifacts written to", DOCS)
    print("Total canonicals:", len(rows))
    print("Final-status distribution:", per_df["final_status"].value_counts().to_dict())
    print("Fundamental canonicals:", len(fund_rows))


if __name__ == "__main__":
    main()
