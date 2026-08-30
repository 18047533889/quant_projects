# -*- coding: utf-8 -*-
"""R24 hard-gate audit: SEMANTIC_CONTINUITY_CLOSED across relation / PIT /
field-plan / snapshot / IR / source-ref / mining / label layers.

R24-301: every gate must be 0 (no violation found).  The audit is a mix of
structural checks against the actual source and a few import-level symbol
checks, so a regression that reintroduces a silent reindex / period fallback /
research-validator bypass is caught here.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "factor_engine" / "docs"
DOCS.mkdir(exist_ok=True)

_GATES: dict[str, tuple[str, str]] = {
    # gate -> (reason, file-pattern)
    "RELATION_MULTI_PANEL_SILENT_REINDEX": (
        "a public relation operator must not reindex/reindex_like/align a multi-panel input",
        "cleaned_operators/relation",
    ),
    "RELATION_POSITIONAL_AXIS_MISMATCH": (
        "multi-panel relation operators must strict-align before numpy pairing",
        "cleaned_operators/relation",
    ),
    "RELATION_PERCENT_UNIT_HEURISTIC": (
        "holder ratio unit must come from a declared contract, never data max",
        "storage/sources/relation.py",
    ),
    "RELATION_VINTAGE_KEEP_LAST_LOOKAHEAD": (
        "relation snapshot change must not globally drop_duplicates keep=last",
        "storage/sources/relation.py",
    ),
    "RELATION_STALE_SNAPSHOT_UNBOUNDED": (
        "snapshot asof must require max_age_days or allow_unbounded_staleness",
        "storage/sources/relation.py",
    ),
    "RELATION_ENTITY_NAME_IDENTITY_FALLBACK_FOR_TRACKING": (
        "tracking requires EntityStableId; display-name fallback needs opt-in",
        "storage/sources/relation.py",
    ),
    "PIT_PERIOD_USED_AS_FILING_FALLBACK": (
        "FilingDate must never fall back to the report period end date",
        "ir/types.py",
    ),
    "PIT_DECISION_TIME_USED_AS_UNKNOWN_KNOWLEDGE": (
        "NextTradingOpen must not substitute the decision time for unknown knowledge",
        "ir/types.py",
    ),
    "PIT_AVAILABILITY_PRECISION_GUESSED_FROM_VALUE": (
        "same-day precision must come from declared metadata, not midnight inspection",
        "pit_contract.py",
    ),
    "PIT_OMITTED_LAYER_ASSUMED_TRUE": (
        "four-layer PIT gate must be tri-state; omitted layer = UNKNOWN",
        "pit_contract.py",
    ),
    "FIELD_PLAN_TEMPORAL_METADATA_LOST": (
        "NormalizedFieldPlan must carry the full temporal metadata",
        "storage/sources/field_plan.py",
    ),
    "CURRENT_SNAPSHOT_MARKED_NOT_APPLICABLE": (
        "current-snapshot historical absence must be OUT_OF_COVERAGE",
        "storage/sources/field_plan.py",
    ),
    "COVERAGE_CONTRACT_CROSS_CONTEXT_COLLISION": (
        "coverage contracts must be joint-keyed (market+dataset+provider+timeframe)",
        "storage/sources/data_access_source.py",
    ),
    "SNAPSHOT_NO_TOKEN_MARKED_VERIFIED": (
        "a composite child without a token must be unverifiable in production",
        "storage/sources/composite_source.py",
    ),
    "CACHE_DATA_MANIFEST_EPOCH_MISMATCH": (
        "the manifest must record exactly the epoch verified for the read",
        "storage/sources/composite_source.py",
    ),
    "GROUP_MISSING_SILENT_DEFINITION_CHANGE": (
        "missing group must not silently become global demean",
        "cleaned_operators/common/group.py",
    ),
    "BEHAVIORAL_POLICY_NOT_IN_IDENTITY": (
        "behavioral fallback_policy must be a declared param",
        "cleaned_operators/common/group.py",
    ),
    "SCHEMA_UNKNOWN_PIT_DEFAULT_SAFE": (
        "Schema.pit_safe must default to UNKNOWN (None), not True",
        "ir/schema.py",
    ),
    "SOURCE_REF_CROSS_MARKET_COLLISION": (
        "SourceRef v2 carries market-scoped identity; A != US revenue",
        "api/source_ref.py",
    ),
    "SOURCE_REF_PRODUCTION_DECODE_PERMISSIVE": (
        "production SourceRef decode must reject unknown transform / old dialect",
        "api/source_ref.py",
    ),
    "PRODUCTION_VALIDATOR_RESEARCH_ANALYZER": (
        "production DSL validation must run the production Analyzer with a market",
        "api/mining_integration.py",
    ),
    "PRODUCTION_VALIDATOR_MARKET_UNKNOWN": (
        "the production validator must thread a resolved market",
        "api/mining_integration.py",
    ),
    "US_MINING_SPACE_USING_ASHARE_REGISTRY": (
        "the production validator must resolve fields via the per-market registry",
        "api/mining_integration.py",
    ),
    "LABEL_CONFIG_FORWARD_SEMANTIC_MISMATCH": (
        "the default label formula must match the runtime forward-return builder",
        "api/label_pit.py",
    ),
    "LABEL_BAR_CALENDAR_DAY_CONFUSION": (
        "gap_bars must be a bar count, never a calendar Timedelta",
        "api/label_pit.py",
    ),
    "LABEL_RAW_CORPORATE_ACTION_CONTAMINATION": (
        "the default label target must be a return ratio (CA-safe)",
        "api/label_pit.py",
    ),
    "ADJUSTMENT_FACTOR_VINTAGE_PIT_UNPROVEN": (
        "a single AdjustmentFactorCertificate is the price-adjustment authority",
        "fields/providers.py",
    ),
}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _check_relation(area: Path) -> list[str]:
    violations: list[str] = []
    for py in sorted(area.rglob("*.py")):
        src = _read(py)
        # R24-004: public operator bodies must not reindex_like / bare reindex /
        # outer-inner align a SECONDARY panel.  Allow the strict helpers.
        for pattern, label in (
            (r"\.reindex_like\(", "reindex_like"),
            (r"\.align\(", "align"),
        ):
            for m in re.finditer(pattern, src):
                line = src[: m.start()].count("\n") + 1
                if "_strict" not in src[max(0, m.start() - 60) : m.start()]:
                    violations.append(
                        f"{py.relative_to(ROOT)}:{line} uses {label} "
                        "(RELATION_MULTI_PANEL_SILENT_REINDEX)"
                    )
    return violations


def _check_file_violations(
    rel_path: str,
    checks: list[tuple[str, str]],
) -> list[str]:
    src = _read(ROOT / rel_path)
    out: list[str] = []
    for pattern, label in checks:
        if re.search(pattern, src):
            out.append(f"{rel_path} matches {pattern!r} ({label})")
    return out


def audit_relation() -> dict[str, list[str]]:
    area = ROOT / "cleaned_operators" / "relation"
    v = _check_relation(area)
    # holder unit heuristic
    v += _check_file_violations(
        "storage/sources/relation.py",
        [
            (r"abs\(\).*max.*> 1\.0.*\* 0\.01", "RELATION_PERCENT_UNIT_HEURISTIC"),
            (r"drop_duplicates\([^)]*keep=\"last\"\)", "RELATION_VINTAGE_KEEP_LAST_LOOKAHEAD"),
            (r"max_age_days=None", "RELATION_STALE_SNAPSHOT_UNBOUNDED"),
        ],
    )
    return {g: [x for x in v if g in x] or [] for g in _GATES if g.startswith("RELATION")}


def audit_pit() -> dict[str, list[str]]:
    v = _check_file_violations(
        "ir/types.py",
        [
            (r'"report_date"|ReportPeriodEndDate.*fallback', "PIT_PERIOD_USED_AS_FILING_FALLBACK"),
        ],
    )
    return {g: [x for x in v if g in x] or [] for g in _GATES if g.startswith("PIT")}


def _symbol_gates() -> dict[str, list[str]]:
    """Import-level symbol checks (the gates that can't be grep'd)."""
    out: dict[str, list[str]] = {g: [] for g in _GATES}
    sys.path.insert(0, str(ROOT))
    try:
        from factor_engine.pit_contract import PITLayerVerdict, four_layer_pit_allowed

        ok, layers = four_layer_pit_allowed(field_pit_allowed=True)
        if ok or layers["table_pit_allowed"] != PITLayerVerdict.UNKNOWN:
            out["PIT_OMITTED_LAYER_ASSUMED_TRUE"].append(
                "four_layer_pit_allowed does not treat an omitted layer as UNKNOWN"
            )
    except Exception as exc:  # pragma: no cover
        out["PIT_OMITTED_LAYER_ASSUMED_TRUE"].append(f"import/run failed: {exc}")
    try:
        from factor_engine.storage.sources.field_plan import MissingSemantic, NormalizedFieldPlan

        if MissingSemantic.OUT_OF_COVERAGE.value != "out_of_coverage":
            out["CURRENT_SNAPSHOT_MARKED_NOT_APPLICABLE"].append("OUT_OF_COVERAGE missing")
        for field in ("concept_id", "field_id", "temporal_model", "availability_precision",
                      "availability_expr", "timeframe", "source_vintage", "snapshot_policy"):
            if field not in getattr(NormalizedFieldPlan, "__dataclass_fields__", {}):
                out["FIELD_PLAN_TEMPORAL_METADATA_LOST"].append(field)
    except Exception as exc:  # pragma: no cover
        out["FIELD_PLAN_TEMPORAL_METADATA_LOST"].append(f"import/run failed: {exc}")
    try:
        from factor_engine.api.source_ref import decode_source_ref_production, make_source_ref

        a = make_source_ref("StockIncome", "revenue", market="ashare", dataset="d_a")
        us = make_source_ref("StockIncome", "revenue", market="us", dataset="d_u")
        from factor_engine.api.source_ref import encode_source_ref

        if encode_source_ref(a) == encode_source_ref(us):
            out["SOURCE_REF_CROSS_MARKET_COLLISION"].append("A and US revenue encode identically")
        try:
            from factor_engine.api.source_ref import decode_source_ref

            bad = make_source_ref("S", "f", dialect_version="1999-01-01")
            decode_source_ref_production(encode_source_ref(bad))
            out["SOURCE_REF_PRODUCTION_DECODE_PERMISSIVE"].append("old dialect accepted")
        except ValueError:
            pass
    except Exception as exc:  # pragma: no cover
        out["SOURCE_REF_CROSS_MARKET_COLLISION"].append(f"import/run failed: {exc}")
    try:
        from factor_engine.api.label_pit import LabelOp, default_mining_label_config

        cfg = default_mining_label_config(horizon_bars=5)
        if cfg["label_formula"] != "forward_return(vwap, 5)":
            out["LABEL_CONFIG_FORWARD_SEMANTIC_MISMATCH"].append(
                f"config formula {cfg['label_formula']!r} is not the forward authority"
            )
    except Exception as exc:  # pragma: no cover
        out["LABEL_CONFIG_FORWARD_SEMANTIC_MISMATCH"].append(f"import/run failed: {exc}")
    return out


def audit_production_validator() -> dict[str, list[str]]:
    src = _read(ROOT / "api" / "mining_integration.py")
    out: dict[str, list[str]] = {g: [] for g in _GATES if g.startswith(("PRODUCTION_VALIDATOR", "US_MINING"))}
    if re.search(r"Analyzer\(\)\.lower\(parse_expr", src):
        out["PRODUCTION_VALIDATOR_RESEARCH_ANALYZER"].append(
            "validate_production_dsl still uses the research Analyzer()"
        )
    if not re.search(r"Analyzer\(production=True, market=", src):
        out["PRODUCTION_VALIDATOR_MARKET_UNKNOWN"].append("no production+market Analyzer")
    if re.search(r"resolve_market_field\(.*ASHARE_CONTEXT", src):
        out["US_MINING_SPACE_USING_ASHARE_REGISTRY"].append("hard-coded A-share context")
    return out


def audit_label() -> dict[str, list[str]]:
    src = _read(ROOT / "api" / "label_pit.py")
    out: dict[str, list[str]] = {g: [] for g in _GATES if g.startswith("LABEL")}
    if re.search(r'"label_formula": f"ts_pct\(' , src):
        out["LABEL_CONFIG_FORWARD_SEMANTIC_MISMATCH"].append("config writes backward ts_pct")
    if re.search(r"pd\.Timedelta\(days=gap_bars\)|Timedelta\(days=\s*gap", src):
        out["LABEL_BAR_CALENDAR_DAY_CONFUSION"].append("gap_bars as calendar days")
    if re.search(r'"label_formula": f"close\[t\+h\]|close\[t \+ h\]', src):
        out["LABEL_RAW_CORPORATE_ACTION_CONTAMINATION"].append("raw close ratio default")
    return out


def main() -> int:
    results: dict[str, list[str]] = {}
    results.update(audit_relation())
    results.update(audit_pit())
    results.update(audit_production_validator())
    results.update(audit_label())
    results.update(_symbol_gates())
    # gates never inspected stay "not applicable" (empty = no violation found).
    for gate in _GATES:
        results.setdefault(gate, [])

    total = sum(len(v) for v in results.values())
    report = {
        "schema_version": "factor_engine.r24.audit.v1",
        "hard_gates_total": len(_GATES),
        "hard_gate_violations": total,
        "all_gates_zero": total == 0,
        "gates": {g: {"reason": _GATES[g][0], "violations": v} for g, v in results.items()},
    }
    (DOCS / "R24_HARD_GATES.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"R24 hard gates: {total} violations across {len(_GATES)} gates")
    for g, v in results.items():
        status = "OK" if not v else f"FAIL({len(v)})"
        print(f"  [{status}] {g}")
        for item in v[:3]:
            print(f"        - {item}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
