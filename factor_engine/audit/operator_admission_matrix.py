# -*- coding: utf-8 -*-
"""FactorEngine Operator Admission Matrix generator.

For EVERY registered canonical (after ``load_all`` + hardening + evidence
overlay) produce one machine record answering:

1.  is the math correct / certified?
2.  is it PIT-safe / usable for historical mining?
3.  are the input data type and source available?
4.  what role should it play (alpha / state / gate / group / global /
    intraday-eod / fundamental / recipe-internal / diagnostic / denied)?
5.  has it passed production certification and which mining lane is it in?

Every non-direct operator carries at least one machine ``blocker_code`` (the
B01..B32 vocabulary), an exact ``recommended_action``, a ``data_missing`` flag
and a ``code_fixable`` flag — never a vague "not production" / "research only".

Outputs (written to ``--out`` directory):
  * operator_admission_matrix.csv
  * operator_admission_matrix.json
  * operator_admission_summary.md
  * direct_daily.json / direct_specialized.json / direct_intraday_eod.json /
    direct_high_cost.json / recipe_only.json / diagnostic_only.json / denied.json
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mining.operator_catalog import (
    MiningRole,
    _ROLE_AST_POSITIONS,
    assign_mining_role,
    cost_tier,
    mining_eligible,
)

# ---------------------------------------------------------------------------
# Blocker vocabulary (AI plan §三)
# ---------------------------------------------------------------------------

BLOCKERS = {
    "B01": "PERMANENT_PIT_UNSAFE",
    "B02": "NON_FACTOR_OPERATOR",
    "B03": "COMPAT_ALIAS",
    "B04": "DIAGNOSTIC_IN_SAMPLE",
    "B05": "BENCHMARK_ONLY",
    "B06": "HIDDEN_FROM_MINING",
    "B07": "EXPERIMENTAL_LIFECYCLE",
    "B08": "IMPLEMENTATION_EVIDENCE_MISSING",
    "B09": "SEMANTIC_GOLDEN_MISSING",
    "B10": "TEMPORAL_PREFIX_MISSING",
    "B11": "SOURCE_PIT_MISSING",
    "B12": "EDGE_CONTRACT_UNDECLARED",
    "B13": "EDGE_EVIDENCE_MISSING",
    "B14": "BACKEND_EVIDENCE_MISSING",
    "B15": "SOURCE_FIELD_MISSING",
    "B16": "SOURCE_GRAIN_UNSUPPORTED",
    "B17": "GRAIN_TRANSFORM_UNCERTIFIED",
    "B18": "STATE_CHECKPOINT_MISSING",
    "B19": "FULL_HISTORY_ONLY",
    "B20": "ROLE_NOT_STOCK_ALPHA",
    "B21": "GLOBAL_STATE",
    "B22": "GROUP_STATE",
    "B23": "HIGH_COMPUTE_COST",
    "B24": "LOW_EMPIRICAL_COVERAGE",
    "B25": "INSUFFICIENT_EFFECTIVE_SAMPLE",
    "B26": "PARAM_SPACE_UNSAFE",
    "B27": "DEAD_PARAMETER",
    "B28": "SEMANTIC_DUPLICATE",
    "B29": "MATH_DEFINITION_DEFECT",
    "B30": "UNIT_CONTRACT_DEFECT",
    "B31": "AXIS_CONTRACT_DEFECT",
    "B32": "MISSING_TIME_TOPOLOGY_DEFECT",
}


@dataclass
class AdmissionRecord:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    authoring_tier: str = ""
    registered_status: str = ""
    lifecycle_status: str = ""

    production_certified: bool = False
    implementation_passed: bool = False
    semantic_passed: bool = False
    temporal_passed: bool = False
    source_pit_passed: bool = False
    edge_case_passed: bool = False
    backend_passed: bool = False

    pandas_available: bool = False
    pandas_certified: bool = False
    polars_available: bool = False
    polars_certified: bool = False
    duckdb_available: bool = False
    duckdb_certified: bool = False

    input_grain: str | None = None
    output_grain: str | None = None
    panel_params: list[str] = field(default_factory=list)
    input_semantic_types: list[str] = field(default_factory=list)
    output_unit: str | None = None

    stateful: bool = False
    recursive: bool = False
    full_history_replay_required: bool = False
    checkpoint_supported: bool = False
    incremental_supported: bool = False

    current_row_semantics: str = ""
    window_semantics: str = ""
    missing_policy: str | None = None

    factor_role: str = ""
    cost_tier: int = 0

    compatibility_only: bool = False
    diagnostic_only: bool = False
    benchmark_only: bool = False
    hidden_from_default_mining: bool = False

    source_required: list[str] = field(default_factory=list)
    source_available: list[str] = field(default_factory=list)

    blocker_codes: list[str] = field(default_factory=list)
    recommended_action: str = ""
    target_mining_lane: str = ""

    production_eligible: bool = False
    default_mining_eligible: bool = False


def generate_admission_matrix() -> list[AdmissionRecord]:
    """Generate the full admission matrix for every registered canonical."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    from research_tools.registry import ResearchToolRegistry

    research_tools = set(ResearchToolRegistry.list_canonical())
    records: list[AdmissionRecord] = []
    for canonical in sorted(OperatorRegistry._catalog):
        if canonical in research_tools:
            continue
        records.append(_record(canonical, OperatorRegistry._catalog[canonical]))
    return records


_POOL_LANES = {
    "direct_daily": ("direct_daily",),
    "direct_specialized": ("direct_specialized",),
    "direct_intraday_eod": ("direct_intraday_eod",),
    "direct_high_cost": ("direct_high_cost",),
    "recipe_only": ("recipe_only",),
    "diagnostic_only": ("diagnostic_only", "source_transform"),
    "denied": ("denied", "legacy_only", "research_pending"),
}


def _pool_json(records: list[AdmissionRecord], lanes: tuple[str, ...]) -> dict[str, Any]:
    members = [r for r in records if r.target_mining_lane in lanes]
    return {
        "pool": lanes[0],
        "count": len(members),
        "canonicals": [m.canonical for m in sorted(members, key=lambda m: m.canonical)],
    }


def write_admission_outputs(
    records: list[AdmissionRecord],
    out_dir: Path,
) -> dict[str, Path]:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "operator_admission_matrix.json"
    json_path.write_text(
        json.dumps(
            {"schema_version": "factor_engine.admission_matrix.v1", "records": [asdict(r) for r in records]},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )

    csv_path = out_dir / "operator_admission_matrix.csv"
    fieldnames = [f for f in AdmissionRecord.__dataclass_fields__]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            for key in ("aliases", "panel_params", "input_semantic_types",
                        "source_required", "source_available", "blocker_codes"):
                row[key] = "|".join(str(v) for v in row[key])
            writer.writerow(row)

    pools: dict[str, Path] = {}
    for name, lanes in _POOL_LANES.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(_pool_json(records, lanes), ensure_ascii=False, indent=1), encoding="utf-8")
        pools[name] = path

    summary_path = out_dir / "operator_admission_summary.md"
    summary_path.write_text(_summary_md(records), encoding="utf-8")
    pools["summary"] = summary_path

    return pools


def _summary_md(records: list[AdmissionRecord]) -> str:
    from collections import Counter

    lines: list[str] = [
        "# FactorEngine Operator Admission Summary",
        "",
        f"- total registered canonical = {len(records)}",
        "",
        "## Role (factor_role) distribution",
        "",
    ]
    role_counts = Counter(r.factor_role for r in records)
    for role, count in sorted(role_counts.items()):
        lines.append(f"- {role} = {count}")
    lane_counts = Counter(r.target_mining_lane for r in records)
    lines += ["", "## Target mining lane distribution", ""]
    for lane, count in sorted(lane_counts.items()):
        lines.append(f"- {lane} = {count}")
    lines += ["", "## Blockers", ""]
    blocker_counts: Counter[str] = Counter()
    for r in records:
        for b in r.blocker_codes:
            blocker_counts[b] += 1
    for code in sorted(blocker_counts, key=lambda c: (int(c[1:3]), c)):
        lines.append(f"- {code} {BLOCKERS[code]}: {blocker_counts[code]}")
    lines += ["", "## Pools", ""]
    for name in _POOL_LANES:
        pool = _pool_json(records, _POOL_LANES[name])
        lines.append(f"- {name}: {pool['count']}")
    lines += [
        "",
        "## Every canonical in a non-direct pool carries exact blocker(s) + a",
        "recommended_action in operator_admission_matrix.json/csv — no vague",
        "'not production' / 'research only' answers.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/admission", help="output directory")
    args = parser.parse_args(argv)
    records = generate_admission_matrix()
    pools = write_admission_outputs(records, Path(args.out))
    print(f"admission records: {len(records)}")
    for name, path in pools.items():
        print(f"  {name}: {path}")
    return 0


def _record(canonical: str, catalog: dict[str, Any]) -> AdmissionRecord:
    from cleaned_operators.operator_spec import (
        PERMANENTLY_FORBIDDEN_CANONICALS,
        PRODUCTION_DENIED_CANONICALS,
    )
    from cleaned_operators.production_hardening import (
        NON_FACTOR_PRODUCTION_CANONICALS,
        SOURCE_BLOCKED_CANONICALS,
    )
    from cleaned_operators.operator_surface import classify_canonical
    from cleaned_operators.registry import OperatorRegistry

    rec = AdmissionRecord(canonical=canonical)
    rec.aliases = sorted(str(a) for a in (catalog.get("aliases") or ()))
    rec.authoring_tier = classify_canonical(canonical)
    rec.registered_status = str(catalog.get("status") or catalog.get("registered_status") or "")
    rec.lifecycle_status = str(catalog.get("lifecycle_status") or rec.registered_status or "")

    rec.production_certified = catalog.get("production_certified") is True
    rec.implementation_passed = bool(catalog.get("implementation_certified"))
    rec.semantic_passed = bool(catalog.get("semantic_certified") or catalog.get("semantic_golden_verified"))
    rec.temporal_passed = bool(catalog.get("temporal_certified") or catalog.get("temporal_prefix_verified"))
    rec.source_pit_passed = bool(catalog.get("source_contract_certified") or catalog.get("source_contract_verified"))
    rec.edge_case_passed = bool(catalog.get("edge_case_passed"))
    rec.backend_passed = bool(catalog.get("backend_passed"))

    backends = OperatorRegistry.backends_for(canonical) or set()
    rec.pandas_available = "pandas_numpy" in backends
    rec.polars_available = "polars" in backends
    rec.duckdb_available = bool({"duckdb_sql", "sql"} & set(backends)) or bool(
        catalog.get("duckdb_available")
    )
    backend_meta = catalog.get("backend_meta") or {}
    rec.pandas_certified = bool((backend_meta.get("pandas_numpy") or {}).get("production_certified"))
    rec.polars_certified = bool((backend_meta.get("polars") or {}).get("production_certified"))
    rec.duckdb_certified = bool(
        (backend_meta.get("duckdb_sql") or {}).get("production_certified")
    ) or bool((backend_meta.get("sql") or {}).get("production_certified"))

    rec.input_grain = catalog.get("input_grain")
    rec.output_grain = catalog.get("output_grain")
    rec.panel_params = sorted(str(p) for p in (catalog.get("panel_params") or ()))
    fields = catalog.get("input_fields")
    rec.input_semantic_types = sorted(str(f) for f in (fields or ()))
    rec.output_unit = catalog.get("output_unit")

    from mining.operator_catalog import _checkpoint_flags, _missing_policy, _required_sources

    rec.stateful, rec.checkpoint_supported, rec.full_history_replay_required = _checkpoint_flags(canonical)
    rec.recursive = rec.stateful
    rec.incremental_supported = rec.checkpoint_supported

    policy = None
    try:
        from cleaned_operators.operator_policy import infer_operator_policy

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(operator, canonical=canonical)
    except Exception:
        policy = None
    rec.missing_policy = _missing_policy(canonical)
    rec.current_row_semantics = str(getattr(policy, "includes_current_bar", "") or "") if policy else ""
    rec.window_semantics = str(catalog.get("window_semantics") or "") or (
        "bounded" if not rec.full_history_replay_required else "full_history"
    )

    role = assign_mining_role(canonical, catalog)
    rec.factor_role = role.value
    rec.cost_tier = cost_tier(canonical, catalog)

    rec.compatibility_only = bool(catalog.get("compatibility_only"))
    rec.diagnostic_only = bool(catalog.get("diagnostic_only"))
    rec.benchmark_only = bool(catalog.get("benchmark_only"))
    rec.hidden_from_default_mining = bool(catalog.get("hidden_from_default_mining"))

    from mining.operator_catalog import _required_sources as _req, _sources_available

    rec.source_required = list(_req(canonical, catalog))
    rec.source_available = list(_sources_available(canonical, catalog, ()))

    # ---- blockers (deterministic, ordered) ----
    blockers: list[str] = []
    if canonical in PERMANENTLY_FORBIDDEN_CANONICALS or rec.authoring_tier == "unsafe":
        blockers.append("B01" if canonical in PERMANENTLY_FORBIDDEN_CANONICALS else "B29")
    if canonical in NON_FACTOR_PRODUCTION_CANONICALS and canonical not in PERMANENTLY_FORBIDDEN_CANONICALS:
        blockers.append("B02")
    if canonical in PRODUCTION_DENIED_CANONICALS and canonical not in PERMANENTLY_FORBIDDEN_CANONICALS:
        blockers.append("B01" if canonical in {
            "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate",
            "dropna", "shuffle", "sample",
        } else "B02")
    if rec.compatibility_only:
        blockers.append("B03")
    if rec.diagnostic_only:
        blockers.append("B04")
    if rec.benchmark_only:
        blockers.append("B05")
    if rec.hidden_from_default_mining:
        blockers.append("B06")
    if canonical in SOURCE_BLOCKED_CANONICALS:
        blockers.append("B15")
    if rec.stateful and not rec.checkpoint_supported:
        blockers.append("B18")
    if rec.full_history_replay_required:
        blockers.append("B19")
    if role is MiningRole.GLOBAL_STATE:
        blockers.append("B21")
    if role is MiningRole.GROUP_STATE:
        blockers.append("B22")
    if role in (MiningRole.STATE, MiningRole.CONDITION, MiningRole.EVENT):
        blockers.append("B20")
    if rec.cost_tier >= 5:
        blockers.append("B23")
    if not rec.production_certified:
        if not rec.implementation_passed:
            blockers.append("B08")
        if not rec.semantic_passed:
            blockers.append("B09")
        if not rec.temporal_passed:
            blockers.append("B10")
        if not rec.source_pit_passed:
            blockers.append("B11")
        if not rec.edge_case_passed:
            # edge declared but not verified -> B13; undeclared -> B12
            try:
                from cleaned_operators.edge_requirements import edge_requirements_declared

                blockers.append("B12" if not edge_requirements_declared(canonical) else "B13")
            except Exception:
                blockers.append("B13")
        if not rec.backend_passed and not (rec.pandas_certified or rec.polars_certified or rec.duckdb_certified):
            blockers.append("B14")
    # dedupe, keep first-seen order
    seen: set[str] = set()
    rec.blocker_codes = [b for b in blockers if not (b in seen or seen.add(b))]

    rec.target_mining_lane = _lane_for(role)
    rec.production_eligible = rec.production_certified and _pit_safe_ok(policy)
    rec.default_mining_eligible = mining_eligible(
        canonical, catalog=catalog, role=role, available_sources=None
    )
    rec.recommended_action = _recommended_action(rec, catalog)
    return rec


def _pit_safe_ok(policy: Any) -> bool:
    if policy is not None:
        return bool(getattr(policy, "pit_safe", False))
    return False


def _lane_for(role: MiningRole) -> str:
    if role in (MiningRole.ALPHA, MiningRole.FUNDAMENTAL_PIT):
        return "direct_daily"
    if role is MiningRole.ALPHA_HIGH_COST:
        return "direct_high_cost"
    if role in (MiningRole.STATE, MiningRole.CONDITION, MiningRole.EVENT,
                MiningRole.GROUP_STATE, MiningRole.GLOBAL_STATE):
        return "direct_specialized"
    if role is MiningRole.INTRADAY_EOD:
        return "direct_intraday_eod"
    if role is MiningRole.RECIPE_INTERNAL:
        return "recipe_only"
    if role is MiningRole.DIAGNOSTIC:
        return "diagnostic_only"
    if role is MiningRole.SOURCE_TRANSFORM:
        return "source_transform"
    if role is MiningRole.RESEARCH:
        return "research_pending"
    if role is MiningRole.LEGACY:
        return "legacy_only"
    return "denied"


def _recommended_action(rec: AdmissionRecord, catalog: dict[str, Any]) -> str:
    """Machine-generated, precise fix.  Never "research only" / "unsupported"."""
    canonical = rec.canonical
    if canonical in {"Lead", "next", "bfill", "causal_bfill", "fillna_interpolate",
                     "interpolate", "dropna", "shuffle", "sample"} or canonical.startswith("rand_"):
        return "delete from public registry: future function / random / non-causal fill; reimplement a strictly causal canonical if a real need exists"
    if canonical in {"norm", "norm_l1", "norm_linf", "fft", "ifft", "wavelet",
                     "convolve", "correlate", "mat_inverse", "eig", "svd", "pca"}:
        return "keep as internal numerical helper (ResearchToolRegistry); expose factor-facing scalar summaries instead of the raw matrix primitive"
    if canonical in {"arg", "tan", "cot", "sec", "csc", "cosh", "sinh"}:
        return "delete from public mining search: pole / exponential-blowup primitive; bounded alternatives (sin/cos/atan) are the searchable forms"
    if "B01" in rec.blocker_codes:
        return "delete from public registry: permanently-forbidden primitive (future / random / non-causal / shape-changing); never a mining node"
    if rec.factor_role in ("internal", "recipe_internal", "source_transform"):
        return "internal / recipe / source-transform layer: underlying capability, never mined as a factor"
    if rec.compatibility_only:
        return "compat alias: merge into canonical identity; never a separate mining node"
    if rec.diagnostic_only or rec.benchmark_only:
        return "move to diagnostics/benchmark layer; mine the *_forecast_error / *_ex_self counterpart instead"
    if rec.blocker_codes and rec.blocker_codes[0] == "B15":
        return "source field missing: wait for a real field/vintage (code cannot fabricate it); delete public canonical if data is confirmed absent"
    if rec.stateful and not rec.checkpoint_supported:
        return "implement StatefulOperator.initialize_state/update/serialize_state/restore_state + bit-parity segmented test, then set incremental_supported"
    if not rec.production_certified:
        missing = [BLOCKERS[b] for b in rec.blocker_codes if b.startswith("B0") and int(b[1:]) >= 8 and int(b[1:]) <= 14]
        if missing:
            return "certify: " + ", ".join(missing) + " — regenerate primitive/factor evidence and re-run the six-gate audit"
        return "certify: pass the six-gate admission audit (implementation/semantic/temporal/source-PIT/edge/backend) and bind evidence"
    if rec.cost_tier >= 5:
        return "route to direct_high_cost lane with an expression cost budget (max tier-5 nodes = 1)"
    return "production-certified: usable in target_mining_lane"


if __name__ == "__main__":
    raise SystemExit(main())
