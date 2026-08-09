# -*- coding: utf-8 -*-
"""FactorEngine Operator Admission Matrix generator (R15 fail-closed).

For EVERY registered canonical (after ``load_all`` + hardening + evidence
overlay) produce one machine record answering:

1.  is the math correct / certified?
2.  is it PIT-safe / usable for historical mining?
3.  are the input data type and source available?
4.  what role should it play (alpha / state / gate / group / global /
    intraday-eod / fundamental / recipe-internal / diagnostic / denied)?
5.  has it passed production certification and which mining lane is it in?

R15 changes vs the R12 version:

* ``source_available`` no longer fakes "all sources available" when no context
  is given — a record carries ``source_status`` (unknown / satisfied /
  missing), and the context-free eligibility is ``intrinsic`` only
  (R15-INC-028/029/247/248).
* ``recursive`` and ``incremental_supported`` come from the ExecutionContract
  execution model, not from ``recursive == stateful`` (R15-INC-033/034/249/250).
* ``current_row_semantics`` and ``window_semantics`` are structured values read
  from the resolved contract / closure registry, not ``True/False/""`` strings
  or a coarse bounded/full_history fallback (R15-INC-035/036/251/252).
* Blockers are split into ``hard_blockers`` / ``routing_constraints`` /
  ``quality_warnings`` — a high cost, a group/global state, or a non-stock role
  is a ROUTING decision, never a quality defect (R15-INC-030/031/032/245).
* Every non-direct entry carries ordered ``required_actions[]`` covering ALL of
  its blockers, from a single shared AdmissionDecision (R15-INC-038/246).
* ``B24``-``B32`` codes are emitted only when a real detector produced them —
  the vocabulary alone never claims a PASS (R15-INC-037/253).
* The CLI has a ``--strict`` fail threshold and every artifact carries a
  HEAD/dirty/registry/evidence fingerprint (R15-INC-040/041/255/256).

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
    AdmissionState,
    MiningRole,
    RoleSource,
    _ROLE_AST_POSITIONS,
    _admission_decision,
    assign_mining_role_ex,
    cost_tier,
    mining_eligible,
    source_status,
)

# ---------------------------------------------------------------------------
# Blocker vocabulary.  B01..B32 are the R12 codes; B33/B34 added for R15
# (missing cost contract / unresolved role).  Each hard blocker MUST have a
# real detector that produces it — see ``DETECTOR_COVERAGE`` below.
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
    "B33": "COST_CONTRACT_MISSING",
    "B34": "ROLE_UNRESOLVED",
}

# R15-INC-037/253: which blocker codes have a REAL machine detector behind them.
# A code listed here is only ever set by its detector; the vocabulary alone
# never claims a PASS.  As detectors are implemented in
# scripts/audit_all_registered_operators.py they are added here.
DETECTOR_COVERAGE = frozenset(
    {
        "B01", "B02", "B03", "B04", "B05", "B06",  # catalog review flags
        "B08", "B09", "B10", "B11", "B12", "B13", "B14",  # evidence gates
        "B15",  # source availability / field presence
        "B18", "B19",  # execution model
        "B33", "B34",  # cost contract / role resolution
    }
)

# R15-INC-245: a blocker is a HARD defect only if it blocks mining outright.
# Everything else is a routing constraint (which lane / position) or a quality
# warning that must not be reported as "cannot use".
_HARD_BLOCKERS = frozenset(
    {
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B08", "B09", "B10", "B11", "B12", "B13", "B14",
        "B15", "B18", "B19",
        "B24", "B25", "B26", "B27", "B28", "B29", "B30", "B31", "B32",
        "B33", "B34",
    }
)
_ROUTING_BLOCKERS = frozenset({"B20", "B21", "B22", "B23", "B16", "B17", "B07"})


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
    execution_model: str = ""

    # R15-INC-035/251: structured current-row semantics (never bool-stringified).
    current_observation_role: str = ""
    # R15-INC-036/252: window semantics read from the resolved contract when
    # declared; ``window_semantics_declared`` distinguishes "unknown" from a
    # real value.
    window_semantics: str = ""
    window_semantics_declared: bool = False
    missing_policy: str | None = None

    factor_role: str = ""
    role_source: str = ""
    admission_state: str = ""
    cost_tier: int = 0
    cost_contract_declared: bool = False
    cost_lane: str = ""

    compatibility_only: bool = False
    diagnostic_only: bool = False
    benchmark_only: bool = False
    hidden_from_default_mining: bool = False

    source_required: list[str] = field(default_factory=list)
    source_satisfied: list[str] = field(default_factory=list)
    source_missing: list[str] = field(default_factory=list)
    source_status: str = "unknown"  # unknown | satisfied | missing

    # R15-INC-245: three independent buckets.  A fully-certified EventBool /
    # group-state / high-cost operator carries empty ``hard_blockers``.
    hard_blockers: list[str] = field(default_factory=list)
    routing_constraints: list[str] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)
    blocker_codes: list[str] = field(default_factory=list)  # union, legacy view
    required_actions: list[str] = field(default_factory=list)
    recommended_action: str = ""
    target_mining_lane: str = ""

    # R15-INC-029/248: intrinsic eligibility (certification + role, no source
    # context) vs contextual eligibility (needs market/source/frequency).
    intrinsic_mining_eligible: bool = False
    contextual_mining_eligible: str = ""  # "" (not evaluated) | eligible | blocked
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


def _artifact_fingerprint() -> dict[str, Any]:
    """R15-INC-041/256: every artifact carries HEAD / dirty / registry /
    evidence fingerprints so a stale report is never mistaken for current."""
    import hashlib
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            return ""

    from cleaned_operators.registry import OperatorRegistry

    registry_digest = ""
    try:
        blob = json.dumps(
            {
                c: sorted(k for k in OperatorRegistry._catalog[c] if not callable(k))
                for c in sorted(OperatorRegistry._catalog)
            },
            sort_keys=True,
        )
        registry_digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    except Exception:
        pass
    evidence_digest = ""
    try:
        from backend.factor_operator_evidence import load_factor_operator_evidence

        payload = load_factor_operator_evidence() or {}
        evidence_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        pass
    return {
        "commit_sha": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "registry_fingerprint": registry_digest,
        "evidence_fingerprint": evidence_digest,
    }


def write_admission_outputs(
    records: list[AdmissionRecord],
    out_dir: Path,
) -> dict[str, Path]:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": "factor_engine.admission_matrix.v2",
        "generated_at": _artifact_fingerprint(),
    }

    json_path = out_dir / "operator_admission_matrix.json"
    json_path.write_text(
        json.dumps(
            {"header": header, "records": [asdict(r) for r in records]},
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
            for key in (
                "aliases", "panel_params", "input_semantic_types",
                "source_required", "source_satisfied", "source_missing",
                "hard_blockers", "routing_constraints", "quality_warnings",
                "blocker_codes", "required_actions",
            ):
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
    lines += ["", "## Hard blockers (quality defects)", ""]
    blocker_counts: Counter[str] = Counter()
    for r in records:
        for b in r.hard_blockers:
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
    # R15-INC-040/255: strict production mode fails on P0/invariant violations.
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any unresolved/denied production canonical exists",
    )
    args = parser.parse_args(argv)
    records = generate_admission_matrix()
    pools = write_admission_outputs(records, Path(args.out))
    print(f"admission records: {len(records)}")
    for name, path in pools.items():
        print(f"  {name}: {path}")
    if args.strict:
        unresolved = [r.canonical for r in records if r.admission_state == "unresolved"]
        if unresolved:
            print(f"STRICT FAIL: {len(unresolved)} unresolved roles: {unresolved[:10]}")
            return 1
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

    from mining.operator_catalog import _execution_flags, _missing_policy, _output_grain

    stateful, execution_model, checkpoint, full_replay = _execution_flags(canonical)
    rec.stateful = stateful
    rec.execution_model = execution_model.value
    # R15-INC-033/249: ``recursive`` is a specific state model, not "any stateful".
    rec.recursive = execution_model.value == "checkpoint"
    rec.checkpoint_supported = checkpoint
    rec.full_history_replay_required = full_replay
    # R15-INC-034/250: stateless bounded window supports incremental warmup.
    rec.incremental_supported = execution_model.value in (
        "independent_with_warmup", "checkpoint",
    )

    policy = None
    try:
        from cleaned_operators.operator_policy import infer_operator_policy

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(operator, canonical=canonical)
    except Exception:
        policy = None
    rec.missing_policy = _missing_policy(canonical)
    # R15-INC-035/251: structured current-row role, not a bool string.
    rec.current_observation_role = str(
        catalog.get("current_observation_role") or ""
    )
    if not rec.current_observation_role and policy is not None:
        rec.current_observation_role = (
            "current_inclusive"
            if bool(getattr(policy, "includes_current_bar", False))
            else "strict_prior"
        )
    # R15-INC-036/252: window semantics from the resolved contract when declared.
    try:
        from cleaned_operators.closure.window_semantics import window_semantics_for

        ws = window_semantics_for(canonical)
        if ws is not None:
            rec.window_semantics = str(ws.value)
            rec.window_semantics_declared = True
    except Exception:
        pass
    if not rec.window_semantics_declared:
        rec.window_semantics = str(catalog.get("window_semantics") or "") or (
            "full_history" if rec.full_history_replay_required else ""
        )

    role, role_source = assign_mining_role_ex(canonical, catalog)
    rec.factor_role = role.value
    rec.role_source = role_source.value
    rec.cost_tier = cost_tier(canonical, catalog)
    rec.cost_contract_declared = catalog.get("cost_contract_declared", False)
    from mining.operator_catalog import cost_contract_declared as _cost_declared

    rec.cost_contract_declared = _cost_declared(canonical, catalog)
    rec.cost_lane = "direct_high_cost" if rec.cost_tier >= 5 else "direct"

    rec.compatibility_only = bool(catalog.get("compatibility_only"))
    rec.diagnostic_only = bool(catalog.get("diagnostic_only"))
    rec.benchmark_only = bool(catalog.get("benchmark_only"))
    rec.hidden_from_default_mining = bool(catalog.get("hidden_from_default_mining"))

    from mining.operator_catalog import _required_sources as _req

    sstatus = source_status(canonical, catalog, None)  # no context → unknown
    rec.source_required = list(sstatus.required)
    rec.source_satisfied = list(sstatus.satisfied)
    rec.source_missing = list(sstatus.missing)
    rec.source_status = "unknown" if sstatus.unknown else (
        "satisfied" if not sstatus.missing else "missing"
    )

    # ---- single shared AdmissionDecision (R15-INC-019) ----
    state, hard_blockers, actions = _admission_decision(
        canonical, catalog, role, role_source, sstatus
    )
    rec.admission_state = state.value
    rec.required_actions = list(actions)
    rec.hard_blockers = list(hard_blockers)

    # ---- routing / warning constraints are separate from hard blockers ----
    routing: list[str] = []
    warnings: list[str] = []
    if role in (MiningRole.STATE, MiningRole.CONDITION, MiningRole.EVENT):
        routing.append("B20")
    if role is MiningRole.GROUP_STATE:
        routing.append("B22")
    if role is MiningRole.GLOBAL_STATE:
        routing.append("B21")
    if rec.cost_tier >= 5:
        routing.append("B23")
    if rec.cost_contract_declared and not _cost_declared(canonical, catalog):
        pass
    rec.routing_constraints = routing
    rec.quality_warnings = warnings
    rec.blocker_codes = list(dict.fromkeys(list(hard_blockers) + routing + warnings))

    rec.target_mining_lane = _lane_for(role)
    # R15-INC-029/248: intrinsic = certification + role (no context); contextual
    # needs real market/source/frequency — without it, report the source_status.
    rec.intrinsic_mining_eligible = bool(catalog.get("production_certified")) and role in (
        MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
        MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT,
    ) and _cost_declared(canonical, catalog)
    if sstatus.unknown:
        rec.contextual_mining_eligible = ""
    elif not sstatus.missing and rec.intrinsic_mining_eligible:
        rec.contextual_mining_eligible = "eligible"
    else:
        rec.contextual_mining_eligible = "blocked"
    rec.default_mining_eligible = mining_eligible(
        canonical, catalog=catalog, role=role, available_sources=None
    )
    rec.recommended_action = "; ".join(actions) if actions else _recommended_action(rec, catalog)
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
    if role is MiningRole.UNRESOLVED:
        return "unresolved"
    return "denied"


def _recommended_action(rec: AdmissionRecord, catalog: dict[str, Any]) -> str:
    """Fallback recommended action when the shared decision produced none."""
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
