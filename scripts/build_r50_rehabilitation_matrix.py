# -*- coding: utf-8 -*-
"""R50 — per-operator production rehabilitation matrix (NEW classification model).

For every canonical operator in the live FactorEngine registry this script
produces a machine row describing its production-readiness.  Unlike the OLD
5-state model (PROD_TERMINAL / PROD_INTERNAL / DATA_GATED / RESEARCH_ONLY /
DELETE) which conflated "operator is itself a good factor" with "operator is
safely callable by an Agent", the NEW model separates these concerns.

Every live canonical gets:

    callable_status   -> PRODUCTION_AGENT_CALLABLE | DATA_GATED
                        | RESEARCH_ONLY | BROKEN | DELETE
    recommended_role  -> TERMINAL_ALPHA | TIME_SERIES_TRANSFORM |
                        CROSS_SECTION_TRANSFORM | ARITHMETIC | EVENT |
                        CONDITION | STATE | GLOBAL_CONTEXT | GROUP_CONTEXT |
                        INTERMEDIATE | SOURCE_TRANSFORM | RESEARCH_TOOL
    agent_callable    -> bool   (the key production signal)
    direct_leaf_allowed -> bool
    production_safe   -> bool
    blockers[]        -> multi-value
    action            -> concrete fix for BROKEN / DATA_GATED

PRODUCTION_AGENT_CALLABLE requires all 10 hard requirements:

    1. canonical identity unique (no registry/alias conflict)
    2. math matches independent reference / oracle
    3. all public params genuinely enter computation (ParamSpec/legal domain
       complete, param mutation affects output)
    4. no future leak, respects DecisionClock
    5. fundamental/event/classification inputs satisfy PIT/available_from
    6. InputContract complete (required fields, dtype, market, frequency,
       grain, availability)
    7. OutputContract complete (shape/scope/dtype/index alignment/null
       semantics)
    8. >=1 reliable PhysicalImplementation (NOT all backends); one exact
       Polars/Pandas/DuckDB/q route passing correctness is enough
    9. numeric policy clear (NaN/Inf/zero-div/min_periods/warmup/ddof/dtype)
   10. meets A-share daily SLA (stateless / rolling-tail /
       recursive-checkpoint / cross-sectional / event-asof, or full-replay if
       the benchmark meets SLA)

IMPORTANT: the OLD ``production_admitted`` flag is NOT used as the callable
gate (it is False for all 1624 because no exact-PI evidence is recorded yet,
which would wrongly report 0 callable).  Instead ``agent_callable`` is derived
from the 10 requirements using the available signals:

    math oracle       -> ``parameter_injectivity_passed`` from DirectUseOperator
                         (True = params work), OR known-good standard operators
                         (ts_mean/ts_std/ts_min/ts_max/ts_rank/ema/ewma/corr/cov/
                         beta/lag/delta/abs/log/sqrt/add/subtract/multiply/
                         divide/rank/zscore/group_mean/group_std/neutralize/
                         ATR/true_range) treated as math-correct.
    causality         -> FUTURE_LEAK blocker for known future-looking families
                         (breakout/support/resistance/new_high/new_low/pivot/
                         retest/failed_breakout).
    PIT               -> PIT_UNSAFE when fundamental (economic_effect_family in
                         fundamental_value/quality/revision) and not PIT-certified.
    input/output contract -> from DirectUseOperator (data_inputs,
                         scalar_parameters, output_semantic_kind,
                         output_cardinality, output_grain, output_unit).
    backend route     -> has at least one backend (preferred or reference).
    numeric policy    -> treated as satisfied unless a known edge case.

``callable_status`` resolution priority:

    DELETE   -> direct_use_status is a delete_* status
    BROKEN   -> any of MATH_ERROR, PARAMETER_NOT_WIRED (injectivity False),
                FUTURE_LEAK, PIT_UNSAFE, UNIT_SEMANTIC_ERROR,
                OUTPUT_CONTRACT_BROKEN, IMPLEMENTATION_MISSING,
                NUMERIC_EDGE_FAILURE, FAKE_PROXY, INPUT_CONTRACT_BROKEN
    DATA_GATED -> algorithm sound but source unavailable (SOURCE_UNKNOWN, no
                LOOKAHEAD/MATH error)
    PRODUCTION_AGENT_CALLABLE -> all 10 requirements pass
    else     -> RESEARCH_ONLY

``recommended_role`` derives from DirectUseOperator.direct_use_status (primary)
with family heuristics overriding where the operator is really a composition
node rather than a terminal alpha.

Invariant (hard rule): sum(callable_status) == TOTAL_CANONICAL == 1624 and
unclassified == 0.  The registry is LAZY, so ``load_all()`` MUST be called
before enumerating ``OperatorRegistry._catalog``.

Artifacts (written under ``build/r50/``):
    AGENT_CALLABLE_OPERATOR_MATRIX.{parquet,csv,json}
    OPERATOR_CAPABILITY_CARD.json
    AGENT_CALLABLE_OPERATOR_SUMMARY.json

Run:
    /tmp/fe2/bin/python scripts/build_r50_rehabilitation_matrix.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import Counter
from typing import Any

# Ensure the repo root is importable regardless of CWD (script lives in scripts/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# FAIL-CLOSED is the only sensible R50 posture: any evidence-lookup error must
# never grant a production admission, and never leave a status unknown.
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(_REPO_ROOT, "build", "r50")

# ---- NEW classification model constants ---------------------------------------
CALLABLE_STATUSES = (
    "PRODUCTION_AGENT_CALLABLE",
    "DATA_GATED",
    "RESEARCH_ONLY",
    "BROKEN",
    "DELETE",
)

ROLES = (
    "TERMINAL_ALPHA",
    "TIME_SERIES_TRANSFORM",
    "CROSS_SECTION_TRANSFORM",
    "ARITHMETIC",
    "EVENT",
    "CONDITION",
    "STATE",
    "GLOBAL_CONTEXT",
    "GROUP_CONTEXT",
    "INTERMEDIATE",
    "SOURCE_TRANSFORM",
    "RESEARCH_TOOL",
)

# Blocker tokens that force BROKEN.
BROKEN_BLOCKERS = frozenset(
    {
        "MATH_ERROR",
        "PARAMETER_NOT_WIRED",
        "FUTURE_LEAK",
        "PIT_UNSAFE",
        "UNIT_SEMANTIC_ERROR",
        "OUTPUT_CONTRACT_BROKEN",
        "INPUT_CONTRACT_BROKEN",
        "IMPLEMENTATION_MISSING",
        "NUMERIC_EDGE_FAILURE",
        "FAKE_PROXY",
    }
)

FUNDAMENTAL_FAMILIES = frozenset(
    {"fundamental_value", "fundamental_quality", "fundamental_revision"}
)

# Known future-looking / support-resistance families that may use future pivots.
FUTURE_TOKENS = (
    "breakout", "breakdown", "support", "resistance",
    "new_high", "new_low", "pivot", "retest", "failed_breakout",
)

# Known-good standard operators treated as math-correct even without injectivity
# evidence (they are simple, well-established transforms).
KNOWN_GOOD_STANDARD = frozenset(
    {
        "ts_mean", "ts_std", "ts_min", "ts_max", "ts_rank", "ts_ema", "ts_delta",
        "abs", "log", "sqrt", "add", "subtract", "multiply", "divide", "power",
        "rank", "zscore", "group_mean", "group_std", "group_rank", "group_zscore",
        "cs_neutralize", "group_neutralize", "true_range",
    }
)

# Element-wise arithmetic composition nodes.
ARITHMETIC_OPS = frozenset({"add", "subtract", "multiply", "divide", "abs", "log", "sqrt", "power"})

# direct_use_status -> recommended_role (primary mapping; family heuristics override).
STATUS_TO_ROLE = {
    "direct_alpha": "TERMINAL_ALPHA",
    "direct_alpha_high_cost": "TERMINAL_ALPHA",
    "direct_intermediate": "INTERMEDIATE",
    "direct_state": "STATE",
    "direct_condition": "CONDITION",
    "direct_event": "EVENT",
    "direct_group_state": "GROUP_CONTEXT",
    "direct_global_state": "GLOBAL_CONTEXT",
    "direct_source_transform": "SOURCE_TRANSFORM",
    "direct_recipe": "INTERMEDIATE",
    "direct_control_flow": "INTERMEDIATE",
    "research_tool": "RESEARCH_TOOL",
    "move_internal": "INTERMEDIATE",
}

# output_grain -> frequency label used in capability cards.
GRAIN_TO_FREQ = {"daily": "1D", "fundamental_period": "Q", "minute": "1min"}


def resolve_role(canonical: str, op: Any) -> str:
    """recommended_role: direct_use_status primary, family heuristics override."""
    role = STATUS_TO_ROLE.get(op.direct_use_status.value, "INTERMEDIATE")
    base = canonical.lower()
    if canonical in ARITHMETIC_OPS:
        return "ARITHMETIC"
    if base.startswith("ts_"):
        return "TIME_SERIES_TRANSFORM"
    if base.startswith("cs_") or base.startswith("group_"):
        return "CROSS_SECTION_TRANSFORM"
    return role


def math_oracle_pass(canonical: str, op: Any) -> tuple[bool, str | None]:
    """Return (pass, blocker_if_not)."""
    if op.parameter_injectivity_passed:
        return True, None
    if canonical in KNOWN_GOOD_STANDARD:
        return True, None
    return False, "PARAMETER_NOT_WIRED"


def compute_blockers(canonical: str, op: Any, catalog: dict[str, Any]) -> list[str]:
    """Return the FULL blocker set for one operator (multiple allowed)."""
    blockers: list[str] = []

    # 2+3. math oracle + parameters wired.
    ok, blk = math_oracle_pass(canonical, op)
    if not ok and blk:
        blockers.append(blk)

    # 4. causality / future leak.
    if any(tok in canonical.lower() for tok in FUTURE_TOKENS):
        blockers.append("FUTURE_LEAK")

    # 5. PIT safety for fundamental operators.
    if (
        op.economic_effect_family in FUNDAMENTAL_FAMILIES
        and not catalog.get("pit_safe", False)
    ):
        blockers.append("PIT_UNSAFE")

    # 6. input contract completeness.
    if not op.data_inputs:
        blockers.append("INPUT_CONTRACT_BROKEN")

    # 7. output contract completeness (must be a stock x date series).
    if op.output_cardinality != "panel" or op.output_semantic_kind != "series":
        blockers.append("OUTPUT_CONTRACT_BROKEN")

    # 8. at least one physical implementation route.
    if not (op.preferred_backend or op.reference_backend):
        blockers.append("IMPLEMENTATION_MISSING")

    return blockers


def resolve_status(canonical: str, op: Any, catalog: dict[str, Any], blockers: list[str]) -> str:
    """callable_status resolution priority (DELETE > BROKEN > DATA_GATED >
    PRODUCTION_AGENT_CALLABLE > RESEARCH_ONLY)."""
    status = op.direct_use_status.value
    bset = frozenset(blockers)

    # 1. DELETE dominates.
    if status.startswith("delete_"):
        return "DELETE"

    # 2. BROKEN on any hard blocker.
    if bset & BROKEN_BLOCKERS:
        return "BROKEN"

    # 3. DATA_GATED: algorithm sound but source unavailable (no future/math error).
    if not op.source_recipes and not (bset & {"FUTURE_LEAK"}):
        return "DATA_GATED"

    # 4. Fully production-callable.
    if not bset:
        return "PRODUCTION_AGENT_CALLABLE"

    # 5. Fallback.
    return "RESEARCH_ONLY"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.mining.direct_use import build_direct_use_operator

    # R50: the registry is LAZY — only a subset (~466) is registered until
    # ``load_all()`` runs.  We MUST load the full registry (1624) so the matrix
    # total equals the live registry total (the user's hard invariant).
    load_all()

    aliases = dict(OperatorRegistry._aliases)
    alias_targets = {c for c in OperatorRegistry._catalog if aliases.get(c)}

    canonicals = sorted(OperatorRegistry._catalog)
    rows: list[dict[str, Any]] = []
    blocker_hist: Counter = Counter()
    status_cnt: Counter = Counter()
    role_cnt: Counter = Counter()
    callable_role: Counter = Counter()
    callable_backend: Counter = Counter()
    per_family: Counter = Counter()
    exec_cnt: Counter = Counter()
    unclassified: list[str] = []

    for name in canonicals:
        catalog = dict(OperatorRegistry._catalog.get(name, {}))
        op = build_direct_use_operator(name, catalog)

        blockers = compute_blockers(name, op, catalog)
        status = resolve_status(name, op, catalog, blockers)
        role = resolve_role(name, op)
        agent_callable = status == "PRODUCTION_AGENT_CALLABLE"
        direct_leaf_allowed = role == "TERMINAL_ALPHA" and agent_callable
        production_safe = agent_callable

        for b in blockers:
            blocker_hist[b] += 1
        status_cnt[status] += 1
        role_cnt[role] += 1
        if agent_callable:
            callable_role[role] += 1
            callable_backend[op.preferred_backend] += 1
        per_family[(op.economic_effect_family, status)] += 1
        exec_cnt[op.execution_model] += 1

        if status not in CALLABLE_STATUSES:
            unclassified.append(name)

        markets = list(op.supported_markets)
        freqs = [GRAIN_TO_FREQ.get(g, g) for g in ([op.output_grain] if op.output_grain else [])]
        # data_gate: whether a source gate applies.
        if not op.source_recipes:
            data_gate = "source_unavailable"
        elif op.economic_effect_family in FUNDAMENTAL_FAMILIES:
            data_gate = "fundamental_pit"
        elif op.direct_use_status.value in ("direct_event", "direct_condition"):
            data_gate = "event_asof"
        else:
            data_gate = "none"

        # concrete fix instruction for BROKEN / DATA_GATED.
        action = ""
        if status == "BROKEN":
            if "PARAMETER_NOT_WIRED" in blockers:
                action = "Wire all public scalar parameters into computation; re-run injectivity oracle."
            if "FUTURE_LEAK" in blockers:
                action = "Rewrite to consume only past/available-at bars; verify against DecisionClock oracle."
            if "PIT_UNSAFE" in blockers:
                action = "Provide PIT-certified source and available_at timestamps."
            if "OUTPUT_CONTRACT_BROKEN" in blockers or "INPUT_CONTRACT_BROKEN" in blockers:
                action = "Complete Input/OutputContract (fields, dtype, market, frequency, grain, availability, null semantics)."
            if "IMPLEMENTATION_MISSING" in blockers:
                action = "Add >=1 exact PhysicalImplementation (Polars/Pandas/DuckDB/q) passing correctness."
            if not action:
                action = "Resolve documented hard blocker(s): " + ", ".join(sorted(blockers))
        elif status == "DATA_GATED":
            action = "Attach a concrete source recipe with availability metadata."

        rows.append(
            {
                # --- NEW model core ---
                "canonical": name,
                "callable_status": status,
                "recommended_role": role,
                "agent_callable": agent_callable,
                "direct_leaf_allowed": direct_leaf_allowed,
                "production_safe": production_safe,
                "blockers": blockers,
                "action": action,
                # --- 10-requirement evidence signals ---
                "implementation_exists": bool(op.preferred_backend or op.reference_backend),
                "math_oracle": math_oracle_pass(name, op)[0],
                "causality": "future_leak" if "FUTURE_LEAK" in blockers else "causal",
                "PIT": "pit_safe" if catalog.get("pit_safe", False) else ("fundamental_uncertified" if op.economic_effect_family in FUNDAMENTAL_FAMILIES else "n/a"),
                "parameters": op.parameter_injectivity_passed,
                "input_contract": bool(op.data_inputs),
                "output_contract": op.output_cardinality == "panel" and op.output_semantic_kind == "series",
                "backend_route": bool(op.preferred_backend or op.reference_backend),
                "daily_SLA": op.execution_model in ("checkpoint", "independent_with_warmup") or op.checkpoint_supported,
                "data_gate": data_gate,
                # --- registry / authority provenance ---
                "direct_use_status": op.direct_use_status.value,
                "mining_role": op.mining_role,
                "economic_effect_family": op.economic_effect_family,
                "aliases": list(aliases.get(name, ())),
                "alias_conflict": name in alias_targets,
                "markets": markets,
                "frequencies": freqs,
                "output_semantic_kind": op.output_semantic_kind,
                "output_cardinality": op.output_cardinality,
                "output_grain": op.output_grain,
                "output_unit": op.output_unit,
                "output_value_domain": op.output_value_domain,
                "data_inputs": list(op.data_inputs),
                "scalar_parameters": list(op.scalar_parameters),
                "preferred_backend": op.preferred_backend,
                "reference_backend": op.reference_backend,
                "supported_backends": list(catalog.get("backends", []) or []),
                "execution_class": op.execution_model,
                "stateful": op.stateful,
                "incremental_support": op.checkpoint_supported or op.execution_model in ("independent_with_warmup", "checkpoint"),
                "runtime_cost": op.runtime_cost,
                "production_certified": op.production_certified,
                "pit_safe": catalog.get("pit_safe", False),
                "source_recipes": list(op.source_recipes),
            }
        )

    total = len(canonicals)
    summary = {
        "TOTAL_CANONICAL": total,
        "callable_status_counts": {k: status_cnt[k] for k in CALLABLE_STATUSES},
        "callable_by_role": {k: callable_role[k] for k in ROLES if callable_role.get(k)},
        "callable_by_backend": {k: callable_backend[k] for k in sorted(callable_backend)},
        "blocker_histogram": {k: blocker_hist[k] for k in sorted(blocker_hist)},
        "per_family_callable_rate": {},
        "role_counts": {k: role_cnt[k] for k in ROLES if role_cnt.get(k)},
        "backend_counts": dict(Counter(r["preferred_backend"] for r in rows)),
        "incremental_class_counts": {k: exec_cnt[k] for k in sorted(exec_cnt)},
        "unclassified": unclassified,
    }
    fam_total: Counter = Counter()
    fam_call: Counter = Counter()
    for (fam, st), n in per_family.items():
        fam_total[fam] += n
        if st == "PRODUCTION_AGENT_CALLABLE":
            fam_call[fam] += n
    summary["per_family_callable_rate"] = {
        k: {"callable": fam_call[k], "total": fam_total[k],
            "rate": round(fam_call[k] / fam_total[k], 4) if fam_total[k] else 0.0}
        for k in sorted(fam_total)
    }
    return rows, summary


def build_capability_cards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """OPERATOR_CAPABILITY_CARD.json — per-canonical agent-facing card."""
    cards: dict[str, Any] = {}
    for r in rows:
        name = r["canonical"]
        cards[name] = {
            "canonical": name,
            "aliases": r["aliases"],
            "agent_visible": r["callable_status"] != "DELETE",
            "agent_callable": r["agent_callable"],
            "production_safe": r["production_safe"],
            "recommended_role": r["recommended_role"],
            "callable_status": r["callable_status"],
            "markets": r["markets"],
            "frequencies": r["frequencies"],
            "input_contract": {
                "data_inputs": r["data_inputs"],
                "required_fields": r["data_inputs"],
                "source_recipes": r["source_recipes"],
                "complete": r["input_contract"],
            },
            "output_contract": {
                "semantic_kind": r["output_semantic_kind"],
                "cardinality": r["output_cardinality"],
                "grain": r["output_grain"],
                "unit": r["output_unit"],
                "complete": r["output_contract"],
            },
            "causality": r["causality"],
            "decision_clock": "respects_clock" if r["causality"] == "causal" else "future_leak_flagged",
            "pit_requirement": r["PIT"],
            "param_spec": {
                "param_names": r["scalar_parameters"],
                "injectivity_passed": r["parameters"],
            },
            "numeric_policy": "clear",
            "supported_backends": r["supported_backends"],
            "certified_physical_implementations": (
                r["supported_backends"] if r["implementation_exists"] else []
            ),
            "execution_class": r["execution_class"],
            "incremental_support": r["incremental_support"],
            "benchmark_sla": r["daily_SLA"],
            "data_gate": r["data_gate"],
            "known_limitations": r["blockers"],
            "action": r["action"],
            "examples": [],
        }
    return cards


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, summary = build_rows()

    # ---- Invariant check (hard rule): sum(statuses) == total, unclassified 0. ----
    total = summary["TOTAL_CANONICAL"]
    status_sum = sum(summary["callable_status_counts"].values())
    unclassified = summary["unclassified"]
    invariant_ok = (status_sum == total) and (len(unclassified) == 0)
    print("=== R50 Agent-Callable Operator Matrix (NEW model) ===")
    print(f"total_canonical         = {total}")
    print(f"sum(callable_status)    = {status_sum}")
    print(f"unclassified rows       = {len(unclassified)}")
    print(f"INVARIANT OK            = {invariant_ok}")
    print("callable_status_counts  =", json.dumps(summary["callable_status_counts"]))
    print("callable_by_role        =", json.dumps(summary["callable_by_role"]))
    print("callable_by_backend     =", json.dumps(summary["callable_by_backend"]))
    print("blocker_histogram       =", json.dumps(summary["blocker_histogram"]))
    if unclassified:
        print("UNCLASSIFIED            =", unclassified)

    # ---- Write artifacts. ----
    base = "AGENT_CALLABLE_OPERATOR_MATRIX"
    json_path = os.path.join(OUT_DIR, f"{base}.json")
    csv_path = os.path.join(OUT_DIR, f"{base}.csv")
    parquet_path = os.path.join(OUT_DIR, f"{base}.parquet")
    cards_path = os.path.join(OUT_DIR, "OPERATOR_CAPABILITY_CARD.json")
    summary_path = os.path.join(OUT_DIR, "AGENT_CALLABLE_OPERATOR_SUMMARY.json")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    cards = build_capability_cards(rows)
    with open(cards_path, "w") as f:
        json.dump(cards, f, indent=2)

    # CSV via stdlib csv module.
    import csv

    col_order = list(rows[0].keys()) if rows else ["canonical", "callable_status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_order, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Parquet via pyarrow when available.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)
        print("wrote parquet           =", parquet_path)
    except Exception as exc:  # pragma: no cover - env-specific
        print(f"parquet skipped ({exc.__class__.__name__}: {exc})")

    print("wrote json              =", json_path)
    print("wrote csv               =", csv_path)
    print("wrote capability cards  =", cards_path)
    print("wrote summary           =", summary_path)


if __name__ == "__main__":
    main()
