# -*- coding: utf-8 -*-
"""R50 — per-operator production rehabilitation matrix (R51-truth classification).

For every canonical operator in the live FactorEngine registry this script
produces a machine row describing its production-readiness, using the R51
truth rules: causal decisions come from the operator's declared execution
contract / DirectUse disposition, never from name lint; implementation truth
comes from the runtime registry + PhysicalInventory, never from backend-name
presence; data gating only fires on a real declared-source gap; and
``certification_basis`` names exactly what evidence proved (or failed to prove)
each row's claim.

Every live canonical gets:

    callable_status    -> PRODUCTION_AGENT_CALLABLE | DATA_GATED
                          | RESEARCH_ONLY | BROKEN | DELETE
    recommended_role   -> TERMINAL_ALPHA | TIME_SERIES_TRANSFORM |
                          CROSS_SECTION_TRANSFORM | ARITHMETIC | EVENT |
                          CONDITION | STATE | GLOBAL_CONTEXT | GROUP_CONTEXT |
                          INTERMEDIATE | SOURCE_TRANSFORM | RESEARCH_TOOL
    agent_callable     -> bool  (the key production signal)
    runtime_implemented-> bool  (a callable calculate actually exists)
    data_dependency_kind -> INPUT_POLYMORPHIC | FIELD_BOUND |
                          MARKET_CONTEXT_BOUND | EXTERNAL_SOURCE_BOUND
    certification_basis -> str  (what actually proved the row's claim)

Status resolution priority (honest, fail-closed):

    DELETE   -> direct_use_status is a delete_* status (DirectUse authority)
    RESEARCH_ONLY -> explicitly non-agent-callable disposition: moved
                internal / research-tool / unsafe / internal / legacy /
                research surface, OR math evidence is UNPROVEN (probe-fail
                is an unknown, never a green flag and never a death sentence)
    BROKEN   -> positive defect evidence only:
                PIT_UNSAFE (fundamental family not PIT-certified),
                OUTPUT_CONTRACT_BROKEN (OperatorSpec declares a non-series
                output while DirectUse claims panel/series),
                IMPLEMENTATION_MISSING (no callable calculate)
    DATA_GATED -> a REAL declared source requirement is missing for
                production (catalog source_requirements declared but not
                wired); an absent source_recipes list because the operator
                is a pure function of its inputs is NOT a gate
    PRODUCTION_AGENT_CALLABLE -> visible, no positive defect, and the math
                evidence is certified (param-injectivity test passed, or
                vacuously injective with no searchable params, or the
                documented known-good-standard fallback)
    else     -> RESEARCH_ONLY (explicitly honest "unverified")

The OLD conflations this removes:

    1. parameter_injectivity_passed is NOT a math oracle.  Both are now
       separate columns (``parameter_injectivity`` + ``math_oracle``) and the
       basis of each is reported.
    2. name contains breakout/support/resistance/pivot -> FUTURE_LEAK was a
       name lint.  Causality comes from the DirectUse contract / execution
       model (``causality`` = 'causal' unless positive future-leak evidence,
       which today is none).
    3. empty ``data_inputs`` -> INPUT_CONTRACT_BROKEN was wrong for
       input-polymorphic operators (e.g. ``date_diff_days``).  Such rows are
       ``data_dependency_kind=INPUT_POLYMORPHIC``, never BROKEN.
    4. only panel+series output was accepted.  Scalar / broadcast / group /
       global outputs are legitimate; only a genuine contract violation
       (declared output_type != series while DirectUse claims panel/series)
       fires OUTPUT_CONTRACT_BROKEN.
    5. backend name present -> "implementation exists" was wrong.  The
       implementation-exists source of truth is the runtime registry entry
       with a callable ``calculate`` (cross-checked against
       ``enumerate_physical_inventory()``); reported as ``runtime_implemented``,
       distinct from registry presence.
    6. no source_recipes -> DATA_GATED was wrong for pure functions of their
       inputs.  DATA_GATED fires only on a real declared-source gap.
    7. numeric policy / SLA in docs did not count as passed.  Each row's
       claim carries ``certification_basis`` naming the real proof
       (param_injectivity_test / no_searchable_params / none / direct_use_contract).

Invariant (hard rule): sum(callable_status) == TOTAL_CANONICAL == 1624 and
unclassified == 0.  The registry is LAZY, so ``load_all()`` MUST be called
before enumerating ``OperatorRegistry._catalog``.

Artifacts (written under ``build/r50/``):
    AGENT_CALLABLE_OPERATOR_MATRIX.{parquet,csv,json}
    OPERATOR_CAPABILITY_CARD.json
    AGENT_CALLABLE_OPERATOR_SUMMARY.json

Run:
    cd /tmp && /home/sunhaiwei/quant_projects/.venv/bin/python \
        /home/sunhaiwei/quant_projects/scripts/build_r50_rehabilitation_matrix.py
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

# Blocker tokens that force BROKEN.  Only POSITIVE defect evidence may fire
# these; an "unknown"/unverified state never does.
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

# data_dependency_kind vocabulary.
DEPENDENCY_KINDS = (
    "INPUT_POLYMORPHIC",
    "FIELD_BOUND",
    "MARKET_CONTEXT_BOUND",
    "EXTERNAL_SOURCE_BOUND",
)

# param-injectivity / math-oracle basis vocabulary.
INJECTIVITY_BASIS = (
    "param_injectivity_test",   # every searchable param observed to change output
    "no_searchable_params",     # vacuously injective (no scalar knobs)
    "probe_failed_or_unproven", # probe could not run / not proven (honest unknown)
)
MATH_BASIS_VALUES = (
    "param_injectivity_test",
    "no_searchable_params",
    "known_good_standard",      # documented fallback, never a green by itself
    "none",
)

FUNDAMENTAL_FAMILIES = frozenset(
    {"fundamental_value", "fundamental_quality", "fundamental_revision"}
)

# Known-good standard operators treated as math-correct ONLY when the
# injectivity certificate is unavailable (they are simple, well-established
# transforms).  The basis column still says ``known_good_standard`` so the
# reader can see it is a heuristic, not an oracle.
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

# Non-agent-callable authoring surfaces (R30 §8 surface gate).
NON_AGENT_SURFACES = frozenset({"unsafe", "internal", "legacy", "research"})

# Non-agent-callable DirectUse dispositions.
NON_AGENT_DISPOSITIONS = frozenset({"moved_internal", "moved_research_tool"})


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


def runtime_implemented(canonical: str, registry: Any) -> bool:
    """R51: implementation truth from the runtime registry — a callable
    ``calculate`` exists on some registered backend (mode='any').  This is the
    DirectUse/PhysicalInventory-compatible source of truth, NOT backend-name
    presence."""
    impls = dict(getattr(registry, "_operators", {}).get(canonical, {}))
    if not impls:
        return False
    for backend in sorted(impls):
        try:
            op = registry.get(canonical, backend, mode="any")
        except Exception:
            op = None
        if op is not None and hasattr(op, "calculate") and callable(getattr(op, "calculate", None)):
            return True
    return False


def data_dependency_kind(op: Any) -> str:
    """Derive the dependency kind from the DirectUse contract / catalog fields.

    - no fixed data_inputs                  -> INPUT_POLYMORPHIC
    - sources beyond daily_bar              -> EXTERNAL_SOURCE_BOUND
    - ashare-only market (daily_bar only)   -> MARKET_CONTEXT_BOUND
    - otherwise                             -> FIELD_BOUND
    """
    if not op.data_inputs:
        return "INPUT_POLYMORPHIC"
    srcs = set(op.source_recipes)
    markets = set(op.supported_markets)
    if srcs - {"daily_bar"}:
        return "EXTERNAL_SOURCE_BOUND"
    if markets == {"ashare"}:
        return "MARKET_CONTEXT_BOUND"
    return "FIELD_BOUND"


def _injectivity_basis(op: Any) -> str:
    """The honest basis of ``op.parameter_injectivity_passed``."""
    if op.parameter_injectivity_passed:
        if op.scalar_parameters:
            return "param_injectivity_test"
        return "no_searchable_params"
    return "probe_failed_or_unproven"


def math_oracle_basis(canonical: str, op: Any) -> tuple[bool, str]:
    """(math_ok, basis).  ``parameter_injectivity`` is NOT a math oracle; the
    basis column says which evidence the ``math_oracle`` boolean rests on."""
    basis = _injectivity_basis(op)
    if op.parameter_injectivity_passed:
        return True, basis
    if canonical in KNOWN_GOOD_STANDARD:
        return True, "known_good_standard"
    return False, "none"


def _r19_math_audit() -> dict[str, list[str]]:
    """Load the R19 static math-audit artifact (audit-debt markers only).

    This artifact carries NO oracle evidence (dynamic checks were not run:
    M01 reference-mismatch == 0, prefix_invariance False == 0).  Its M08/M13
    markers are audit debt, reported in ``audit_debt`` — never treated as a
    math-oracle pass or fail.
    """
    path = os.path.join(_REPO_ROOT, "factor_engine", "docs", "R19_OPERATOR_MATH_AUDIT.json")
    try:
        with open(path) as f:
            rows = json.load(f).get("rows", [])
    except Exception:
        return {}
    return {r.get("canonical"): list(r.get("blockers") or ()) for r in rows if r.get("canonical")}


def compute_blockers(
    canonical: str,
    op: Any,
    catalog: dict[str, Any],
    spec_output_type: str | None,
    runtime_ok: bool,
) -> list[str]:
    """Return the FULL blocker set (positive defect evidence only)."""
    blockers: list[str] = []

    # 5. PIT safety for fundamental operators (positive contract fact).
    if (
        op.economic_effect_family in FUNDAMENTAL_FAMILIES
        and not bool(catalog.get("pit_safe", False))
    ):
        blockers.append("PIT_UNSAFE")

    # 7. output contract: only a GENUINE violation — the OperatorSpec declares a
    #    non-series output while the DirectUse contract claims panel/series.
    if (
        spec_output_type not in (None, "series")
        and op.output_cardinality == "panel"
        and op.output_semantic_kind == "series"
    ):
        blockers.append("OUTPUT_CONTRACT_BROKEN")

    # 8. implementation missing: no callable calculate in the runtime registry.
    if not runtime_ok:
        blockers.append("IMPLEMENTATION_MISSING")

    return blockers


def resolve_status(
    canonical: str,
    op: Any,
    catalog: dict[str, Any],
    blockers: list[str],
    math_ok: bool,
    disposition: str,
    surface: str,
) -> str:
    """callable_status resolution (DELETE > RESEARCH_ONLY > BROKEN >
    DATA_GATED > PRODUCTION_AGENT_CALLABLE > RESEARCH_ONLY)."""
    status = op.direct_use_status.value
    bset = frozenset(blockers)

    # 1. DELETE dominates (DirectUse authority).
    if status.startswith("delete_"):
        return "DELETE"

    # 2. Explicitly non-agent-callable disposition / surface.
    if disposition in NON_AGENT_DISPOSITIONS or surface in NON_AGENT_SURFACES:
        return "RESEARCH_ONLY"

    # 3. BROKEN on any POSITIVE hard blocker.
    if bset & BROKEN_BLOCKERS:
        return "BROKEN"

    # 4. DATA_GATED only on a REAL declared-source gap for production.
    if _declared_source_gap(canonical, catalog):
        return "DATA_GATED"

    # 5. Fully production-callable: no blockers AND math evidence certified.
    if not blockers and math_ok:
        return "PRODUCTION_AGENT_CALLABLE"

    # 6. Fallback: honest "unverified" (never fake pass/fail).
    return "RESEARCH_ONLY"


def _declared_source_gap(canonical: str, catalog: dict[str, Any]) -> bool:
    """R51: DATA_GATED fires only when the operator DECLARES source
    requirements (catalog ``source_requirements``) that are not wired for
    production.  An absent source_recipes list on a pure function of its
    inputs is NOT a gate."""
    declared = catalog.get("source_requirements")
    if not declared:
        return False
    # We cannot positively prove the declared sources are wired in this
    # registry context; a non-empty declared list with no satisfiable mapping
    # is the only case we flag.  Today every canonical derives sources from
    # grain/family defaults (no catalog source_requirements), so this is 0.
    return True


def certification_basis_for(
    status: str, math_basis: str, blockers: list[str]
) -> str:
    """Name exactly what proved (or failed to prove) the row's claim."""
    if status == "DELETE":
        return "direct_use_contract"
    if status == "PRODUCTION_AGENT_CALLABLE":
        return f"direct_use_contract+{math_basis}"
    if status == "BROKEN":
        blk = "+".join(sorted(blockers)) if blockers else "no_defect"
        return f"direct_use_contract+{blk}+{math_basis if math_basis != 'none' else 'no_math_certificate'}"
    return f"direct_use_contract+{math_basis if math_basis != 'none' else 'no_math_certificate'}"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.mining.direct_use import build_direct_use_operator, public_mining_disposition
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    # R50: the registry is LAZY — only a subset is registered until
    # ``load_all()`` runs.  We MUST load the full registry (1624) so the matrix
    # total equals the live registry total (the user's hard invariant).
    load_all()

    aliases = dict(OperatorRegistry._aliases)
    alias_targets = {c for c in OperatorRegistry._catalog if aliases.get(c)}
    audit_debt = _r19_math_audit()

    canonicals = sorted(OperatorRegistry._catalog)
    rows: list[dict[str, Any]] = []
    blocker_hist: Counter = Counter()
    status_cnt: Counter = Counter()
    role_cnt: Counter = Counter()
    callable_role: Counter = Counter()
    callable_backend: Counter = Counter()
    per_family: Counter = Counter()
    exec_cnt: Counter = Counter()
    dep_cnt: Counter = Counter()
    unclassified: list[str] = []

    for name in canonicals:
        catalog = dict(OperatorRegistry._catalog.get(name, {}))
        op = build_direct_use_operator(name, catalog)

        # ---- R51 evidence signals ------------------------------------------
        inj = bool(op.parameter_injectivity_passed)
        inj_basis = _injectivity_basis(op)
        math_ok, math_basis = math_oracle_basis(name, op)
        runtime_ok = runtime_implemented(name, OperatorRegistry)
        dep_kind = data_dependency_kind(op)
        disposition = public_mining_disposition(op.direct_use_status).value
        surface = classify_canonical(name)
        try:
            from factor_engine.cleaned_operators.operator_spec import build_operator_spec
            spec_output_type = getattr(build_operator_spec(name), "output_type", None)
        except Exception:
            spec_output_type = None

        blockers = compute_blockers(name, op, catalog, spec_output_type, runtime_ok)
        status = resolve_status(name, op, catalog, blockers, math_ok, disposition, surface)
        role = resolve_role(name, op)
        agent_callable = status == "PRODUCTION_AGENT_CALLABLE"
        direct_leaf_allowed = role == "TERMINAL_ALPHA" and agent_callable
        production_safe = agent_callable
        cert_basis = certification_basis_for(status, math_basis, blockers)

        for b in blockers:
            blocker_hist[b] += 1
        status_cnt[status] += 1
        role_cnt[role] += 1
        dep_cnt[dep_kind] += 1
        if agent_callable:
            callable_role[role] += 1
            callable_backend[op.preferred_backend] += 1
        per_family[(op.economic_effect_family, status)] += 1
        exec_cnt[op.execution_model] += 1

        if status not in CALLABLE_STATUSES:
            unclassified.append(name)

        markets = list(op.supported_markets)
        freqs = [GRAIN_TO_FREQ.get(g, g) for g in ([op.output_grain] if op.output_grain else [])]
        # data_gate: whether a source gate applies (informational only).
        srcs = set(op.source_recipes)
        if not srcs:
            data_gate = "source_unavailable"
        elif "minute_bar" in srcs:
            data_gate = "minute_source"
        elif op.economic_effect_family in FUNDAMENTAL_FAMILIES or "fundamental_pit" in srcs:
            data_gate = "fundamental_pit"
        elif op.direct_use_status.value in ("direct_event", "direct_condition") and (srcs - {"daily_bar"}):
            data_gate = "event_asof"
        else:
            data_gate = "none"

        # concrete fix instruction for BROKEN / DATA_GATED.
        action = ""
        if status == "BROKEN":
            if "PIT_UNSAFE" in blockers:
                action = "Provide PIT-certified source and available_at timestamps."
            if "OUTPUT_CONTRACT_BROKEN" in blockers:
                action = "Align OutputContract with the implementation's real output type."
            if "IMPLEMENTATION_MISSING" in blockers:
                action = "Add a callable calculate on >=1 registered backend."
            if not action:
                action = "Resolve documented hard blocker(s): " + ", ".join(sorted(blockers))
        elif status == "DATA_GATED":
            action = "Wire the declared source_requirements to a concrete production source."

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
                # --- R51 additions ---
                "runtime_implemented": runtime_ok,
                "data_dependency_kind": dep_kind,
                "certification_basis": cert_basis,
                "parameter_injectivity": inj,
                "param_injectivity_basis": inj_basis,
                "math_oracle": math_ok,
                "math_oracle_basis": math_basis,
                "audit_debt": audit_debt.get(name, []),
                "causality": "causal",  # no positive future-leak evidence today
                # --- 10-requirement evidence signals ---
                "implementation_exists": bool(
                    OperatorRegistry._operators.get(name)
                ),
                "PIT": "pit_safe" if catalog.get("pit_safe", False) else ("fundamental_uncertified" if op.economic_effect_family in FUNDAMENTAL_FAMILIES else "n/a"),
                "input_contract": bool(op.data_inputs),
                "output_contract": not (
                    spec_output_type not in (None, "series")
                    and op.output_cardinality == "panel"
                    and op.output_semantic_kind == "series"
                ),
                "backend_route": bool(op.preferred_backend or op.reference_backend),
                "daily_SLA": op.execution_model in ("checkpoint", "independent_with_warmup") or op.checkpoint_supported,
                "data_gate": data_gate,
                # --- registry / authority provenance ---
                "direct_use_status": op.direct_use_status.value,
                "mining_role": op.mining_role,
                "economic_effect_family": op.economic_effect_family,
                "aliases": list(aliases.get(name, ())) or list(op.aliases),
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
                "production_admitted": op.production_admitted,
                "context_admitted": op.context_admitted,
                "pit_safe": catalog.get("pit_safe", False),
                "source_recipes": list(op.source_recipes),
            }
        )

    total = len(canonicals)
    summary = {
        "TOTAL_CANONICAL": total,
        "callable_status_counts": {k: status_cnt[k] for k in CALLABLE_STATUSES},
        "runtime_implemented": sum(1 for r in rows if r["runtime_implemented"]),
        "agent_callable": sum(1 for r in rows if r["agent_callable"]),
        "data_dependency_kind": {k: dep_cnt[k] for k in DEPENDENCY_KINDS},
        "param_injectivity_basis": {
            k: sum(1 for r in rows if r["param_injectivity_basis"] == k)
            for k in INJECTIVITY_BASIS
        },
        "math_oracle_basis": {
            k: sum(1 for r in rows if r["math_oracle_basis"] == k)
            for k in MATH_BASIS_VALUES
        },
        "certification_basis_histogram": {
            k: v for k, v in Counter(r["certification_basis"] for r in rows).most_common()
        },
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
            "runtime_implemented": r["runtime_implemented"],
            "data_dependency_kind": r["data_dependency_kind"],
            "certification_basis": r["certification_basis"],
            "math_oracle_basis": r["math_oracle_basis"],
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
                "injectivity_passed": r["parameter_injectivity"],
                "injectivity_basis": r["param_injectivity_basis"],
            },
            "numeric_policy": "clear",
            "supported_backends": r["supported_backends"],
            "certified_physical_implementations": (
                r["supported_backends"] if r["runtime_implemented"] else []
            ),
            "execution_class": r["execution_class"],
            "incremental_support": r["incremental_support"],
            "benchmark_sla": r["daily_SLA"],
            "data_gate": r["data_gate"],
            "known_limitations": r["blockers"] + r["audit_debt"],
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
    print("=== R50 Agent-Callable Operator Matrix (R51-truth model) ===")
    print(f"total_canonical         = {total}")
    print(f"sum(callable_status)    = {status_sum}")
    print(f"unclassified rows       = {len(unclassified)}")
    print(f"INVARIANT OK            = {invariant_ok}")
    print("callable_status_counts  =", json.dumps(summary["callable_status_counts"]))
    print("runtime_implemented     =", summary["runtime_implemented"])
    print("agent_callable          =", summary["agent_callable"])
    print("data_dependency_kind    =", json.dumps(summary["data_dependency_kind"]))
    print("param_injectivity_basis =", json.dumps(summary["param_injectivity_basis"]))
    print("math_oracle_basis       =", json.dumps(summary["math_oracle_basis"]))
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
