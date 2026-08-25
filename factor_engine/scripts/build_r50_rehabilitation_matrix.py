# -*- coding: utf-8 -*-
"""R50 — per-operator production rehabilitation matrix.

For every canonical operator in the live FactorEngine registry this script
produces a machine row describing its production-readiness dimensions, a list
of blockers (possibly several per operator), and a final *rehabilitation fate*
drawn from a 5-state model:

    PROD_TERMINAL / PROD_INTERNAL / DATA_GATED / RESEARCH_ONLY / DELETE

Invariant (the user's hard rule): every row must get a concrete final_fate and
the sum of all fates MUST equal the live registry total — never UNKNOWN /
UNCLASSIFIED.

All row fields are read from the R18 direct-use authority
(``factor_engine.mining.direct_use.build_direct_use_operator``) plus the raw
registry catalog entry.  The script is read-only over the registry; it writes
only its own artifacts under ``build/r50/``.

Artifacts:
    R50_OPERATOR_PRODUCTION_REHABILITATION_MATRIX.{parquet,csv,json}
    R50_OPERATOR_PRODUCTION_REHABILITATION_SUMMARY.json

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
# never grant a production admission, and never leave a fate unknown.
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(_REPO_ROOT, "build", "r50")

FATES = ("PROD_TERMINAL", "PROD_INTERNAL", "DATA_GATED", "RESEARCH_ONLY", "DELETE")
DELETE_PREFIXES = ("delete_",)
HARD_TERMINAL_BLOCKERS = frozenset({"LOOKAHEAD", "PIT_UNKNOWN", "MATHEMATICALLY_WRONG"})

# Known level / scale-sensitive price-level canonicals that are NOT
# scale-invariant.  Matches the R18 ``_PRICE_LEVEL_INTERMEDIATE_OPS`` family.
KNOWN_LEVEL_OPS = frozenset(
    {
        "rolling_vwap", "true_range", "ATR_WILDER",
        "donchian_upper", "donchian_lower", "donchian_mid",
        "KeltnerMid", "KeltnerUpper", "KeltnerLower",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "KAMA", "DEMA", "TEMA", "PSAR", "Supertrend",
        "ts_prev_high", "ts_prev_low",
        "ts_last_pivot_high", "ts_last_pivot_low",
        "ts_nth_pivot_high", "ts_nth_pivot_low",
        "ts_resistance_level", "ts_support_level",
        "candle_body", "candle_abs_body", "candle_range",
        "candle_upper_shadow", "candle_lower_shadow", "candle_gap",
        "ts_swing_amplitude", "ts_channel_width", "ts_consolidation_width",
        "ts_ema", "ts_sma", "ts_wma", "ts_hma", "ts_dema",
        "bollinger_upper", "bollinger_lower", "bollinger_mid",
        "close", "open", "high", "low", "volume", "amount", "market_cap",
        "avg_price", "close_p", "open_p", "high_p", "low_p",
    }
)

# Known future-looking / support-resistance families that may use future pivots.
LOOKAHEAD_TOKENS = ("breakout", "breakdown", "support", "resistance",
                    "new_high", "new_low", "pivot", "channel", "donchian")

# Known in-sample diagnostics — RESEARCH_TOOL by authority; flagged IN_SAMPLE_MODEL.
IN_SAMPLE_MODEL_OPS = frozenset(
    {
        "ts_ar_fitted_value", "ts_ar_in_sample_resid",
        "ts_poly2_coeff", "ts_poly2_resid",
        "ts_quantile_regression_beta", "ts_quantile_regression_coeff",
        "ts_quantile_regression_slope", "ts_quantile_regression_resid",
        "ts_multi_regression_coeff", "ts_multi_regression_resid",
        "ts_multi_regression_resid_z", "ts_multi_regression_r2",
        "ts_huber_regression_coeff", "ts_huber_regression_in_sample_resid",
        "ts_ridge_regression_coeff", "ts_ridge_regression_in_sample_resid",
        "ts_expectile_regression_coeff", "ts_expectile_regression_resid",
        "ts_ar_forecast", "ts_ar_innovation", "ts_ar_innovation_z",
        "cs_beta_to_market", "cs_alpha_to_market",
        "intra_idiosyncratic_variance", "intra_realized_beta",
        "intra_realized_correlation", "micro_bvc_vpin",
    }
)

FUNDAMENTAL_FAMILIES = frozenset(
    {"fundamental_value", "fundamental_quality", "fundamental_revision"}
)


def compute_blockers(op, catalog: dict[str, Any], canonical: str) -> list[str]:
    """Return the FULL blocker set for one operator (multiple allowed)."""
    blockers: list[str] = []

    status = op.direct_use_status.value
    if status not in ("direct_alpha", "direct_alpha_high_cost", "direct_recipe"):
        blockers.append("ROLE_NOT_TERMINAL")

    # RAW_LEVEL — non scale-invariant level output.
    level = (
        op.output_value_domain == "price_level"
        or canonical in KNOWN_LEVEL_OPS
    )
    if level:
        blockers.append("RAW_LEVEL")

    # CROSS_SECTION_DEGENERATE — broadcast a single value across the cross-section.
    if op.mining_role in ("global_state", "group_state"):
        blockers.append("CROSS_SECTION_DEGENERATE")

    # LOOKAHEAD — family heuristics flagged for oracle verification.
    if any(tok in canonical.lower() for tok in LOOKAHEAD_TOKENS):
        blockers.append("LOOKAHEAD")

    # PIT_UNKNOWN — fundamental operator without production/PIT certification.
    if (
        not op.production_certified
        and op.economic_effect_family in FUNDAMENTAL_FAMILIES
    ):
        blockers.append("PIT_UNKNOWN")

    # UNIT_UNKNOWN — no declared output unit.
    if op.output_unit is None:
        blockers.append("UNIT_UNKNOWN")

    # SOURCE_UNKNOWN — no source recipes / source-status missing.
    if not op.source_recipes:
        blockers.append("SOURCE_UNKNOWN")

    # PARAMETER_NON_INJECTIVE.
    if not op.parameter_injectivity_passed:
        blockers.append("PARAMETER_NON_INJECTIVE")

    # NUMERIC_UNCERTIFIED — no independent numeric oracle evidence recorded.
    if not op.production_certified:
        blockers.append("NUMERIC_UNCERTIFIED")

    # IN_SAMPLE_MODEL — known in-sample diagnostic.
    if canonical in IN_SAMPLE_MODEL_OPS:
        blockers.append("IN_SAMPLE_MODEL")

    # NO_INCREMENTAL_PATH — full-history / warmup replay without a checkpoint.
    if (
        op.execution_model == "full_history"
        or (op.stateful and not op.checkpoint_supported)
    ):
        blockers.append("NO_INCREMENTAL_PATH")

    # BACKEND_UNCERTIFIED — no exact PI production route.
    if not op.production_admitted:
        blockers.append("BACKEND_UNCERTIFIED")

    # HIGH_COST — expensive family.
    if op.runtime_cost >= 3:
        blockers.append("HIGH_COST")

    # OUTPUT_WRONG_SHAPE — not stock×date series.
    if op.output_cardinality == "broadcast" or op.output_semantic_kind != "series":
        blockers.append("OUTPUT_WRONG_SHAPE")

    # DUPLICATE — the canonical is the target of another canonical's alias.
    if canonical in ALIAS_TARGETS:
        blockers.append("DUPLICATE")

    return blockers


def resolve_fate(opts, blockers: list[str]) -> str:
    """5-state fate.  MUST return one of OUT — never UNKNOWN."""
    status = opts.direct_use_status.value
    bset = frozenset(blockers)

    # 1. DELETE dominates every other verdict.
    if status.startswith("delete_"):
        return "DELETE"
    # 2. Research / internal tooling is never production.
    if status in ("research_tool", "move_internal"):
        return "RESEARCH_ONLY"
    # 3. Full terminal production.
    if (
        opts.production_admitted
        and opts.terminal_usable
        and not (bset & HARD_TERMINAL_BLOCKERS)
    ):
        return "PROD_TERMINAL"
    # 4. Internal production.
    if opts.production_admitted and opts.composition_usable:
        return "PROD_INTERNAL"
    # 5. Data-gated: sound algorithm but missing a source.
    if "SOURCE_UNKNOWN" in bset and not (bset & {"LOOKAHEAD", "MATHEMATICALLY_WRONG"}):
        return "DATA_GATED"
    # 6. Everything else.
    return "RESEARCH_ONLY"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # R50: the registry is LAZY — only a subset (~466) is registered until
    # ``load_all()`` runs.  The incremental ledger and evidence generator both
    # call load_all() and see the FULL registry (1624).  To make the matrix
    # total equal the live registry total (the user's hard invariant), we MUST
    # load the full registry here too, or the matrix silently covers only the
    # lazy subset and the fate sum never matches the ledger.
    load_all()

    global ALIAS_TARGETS
    aliases = dict(OperatorRegistry._aliases)
    ALIAS_TARGETS = {c for c in OperatorRegistry._catalog if aliases.get(c)}

    canonicals = sorted(OperatorRegistry._catalog)
    from factor_engine.mining.direct_use import build_direct_use_operator

    rows: list[dict[str, Any]] = []
    blocker_hist = Counter()
    fates: Counter[str] = Counter()
    per_family: Counter = Counter()
    backend_cnt: Counter = Counter()
    exec_cnt: Counter = Counter()
    unclassified: list[str] = []

    for name in canonicals:
        catalog = dict(OperatorRegistry._catalog.get(name, {}))
        op = build_direct_use_operator(name, catalog)
        blockers = compute_blockers(op, catalog, name)
        fate = resolve_fate(op, blockers)

        for b in blockers:
            blocker_hist[b] += 1
        fates[fate] += 1
        per_family[(op.economic_effect_family, fate)] += 1
        backend_cnt[op.preferred_backend] += 1
        exec_cnt[op.execution_model] += 1

        if fate not in FATES:
            unclassified.append(name)

        rows.append(
            {
                "canonical": name,
                "direct_use_status": op.direct_use_status.value,
                "mining_role": op.mining_role,
                "terminal_allowed": op.terminal_allowed,
                "production_admitted": op.production_admitted,
                "production_certified": op.production_certified,
                "directly_usable": op.directly_usable,
                "mining_visible": op.mining_visible,
                "composition_usable": op.composition_usable,
                "terminal_usable": op.terminal_usable,
                "production_terminal_usable": op.production_terminal_usable,
                "context_admitted": op.context_admitted,
                "economic_effect_family": op.economic_effect_family,
                "semantic_redundancy_group": op.semantic_redundancy_group,
                "monotonic_transform_class": op.monotonic_transform_class,
                "output_semantic_kind": op.output_semantic_kind,
                "output_unit": op.output_unit,
                "output_cardinality": op.output_cardinality,
                "output_value_domain": op.output_value_domain,
                "output_grain": op.output_grain,
                "supported_markets": list(op.supported_markets),
                "source_recipes": list(op.source_recipes),
                "stateful": op.stateful,
                "execution_model": op.execution_model,
                "checkpoint_supported": op.checkpoint_supported,
                "runtime_cost": op.runtime_cost,
                "memory_cost": op.memory_cost,
                "preferred_backend": op.preferred_backend,
                "reference_backend": op.reference_backend,
                "search_prior": op.search_prior,
                "family_budget": op.family_budget,
                "cost_budget": op.cost_budget,
                "blockers": blockers,
                "final_fate": fate,
            }
        )

    total = len(canonicals)
    promotion = fates["PROD_TERMINAL"] + fates["PROD_INTERNAL"]
    summary = {
        "total_canonical": total,
        "fate_counts": {k: fates[k] for k in FATES},
        "blocker_histogram": {k: blocker_hist[k] for k in sorted(blocker_hist)},
        "promotion_rate": round(promotion / total, 4) if total else 0.0,
        "per_family_promotion": {},
        "backend_counts": {k: backend_cnt[k] for k in sorted(backend_cnt)},
        "incremental_class_counts": {k: exec_cnt[k] for k in sorted(exec_cnt)},
        "unclassified": unclassified,
    }
    # per_family_promotion: family -> {fate: count}
    fam_map: dict[str, dict[str, int]] = {}
    for (fam, fate), n in per_family.items():
        fam_map.setdefault(fam, {}).setdefault(fate, 0)
        fam_map[fam][fate] += 1
    summary["per_family_promotion"] = {k: fam_map[k] for k in sorted(fam_map)}
    return rows, summary


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, summary = build_rows()

    # ---- Invariant check (hard rule): sum(fates) == total and no UNKNOWN. ----
    total = summary["total_canonical"]
    fate_sum = sum(summary["fate_counts"].values())
    unclassified = summary["unclassified"]
    invariant_ok = (fate_sum == total) and (len(unclassified) == 0)
    print("=== R50 Rehabilitation Matrix ===")
    print(f"total_canonical        = {total}")
    print(f"sum(fate_counts)       = {fate_sum}")
    print(f"unclassified rows      = {len(unclassified)}")
    print(f"INVARIANT OK           = {invariant_ok}")
    print("fate_counts            =", json.dumps(summary["fate_counts"]))
    print("blocker_histogram      =", json.dumps(summary["blocker_histogram"]))
    print("promotion_rate         =", summary["promotion_rate"])
    print("backend_counts         =", json.dumps(summary["backend_counts"]))
    print("incremental_class_counts=", json.dumps(summary["incremental_class_counts"]))
    if unclassified:
        print("UNCLASSIFIED            =", unclassified)

    # ---- Write artifacts. ----
    json_path = os.path.join(OUT_DIR, "R50_OPERATOR_PRODUCTION_REHABILITATION_MATRIX.json")
    csv_path = os.path.join(OUT_DIR, "R50_OPERATOR_PRODUCTION_REHABILITATION_MATRIX.csv")
    parquet_path = os.path.join(OUT_DIR, "R50_OPERATOR_PRODUCTION_REHABILITATION_MATRIX.parquet")
    summary_path = os.path.join(OUT_DIR, "R50_OPERATOR_PRODUCTION_REHABILITATION_SUMMARY.json")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # CSV via stdlib csv module.
    import csv

    if rows:
        col_order = list(rows[0].keys())
    else:
        col_order = ["canonical", "final_fate"]
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
        print("wrote parquet         =", parquet_path)
    except Exception as exc:  # pragma: no cover - env-specific
        print(f"parquet skipped ({exc.__class__.__name__}: {exc})")

    print("wrote json            =", json_path)
    print("wrote csv             =", csv_path)
    print("wrote summary         =", summary_path)


if __name__ == "__main__":
    main()
