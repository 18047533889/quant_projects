# -*- coding: utf-8 -*-
"""R26 hard-gate audit: operator hidden-correctness gates.

R26-205: every gate must be 0 (no violation found).  The audit is a mix of
structural checks against the actual source + a few import-level symbol checks,
so a regression that reintroduces a target-dependent tie break / unknown->false
pattern / observed-row-as-clock / inferred bar width / unbounded state /
inf feasible default / undefined->0 estimator is caught here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "factor_engine" / "docs"
DOCS.mkdir(exist_ok=True)

_GATES: dict[str, str] = {
    "TARGET_DEPENDENT_TIE_BREAKS": "ts_chatterjee_xi must not sort tied-X rows by the response y",
    "UNKNOWN_TO_FALSE_PATTERN_BUGS": "a NaN pattern input must not collapse into a confirmed False / 0",
    "INTRADAY_OBSERVED_ROW_AS_CLOCK_BUGS": "intraday time positions must use official slot ordinals, not observed-row indices",
    "INTRADAY_INFERRED_BAR_WIDTH_FROM_OBSERVED_DELTAS": "bar width must come from a declared contract, never observed deltas",
    "INTRADAY_UNDETECTED_PHYSICAL_GAPS": "physically absent minutes must be explicit missing slots, never invisible NaN",
    "SESSION_ENDPOINT_SUBSTITUTION_BUGS": "segment / lunch-gap endpoints must be exact official minutes by default",
    "UNDECLARED_UNBOUNDED_STATE": "ffill / carried state / pivot age must be bounded by an explicit lookback",
    "UNDERDECLARED_NESTED_HISTORY": "nested window history must be declared as a sum, not max",
    "DEFAULT_GUARANTEED_INFEASIBLE_OPERATORS": "registered defaults must be runtime-feasible (not guaranteed raise / all-NaN)",
    "UNDEFINED_ESTIMATOR_MAPPED_TO_NORMAL_ZERO": "an undefined estimator (e.g. positive Roll covariance) must be NaN, never 0",
    "NUMERIC_CANCELLATION_BLOCKERS": "sum(x²)-sum(x)²/n style SSE must be recentered (no catastrophic cancellation)",
    "PRELOG_OVERFLOW_UNDERFLOW_BLOCKERS": "log-domain computation must not materialise abs(λ)² / abs(b)²",
    "SIGNED_PROBABILITY_MISLABELED_AS_PROBABILITY": "conditional-probability differences must be declared signed, never [0,1] probability",
    "DIMENSIONALLY_INVALID_OPERATOR_MODES": "directional-change scale units must be mode-fixed (PriceDistance vs DimensionlessVol)",
    "INVALID_COMPOSITION_SCHEMAS": "composition schemas must be genuine part-whole; financial_statement flow fields are rejected",
    "SURROGATE_MISSING_TOPOLOGY_DRIFT": "TE surrogates must fix the NaN mask and rearrange only finite positions",
    "PCA_UNNECESSARY_RANK_REJECTION": "PCA residual must require rank >= k (not k+1)",
    "R26_UNREVIEWED_STATIC_HAZARD_HITS": "every static hazard hit must be classified (remaining_unreviewed == 0)",
}


def _read(path: str) -> str:
    try:
        return (ROOT / path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _violations(gate: str) -> list[str]:
    out: list[str] = []

    if gate == "TARGET_DEPENDENT_TIE_BREAKS":
        src = _read("cleaned_operators/dependence_ext.py")
        if re.search(r"np\.lexsort\(\(yv, xv\)\)", src):
            out.append("ts_chatterjee_xi still sorts tied-X by y (np.lexsort((yv,xv)))")

    elif gate == "UNKNOWN_TO_FALSE_PATTERN_BUGS":
        src = _read("cleaned_operators/price_volume/technical_extensions.py")
        for pat, name in [
            (r"\.astype\(float\).*\.where|where.*\.astype\(float\)", None),
        ]:
            pass
        # cdl_* must route through _cdl_valid / _cdl_signed (tri-state)
        if not re.search(r"def _cdl_valid", src) or not re.search(r"def _cdl_signed", src):
            out.append("candlestick tri-state helpers missing")
        for op in ["_cdl_doji", "_cdl_hammer", "_cdl_spinning_top", "_cdl_outside_bar",
                   "_cdl_engulfing", "_cdl_inside_bar", "_cdl_marubozu"]:
            # find the function and ensure it calls _cdl_signed or _cdl_valid
            m = re.search(rf"def {op}\(.*?(?=\ndef |\Z)", src, re.S)
            if m and "_cdl_signed" not in m.group(0) and "_cdl_valid" not in m.group(0):
                out.append(f"{op} does not tri-state via _cdl_valid/_cdl_signed")
        if re.search(r"return np\.sign\(body\)\.replace\(0\s*,\s*1\)", src):
            out.append("spinning-top zero body still forced +1")

    elif gate == "INTRADAY_OBSERVED_ROW_AS_CLOCK_BUGS":
        src = _read("cleaned_operators/microstructure/intraday_agg.py")
        if re.search(r"idx\[target\]\s*/\s*float\(total\)", src):
            out.append("intra_high_time/low_time still use observed-row index")
        if re.search(r"_position_of\(.*idx\[target\]", src):
            out.append("_position_of uses raw index instead of slot_id")

    elif gate == "INTRADAY_INFERRED_BAR_WIDTH_FROM_OBSERVED_DELTAS":
        src = _read("cleaned_operators/microstructure/intraday_agg.py")
        if re.search(r"def _bar_width_minutes", src):
            out.append("_bar_width_minutes (observed-delta inference) still present")

    elif gate == "INTRADAY_UNDETECTED_PHYSICAL_GAPS":
        src = _read("runtime/session_panel.py")
        if not re.search(r"class SessionPanel", src) or not re.search(r"is_present", src):
            out.append("central SessionPanel / presence tracking missing")

    elif gate == "SESSION_ENDPOINT_SUBSTITUTION_BUGS":
        src = _read("cleaned_operators/microstructure/intraday_agg.py")
        if re.search(r"finite\[-1\]\s*/\s*finite\[0\]", src):
            out.append("segment return still uses last/first finite")
        if re.search(r"mc\[-1\]|ao\[0\]", src) and "recent_valid" not in src:
            out.append("lunch gap still uses first/last finite without explicit policy")
        if not re.search(r'endpoint_policy.*"exact"', src):
            out.append("no explicit exact endpoint policy")

    elif gate == "UNDECLARED_UNBOUNDED_STATE":
        src = _read("cleaned_operators/price_volume/technical_extensions.py")
        if re.search(r"confirmed\.ffill\(\)", src):
            out.append("pivot last-value uses unbounded ffill")
        if "pivot_lookback_bars" not in src:
            out.append("pivot family has no bounded lookback")

    elif gate == "UNDERDECLARED_NESTED_HISTORY":
        src = _read("cleaned_operators/activity_clock.py")
        if "scale_window + max_lookback" not in src:
            out.append("activity-clock history formula not declared as scale_window + max_lookback")

    elif gate == "DEFAULT_GUARANTEED_INFEASIBLE_OPERATORS":
        src = _read("cleaned_operators/state_geometry.py")
        if re.search(r"def _calculate_series\(\s*self, x: pd\.DataFrame, window: int = 120", src):
            out.append("multiscale PE slope default window=120 is guaranteed-infeasible for order=3")
        if "window: int = 256" not in src:
            out.append("multiscale PE slope does not use a feasible default window")

    elif gate == "UNDEFINED_ESTIMATOR_MAPPED_TO_NORMAL_ZERO":
        src = _read("cleaned_operators/spread_estimators.py")
        if re.search(r"2\.0 \* np\.sqrt\(max\(-cov, 0\.0\)\)", src):
            out.append("Roll spread still maps positive covariance to 0")

    elif gate == "NUMERIC_CANCELLATION_BLOCKERS":
        src = _read("cleaned_operators/glr_change.py")
        if re.search(r"pss\[n\]\s*-\s*ps\[n\]\s*\*\s*ps\[n\]\s*/\s*n", src):
            out.append("GLR still uses raw prefix SSE (cancellation)")
        if "vc = v - c" not in src:
            out.append("GLR missing recentered prefix moments")

    elif gate == "PRELOG_OVERFLOW_UNDERFLOW_BLOCKERS":
        src = _read("cleaned_operators/dmd.py")
        if re.search(r"np\.abs\(eig_vals\) \*\* 2", src):
            out.append("DMD materialises rho = |lambda|**2")
        if re.search(r"np\.log\(abs_b \*\* 2\)", src):
            out.append("DMD materialises abs_b**2 before log")

    elif gate == "SIGNED_PROBABILITY_MISLABELED_AS_PROBABILITY":
        src = _read("cleaned_operators/advanced_quantile_dynamics.py")
        if re.search(r'"unit": "probability"', src):
            out.append("extremogram still declared as [0,1] probability")

    elif gate == "DIMENSIONALLY_INVALID_OPERATOR_MODES":
        src = _read("cleaned_operators/directional_change.py")
        if "scale_mode" not in src:
            out.append("directional change missing absolute/relative scale mode")

    elif gate == "INVALID_COMPOSITION_SCHEMAS":
        src = _read("cleaned_operators/composition.py")
        if '"revenue": "financial_statement"' in src or '"net_income": "financial_statement"' in src:
            out.append("invalid financial_statement part-whole family still present")

    elif gate == "SURROGATE_MISSING_TOPOLOGY_DRIFT":
        src = _read("cleaned_operators/advanced_information.py")
        if not re.search(r"finite_idx = np\.isfinite\(sw\)", src):
            out.append("effective TE surrogate does not fix the NaN mask")

    elif gate == "PCA_UNNECESSARY_RANK_REJECTION":
        src = _read("cleaned_operators/advanced_intraday.py")
        if re.search(r"if _numerical_rank\(s\) < k \+ 1:\n\s+continue  # numerical rank < k\+1 -> projection", src):
            out.append("PCA residual still requires rank >= k+1")

    elif gate == "R26_UNREVIEWED_STATIC_HAZARD_HITS":
        hits = DOCS / "R26_STATIC_OPERATOR_HAZARD_SCAN.json"
        if hits.exists():
            data = json.loads(hits.read_text(encoding="utf-8"))
            unreviewed = [h for h in data.get("hits", []) if h.get("hazard_class") == "NEEDS_MANUAL_REVIEW"]
            if unreviewed:
                out.append(f"{len(unreviewed)} hazard hits remain NEEDS_MANUAL_REVIEW")
        else:
            out.append("R26_STATIC_OPERATOR_HAZARD_SCAN.json not generated")
    return out


def main() -> int:
    results: dict[str, list[str]] = {}
    for gate in _GATES:
        results[gate] = _violations(gate)
    total = sum(len(v) for v in results.values())
    report = {
        "schema_version": "factor_engine.r26.audit.v1",
        "hard_gates_total": len(_GATES),
        "hard_gate_violations": total,
        "all_gates_zero": total == 0,
        "gates": {g: {"reason": _GATES[g], "violations": v} for g, v in results.items()},
    }
    (DOCS / "R26_HARD_GATES.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"R26 hard gates: {total} violations across {len(_GATES)} gates")
    for g, v in results.items():
        status = "OK" if not v else f"FAIL({len(v)})"
        print(f"  [{status}] {g}")
        for item in v[:3]:
            print(f"        - {item}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
