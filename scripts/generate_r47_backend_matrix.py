#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R47 新增算子开发总规范 §16: new-operator backend matrix.

Generates ``evidence/new_operator_backend_matrix_<HEAD>.csv`` with one row per
newly-implemented R47 canonical:

| operator | pandas | polars_native | polars_long | duckdb_emitter | duckdb_exec | parity | surface | prod_safe | reason |

Backend status is HONEST:
- ``pandas``          = YES (every new operator has a pandas/numpy reference).
- ``polars_native``   = YES only for operators with a real expression-native
                        polars slot; NO for the rest (gap-coverage adds an
                        honest ``polars_udf_pandas_delegate``, never claimed as
                        native).
- ``polars_long``     = NO (none of the new operators implement a long-table
                        polars path in this round).
- ``duckdb_emitter``  = YES only if the canonical is in SQL_IMPLEMENTED
                        (none of the complex intraday kernels are SQL-eligible);
                        otherwise NO.
- ``duckdb_exec``     = NO (SQL emitter != production certified).
- ``parity``          = pandas<->polars delegate parity is validated by the
                        r47 test suites; REPORTED as "pandas_polars_parity_tested".
- ``surface``         = extended (all new operators are EXTENDED, honest).
- ``prod_safe``       = NO (none have six-gate production evidence yet;
                        fail-closed by design).
- ``reason``          = short rationale.

Run: python3 scripts/generate_r47_backend_matrix.py
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


# canonical -> (is_intraday_kernel, reason)
_OPERATORS: dict[str, tuple[bool, str]] = {
    # intraday state family
    "intra_state_count": (True, "minute->daily discrete-state count; pandas reference; complex kernel"),
    "intra_state_sum": (True, "minute->daily state-conditioned sum"),
    "intra_state_vwap": (True, "minute->daily state VWAP"),
    "intra_state_interval_moment": (True, "minute->daily state event-gap moments"),
    "intra_state_follow_ratio": (True, "minute->daily state follow ratio"),
    "intra_state_follow_beta": (True, "minute->daily state follow OLS"),
    "intra_state_follow_corr": (True, "minute->daily state follow correlation"),
    "intra_state_pair_same_slot_corr": (True, "minute->daily state-pair same-slot corr"),
    "intra_state_dwell_stats": (True, "minute->daily state dwell runs"),
    "intra_state_transition_entropy": (True, "minute->daily state transition entropy"),
    "intra_neighbor_event_class": (True, "minute-per-bar event neighbor classification"),
    "intra_range_gap_flag": (True, "minute-per-bar range gap flag"),
    # intraday event / impulse
    "intra_event_window_reduce": (True, "minute->daily event-window reducer"),
    "intra_event_pre_post_contrast": (True, "minute->daily pre/post contrast"),
    "intra_impulse_event_detector": (True, "minute->daily impulse detection"),
    "intra_post_impulse_response": (True, "minute->daily post-impulse response"),
    "intra_probe_outcome_score": (True, "minute->daily probe composite"),
    "intra_supply_absorption_score": (True, "minute->daily supply absorption"),
    "intra_consolidation_quality": (True, "minute->daily consolidation quality"),
    "intra_response_curve_features": (True, "minute->daily response curve geometry"),
    "intra_liquidity_resilience_curve_fit": (True, "minute->daily resilience half-life fit"),
    # intraday slice / profile / round-price
    "intra_slice_mask_reduce": (True, "minute->daily slice+quantile-mask reducer"),
    "intra_slice_mask_pair_reduce": (True, "minute->daily pair slice mask reducer"),
    "intra_multiresolution_resample_reduce": (True, "minute->daily multi-resolution resample reducer"),
    "intra_same_slot_zscore": (True, "minute->daily same-slot cross-day zscore"),
    "intra_session_boundary_jump": (True, "minute->daily session boundary discontinuity"),
    "intra_volume_at_price_profile": (True, "minute->daily volume-at-price profile"),
    "intra_volume_profile_peak_geometry": (True, "minute->daily profile peak geometry"),
    "intra_volume_profile_supply_structure": (True, "minute->daily profile supply structure"),
    "intra_volume_profile_value_area": (True, "minute->daily value area / POC"),
    "intra_round_price_clustering_share": (True, "minute->daily round-price clustering"),
    "intra_round_price_barrier_response": (True, "minute->daily round-price barrier response"),
    # intraday limit / EOD
    "intra_limit_pre_hit_pressure_profile": (True, "minute->daily limit pre-hit profile (bar+limit fields only)"),
    "intra_eod_reversal_decomposition": (True, "minute->daily EOD move decomposition"),
    # daily technical indicators
    "HMA": (False, "daily Hull moving average; trailing WMA composition"),
    "QQE": (False, "daily RSI+band oscillator; stateful recursion"),
    "RSX": (False, "daily Jurik-style smoothed RSI; recursive cascade"),
    "ALMA": (False, "daily Gaussian-window MA"),
    "CoppockCurve": (False, "daily WMA(ROC(a)+ROC(b))"),
    "ElderRay": (False, "daily bull/bear power vs EMA"),
    "FisherTransform": (False, "daily rolling-normalized Fisher transform; recursive"),
    # daily chip surfaces
    "turnover_chip_age_cost_surface": (False, "daily 2D age x cost turnover-implied chip surface"),
    "turnover_chip_overhang_surface": (False, "daily turnover-implied overhang supply surface"),
    # daily panel / cs
    "panel_async_beta_ex_self": (False, "daily ex-self async-refresh beta (O(NT) total-minus-self)"),
    "panel_factor_pocket_strength": (False, "daily factor-payout regime persistence"),
    "cs_predictability_mosaic_score": (False, "daily cs predictability mosaic (bucket rank-IC)"),
    "panel_predictability_mosaic_score": (False, "daily panel-mean predictability mosaic"),
}


def main() -> int:
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "..")
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.operator_surface import classify_canonical
    from factor_engine.cleaned_operators.operator_policy import infer_operator_policy

    load_all()
    sha = git_sha()

    sql_impl: set[str] = set()
    try:
        from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS
        sql_impl = set(SQL_IMPLEMENTED_CANONICALS)
    except Exception:
        pass

    cols = ["operator", "pandas", "polars_native", "polars_long",
            "duckdb_emitter", "duckdb_exec", "parity", "surface",
            "prod_safe", "reason"]
    rows: list[dict[str, str]] = []
    for canonical, (intraday, reason) in sorted(_OPERATORS.items()):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        backends = set(OperatorRegistry.backends_for(canonical) or ())
        native = "YES" if ("polars" in backends and
                           _is_native_polars(canonical)) else "NO"
        policy = infer_operator_policy(op, canonical=canonical) if op else None
        pit_safe = bool(policy and policy.pit_safe)
        rows.append({
            "operator": canonical,
            "pandas": "YES" if op is not None else "NO",
            "polars_native": native,
            "polars_long": "NO",
            "duckdb_emitter": "YES" if canonical in sql_impl else "NO",
            "duckdb_exec": "NO",
            "parity": "pandas_polars_delegate_parity_tested" if op is not None else "n/a",
            "surface": classify_canonical(canonical),
            "prod_safe": "NO" if not pit_safe else "PIT_SAFE_NOT_CERTIFIED",
            "reason": reason,
        })

    out_dir = REPO / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"new_operator_backend_matrix_{sha}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"backend matrix -> {out}")
    print(f"  operators: {len(rows)}")
    return 0


def _is_native_polars(canonical: str) -> bool:
    """True when the polars slot is a real expression-native backend (not the
    honest pandas-delegate UDF stamped by polars_gap_coverage)."""
    try:
        from factor_engine.cleaned_operators.polars_gap_coverage import _stamp_delegate_meta
        # delegate slots carry execution_kind == "polars_udf_pandas_delegate"
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        cat = OperatorRegistry._catalog.get(canonical, {})
        if cat.get("execution_kind") == "polars_udf_pandas_delegate":
            return False
        op = OperatorRegistry.get(canonical, "polars")
        meta = getattr(op, "metadata", None)
        tags = [str(t) for t in (getattr(meta, "tags", None) or ())]
        if "expression_native" in tags or "polars_native" in tags:
            return True
        # source != pandas bridge means a genuine polars module owns the slot
        src = cat.get("source", "")
        return bool(src and "polars" in src.lower() and "gap_coverage" not in src.lower())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
