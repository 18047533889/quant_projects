# -*- coding: utf-8 -*-
"""R26-139..143: machine-readable R26 artifacts.

Generates:
* R26_OPERATOR_CORRECTNESS_MATRIX.csv/json/md  — per-canonical status columns;
* R26_INTRADAY_PHYSICAL_CLOCK_MATRIX.csv        — minute canonical physical-clock
  contract columns;
* R26_ESTIMATOR_DEFINITION_AUDIT.json           — estimator definition coherence;
* R26_STATE_HISTORY_AUDIT.json                  — bounded / unbounded state memory;
* R26_PATTERN_TRISTATE_AUDIT.csv                — pattern tri-state outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def main() -> int:
    # ---- R26_OPERATOR_CORRECTNESS_MATRIX ----
    rows = [
        {"canonical": "ts_chatterjee_xi", "family": "dependence", "math_status": "ok",
         "tie_status": "target_independent", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "y-independent stable tie ordering (R26-005..007)", "evidence": "test_dependence_ties"},
        {"canonical": "cs_rank_copula_mi", "family": "information", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "Jeffreys-only estimator, Miller-Madow removed (R26-009..011)", "evidence": "test_dependence_ties"},
        {"canonical": "intra_realized_variance", "family": "intraday", "math_status": "ok",
         "tie_status": "ok", "missing_status": "explicit_slots", "physical_clock_status": "official_grid",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "SessionPanel grid alignment + gated log returns (R26-013..020)", "evidence": "test_intraday_physical_clock"},
        {"canonical": "intra_path_efficiency", "family": "intraday", "math_status": "ok",
         "tie_status": "ok", "missing_status": "break_on_gap", "physical_clock_status": "official_grid",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "contiguous complete run, no gap bridging (R26-028/029)", "evidence": "test_intraday_physical_clock"},
        {"canonical": "intra_high_time", "family": "intraday", "math_status": "ok",
         "tie_status": "latest_extreme", "missing_status": "explicit_slots", "physical_clock_status": "slot_ordinal",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "official slot ordinal, not observed row index (R26-030)", "evidence": "test_intraday_physical_clock"},
        {"canonical": "intra_lunch_gap_return", "family": "intraday", "math_status": "ok",
         "tie_status": "ok", "missing_status": "exact_endpoints", "physical_clock_status": "official_grid",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "EndpointPolicy.EXACT 11:30/13:01 (R26-027)", "evidence": "test_intraday_physical_clock"},
        {"canonical": "intra_limit_first_hit_time", "family": "intraday", "math_status": "ok",
         "tie_status": "ok", "missing_status": "tri_state", "physical_clock_status": "slot_ordinal",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "per-side OHLC + slot ordinal + tri-state limit (R26-031..034)", "evidence": "test_intraday_physical_clock"},
        {"canonical": "session_event_recovery_score", "family": "intraday", "math_status": "ok",
         "tie_status": "ok", "missing_status": "break_censor", "physical_clock_status": "official_grid",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "min_events=3 + EventMissingPolicy + session-close proof (R26-047..051)", "evidence": "test_r11_round3_intraday"},
        {"canonical": "ts_hill_tail_index", "family": "tail", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "positive-level lower tail unsupported; loss-magnitude Hill (R26-060..062)", "evidence": "test_tail_estimators"},
        {"canonical": "ts_roll_effective_spread", "family": "liquidity", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "positive covariance -> NaN, positive-price gate (R26-069..071)", "evidence": "test_tail_estimators"},
        {"canonical": "ts_glr_mean_shift_score", "family": "change_point", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "recentered", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "recentered prefix moments (R26-065..068)", "evidence": "test_r11_round3_glr_spread"},
        {"canonical": "AROON_up", "family": "technical", "math_status": "ok",
         "tie_status": "latest_extreme", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "latest-extreme tie policy (R26-076..078)", "evidence": "test_technical_ties_patterns"},
        {"canonical": "ts_last_pivot_high", "family": "technical", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "bounded_lookback", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "pivot_lookback_bars bounds the state machine (R26-079..081)", "evidence": "test_technical_extensions"},
        {"canonical": "cdl_spinning_top", "family": "candle", "math_status": "ok",
         "tie_status": "ok", "missing_status": "tri_state", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "neutral-direction event -> NaN, never +1 (R26-085/086)", "evidence": "test_technical_ties_patterns"},
        {"canonical": "event_interval_memory", "family": "event_interval", "math_status": "ok",
         "tie_status": "ok", "missing_status": "break", "physical_clock_status": "bar_window",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "BAR window semantics + doc==kernel min-support (R26-088..091)", "evidence": "test_event_interval_clock"},
        {"canonical": "ts_multiscale_permutation_entropy_slope", "family": "state_geometry", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "feasible default window=256 (R26-093..095)", "evidence": "audit_r26_default_feasibility"},
        {"canonical": "composition_entropy", "family": "composition", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "part_whole_only",
         "fix": "invalid financial_statement schema removed (R26-096..102)", "evidence": "test_composition_schema"},
        {"canonical": "ts_effective_transfer_entropy", "family": "information", "math_status": "ok",
         "tie_status": "ok", "missing_status": "fixed_mask", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "fixed NaN mask, finite-only surrogate rearrangement (R26-103/104)", "evidence": "test_r11_te_peak_binning"},
        {"canonical": "ts_quantile_crossing_spectral_concentration", "family": "quantile", "math_status": "ok",
         "tie_status": "ok", "missing_status": "contiguous_run_checked", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "post-truncation length re-check (R26-108/109)", "evidence": "test_market_language"},
        {"canonical": "ts_extremogram", "family": "quantile", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "signed_probability_difference", "economic_semantics_status": "ok",
         "fix": "signed probability difference unit (R26-110)", "evidence": "test_market_language"},
        {"canonical": "ts_dc_event_rate", "family": "directional_change", "math_status": "ok",
         "tie_status": "ok", "missing_status": "clock_observable_denom", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "ok", "default_feasibility_status": "PASS",
         "unit_semantics_status": "scale_mode_split", "economic_semantics_status": "ok",
         "fix": "scale_mode absolute/relative + clock-observable denominator (R26-113..117)", "evidence": "test_r11_round2_directional_change"},
        {"canonical": "ts_dmd_energy_concentration", "family": "dmd", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "log_domain", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "log-domain rho/energy, fail-closed on all-zero (R26-122..125)", "evidence": "test_dmd_log_numerics"},
        {"canonical": "intraday_impact_decay_rate", "family": "intraday", "math_status": "ok",
         "tie_status": "ok", "missing_status": "ok", "physical_clock_status": "n/a",
         "state_history_status": "ok", "numeric_stability_status": "censor_floor", "default_feasibility_status": "PASS",
         "unit_semantics_status": "ok", "economic_semantics_status": "ok",
         "fix": "detection-floor censor keeps recovery evidence (R26-057..059)", "evidence": "test_r26_impact_decay"},
    ]
    _write_matrix(rows)

    # ---- R26_INTRADAY_PHYSICAL_CLOCK_MATRIX ----
    intra = [
        {"canonical": "intra_segment_return", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "explicit_slot", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "exact", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "unique_valid_slots", "time_output_uses_slot_id": "n/a", "PASS/FAIL": "PASS"},
        {"canonical": "intra_realized_variance", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "explicit_slot", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "n/a", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "unique_valid_slots", "time_output_uses_slot_id": "n/a", "PASS/FAIL": "PASS"},
        {"canonical": "intra_path_efficiency", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "break_run", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "n/a", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "contiguous_run", "time_output_uses_slot_id": "n/a", "PASS/FAIL": "PASS"},
        {"canonical": "intra_high_time", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "explicit_slot", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "n/a", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "n/a", "time_output_uses_slot_id": "yes", "PASS/FAIL": "PASS"},
        {"canonical": "intra_lunch_gap_return", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "exact_endpoint", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "exact", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "n/a", "time_output_uses_slot_id": "n/a", "PASS/FAIL": "PASS"},
        {"canonical": "intra_limit_first_hit_time", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "tri_state", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "n/a", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "n/a", "time_output_uses_slot_id": "yes", "PASS/FAIL": "PASS"},
        {"canonical": "session_event_recovery_score", "source_frequency": "minute", "declared_bar_width": "1min",
         "official_grid_required": "yes", "missing_timestamp_policy": "break_censor", "duplicate_policy": "DQ_fail",
         "endpoint_policy": "n/a", "session_tz_policy": "session_local", "early_close_policy": "calendar",
         "coverage_policy": "session_close_required", "time_output_uses_slot_id": "yes", "PASS/FAIL": "PASS"},
    ]
    _write_intraday(intra)

    # ---- R26_ESTIMATOR_DEFINITION_AUDIT ----
    est = {
        "ts_chatterjee_xi": {"definition": "tie-aware Chatterjee, y-independent tie order",
                             "status": "coherent", "r26_fix": "target-independent tie handling"},
        "cs_rank_copula_mi": {"definition": "Jeffreys-smoothed plug-in MI (single estimator)",
                              "status": "coherent", "r26_fix": "removed Miller-Madow double correction"},
        "ts_hill_tail_index": {"definition": "Hill on exceedance magnitudes; positive-level lower unsupported",
                               "status": "coherent", "r26_fix": "lower-tail domain gate"},
        "ts_roll_effective_spread": {"definition": "classic Roll 2*sqrt(-Cov); Cov>=0 undefined",
                                     "status": "coherent", "r26_fix": "positive covariance -> NaN"},
        "ts_glr_mean_shift_score": {"definition": "change-point GLR with recentered SSE",
                                    "status": "coherent", "r26_fix": "recentered prefix moments"},
        "ts_dmd_energy_concentration": {"definition": "log-domain mode energy; no overflow/underflow",
                                        "status": "coherent", "r26_fix": "log_rho/log_b2"},
        "ts_quantile_regression_beta": {"definition": "conditional quantile slope Q_y(q|x)",
                                        "status": "coherent", "r26_fix": "doc corrected (not tail-state OLS)"},
        "ts_extremogram": {"definition": "signed probability difference",
                           "status": "coherent", "r26_fix": "unit corrected"},
        "intraday_quantile_curve_pca_residual": {"definition": "top-k PCA projection residual; rank>=k",
                                                 "status": "coherent", "r26_fix": "rank gate k not k+1"},
        "ts_effective_transfer_entropy": {"definition": "TE - E[surrogate] with fixed NaN mask",
                                          "status": "coherent", "r26_fix": "finite-only surrogate rearrangement"},
    }
    (DOCS / "R26_ESTIMATOR_DEFINITION_AUDIT.json").write_text(json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- R26_STATE_HISTORY_AUDIT ----
    state = {
        "ts_last_pivot_high": {"actual_state_memory": "bounded by pivot_lookback_bars",
                               "declared_history": "left + right + pivot_lookback_bars",
                               "bounded": True, "checkpoint": False, "full_replay": False,
                               "max_age": "pivot_lookback_bars", "full-vs-warmup parity": "yes (bounded)"},
        "ts_pivot_high_age": {"actual_state_memory": "bounded by pivot_lookback_bars",
                              "declared_history": "left + right + pivot_lookback_bars",
                              "bounded": True, "checkpoint": False, "full_replay": False,
                              "max_age": "pivot_lookback_bars", "full-vs-warmup parity": "yes (bounded)"},
        "session_event_recovery_score": {"actual_state_memory": "per-session grid + refractory",
                                         "declared_history": "session grid", "bounded": True,
                                         "checkpoint": False, "full_replay": False,
                                         "max_age": "session", "full-vs-warmup parity": "yes"},
        "intra_limit_reopen_count": {"actual_state_memory": "per-session grid state machine",
                                     "declared_history": "session grid", "bounded": True,
                                     "checkpoint": False, "full_replay": False,
                                     "max_age": "session", "full-vs-warmup parity": "yes"},
    }
    (DOCS / "R26_STATE_HISTORY_AUDIT.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- R26_PATTERN_TRISTATE_AUDIT ----
    patterns = [
        {"canonical": "cdl_doji", "nan_input": "NaN", "confirmed_no_pattern": "0", "tri_state": "yes"},
        {"canonical": "cdl_spinning_top", "nan_input": "NaN", "neutral_event": "NaN (not +1)", "tri_state": "yes"},
        {"canonical": "cdl_outside_bar", "nan_input": "NaN", "neutral_event": "NaN (not 0)", "tri_state": "yes"},
        {"canonical": "cdl_engulfing", "nan_input": "NaN", "two_day_requires_both": "yes", "tri_state": "yes"},
        {"canonical": "cdl_inside_bar", "nan_input": "NaN", "two_day_requires_both": "yes", "tri_state": "yes"},
        {"canonical": "intra_limit_duration", "nan_input": "tri_state", "unknown_limit": "NaN", "tri_state": "yes"},
    ]
    _write_patterns(patterns)

    # ---- R26_OPERATOR_CORRECTNESS_MATRIX.md ----
    _write_md(rows)
    return 0


def _write_matrix(rows):
    fields = list(rows[0].keys())
    with open(DOCS / "R26_OPERATOR_CORRECTNESS_MATRIX.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (DOCS / "R26_OPERATOR_CORRECTNESS_MATRIX.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_intraday(rows):
    fields = list(rows[0].keys())
    with open(DOCS / "R26_INTRADAY_PHYSICAL_CLOCK_MATRIX.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_patterns(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(DOCS / "R26_PATTERN_TRISTATE_AUDIT.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_md(rows):
    lines = ["# R26 Operator Correctness Matrix\n",
             "| canonical | family | tie | missing | clock | state | numeric | default | unit | econ | fix |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['canonical']} | {r['family']} | {r['tie_status']} | "
                     f"{r['missing_status']} | {r['physical_clock_status']} | {r['state_history_status']} | "
                     f"{r['numeric_stability_status']} | {r['default_feasibility_status']} | "
                     f"{r['unit_semantics_status']} | {r['economic_semantics_status']} | {r['fix']} |")
    (DOCS / "R26_OPERATOR_CORRECTNESS_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
