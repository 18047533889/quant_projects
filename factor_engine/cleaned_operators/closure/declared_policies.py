# -*- coding: utf-8 -*-
"""Central MissingPolicy / WindowSemantics declarations (Master Spec P0).

The operators below were fixed in the Round-14 estimator audit batch (or in
adjacent rounds); this module records their authoritative missing-value and
window-counting semantics so the Semantic Closure audit (Part BY) can see the
contracts instead of guessing.  Declarations are keyed by canonical and live in
the closure side-registry — no operator file is mutated here.

The declarations encode three cross-cutting round-14 decisions:

* **stale current-row → NaN** (``CURRENT_REQUIRED`` + ``CONTIGUOUS_FULL_WINDOW``):
  Hankel / multifractal / persistence-topology / structural-level operators must
  not re-serve the previous block's result when the current observation is
  missing (Part C-11/12).
* **contiguous trailing window** for embedding / spectral / recurrence estimators
  (Part D-17): a gap inside the window invalidates the statistic.
* **coverage-gated partial windows** for pair-based estimators (kernel / MI / TE /
  HVG / intrinsic-dim / Lyapunov / rough-vol): partial data is legitimate ONLY
  against a declared min-pair / min-coverage gate (Round-14 support gates).
"""
from __future__ import annotations

from cleaned_operators.closure.axis_contract import declare_axis_contract
from cleaned_operators.closure.missing_policy import (
    MissingPolicy,
    declare_current_required_family,
    declare_missing_policy,
)
from cleaned_operators.closure.window_semantics import (
    WindowSemantics,
    declare_window_semantics,
)

# CURRENT_REQUIRED + CONTIGUOUS_FULL_WINDOW: trailing-embedding estimators whose
# output describes the CURRENT block / state and must fail closed on a current
# missing row (round-14 stale current-row fix).
_CONTIGUOUS_CURRENT = (
    "ts_hankel_effective_rank",
    "ts_hankel_singular_gap",
    "ts_ssa_reconstruction_residual",
    "ts_generalized_hurst_exponent",
    "ts_generalized_hurst_spread_q1_q4",
    "ts_multifractal_curvature",
    "ts_multifractal_spectrum_width",
    "ts_persistence_entropy_h0",
    "ts_persistence_entropy_h1",
    "ts_nearest_structural_level_distance",
    "ts_structural_level_density",
    "ts_structural_level_strength",
)

# CURRENT_REQUIRED + MIN_SUPPORT_WINDOW: current-state estimators that combine a
# contiguous tail with a coverage gate.
_CURRENT_MINSUPPORT = (
    "ts_state_density",
    "ts_multiscale_permutation_entropy_slope",
    "ts_ordinal_irreversibility",
)

# WINDOW_VALID + MIN_SUPPORT_WINDOW: trailing statistics computed on the finite
# subset of the window, gated by min_periods / min-pair coverage.
_MINSUPPORT_VALID = (
    "ts_vol_pvariation_roughness",
    "ts_vol_scaling_break",
    "ts_hartigan_dip",
    "ts_l_kurtosis",
    "ts_l_skewness",
    "ts_expectile",
    "ts_expectile_beta",
    "ts_binned_response_curvature",
    "ts_binned_response_monotonicity",
    "ts_response_slope_asymmetry",
    "ts_hvg_assortativity",
    "ts_hvg_clustering_coefficient",
    "ts_hvg_degree_entropy",
    "ts_hvg_forward_backward_asymmetry",
    "ts_hvg_motif_entropy",
    "ts_spectral_entropy",
    "ts_dominant_cycle_period",
    "ts_return_spectral_entropy",
    "ts_detrended_level_spectral_entropy",
    "ts_quantile_range",
    "ts_robust_zscore",
    "ts_trimmed_mean",
    "ts_expected_shortfall",
    "ts_lower_partial_moment",
    "ts_upper_partial_moment",
    "ts_quantile_skew",
    "ts_quantile_kurtosis",
    "ts_tail_ratio",
    "ts_extreme_cluster_ratio",
    "ts_sample_entropy",
    "ts_permutation_entropy",
    "ts_permutation_transition_entropy",
    "ts_weighted_permutation_entropy",
    "ts_higuchi_fractal_dimension",
    "ts_hurst_dfa",
    "ts_variogram_slope",
    "ts_autocorr_decay_half_life",
    "ts_forbidden_ordinal_pattern_ratio",
    "ts_lempel_ziv_complexity",
    "ts_markov_persistence",
    "ts_markov_spectral_gap",
    "ts_markov_state_entropy",
    "ts_markov_transition_surprisal",
    "ts_markov_stationary_surprisal",
    "ts_markov_mean_first_passage_time",
    "ts_markov_committor",
    "ts_markov_entropy_production",
    "ts_active_information_storage",
    "ts_kramers_moyal_local_stability",
    "ts_km_diffusion_gradient",
    "ts_km_equilibrium_distance",
    "ts_km_quasipotential_depth",
)

# PAIRWISE_VALID + MIN_SUPPORT_WINDOW: pair / two-input estimators that align
# finite pairs (never compress the time axis silently) and gate on pair count.
_PAIRWISE_MINSUPPORT = (
    "ts_residualized_hsic",
    "ts_bds_statistic",
    "ts_rolling_sr_gaussian_mean_shift_score",
    "ts_sr_gaussian_mean_shift_score",
)

# R16-038: PHYSICAL-AXIS estimators (TE / kernel Granger / cross-spectrum /
# intrinsic-dimension / Lyapunov / bicoherence) depend on the physical lag /
# frequency between rows.  Dropping missing rows and re-indexing as continuous
# time would RE-CONNECT lag/frequency across a gap, so they are declared
# BREAK + CONTIGUOUS_FULL_WINDOW — a gap invalidates the estimator instead of
# silently re-pairing.
_PHYSICAL_AXIS_BREAK = (
    "ts_delay_intrinsic_dimension",
    "ts_local_lyapunov_exponent",
    "ts_kernel_granger_score",
    "ts_transfer_entropy",
    "ts_transfer_entropy_peak_excess",
    "ts_transfer_entropy_peak_lag",
    "ts_transfer_entropy_peak_strength",
    "ts_effective_transfer_entropy",
    "ts_cross_spectral_coherence",
    "ts_cross_spectral_phase",
    "ts_bicoherence_max",
    "ts_bicoherence_top_decile_excess",
    "ts_bicoherence_top_decile_mean",
)

# FULL_WINDOW: statistic whose precision depends on the full declared sample —
# a partial window is NEVER emitted as the same statistic (Part D-17).
_FULL_WINDOW = (
    "ts_evt_threshold_stability",
    "ts_pickands_tail_index",
    "ts_glr_mean_shift_score",
    "ts_glr_variance_shift_score",
    "ts_pettitt_change_score",
)

# CONTIGUOUS_FULL_WINDOW (no current-state requirement): recurrence / RQA
# estimators over a contiguous trailing block.
_CONTIGUOUS_FULL = (
    "ts_recurrence_rate",
    "ts_recurrence_diagonal_entropy",
    "ts_recurrence_divergence",
    "ts_recurrence_trapping_time",
    "ts_recurrence_determinism",
    "ts_recurrence_laminarity",
    "ts_recurrence_longest_vertical_length",
    "ts_recurrence_mean_diagonal_length",
    "ts_rqa_determinism_fixed_rr",
    "ts_rqa_laminarity_fixed_rr",
    "ts_first_passage_bias",
    "ts_first_passage_conditional_time",
    "ts_first_passage_hit_probability",
)

# CENSOR + CONTIGUOUS_FULL_WINDOW: marked-event operators where a missing mark
# is UNKNOWN (not zero) and censors the coupling statistic (Part O-70).
_CENSOR_CONTIGUOUS = (
    "event_interval_mark_coupling",
    "event_mark_autocorr",
)

# BREAK + EVENT_COUNT_WINDOW: event-clock histories.
_EVENT_CLOCK = (
    "event_count",
    "event_counting",
    "event_allan_factor",
    "event_allan_log_mean",
    "event_allan_scaling_slope",
)


def _declare_safely(canon: str, *, policy: MissingPolicy | None = None,
                    window: WindowSemantics | None = None,
                    family: str | None = None) -> None:
    """R16-037: conflict-DETECTING declaration.

    A later-loaded central declaration must NEVER silently overwrite a LOCAL
    operator-file contract with a different value.  If a non-matching
    declaration already exists, this raises (load/release fails loud); a
    matching existing declaration is an idempotent no-op; an absent one is
    declared fresh.
    """
    from cleaned_operators.closure.missing_policy import missing_policy_for
    from cleaned_operators.closure.window_semantics import window_semantics_for

    if policy is not None:
        existing = missing_policy_for(canon)
        if existing is not None and existing is not policy:
            raise RuntimeError(
                f"R16-037 declared-policy conflict for {canon}: operator/local "
                f"declares missing_policy={existing.value}, central side-registry "
                f"wants {policy.value}"
            )
        declare_missing_policy(canon, policy, replace=True)
    if window is not None:
        existing = window_semantics_for(canon)
        if existing is not None and existing is not window:
            raise RuntimeError(
                f"R16-037 declared-policy conflict for {canon}: operator/local "
                f"declares window_semantics={existing.value}, central side-registry "
                f"wants {window.value}"
            )
        declare_window_semantics(canon, window, replace=True)
    if family is not None:
        declare_current_required_family(canon, family)


def declare_all() -> None:
    """Idempotent batch declaration (safe to call from a test / gate).

    R16-037: every declaration is conflict-DETECTED — a local operator-file
    contract with a DIFFERENT value fails loud instead of being silently
    overridden by this side-registry.
    """
    for canon in _CONTIGUOUS_CURRENT:
        _declare_safely(canon, policy=MissingPolicy.CURRENT_REQUIRED,
                        window=WindowSemantics.CONTIGUOUS_FULL_WINDOW, family="current_state")
    for canon in _CURRENT_MINSUPPORT:
        _declare_safely(canon, policy=MissingPolicy.CURRENT_REQUIRED,
                        window=WindowSemantics.MIN_SUPPORT_WINDOW, family="current_state")
    for canon in _MINSUPPORT_VALID:
        _declare_safely(canon, policy=MissingPolicy.WINDOW_VALID,
                        window=WindowSemantics.MIN_SUPPORT_WINDOW)
    for canon in _PAIRWISE_MINSUPPORT:
        _declare_safely(canon, policy=MissingPolicy.PAIRWISE_VALID,
                        window=WindowSemantics.MIN_SUPPORT_WINDOW)
    for canon in _PHYSICAL_AXIS_BREAK:
        # R16-038: physical lag/frequency must not be re-connected across a gap.
        _declare_safely(canon, policy=MissingPolicy.BREAK,
                        window=WindowSemantics.CONTIGUOUS_FULL_WINDOW)
    for canon in _FULL_WINDOW:
        _declare_safely(canon, policy=MissingPolicy.WINDOW_VALID,
                        window=WindowSemantics.FULL_WINDOW)
    for canon in _CONTIGUOUS_FULL:
        _declare_safely(canon, policy=MissingPolicy.BREAK,
                        window=WindowSemantics.CONTIGUOUS_FULL_WINDOW)
    for canon in _CENSOR_CONTIGUOUS:
        _declare_safely(canon, policy=MissingPolicy.CENSOR,
                        window=WindowSemantics.CONTIGUOUS_FULL_WINDOW)
    for canon in _EVENT_CLOCK:
        _declare_safely(canon, policy=MissingPolicy.BREAK,
                        window=WindowSemantics.EVENT_COUNT_WINDOW)

    # SameAxis contracts for the genuinely multi-input estimators
    # (Part BM-259): the two panel inputs must share the exact row index AND the
    # exact instrument columns before any pair alignment.
    for canon in (
        "ts_kernel_granger_score",
        "ts_residualized_hsic",
        "ts_transfer_entropy",
        "ts_effective_transfer_entropy",
        "ts_cross_spectral_coherence",
        "ts_cross_spectral_phase",
        "event_interval_mark_coupling",
    ):
        declare_axis_contract(canon, ("same_index", "same_columns", "unique_index", "unique_columns"))


declare_all()
