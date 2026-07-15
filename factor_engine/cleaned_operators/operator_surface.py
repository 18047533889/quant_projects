# -*- coding: utf-8 -*-
"""Public DSL surface policy for daily factor generation.

The runtime registry remains the compatibility and execution index.  This
module controls which names may be used in newly submitted DSL formulas.
"""
from __future__ import annotations

from typing import Literal

OperatorSurface = Literal["daily", "research", "unsafe", "legacy", "all"]

# Operators that are non-causal, backward-looking from the future, or random.
# They remain importable only from ``research_operators`` with explicit unsafe
# opt-in, never from the normal factor DSL.
UNSAFE_CANONICALS: frozenset[str] = frozenset(
    {
        "Lead",
        "next",
        "bfill",
        "fillna_interpolate",
        "shuffle",
        "sample",
        "rand_exp",
        "rand_lognormal",
        "rand_normal",
        "rand_poisson",
        "rand_uniform",
    }
)

# Useful research/statistical utilities, but not operators whose normal output
# is a daily scalar factor panel.  They are exposed by ``research_operators``.
RESEARCH_ONLY_CANONICALS: frozenset[str] = frozenset(
    {
        # Statistical tests and diagnostics.
        "bartlett_test",
        "chi_square_test",
        "corr_test",
        "durbin_watson_test",
        "granger_causality",
        "jarque_bera_test",
        "kendall_corr_test",
        "kpss_test",
        "ks_test",
        "levene_test",
        "lilliefors_test",
        "spearman_corr_test",
        "stationarity_test",
        "ttest_one_sample",
        "ttest_paired",
        "ttest_two_samples",
        # Probability distributions.
        "cdf_chi2",
        "cdf_f",
        "cdf_normal",
        "cdf_t",
        "pdf_chi2",
        "pdf_f",
        "pdf_normal",
        "pdf_t",
        "quantile_normal",
        "quantile_t",
        # Complex numbers and linear algebra.
        "complex",
        "conj",
        "real",
        "imag",
        "polar",
        "phase",
        "eig",
        "svd",
        "pca",
        "lu_decompose",
        "qr_decompose",
        "mat_add",
        "mat_subtract",
        "mat_multiply",
        "mat_transpose",
        "mat_inverse",
        "mat_determinant",
        "mat_rank",
        "norm",
        "norm_l1",
        "norm_linf",
        # Signal processing.  Boundary and causality policy must be explicit.
        "fft",
        "ifft",
        "wavelet",
        "wavelet_denoise",
        "convolve",
        "correlate",
        "decimate",
        "filter_lowpass",
        "filter_highpass",
        "filter_bandpass",
        "filter_notch",
        "interpolate",
        "unwrap",
    }
)

# Old convenience canonicals retained for direct registry compatibility.  New
# formulas use the canonical on the right side of ``_dedupe.py`` instead.
LEGACY_ONLY_CANONICALS: frozenset[str] = frozenset(
    {
        "inv",
        "reciprocal",
        "fmax",
        "fmin",
        "sqr",
        "cube",
        "cumulative_max",
        "cumulative_mean",
        "cumulative_min",
    }
)

# Alias names that resolve to an allowed canonical after deduplication but must
# still be hidden from the normal daily DSL.
HIDDEN_DAILY_NAMES: frozenset[str] = frozenset(
    {
        "inv",
        "reciprocal",
        "fmax",
        "fmin",
        "sqr",
        "cube",
        "cumulative_max",
        "cumulative_mean",
        "cumulative_min",
    }
)


def classify_canonical(canonical: str) -> str:
    """Return the single public surface classification for a canonical name."""
    if canonical in UNSAFE_CANONICALS:
        return "unsafe"
    if canonical in RESEARCH_ONLY_CANONICALS:
        return "research"
    if canonical in LEGACY_ONLY_CANONICALS:
        return "legacy"
    return "daily"


def is_dsl_name_allowed(name: str, canonical: str, *, surface: OperatorSurface = "daily") -> bool:
    """Return whether a registry name is visible on the requested DSL surface."""
    if surface == "all":
        return True
    category = classify_canonical(canonical)
    if surface == "daily":
        return category == "daily" and name not in HIDDEN_DAILY_NAMES
    if surface == "research":
        return category == "research"
    if surface == "unsafe":
        return category == "unsafe"
    if surface == "legacy":
        return category == "legacy" or name in HIDDEN_DAILY_NAMES
    raise ValueError(f"unknown operator surface: {surface!r}")


def surface_summary(canonicals: list[str]) -> dict[str, int]:
    """Count canonical operators by surface classification."""
    out = {"daily": 0, "research": 0, "unsafe": 0, "legacy": 0}
    for canonical in canonicals:
        out[classify_canonical(canonical)] += 1
    return out
