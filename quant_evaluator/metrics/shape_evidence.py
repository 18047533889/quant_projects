"""Shape evidence metrics (plan §13.4 / §14.3 / §14.4, R61-FI-021).

This module adds the shape metrics the plan requires that did not already
exist as registered ids (the pre-existing family in
:mod:`quant_evaluator.metrics.quantile_shape` — monotonicity / curvature /
tail_asymmetry / adjacent_spread / extreme_cliff / top+bottom cliff — is
REUSED, never re-implemented here).

All kernels are pure CPU reference implementations over the SAME input all
shape metrics consume: a `QuantileReturnArtifact`'s ``values``, an
``(n_quantiles, F)`` matrix of time-averaged forward return per quantile
bucket per factor.  Quantile 0 = LOWEST factor-value group, quantile
``n-1`` = HIGHEST (QE2-P0-001).  Each function returns a per-factor scalar
array of shape ``(F,)`` and returns ``NaN`` (never 0) when the profile does
not carry enough finite quantile returns — missing evidence maps to the
existing ``EvidenceStatus`` vocabulary at the caller, never a fabricated 0.

R61 plan §13.4 / §14.4 semantic contracts honoured:

- U-shape detection may NOT be ``RankIC ≈ 0`` (§14.3).  ``u_shape_score``
  combines the plan's ingredients: middle bins underperform both tails
  (U template fit), positive curvature, and a requirement that the U
  template fit EXCEEDS the monotone template fit.
- Tail metrics preserve orientation metadata: the sign convention is made
  explicit per metric (``top_tail_slope`` is the top segment of the curve
  *as drawn from quantile 0 to n-1*, which is a positive slope for a
  positively inclined factor), and direction metadata is carried on the
  registry ``MetricSpec`` (plan §14.4: do not assume all factors positive
  until orientation-normalized).
- A robust tail cliff variant ``Q_K - mean(Q_(K-3..K-1))`` (plan §14.4)
  is added ONLY where the existing one-bin ``top_quantile_cliff`` /
  ``bottom_quantile_cliff`` semantics are different, and is named to avoid
  collision (``top_quantile_cliff_robust`` / ``bottom_quantile_cliff_robust``).

Adaptive bin resolution (plan §14.1) lives in
:mod:`quant_evaluator.contracts.adaptive_bins_policy`; the
``adaptive_quantile_count`` metric here computes it from the artifact's
provenance (the number of quantiles actually used to build the profile) and
records that actual count — the fallback selection itself is provenance
input (the artifact builder chooses the count via the policy), this metric
REPORTS what was used so consumers never assume 20.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_u_shape_score",
    "compute_inverted_u_score",
    "compute_adaptive_quantile_count",
    "compute_top_tail_slope",
    "compute_bottom_tail_slope",
    "compute_tail_vs_middle_contrast",
    "compute_left_right_asymmetry",
    "compute_linear_trend_score",
    "compute_shape_stability",
    "compute_shape_regime_stability",
    "compute_shape_bootstrap_confidence",
    "compute_top_quantile_cliff_robust",
    "compute_bottom_quantile_cliff_robust",
    "SHAPE_DIRECTION_FORWARD",
    "SHAPE_DIRECTION_REVERSED",
]


# Orientation metadata (plan §14.4).  The NEW shape family does not assume
# a factor's orientation: a metric's ``MetricSpec.direction`` names which
# value of the metric is better, and these constants let downstream
# consumers name the SIGN convention of curve segments.
SHAPE_DIRECTION_FORWARD = "forward"   # quantile 0=lowest value -> n-1=highest value
SHAPE_DIRECTION_REVERSED = "reversed"  # quantile 0=highest value -> n-1=lowest value


def _as_matrix(qr: np.ndarray) -> np.ndarray:
    """Coerce to a float64 (n_quantiles, F) matrix."""
    m = np.asarray(qr, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    return m


def _finite_columns_exact(
    m: np.ndarray,
    required: int,
) -> np.ndarray:
    """Boolean (F,) mask of columns with EXACTLY ``required`` finite returns."""
    return np.sum(np.isfinite(m), axis=0) == required


def _full_columns(m: np.ndarray, required: int) -> np.ndarray:
    """Boolean (F,) mask of columns where the first ``required`` quantiles are all finite."""
    return np.all(np.isfinite(m[:required]), axis=0)


# ---------------------------------------------------------------------------
# U / inverted-U (plan §14.3)
# ---------------------------------------------------------------------------


def _template_fit_x(nq: int) -> np.ndarray:
    """Normalised quantile coordinate array for template regression, (nq,)."""
    if nq <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, nq)


def _fit_variance_explained(y: np.ndarray, template: np.ndarray) -> float:
    """R^2 of the scalar offset+scale OLS fit ``a + b*template`` against ``y``."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    n = y.shape[0]
    if n < 3:
        return np.nan
    X = np.stack([np.ones(n), t], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else np.nan
    return 1.0 - ss_res / ss_tot


def compute_u_shape_score(qr: np.ndarray) -> np.ndarray:
    """U-shape score per factor, (F,).

    Plan §14.3 — NOT ``RankIC ≈ 0``.  Combines three ingredients into
    [0, 1]:

    1. *Middle-underperforms-tails* (U template fit): fit ``a + b*U(x)``
       where ``U(x) = (x - c)^2`` (c = centre) to the quantile-return
       profile and take positive-bounded R^2.
    2. *Curvature*: fraction of the interior second differences that are
       positive (convex).  A U profile is convex; an inverted-U is concave.
    3. *U-template-beats-monotone-template*: the U R^2 must EXCEED the
       monotone-linear R^2 (plan §14.3 ingredient "U template fit >
       monotonic template fit").  A factor with strong monotone slope is
       NOT a U even if its U-fit is high.

    NaN when the profile has fewer than 4 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    x = _template_fit_x(nq)
    ut = (x - 0.5) ** 2
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        if np.sum(finite) < 4:
            continue
        y = col[finite]
        xu = ut[finite]
        xm = x[finite]
        if np.ptp(y) == 0.0:
            out[f] = 0.0
            continue
        r2_u = _fit_variance_explained(y, xu)
        r2_m = _fit_variance_explained(y, xm)
        if not (np.isfinite(r2_u) and np.isfinite(r2_m)):
            continue
        # Convex curvature fraction over interior positions.
        interior = finite[1:-1] & finite[:-2] & finite[2:]
        if not np.any(interior):
            continue
        d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
        curv = float(np.mean(d2 > 0.0))
        # Ingredient 3: U template must beat the monotone template.
        if r2_u <= r2_m:
            out[f] = 0.0
            continue
        # A hill profile is concave; without convex curvature the U label
        # must not win even though the sign-flipped template fits.
        if curv < 0.5:
            out[f] = 0.0
            continue
        u_fit = max(r2_u, 0.0)
        out[f] = float(0.5 * u_fit + 0.3 * curv + 0.2 * (u_fit - r2_m))
    return out


def compute_inverted_u_score(qr: np.ndarray) -> np.ndarray:
    """Inverted-U score per factor, (F,).

    Mirrors :func:`compute_u_shape_score` with the opposite curvature
    (concave: interior second differences NEGATIVE) and the opposite
    template (``-U(x)`` inverted-U template, concave).  NaN when fewer than
    4 finite quantile returns.

    NOTE on template fitting with a free scale: fitting ``a + b*T`` to ``y``
    with ``T = -(x-0.5)^2`` also fits ``-T`` perfectly when ``b`` is free
    (``b`` flips sign), so an inverted-U TEMPLATE fit alone cannot separate
    U from inverted-U — the concave-curvature term is the separator.  A pure
    U profile therefore reports a non-zero inverted-U score only from the
    template term; the concave fraction keeps it below the U score.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    x = _template_fit_x(nq)
    it = -(x - 0.5) ** 2
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        if np.sum(finite) < 4:
            continue
        y = col[finite]
        xi = it[finite]
        xm = x[finite]
        if np.ptp(y) == 0.0:
            out[f] = 0.0
            continue
        r2_i = _fit_variance_explained(y, xi)
        r2_m = _fit_variance_explained(y, xm)
        if not (np.isfinite(r2_i) and np.isfinite(r2_m)):
            continue
        interior = finite[1:-1] & finite[:-2] & finite[2:]
        if not np.any(interior):
            continue
        d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
        curv = float(np.mean(d2 < 0.0))
        # A U profile is convex; without concave curvature the inverted-U
        # label must not win even though the sign-flipped template fits.
        if curv < 0.5:
            out[f] = 0.0
            continue
        if r2_i <= r2_m:
            out[f] = 0.0
            continue
        inv_fit = max(r2_i, 0.0)
        out[f] = float(0.5 * inv_fit + 0.3 * curv + 0.2 * (inv_fit - r2_m))
    return out


# ---------------------------------------------------------------------------
# Adaptive bin count (plan §14.1)
# ---------------------------------------------------------------------------


def compute_adaptive_quantile_count(factor_batch, policy=None, tie_policy="max"):
    """Resolve a tie-aware fixed quantile count independently per factor.

    The public path accepts a :class:`FactorBatch`, applies the canonical
    quantile tie rule on every date, and returns a ``ScalarMetricArtifact``
    whose provenance contains the complete per-date feasibility evidence.
    A candidate Q is selected only when every date has all Q occupied buckets
    with the policy's minimum effective names.  Missing dates are recorded and
    make the fixed-Q comparison insufficient; they are never dropped.

    Plain quantile profiles remain accepted for backwards-compatible direct
    helper use, where this function only reports their already-built row count.
    """
    if not (hasattr(factor_batch, "time_axis") and hasattr(factor_batch, "validity")):
        qr_or_artifact = factor_batch
        if hasattr(qr_or_artifact, "n_quantiles"):
            nq = int(qr_or_artifact.n_quantiles)
            values = np.asarray(qr_or_artifact.values, dtype=np.float64)
        else:
            values = np.asarray(qr_or_artifact, dtype=np.float64)
            if values.ndim == 1:
                values = values[:, None]
            nq = int(values.shape[0])
        if values.ndim != 2 or values.shape[0] == 0:
            return np.array([np.nan], dtype=np.float64)
        return np.where(np.all(np.isfinite(values), axis=0), float(nq), np.nan)

    from quant_evaluator.contracts.adaptive_bins_policy import (
        AdaptiveBinsDateEvidence,
        AdaptiveBinsFactorEvidence,
        AdaptiveBinsPolicy,
        AdaptiveBinsResolution,
    )
    from quant_evaluator.contracts.axis_refs import FactorAxisRef
    from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact
    from quant_evaluator.contracts.quantile_policy import validate_tie_policy
    from quant_evaluator.metrics.quantile import assign_quantiles_batch

    if policy is None:
        policy = AdaptiveBinsPolicy()
    elif not isinstance(policy, AdaptiveBinsPolicy):
        from collections.abc import Mapping
        if isinstance(policy, Mapping):
            policy = AdaptiveBinsPolicy.from_dict(policy)
    if not isinstance(policy, AdaptiveBinsPolicy):
        raise TypeError("policy must be AdaptiveBinsPolicy")
    tie_policy_value = validate_tie_policy(tie_policy).value
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)

    candidates = policy.candidate_bin_counts()
    assignments = {
        q: assign_quantiles_batch(values, n_quantiles=q, method=tie_policy_value)
        for q in candidates
    }
    if values.shape[2] == 1:
        assignments = {q: a[:, :, None] for q, a in assignments.items()}

    output = np.full(factor_batch.num_factors, np.nan, dtype=np.float64)
    factor_evidence = []
    observation_counts = []
    for f, factor_id in enumerate(factor_batch.factor_ids):
        date_rows = []
        candidate_minima = {q: [] for q in candidates}
        for t in range(factor_batch.num_times):
            finite = np.isfinite(values[t, :, f])
            finite_names = int(finite.sum())
            distinct = int(np.unique(values[t, finite, f]).size) if finite_names else 0
            observed = []
            for q in candidates:
                assigned = assignments[q][t, :, f]
                valid_bins = assigned[assigned >= 0]
                minimum = (
                    int(np.bincount(valid_bins, minlength=q).min())
                    if valid_bins.size else None
                )
                observed.append((q, minimum))
                candidate_minima[q].append(minimum)
            applicable = finite_names > 0
            date_rows.append(AdaptiveBinsDateEvidence(
                date_index=t,
                applicable=applicable,
                finite_names=finite_names,
                distinct_levels=distinct,
                candidate_min_bucket_counts=tuple(observed),
                reason="evaluated" if applicable else "missing_factor_values",
            ))

        selected = None
        selected_minimum = None
        for q in candidates:
            counts = candidate_minima[q]
            if counts and all(
                count is not None and count >= policy.min_effective_names_per_bin
                for count in counts
            ):
                selected = q
                selected_minimum = float(min(counts))
                break
        reason = (
            "preferred" if selected == policy.preferred_bins
            else "fallback" if selected is not None
            else "insufficient"
        )
        resolution = AdaptiveBinsResolution(
            bin_count=selected,
            reason=reason,
            min_names_per_bin=selected_minimum,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
        )
        if selected is not None:
            output[f] = float(selected)
        observation_counts.append(sum(row.applicable for row in date_rows))
        factor_evidence.append(AdaptiveBinsFactorEvidence(
            factor_id=factor_id,
            resolution=resolution,
            tie_policy=tie_policy_value,
            comparison_policy="largest_fixed_q_feasible_on_every_date",
            dates=tuple(date_rows),
        ))

    return ScalarMetricArtifact(
        metric_id="adaptive_quantile_count",
        domain="quantile_shape",
        values=output,
        factor_axis=FactorAxisRef(tuple(factor_batch.factor_ids)),
        provenance={
            "adaptive_bins_policy": {
                "policy_id": policy.policy_id,
                "policy_version": policy.policy_version,
                "preferred_bins": policy.preferred_bins,
                "fallback_bins": tuple(policy.fallback_bins),
                "min_effective_names_per_bin": policy.min_effective_names_per_bin,
            },
            "adaptive_bins_coverage": tuple(row.to_dict() for row in factor_evidence),
            "observation_counts": tuple(observation_counts),
        },
    )


# ---------------------------------------------------------------------------
# Tail metrics with direction metadata (plan §14.4)
# ---------------------------------------------------------------------------


def _top_tail_slope_value(col: np.ndarray, n_adj: int = 2) -> float:
    """Slope of the last ``n_adj+1`` finite quantile points, direction-agnostic.

    Returns NaN when the top segment is not fully finite.  The slope is
    ``mean(diff(segment))``.
    """
    seg = col[-n_adj - 1:]
    if not np.all(np.isfinite(seg)):
        return np.nan
    diffs = np.diff(seg)
    if diffs.size == 0:
        return np.nan
    return float(np.mean(diffs))


def _bottom_tail_slope_value(col: np.ndarray, n_adj: int = 2) -> float:
    """Slope of the first ``n_adj+1`` finite quantile points."""
    seg = col[: n_adj + 1]
    if not np.all(np.isfinite(seg)):
        return np.nan
    diffs = np.diff(seg)
    if diffs.size == 0:
        return np.nan
    return float(np.mean(diffs))


def compute_top_tail_slope(qr: np.ndarray) -> np.ndarray:
    """Top-tail slope per factor, (F,).

    Mean adjacent return difference over the TOP segment of the curve as
    drawn (quantile ``K-1-k .. K-1``).  For a positively inclined factor
    this is POSITIVE (returns keep rising into the top bucket); for a
    factor with a top-tail cliff it is LARGE.  Direction metadata is on the
    registry spec (``neutral`` — the metric measures the profile's top
    segment, sign is informative, not good/bad).  NaN when the top 3
    quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    for f in range(F):
        out[f] = _top_tail_slope_value(m[:, f], n_adj=2)
    return out


def compute_bottom_tail_slope(qr: np.ndarray) -> np.ndarray:
    """Bottom-tail slope per factor, (F,).

    Mean adjacent return difference over the BOTTOM segment of the curve as
    drawn (quantile 0..2).  For a positively inclined factor this is
    POSITIVE (returns rise out of the bottom bucket).  NaN when the first
    3 quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    for f in range(F):
        out[f] = _bottom_tail_slope_value(m[:, f], n_adj=2)
    return out


def compute_tail_vs_middle_contrast(qr: np.ndarray) -> np.ndarray:
    """Tail-versus-middle contrast per factor, (F,).

    ``mean(|tail returns - middle return|)`` where the middle return is the
    median quantile's return and tails are the top and bottom quartiles of
    the quantile range.  A U-shaped profile has HIGH contrast (tails away
    from the middle); a flat profile has LOW contrast.  Always >= 0; NaN
    when the middle/tail quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 6:
        return out
    mid = nq // 2
    # Tails: quantile indices strictly BELOW lo and strictly ABOVE hi,
    # defined so that a U (or inverted-U) puts its extremal mass at the
    # edges.  ``mid`` is the interior reference point (median quantile).
    lo = max(1, min(mid - 1, int(round(nq * 0.25))))
    hi = min(nq - 1, max(mid + 1, int(round(nq * 0.75))))
    # Guarantee non-empty tails on both sides of the middle.
    if lo >= mid or hi <= mid:
        lo = mid - 1
        hi = mid + 1
    for f in range(F):
        col = m[:, f]
        if not (np.isfinite(col[mid]) and np.all(np.isfinite(col[:lo]))
                and np.all(np.isfinite(col[hi:]))):
            continue
        tail_vals = np.concatenate([col[:lo], col[hi:]])
        out[f] = float(np.mean(np.abs(tail_vals - col[mid])))
    return out


def compute_left_right_asymmetry(qr: np.ndarray) -> np.ndarray:
    """Left-right asymmetry of the quantile profile per factor, (F,).

    Mean right-minus-left mirrored bucket contrast. Symmetric U and inverted-U
    profiles are zero; curvature is not asymmetry. The central bucket (odd Q)
    is excluded. All mirrored pairs must be finite; direction is descriptive.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    mid = nq // 2
    for f in range(F):
        col = m[:, f]
        bottom = col[:mid]
        top = col[-mid:][::-1]
        if not np.all(np.isfinite(np.concatenate([bottom, top]))):
            continue
        out[f] = float(np.mean(top - bottom))
    return out


def compute_linear_trend_score(qr: np.ndarray) -> np.ndarray:
    """Linear-trend score of the quantile profile per factor, (F,).

    Pearson correlation of the profile with the linear quantile coordinate,
    bounded to [-1, 1].  ``+1`` = perfectly monotone increasing, ``-1`` =
    perfectly monotone decreasing, ~0 = flat or U-shaped (the U-shape kite
    is separated by ``u_shape_score``).  Direction ``higher_is_better``.
    NaN when fewer than 3 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    x = _template_fit_x(nq)
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        if np.sum(finite) < 3:
            continue
        y = col[finite]
        xf = x[finite]
        if np.ptp(y) == 0.0:
            out[f] = 0.0
            continue
        r = np.corrcoef(xf, y)[0, 1]
        out[f] = float(r) if np.isfinite(r) else np.nan
    return out


# ---------------------------------------------------------------------------
# Shape stability (plan §13.4)
# ---------------------------------------------------------------------------

# Shape-stability kernels require per-window quantile profiles.  They accept
# either a 2-D ``(n_quantiles, F)`` time-averaged profile aggregated directly
# (a single window), or a 3-D ``(W, n_quantiles, F)`` array of per-window
# profiles (W = windows).  The scalar burden is on the caller to build the
# window panels; the runtime's compute_fn convention passes the raw array.


def _windows_or_single(m: np.ndarray):
    """Return (windows, n_quantiles, F) view and whether multi-window."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 2:
        return m.reshape(1, m.shape[0], m.shape[1]), False
    if m.ndim == 3:
        return m, True
    raise ValueError(
        f"shape-stability metrics expect (n_quantiles, F) or (W, n_quantiles, F), "
        f"got shape {m.shape}"
    )


def _per_window_profile(
    windows: np.ndarray,
) -> np.ndarray:
    """Mean quantile-return across the window axis, (n_quantiles, F)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(windows, axis=0)


def compute_shape_stability(qr: np.ndarray) -> np.ndarray:
    """Quantile-shape stability across windows per factor, (F,).

    For a 3-D ``(W, n_quantiles, F)`` input: mean Fisher-z-corrected
    correlation of each window's profile against the leave-one-window-out
    mean profile, inverse transformed to correlation units. ``1`` = stable,
    low = shape churns.  For a 2-D single profile input this returns NaN
    with an honest ``unsupported`` (single-profile stability is undefined —
    there is no second observation), never 0 or 1.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if nw < 2:
        # Single profile: stability is a missing-evidence scalar.
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    F = windows.shape[2]
    out = np.full(F, np.nan)
    for f in range(F):
        corrs = []
        for w in range(nw):
            mp = _per_window_profile(np.delete(windows, w, axis=0))[:, f]
            wp = windows[w, :, f]
            finite = np.isfinite(wp) & np.isfinite(mp)
            if np.sum(finite) < 3:
                continue
            if np.ptp(wp[finite]) == 0.0 or np.ptp(mp[finite]) == 0.0:
                continue
            r = np.corrcoef(wp[finite], mp[finite])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
        if corrs:
            out[f] = float(np.tanh(np.mean(np.arctanh(np.clip(corrs, -1 + 1e-9, 1 - 1e-9)))))
    return out


def compute_shape_regime_stability(qr: np.ndarray) -> np.ndarray:
    """Shape stability across windows, aggregated within/against the mean, (F,).

    ## This function serves a separate metric id from "shape_stability"
    ## (id ``shape_regime_stability``, plan §13.4 "shape appears across
    ## multiple windows").  Implementation note: this metric is currently
    ## registered as ``UNSUPPORTED`` configuration in the presence of a
    ## single profile (like ``shape_stability``).  The kernel contract below
    ## IS the CPU reference for the multi-window case and matches
    ## ``compute_shape_stability``'s value distribution so both resolve the
    ## same "shape across windows" question with different cost/eyeballing.

    For a 3-D ``(W, n_quantiles, F)`` input: mean pairwise correlation of
    consecutive windows (regime version of shape stability) — measures the
    coherence of the profile as the regime rolls, damping a single-common
    shape that is stable overall but regime-uncorrelated.  NaN for a
    single-profile input.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if nw < 3:
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    F = windows.shape[2]
    out = np.full(F, np.nan)
    for f in range(F):
        corrs = []
        for w in range(nw - 1):
            a = windows[w, :, f]
            b = windows[w + 1, :, f]
            finite = np.isfinite(a) & np.isfinite(b)
            if np.sum(finite) < 3:
                continue
            if np.ptp(a[finite]) == 0.0 or np.ptp(b[finite]) == 0.0:
                continue
            r = np.corrcoef(a[finite], b[finite])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
        if corrs:
            out[f] = float(np.tanh(np.mean(np.arctanh(np.clip(corrs, -1 + 1e-9, 1 - 1e-9)))))
    return out


def compute_shape_bootstrap_confidence(qr: np.ndarray, block_length: int = 2,
                                      resamples: int = 100, random_seed: int = 0,
                                      agreement_threshold: float = .5) -> np.ndarray:
    """Descriptive moving-block bootstrap rank-agreement frequency, not U probability.

    For a 3-D ``(W, n_quantiles, F)`` input: fraction of bootstrap resamples
    (across the W windows) whose quantile-rank order (Spearman of the
    profile) reproduces the overall mean profile's rank order.  ``1`` = the
    shape's ordering is reproduced in every resample (high confidence),
    ~0.5 = noisy.  Cost is CHEAP (W is small).  Deterministic via
    ``random_seed``.  NaN for single-profile input.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not np.isfinite(agreement_threshold) or not -1 <= agreement_threshold <= 1:
        raise ValueError("agreement_threshold must be in [-1,1]")
    rng = np.random.default_rng(random_seed)
    if nw < 3:
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    if block_length > nw:
        raise ValueError("block_length cannot exceed window count")
    from scipy.stats import rankdata
    # One request-level draw schedule for every factor: adding/reordering other
    # factors cannot change a factor's evidence through RNG consumption.
    starts = rng.integers(0, nw-block_length+1, size=(resamples,int(np.ceil(nw/block_length))))
    bootstrap_indices = (starts[:,:,None]+np.arange(block_length)).reshape(resamples,-1)[:,:nw]
    F = windows.shape[2]
    out = np.full(F, np.nan)
    mean_profile = _per_window_profile(windows)
    for f in range(F):
        mp = mean_profile[:, f]
        if np.sum(np.isfinite(mp)) < 3:
            continue
        agreed = 0
        drawn = 0
        for idx in bootstrap_indices:
            sample = windows[idx, :, f]
            with np.errstate(invalid="ignore"):
                sp = np.nanmean(sample, axis=0)
            finite = np.isfinite(sp) & np.isfinite(mp)
            if np.sum(finite) < 3:
                continue
            if np.ptp(sp[finite]) == 0.0 or np.ptp(mp[finite]) == 0.0:
                continue
            r = np.corrcoef(
                rankdata(sp[finite],method="average"),
                rankdata(mp[finite],method="average"),
            )[0, 1]
            drawn += 1
            if np.isfinite(r) and r >= agreement_threshold:
                agreed += 1
        if drawn:
            out[f] = agreed / drawn
    return out


# ---------------------------------------------------------------------------
# Robust tail cliffs (plan §14.4: robust vs several adjacent bins).
# Existing one-bin cliffs are ``top_quantile_cliff`` / ``bottom_quantile_cliff``.
# The robust variants below use ``Q_K - mean(Q_(K-3..K-1))`` / symmetric —
# a DIFFERENT semantic, hence separate ids (no collision).
# ---------------------------------------------------------------------------


def compute_top_quantile_cliff_robust(qr: np.ndarray) -> np.ndarray:
    """Robust top cliff: ``ret[top] - mean(ret[K-3:K-1])`` per factor, (F,).

    Plan §14.4: single-bin cliffs are noisy; the robust variant contrasts
    the top bucket against the mean of the three DIFFERENT prior buckets
    (not ``Q_K - Q_(K-1)``).  Direction ``higher_is_better`` (big positive
    jump into the top bucket).  NaN when the top 4 quantile returns are not
    all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    for f in range(F):
        col = m[:, f]
        seg = col[-4:]
        if not np.all(np.isfinite(seg)):
            continue
        out[f] = seg[-1] - float(np.mean(seg[:-1]))
    return out


def compute_bottom_quantile_cliff_robust(qr: np.ndarray) -> np.ndarray:
    """Robust bottom cliff: ``ret[1] - mean(ret[1:4])`` per factor, (F,).

    Plan §14.4 symmetric variant: contrast the FIRST bucket against the mean
    of the three next buckets.  Direction ``higher_is_better``.  NaN when
    the bottom 4 quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    for f in range(F):
        col = m[:, f]
        seg = col[:4]
        if not np.all(np.isfinite(seg)):
            continue
        out[f] = seg[1] - float(np.mean(seg[1:4]))
    return out
