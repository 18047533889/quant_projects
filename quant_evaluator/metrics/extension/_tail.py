"""Shared §6.6 upper-tail primitives for the extension metric families.

Spec: ``qe_opt/extension_plan.md`` (QE-EXT-SPEC-1.0).

- §6.6 ``UpperTailMean(x, q)`` is the mean of the worst ``1 - q`` probability
  mass of the empirical distribution, with boundary ties splitting the
  remaining fractional mass equally (NOT a ">= interpolated quantile" filter
  and never earliest-tied-wins).  This keeps tied-date permutations from
  changing conditional results.
- §1.1 precision clause: all reductions are FP64; no fastmath, no FP16/BF16.
- §1.3 dtype gate: bool / complex / object / string inputs are rejected with
  ``TypeError`` (matches the repo-wide dtype-rejection convention).
"""

from __future__ import annotations

import math

import numpy as np

_REAL_KINDS = frozenset("fiu")


def as_real_f64(values, name, *, ndim=None):
    """Coerce a real numeric array to float64; reject non-real dtypes."""
    raw = np.asarray(values)
    if raw.dtype.kind not in _REAL_KINDS:
        raise TypeError(
            f"{name} must be a real numeric array (dtype kind in "
            f"{sorted(_REAL_KINDS)}), got dtype {raw.dtype}"
        )
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-dimensional, got shape {raw.shape}"
        )
    return raw.astype(np.float64, copy=True)


def check_confidence(confidence, name="tail_confidence"):
    """Validate a strictly-between-0-and-1 real scalar confidence level."""
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(
        confidence, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar in (0, 1), got {confidence!r}")
    value = float(confidence)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}")
    return value


def check_min_tail_mass(min_tail_mass):
    """Validate the observation-equivalent minimum tail weight mass."""
    if isinstance(min_tail_mass, (bool, np.bool_)) or not isinstance(
        min_tail_mass, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "min_tail_mass must be a real observation-equivalent weight, "
            f"got {min_tail_mass!r}"
        )
    mass = float(min_tail_mass)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"min_tail_mass must be positive finite, got {mass!r}")
    return mass


def resolved_tail_mass(n, confidence):
    """``(1 - q) * n`` with the §6.6 8-eps snap to a nearby integer.

    Prevents cross-platform off-by-one when the fractional mass lands within
    rounding distance of an integer.
    """
    mass = (1.0 - confidence) * int(n)
    nearest = round(mass)
    if nearest >= 1 and abs(mass - nearest) <= 8.0 * np.finfo(float).eps * max(1.0, mass):
        mass = float(nearest)
    return mass


def tail_weights_1d(x, confidence):
    """Reference boundary weights for one finite 1-D series (§6.6 / §12 oracle).

    Weights sum to exactly the tail mass ``m``: strictly-above-cutoff dates
    get weight 1, below-cutoff weight 0, all cutoff-tied dates split the
    remaining mass equally.
    """
    if x.ndim != 1 or x.size == 0:
        raise ValueError("tail_weights_1d requires a nonempty 1-D series")
    confidence = check_confidence(confidence)
    if not bool(np.isfinite(x).all()):
        raise ValueError("tail_weights_1d requires finite values")
    mass = resolved_tail_mass(x.size, confidence)
    index = min(x.size - 1, math.ceil(mass) - 1)
    cutoff = np.sort(x)[::-1][index]
    above = x > cutoff
    tied = x == cutoff
    w = above.astype(np.float64)
    n_above = int(above.sum())
    n_tied = int(tied.sum())
    if n_tied:
        w[tied] = (mass - n_above) / n_tied
    return w


def tail_weights_batch(x, confidence):
    """Vectorized boundary weights for finite (T,) or (T, F) inputs."""
    x2 = x if x.ndim == 2 else x[:, None]
    mass = resolved_tail_mass(x2.shape[0], confidence)
    index = min(x2.shape[0] - 1, math.ceil(mass) - 1)
    cutoff = np.sort(x2, axis=0)[::-1][index]
    above = x2 > cutoff
    tied = x2 == cutoff
    frac = (mass - above.sum(axis=0)) / tied.sum(axis=0)
    w = above.astype(np.float64) + tied * frac
    return w if x.ndim == 2 else w[:, 0]


def upper_tail_mean_1d(x, confidence):
    """Reference §6.6 upper-tail mean of one finite 1-D series."""
    w = tail_weights_1d(x, confidence)
    return float(w @ x / w.sum())


def upper_tail_mean_batch(x, confidence):
    """Vectorized §6.6 upper-tail mean on finite (T,) or (T, F) inputs."""
    w = tail_weights_batch(x, confidence)
    values = (w * x).sum(axis=0) / w.sum(axis=0)
    return float(values[0]) if x.ndim == 1 else values
