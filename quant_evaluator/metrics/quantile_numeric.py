"""Numerically guarded quantile means, shared by CPU implementations.

Retain ordinary Float64 accumulation when its conservative absolute rounding
bound is <= 1e-12 per mean. Recompute only risky buckets with accurate summation;
exact rational arithmetic handles overflow and rounds the mean once. This is not
a GPU implementation or a claim of exact rounding for the ordinary fast path.
"""
from fractions import Fraction
import math

import numpy as np


def label_sum_error_bounds(labels):
    """Upper bound for sequential summation error, once per label row.

    N bounds every bucket's length. A factor of two also covers rounding of the
    bound computation. Extreme rows fail closed to the stable path. Invalid
    labels do not participate. Retained bounds are O(T), row temporaries O(N);
    no T*N*F working array is allocated here.
    """
    n = labels.shape[1]
    eps = np.finfo(np.float64).eps
    coefficient = 2.0 * (n * eps / (1.0 - n * eps)) if n * eps < 1 else math.inf
    bounds = np.zeros(labels.shape[0], dtype=np.float64)
    for t, row in enumerate(labels):
        finite = row[np.isfinite(row)]
        if finite.size:
            maximum = float(np.max(np.abs(finite)))
            if 0 < maximum < np.finfo(np.float64).tiny:
                bounds[t] = math.inf
            elif maximum > np.finfo(np.float64).max / max(n, 1):
                bounds[t] = math.inf
            elif maximum:
                # Sum absolute magnitudes with scaling, avoiding bound overflow
                # and the excessively loose N*max bound on wide financial rows.
                scaled_sum = float(np.sum(np.abs(finite) / maximum))
                bounds[t] = maximum * (coefficient * scaled_sum)
    return bounds


def stable_finite_mean(values):
    """Correctly rounded mean of a nonempty finite Float64 risk bucket.

    Rounding the sum before division (including fsum) can double round. This
    exceptional path divides an exact rational sum before Float64 conversion.
    """
    n = len(values)
    total = sum((Fraction.from_float(float(v)) for v in values), Fraction())
    return float(total / n)


def repair_bucket_means(q_ids, labels, means, counts, min_assets, error_bound):
    """Repair a writable Q-vector in place; IDs/masks/counts stay unchanged."""
    sufficient = counts >= min_assets
    risky = sufficient & ((error_bound / np.maximum(counts, 1) > 1e-12)
                          | ~np.isfinite(means))
    if not np.any(risky):
        return
    finite_labels = labels[np.isfinite(labels)]
    finance_scale = np.all(np.abs(finite_labels) <= 1.0)
    for q in np.flatnonzero(risky):
        bucket = labels[(q_ids == q) & np.isfinite(labels)]
        # A full-row bound can be very loose for small/tied buckets. Refine
        # with the same conservative arithmetic on exactly the selected
        # finite observations, using their actual length. This changes no
        # error contract: exact repair is still required above 1e-12, for
        # overflow/subnormal risk, or when the computed mean is nonfinite.
        # Non-financial/mixed-scale rows retain the previous exact-risk path,
        # including the independently verified double-rounding regression.
        if np.isfinite(means[q]) and finance_scale:
            bucket_bound = label_sum_error_bounds(bucket.reshape(1, -1))[0]
            if bucket_bound / max(len(bucket), 1) <= 1e-12:
                continue
        means[q] = stable_finite_mean(bucket)


def repair_quantile_panel(q_ids, labels, returns, counts, min_assets):
    """Postprocess a CPU kernel's T*Q*F output with its exact assignments."""
    bounds = label_sum_error_bounds(labels)
    for t, bound in enumerate(bounds):
        sufficient = counts[t] >= min_assets
        if not np.any(sufficient):
            continue
        if (bound / np.min(counts[t][sufficient]) <= 1e-12
                and np.all(np.isfinite(returns[t][sufficient]))):
            continue
        for f in range(q_ids.shape[2]):
            repair_bucket_means(q_ids[t, :, f], labels[t], returns[t, :, f],
                                counts[t, :, f], min_assets, bound)
