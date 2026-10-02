"""Exact sign-antithetic RankIC cache support for adjacent TRAIN proposals.

The helper is deliberately separate from candidate selection. A caller may use
it only after the positive-sign candidate has been evaluated on TRAIN and has
the same pairwise-availability mask as its negative-sign sibling.
"""
from __future__ import annotations

import numpy as np


def _validate_candidate_pair(candidate_values, common_mask, target, minimum_assets):
    values = np.asarray(candidate_values)
    common = np.asarray(common_mask)
    if values.ndim != 2 or values.dtype.kind != "f" or min(values.shape, default=0) < 1:
        raise ValueError("candidate_values must be a 2D real floating-point panel")
    if common.shape != values.shape or common.dtype != np.bool_:
        raise ValueError("common_mask must be a matching boolean panel")
    if type(minimum_assets) is not int or minimum_assets < 1:
        raise ValueError("minimum_assets must be a positive integer")
    label_values = np.asarray(target.values)
    if label_values.shape != values.shape:
        raise ValueError("candidate and target panels must have identical shapes")
    label_validity = target.validity
    if label_validity is not None and label_validity.shape != values.shape:
        raise ValueError("target validity must match candidate panel")

    pair_validity = np.isfinite(values) & np.isfinite(label_values)
    if label_validity is not None:
        pair_validity &= label_validity
    if np.any(common & ~pair_validity):
        raise ValueError("common_mask may include only finite, valid candidate-label pairs")
    return values, common


def _negated_candidate_key(values, common, target, minimum_assets):
    """Use the canonical candidate-key serializer for the sign-flipped panel."""
    from factor_optimizer.research_batch import _candidate_ic_key

    # One bounded 2D working panel is sufficient. Pass it through the existing
    # serializer so future identity changes remain authoritative in one place.
    negative_effective = np.full(values.shape, np.nan, dtype=values.dtype)
    np.negative(values, out=negative_effective, where=common)
    return _candidate_ic_key(negative_effective, target, minimum_assets)


def cache_negated_candidate_ic(
    candidate_cache,
    candidate_values,
    common_mask,
    target,
    minimum_assets,
    positive_ic,
):
    """Cache exact sign-opposite IC and return its normal key, or None to fall back.

    ``candidate_values`` is the unmasked positive candidate panel. ``common_mask``
    must be the exact RAW/candidate/label availability mask used by paired IC.
    The sign flip preserves that mask. This function never reads VALIDATION or
    TEST data; its caller supplies only the TRAIN target and TRAIN IC series.
    """
    from factor_optimizer.research_batch import PairICCache

    if not isinstance(candidate_cache, PairICCache):
        raise TypeError("candidate_cache must be PairICCache")
    values = np.asarray(candidate_values)
    ic = np.asarray(positive_ic)
    if values.ndim != 2:
        raise ValueError("candidate_values must be a 2D panel")
    if ic.shape != (values.shape[0],) or ic.dtype.kind != "f":
        raise ValueError("positive_ic must be a floating-point vector aligned to candidate rows")
    if np.isinf(ic).any():
        raise ValueError("positive_ic cannot contain infinities")
    values, common = _validate_candidate_pair(
        values, common_mask, target, minimum_assets)
    # Floating-point dot products can canonicalize an exact zero covariance to
    # +0.0 for both signs. Negating the positive series would then produce -0.0,
    # so leave this candidate uncached and let the exact negative path run.
    if np.any(np.isfinite(ic) & (ic == 0.0)):
        return None
    key = _negated_candidate_key(values, common, target, minimum_assets)
    negative_ic = -ic.copy()
    # QE initializes unavailable observations with a canonical positive NaN;
    # preserve that representation rather than caching sign-flipped NaN payloads.
    negative_ic[~np.isfinite(ic)] = np.nan
    candidate_cache.put(key, negative_ic)
    return key


__all__ = ["cache_negated_candidate_ic"]
