"""Shared bounded identities for source-route output receipts.

Hashes cover raw array bytes in canonical C order, finite-mask bytes, and
little-endian int64 observation counts. Chunked iteration keeps peak scratch
memory bounded independently of the evaluated panel size.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

import numpy as np

from quant_evaluator.runtime.source_metric_catalog import SOURCE_SERIES_METRICS

_HASH_CHUNK = 65_536


@dataclass(frozen=True)
class MetricOutputIdentity:
    values_sha256: str
    finite_mask_sha256: str
    observation_counts_sha256: str


def hash_array_bytes(values: np.ndarray) -> str:
    """SHA-256 of raw array bytes visited in canonical C order."""
    digest = hashlib.sha256()
    iterator = np.nditer(values, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C", buffersize=_HASH_CHUNK)
    for chunk in iterator:
        digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def hash_finite_mask(values: np.ndarray) -> str:
    """SHA-256 of C-order uint8 finite-mask bytes."""
    digest = hashlib.sha256()
    iterator = np.nditer(values, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C", buffersize=_HASH_CHUNK)
    for chunk in iterator:
        digest.update(np.isfinite(chunk).astype(np.uint8, copy=False).tobytes())
    return digest.hexdigest()


def normalize_counts_array(counts: np.ndarray) -> np.ndarray:
    """Use the established C-order little-endian int64 count encoding."""
    return np.asarray(counts, dtype="<i8", order="C")


def metric_output_identity(values: np.ndarray, counts: np.ndarray) -> MetricOutputIdentity:
    normalized_counts = normalize_counts_array(counts)
    return MetricOutputIdentity(
        values_sha256=hash_array_bytes(values),
        finite_mask_sha256=hash_finite_mask(values),
        observation_counts_sha256=hash_array_bytes(normalized_counts),
    )


def matches_profile_output_identity(output, profile, context, *, metrics,
                                    expected_factor_ids) -> bool:
    """Compare a live bundle's full scalar/series outputs to one backend receipt."""
    try:
        selected = tuple(metrics)
        factor_ids = getattr(output, "factor_ids", None)
        factor_count = context.request_shape[-1]
        if selected != tuple(context.metric_ids):
            return False
        if (type(factor_ids) is not tuple or factor_ids != expected_factor_ids
                or len(factor_ids) != factor_count
                or any(type(value) is not str or not value for value in factor_ids)
                or len(set(factor_ids)) != factor_count):
            return False
        scalar = output.scalar_metrics
        series = output.series_metrics
        vector = output.vector_metrics
        counts_map = output.observation_counts
        if not all(isinstance(item, Mapping) for item in (scalar, series, vector, counts_map)):
            return False
        receipts = profile.outputs
        if (type(receipts) is not tuple
                or tuple(item.metric_id for item in receipts) != selected):
            return False
        T, _, F = context.request_shape
        for receipt in receipts:
            metric = receipt.metric_id
            if metric in vector:
                return False
            in_scalar, in_series = metric in scalar, metric in series
            if in_scalar == in_series or ((metric in SOURCE_SERIES_METRICS) != in_series):
                return False
            values = series[metric] if in_series else scalar[metric]
            expected_shape = (T, F) if in_series else (F,)
            if (not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu"
                    or values.shape != expected_shape):
                return False
            counts = counts_map.get(metric)
            if (not isinstance(counts, np.ndarray) or counts.shape != (F,)
                    or counts.dtype.kind not in "iu" or np.any(counts < 0)):
                return False
            if (counts.dtype.kind == "u"
                    and np.any(counts > np.iinfo(np.int64).max)):
                return False
            identity = metric_output_identity(values, counts)
            if (identity.values_sha256 != receipt.values_sha256
                    or identity.finite_mask_sha256 != receipt.finite_mask_sha256
                    or identity.observation_counts_sha256
                    != receipt.observation_counts_sha256):
                return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False
