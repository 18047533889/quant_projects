"""Scale-safe unit vectors for cosine search backend boundaries."""
from __future__ import annotations

import numpy as np


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Normalize real finite rows before narrowing to backend float32.

    Scaling by each row's largest magnitude avoids overflow and underflow
    in the squared norm. The returned float64 vectors preserve direction;
    this helper never modifies caller storage.
    """
    if (not isinstance(values, np.ndarray) or values.ndim != 2
            or values.shape[1] == 0 or values.dtype.kind not in "fiu"):
        raise ValueError("embeddings must be a real numeric matrix")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite values")
    work = values.astype(np.result_type(values.dtype, np.float64), copy=False)
    scales = np.max(np.abs(work), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("zero vectors are not valid embeddings")
    scaled = work / scales
    norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return np.ascontiguousarray(scaled / norms, dtype=np.float64)


def _canonical_cosine(value: float) -> float:
    """Clamp float32 roundoff at cosine endpoints; reject invalid scores."""
    value = float(value)
    tolerance = 16 * np.finfo(np.float32).eps
    if not np.isfinite(value) or abs(value) > 1. + tolerance:
        raise ValueError("backend cosine score is outside its finite numeric envelope")
    return float(np.clip(value, -1., 1.))
