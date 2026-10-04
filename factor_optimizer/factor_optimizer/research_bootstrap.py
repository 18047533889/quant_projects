"""Small deterministic moving-block bootstrap helper for research comparisons."""
from __future__ import annotations

import math
import numbers

import numpy as np


def _strict_int(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def moving_block_lower_bound(
    differences,
    *,
    block_length: int,
    horizon: int,
    minimum_validation_days: int,
    bootstrap_draws: int,
    seed: int,
    confidence_level: float,
) -> float | None:
    """Return the seeded lower-tail mean bound using moving time blocks.

    Input order and dtype are retained. Non-finite observations remain in the
    timeline during block sampling, then are omitted from each draw's mean.
    """
    for name, value in (
        ("block_length", block_length),
        ("horizon", horizon),
        ("minimum_validation_days", minimum_validation_days),
        ("bootstrap_draws", bootstrap_draws),
    ):
        _strict_int(name, value, minimum=1)
    _strict_int("seed", seed, minimum=0)
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, numbers.Real):
        raise ValueError("confidence_level must be finite and in (0,1)")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and in (0,1)")

    try:
        values = np.asarray(differences)
    except (TypeError, ValueError) as exc:
        raise ValueError("differences must be one-dimensional real numeric data") from exc
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError("differences must be one-dimensional real numeric data")

    length = max(block_length, horizon)
    n = len(values)
    if n < 3 * length:
        return None

    rng = np.random.default_rng(seed)
    starts_per_draw = math.ceil(n / length)
    offsets = np.arange(length)
    block_indices = np.empty((starts_per_draw, length), dtype=np.intp)
    sampled = np.empty(n, dtype=values.dtype)
    draws = []
    for _ in range(bootstrap_draws):
        starts = rng.integers(0, n - length + 1, size=starts_per_draw)
        np.add(starts[:, None], offsets, out=block_indices)
        np.take(values, block_indices.reshape(-1)[:n], out=sampled)
        finite = sampled[np.isfinite(sampled)]
        if len(finite) < minimum_validation_days:
            return None
        draws.append(float(finite.mean()))
    return float(np.quantile(draws, (1 - confidence_level) / 2))
