"""Bounded independent Pearson-chain reference, outside performance timings.

No QE numeric kernels are used. SciPy provides ordinary row correlations;
Decimal centered moments handle near-constant rows without subtractive loss.
The caller supplies a fresh sequential source and owns its close lifecycle.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import warnings

import numpy as np
from scipy.stats import pearsonr

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api.factor_source import _source_request_fingerprint
from quant_evaluator.contracts.factor_tile_source import (
    capture_factor_tile_source, read_validated_factor_tile,
    admitted_source_tile_limit,
)
from quant_evaluator.contracts.label_bundle import LabelBundle

PEARSON_CHAIN = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")


def _decimal_correlation(x, y):
    with localcontext() as ctx:
        ctx.prec = 80
        dx = [Decimal.from_float(float(v)) for v in x]
        dy = [Decimal.from_float(float(v)) for v in y]
        mx, my = sum(dx) / len(dx), sum(dy) / len(dy)
        dx, dy = [v - mx for v in dx], [v - my for v in dy]
        xx, yy = sum(v * v for v in dx), sum(v * v for v in dy)
        if not xx or not yy:
            return math.nan
        return float(sum(a * b for a, b in zip(dx, dy)) / (xx * yy).sqrt())


def _exact_affine_endpoint_sign(x, y):
    """Return +/-1 only when the stored binary floats are exactly affine."""
    fx = [Fraction.from_float(float(v)) for v in x]
    fy = [Fraction.from_float(float(v)) for v in y]
    anchor = next((i for i in range(1, len(fx)) if fx[i] != fx[0]), None)
    if anchor is None:
        return None
    slope = (fy[anchor] - fy[0]) / (fx[anchor] - fx[0])
    if slope == 0:
        return None
    intercept = fy[0] - slope * fx[0]
    if all(yi == slope * xi + intercept for xi, yi in zip(fx, fy)):
        return 1 if slope > 0 else -1
    return None


def reference_row_pearson(x, y, *, min_assets=20):
    """Finite vectors already filtered by caller's pairwise validity mask."""
    if type(min_assets) is not int or min_assets < 2:
        raise ValueError("min_assets must be an integer >= 2")
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("reference rows must have matching one-dimensional shapes")
    if any(v.dtype not in (np.dtype("float32"), np.dtype("float64")) for v in (x, y)):
        raise TypeError("reference rows require Float32 or Float64")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("reference rows must be pairwise finite")
    # Float32 storage must not imply Float32 reference arithmetic.
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if len(x) < min_assets or len(x) < 2:
        return math.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan
    # Algebraic identities, independently checked before SciPy normalization:
    # its ulp noise must not manufacture dispersion for a perfect IC series.
    if np.array_equal(x, y):
        return 1.0
    if np.array_equal(x, -y):
        return -1.0
    sx, sy = float(np.max(np.abs(x))), float(np.max(np.abs(y)))
    # Scaling is safe for ordinary rows and avoids overflow of raw sums.
    ax, ay = x / sx, y / sy
    near_constant = (float(np.max(ax) - np.min(ax)) < 1e-12
                     or float(np.max(ay) - np.min(ay)) < 1e-12)
    if near_constant:
        value = _decimal_correlation(x, y)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = float(pearsonr(ax, ay).statistic)
    # Only perform rational proof for numerically endpoint-like candidates.
    # Near-perfect but non-affine data retains the independent numeric result.
    if math.isfinite(value) and abs(value) >= 1.0 - 1e-12:
        exact_sign = _exact_affine_endpoint_sign(x, y)
        if exact_sign is not None:
            return float(exact_sign)
    if not math.isfinite(value):
        raise ValueError("independent Pearson oracle produced nonfinite correlation")
    return max(-1.0, min(1.0, value))


def reference_source_pearson_chain(source, labels, *, max_tile_size=1,
                                   max_result_bytes=64 * 1024**2):
    """Validate every time/factor row with at most one admitted tile in memory.

    Defaults mirror the public source API: min_assets=20; mean minimum 1;
    sample std/IR minimum 20; constant-series IR is undefined. Counts always
    count finite daily correlations, including when a summary is unavailable.
    Only the Pearson chain is certified by this reference, not other metrics.
    """
    for name, value in (("max_tile_size", max_tile_size),
                        ("max_result_bytes", max_result_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if not isinstance(labels, LabelBundle):
        raise TypeError("labels must be LabelBundle")
    meta = capture_factor_tile_source(source)
    if np.dtype(meta.dtype) not in (np.dtype("float32"), np.dtype("float64")) or (
            labels.values.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise TypeError("Pearson oracle supports Float32/Float64 source and labels only")
    T, N, F = meta.time_axis.size, meta.asset_axis.size, len(meta.factor_ids)
    if labels.values.shape != (T, N):
        raise ValueError("label shape does not match source")
    if not np.array_equal(np.asarray(labels.decision_time), meta.time_axis.values):
        raise ValueError("label time coordinates do not match source")
    if (labels.asset_axis is None or labels.asset_axis.values is None
            or labels.asset_axis.name != meta.asset_axis.name
            or labels.asset_axis.dtype != meta.asset_axis.dtype
            or not np.array_equal(
                labels.asset_axis.values, meta.asset_axis.values)):
        raise ValueError("label asset coordinates do not match source")
    # Output only, never a full factor cube. Scalar + count reserves included.
    # This is a returned-array budget, not a peak RSS or SciPy scratch bound.
    if 8 * F * (T + 3 + 4) > max_result_bytes:
        raise MemoryError("independent oracle result exceeds budget")
    series = np.full((T, F), np.nan, dtype=np.float64)
    width = min(max_tile_size, admitted_source_tile_limit(
        meta.max_tile_size, getattr(source, "admitted_max_tile_size", None)))
    for start in range(0, F, width):
        tile = read_validated_factor_tile(source, meta, start, min(start + width, F))
        batch = tile.batch
        for t in range(T):
            y = labels.values[t]
            label_mask = np.isfinite(y)
            if labels.validity is not None:
                label_mask &= labels.validity[t]
            for f in range(batch.num_factors):
                x = batch.values[t, :, f]
                mask = label_mask & np.isfinite(x)
                if batch.validity is not None:
                    mask &= batch.validity[t, :, f]
                series[t, start + f] = reference_row_pearson(x[mask], y[mask])
                del x  # Do not keep a previous tile alive through a row view.
        del batch, tile
    out = BatchEvaluationBundle(meta.factor_ids, labels.target_id)
    out.metadata = {
        "source_request_fingerprint": _source_request_fingerprint(meta, labels, PEARSON_CHAIN),
        "reference_method": "independent_scipy_decimal80_pearson_v1",
        "coverage_scope": "every_time_factor_row",
        "source_snapshot_id": meta.snapshot_id,
    }
    out.series_metrics["pearson_ic_series"] = series
    for metric in ("pearson_ic", "pearson_ic_std", "pearson_ic_ir"):
        out.scalar_metrics[metric] = np.full(F, np.nan)
    counts = np.isfinite(series).sum(axis=0, dtype=np.int64)
    for metric in PEARSON_CHAIN:
        out.observation_counts[metric] = counts.copy()
    for f in range(F):
        finite = series[np.isfinite(series[:, f]), f]
        if not finite.size:
            continue
        mean = math.fsum(float(v) for v in finite) / finite.size
        out.scalar_metrics["pearson_ic"][f] = mean
        if finite.size >= 20:
            constant = bool(np.all(finite == finite[0]))
            std = (0.0 if constant else math.sqrt(
                math.fsum((float(v) - mean)**2 for v in finite) / (finite.size - 1)))
            out.scalar_metrics["pearson_ic_std"][f] = std
            if not constant and std > 0:
                out.scalar_metrics["pearson_ic_ir"][f] = mean / std
    return out
