# -*- coding: utf-8 -*-
"""R11 regression: pairwise bicor correlation matrix must be PSD-projected.

A pairwise biweight midcorrelation matrix is NOT guaranteed positive-semidefinite
(each off-diagonal is estimated independently on its own pair).  Before R11 the
spectral consumers of ``cleaned_operators/feature_geometry`` clipped negative
eigenvalues to 0 with ``np.maximum(w, 0.0)`` — silently dropping eigenvalue mass
(breaking the trace / mode-share / effective-rank identity) — while
``_dominant_direction`` and ``_subspace_rotation_chunk`` still read the RAW
unprojected matrix for the eigenvector, so the eigen-metrics and the subspace
direction came from DIFFERENT matrices.

R11 introduces ``_nearest_psd_correlation`` (Higham-style alternating
projection) and routes every spectral consumer through one canonical projected
matrix ``_canonical_corr``.

Also asserts the R11 unit fix: ``ts_feature_effective_rank`` outputs an
effective DIMENSION ``exp(-sum p log p)`` in [1, 3], not an ordinal rank, so its
metadata unit is ``dimensionless``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.feature_geometry import (
    _canonical_corr,
    _corr_eigenvalues,
    _dominant_direction,
    _nearest_psd_correlation,
    TsFeatureEffectiveRank,
)


def _frame(values: np.ndarray, ncols: int = 1) -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    cols = v.shape[1] if ncols is None else ncols
    return pd.DataFrame(
        v,
        index=pd.date_range("2024-01-01", periods=v.shape[0], freq="B"),
        columns=[f"S{i}" for i in range(cols)],
    )


def _z3(seed: int = 0) -> np.ndarray:
    """3-column feature z-matrix: two strongly-correlated modes + one independent.

    Robust (median/MAD) z-scores, matching the kernel's pre-processing, so the
    pairwise bicor matrix and its dominant mode are meaningful.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, 80)
    raw = np.column_stack(
        [
            base + 0.1 * rng.normal(0.0, 1.0, 80),
            base + 0.1 * rng.normal(0.0, 1.0, 80),
            rng.normal(0.0, 1.0, 80),
        ]
    )
    med = np.median(raw, axis=0)
    mad = np.median(np.abs(raw - med), axis=0)
    scale = np.where(mad > 0.0, 1.4826 * mad, np.std(raw, axis=0))
    return (raw - med) / scale


# ---------------------------------------------------------------------------
# ISSUE 1 — PSD projection of the pairwise robust correlation matrix
# ---------------------------------------------------------------------------
def test_canonical_corr_is_symmetric_unit_diag_psd():
    """(a) synthetic 3-col matrix -> symmetric, unit diagonal, eigenvalues >= -1e-10."""
    c = _canonical_corr(_z3(1))
    assert c is not None
    assert c.shape == (3, 3)
    assert np.allclose(c, c.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(c), 1.0, atol=1e-12)
    ev = np.linalg.eigvalsh(c)
    assert np.all(ev >= -1e-10), ev


def test_corr_eigenvalues_conserve_trace_no_silent_clip():
    """(b) sum of eigenvalues == trace of the SAME projected matrix (no clipping away)."""
    z = _z3(2)
    w = _corr_eigenvalues(z)
    c = _canonical_corr(z)
    assert w is not None and c is not None
    # Pre-R11, clipping negative eigenvalues to 0 broke sum(λ) == trace(C).
    assert abs(float(w.sum()) - float(np.trace(c)) < 1e-9


def test_dominant_direction_shares_eigenvalue_matrix():
    """(c) dominant-direction eigenvector's eigenvalue == largest eigenvalue from
    ``_corr_eigenvalues`` — direction and mode-share come from ONE projected matrix."""
    z = _z3(3)
    v1 = _dominant_direction(z)
    w = _corr_eigenvalues(z)
    c = _canonical_corr(z)
    assert v1 is not None and w is not None and c is not None
    # The direction returned by the kernel must be the top eigenvector of the
    # SAME projected matrix feeding the eigen-metrics.
    _, v = np.linalg.eigh(c)
    v1_ref = v[:, -1]
    k = int(np.argmax(np.abs(v1_ref)))
    if v1_ref[k] < 0:
        v1_ref = -v1_ref
    np.testing.assert_allclose(v1, v1_ref, atol=1e-12)
    assert abs(float(w[-1]) - float(np.linalg.eigvalsh(c)[-1]) < 1e-12


def test_adversarial_non_psd_pairwise_corr_projected():
    """(d) 0.9/0.9/-0.9 triangle is genuinely indefinite; the projection makes it
    PSD while keeping a unit diagonal and staying close to the input."""
    c_raw = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    assert np.min(np.linalg.eigvalsh(c_raw) < -1e-6  # genuinely indefinite
    c_proj = _nearest_psd_correlation(c_raw)
    np.testing.assert_allclose(np.diag(c_proj), 1.0, atol=1e-12)
    assert np.allclose(c_proj, c_proj.T, atol=1e-12)
    ev = np.linalg.eigvalsh(c_proj)
    assert np.all(ev >= -1e-10), ev
    # nearest-correlation flavour: a modest perturbation, not a wholesale rewrite
    assert np.linalg.norm(c_proj - c_raw, ord="fro") < 1.0


# ---------------------------------------------------------------------------
# ISSUE 2 — ts_feature_effective_rank unit is dimensionless (effective dimension)
# ---------------------------------------------------------------------------
def test_effective_rank_metadata_unit_dimensionless():
    unit_tags = [t for t in TsFeatureEffectiveRank.metadata.tags if t.startswith("unit:")]
    assert unit_tags == ["unit:dimensionless"], unit_tags
    assert any(
        t == "semantic_kind:effective_dimension"
        for t in TsFeatureEffectiveRank.metadata.tags
    ), TsFeatureEffectiveRank.metadata.tags


def test_effective_rank_bounds_on_synthetic_fields():
    rng = np.random.default_rng(5)
    n = 80
    base = rng.normal(0.0, 1.0, n)
    f1 = _frame(base + 0.1 * rng.normal(0.0, 1.0, n))
    f2 = _frame(base + 0.1 * rng.normal(0.0, 1.0, n))
    f3 = _frame(rng.normal(0.0, 1.0, n))
    out = TsFeatureEffectiveRank().calculate(f1, f2, f3, window=60)
    vals = out.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert finite.min() >= 1.0 - 1e-9
    assert finite.max() <= 3.0 + 1e-9
