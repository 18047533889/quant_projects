"""Real three-feature tangent-plane contracts and an independent eigen oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _frames():
    rng = np.random.default_rng(4041)
    idx = pd.date_range("2024-01-01", periods=6)
    return [pd.DataFrame(rng.normal(size=(6, 32)), index=idx,
                         columns=[f"S{i}" for i in range(32)]) for _ in range(3)]


def _oracle(frames, k=20):
    expected = np.full(frames[0].shape, np.nan)
    # Independent pandas average ranks and symmetric eigendecomposition.
    ranks = [(f.rank(axis=1, method="average") - 0.5).div(f.count(axis=1), axis=0)
             for f in frames]
    for row in range(len(frames[0])):
        points = np.stack([f.iloc[row].to_numpy() for f in ranks], axis=1)
        valid = np.isfinite(points).all(axis=1)
        for i in np.flatnonzero(valid):
            peers = np.flatnonzero(valid & (np.arange(len(points)) != i))
            if len(peers) < k:
                continue
            distances = np.linalg.norm(points[peers] - points[i], axis=1)
            peers = peers[distances <= np.sort(distances)[k - 1]]
            center = points[peers].mean(axis=0)
            centered = points[peers] - center
            eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
            if eigenvalues[-1] <= 1e-24 or eigenvalues[-2] / eigenvalues[-1] < 0.05**2:
                continue
            scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
            expected[row, i] = abs((points[i] - center) @ vectors[:, 0]) / scale
    return pd.DataFrame(expected, index=frames[0].index, columns=frames[0].columns)


def test_tangent_matches_independent_formula_and_keyword_call():
    frames = _frames()
    op = OperatorRegistry.get("cs_knn_tangent_residual")
    actual = op.calculate(*frames)
    assert np.isfinite(actual.to_numpy()).all()
    pd.testing.assert_frame_equal(actual, _oracle(frames), atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(actual, op.calculate(f1=frames[0], f2=frames[1], f3=frames[2]))


def test_tangent_missing_and_insufficient_breadth():
    frames = _frames()
    frames[1].iloc[0, 0] = np.nan
    frames[2].iloc[1, :] = np.nan
    op = OperatorRegistry.get("cs_knn_tangent_residual")
    actual = op.calculate(*frames)
    pd.testing.assert_frame_equal(actual, _oracle(frames), atol=1e-10, rtol=1e-10)
    assert np.isnan(actual.iloc[0, 0])
    assert actual.iloc[1].isna().all()
    assert op.calculate(*(f.iloc[:, :20] for f in frames)).isna().all().all()


def test_tangent_deterministic_prefix_and_permutation():
    frames = _frames()
    op = OperatorRegistry.get("cs_knn_tangent_residual")
    actual = op.calculate(*frames)
    pd.testing.assert_frame_equal(actual, op.calculate(*frames))
    pd.testing.assert_frame_equal(actual.iloc[:3], op.calculate(*(f.iloc[:3] for f in frames)))
    reordered = op.calculate(*(f.iloc[:, ::-1] for f in frames)).iloc[:, ::-1]
    pd.testing.assert_frame_equal(actual, reordered, atol=1e-10, rtol=1e-10)


def test_tangent_declares_and_validates_input_contract():
    frames = _frames()
    op = OperatorRegistry.get("cs_knn_tangent_residual")
    assert tuple(op.metadata.panel_params) == ("f1", "f2", "f3")
    assert tuple(op.metadata.scalar_params) == ("k",)
    with pytest.raises(OperatorParameterError, match="missing"):
        op.calculate(frames[0])
    for bad_k in (1, 19, 20.5, True):
        with pytest.raises((ValueError, TypeError)):
            op.calculate(*frames, k=bad_k)
