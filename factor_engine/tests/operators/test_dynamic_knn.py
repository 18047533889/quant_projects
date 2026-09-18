from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None
    return op


def _panels(rows=5):
    columns = list("ABCDEFGH")
    index = pd.date_range("2024-01-01", periods=rows)
    asset = np.arange(8, dtype=float)
    time = np.arange(rows, dtype=float)[:, None]
    target = pd.DataFrame(10 * time + asset[None, :] ** 2, index=index, columns=columns)
    f1 = pd.DataFrame(asset[None, :] + 0.1 * time, index=index, columns=columns)
    f2 = pd.DataFrame(
        np.array([0, 2, 1, 3, 5, 4, 7, 6])[None, :] - 0.2 * time,
        index=index, columns=columns,
    )
    f3 = pd.DataFrame(
        np.array([0, 0, 1, 1, 2, 2, 3, 3])[None, :] + 0.3 * time,
        index=index, columns=columns,
    )
    return target, f1, f2, f3


def _call(name, target, f1, f2, f3, k=3, lag=1):
    if name == "cs_knn_neighbor_retention":
        return _op(name).calculate(f1, f2, f3, k=k, lag=lag)
    return _op(name).calculate(target, f1, f2, f3, k=k)


def _oracle_neighbors(f1, f2, f3, row, asset, k):
    vectors = []
    for frame in (f1, f2, f3):
        values = frame.iloc[row].to_numpy(dtype=float)
        finite = np.isfinite(values)
        ranks = pd.Series(values[finite]).rank(method="average").to_numpy()
        ranked = np.full(len(values), np.nan)
        ranked[finite] = (ranks - 0.5) / finite.sum()
        vectors.append(ranked)
    features = np.column_stack(vectors)
    valid = np.all(np.isfinite(features), axis=1)
    distance = np.sqrt(np.sum((features - features[asset]) ** 2, axis=1))
    candidates = np.flatnonzero(valid)
    candidates = candidates[candidates != asset]
    kth = np.sort(distance[candidates], kind="stable")[k - 1]
    return candidates[distance[candidates] <= kth]


NAMES = (
    "cs_knn_peer_mean_ex_self",
    "cs_knn_neighbor_retention",
    "cs_knn_graph_dirichlet_energy",
)


@pytest.mark.parametrize("name", NAMES)
def test_dynamic_knn_legal_eight_asset_inputs_are_finite(name):
    target, f1, f2, f3 = _panels()
    result = _call(name, target, f1, f2, f3)
    assert result.shape == target.shape
    assert np.isfinite(result.iloc[1:].to_numpy()).all()


def test_peer_mean_and_energy_match_independent_distance_oracle():
    target, f1, f2, f3 = _panels()
    peer = _call("cs_knn_peer_mean_ex_self", target, f1, f2, f3)
    energy = _call("cs_knn_graph_dirichlet_energy", target, f1, f2, f3)
    row, asset, k = 2, 3, 3
    neighbors = _oracle_neighbors(f1, f2, f3, row, asset, k)
    assert len(neighbors) >= k
    assert asset not in neighbors
    values = target.iloc[row, neighbors].to_numpy(dtype=float)
    assert peer.iloc[row, asset] == pytest.approx(values.mean())
    assert energy.iloc[row, asset] == pytest.approx(
        np.mean((values - target.iloc[row, asset]) ** 2)
    )


@pytest.mark.parametrize(
    "name", ["cs_knn_peer_mean_ex_self", "cs_knn_graph_dirichlet_energy"]
)
def test_dynamic_knn_missing_feature_and_target_fail_closed(name):
    target, f1, f2, f3 = _panels()
    f2.iloc[2, 0] = np.nan
    target.iloc[3, 1] = np.nan
    result = _call(name, target, f1, f2, f3)
    assert np.isnan(result.iloc[2, 0])
    assert np.isnan(result.iloc[3, 1])
    assert np.isfinite(result.iloc[2, 1:]).any()


@pytest.mark.parametrize("name", NAMES)
def test_dynamic_knn_all_nan_features_return_all_nan(name):
    target, f1, f2, f3 = _panels()
    f1.iloc[:] = np.nan
    assert _call(name, target, f1, f2, f3).isna().all().all()


@pytest.mark.parametrize("name", NAMES)
def test_dynamic_knn_is_prefix_causal_and_deterministic(name):
    panels = _panels(rows=7)
    baseline = _call(name, *panels)
    changed = [frame.copy() for frame in panels]
    for frame in changed:
        frame.iloc[5:] = frame.iloc[5:].to_numpy()[:, ::-1] + 1000
    rerun = _call(name, *changed)
    pd.testing.assert_frame_equal(baseline.iloc[:5], rerun.iloc[:5])
    pd.testing.assert_frame_equal(baseline, _call(name, *panels))


def test_peer_mean_includes_every_boundary_tie_and_excludes_self():
    target, f1, f2, f3 = _panels(rows=1)
    for frame in (f1, f2, f3):
        frame.iloc[0] = 1.0
    result = _call("cs_knn_peer_mean_ex_self", target, f1, f2, f3, k=3)
    values = target.iloc[0].to_numpy(dtype=float)
    for asset in range(len(values)):
        oracle = np.delete(values, asset).mean()
        assert result.iloc[0, asset] == pytest.approx(oracle)


def test_retention_identical_consecutive_graph_is_one():
    target, f1, f2, f3 = _panels(rows=2)
    for frame in (f1, f2, f3):
        frame.iloc[1] = frame.iloc[0]
    result = _call("cs_knn_neighbor_retention", target, f1, f2, f3)
    np.testing.assert_allclose(result.iloc[1], 1.0)


def test_dynamic_knn_metadata():
    for name in NAMES:
        metadata = _op(name).metadata
        assert metadata.name == name
        assert all(feature in metadata.param_names for feature in ("f1", "f2", "f3"))
