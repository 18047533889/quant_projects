"""Independent rank/distance oracle and physical graph-lifetime checks."""
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata


def _module():
    from factor_engine.cleaned_operators import dynamic_knn
    return dynamic_knn


def _oracle(features, k, lag):
    graphs = []
    for row in features:
        ranks = np.full_like(row, np.nan)
        for j in range(row.shape[1]):
            valid = np.isfinite(row[:, j])
            if valid.sum() >= 2:
                ranks[valid, j] = (rankdata(row[valid, j]) - 0.5) / valid.sum()
        active = np.flatnonzero(np.isfinite(ranks).all(axis=1))
        graph = [set() for _ in row]
        if active.size > k:
            for i in active:
                distances = {
                    # Preserve the declared binary64 distance arithmetic:
                    # BLAS norm can round boundary ties differently.
                    int(j): float(np.sqrt(np.sum((ranks[i] - ranks[j]) ** 2)))
                    for j in active if j != i
                }
                radius = sorted(distances.values())[k - 1]
                graph[i] = {j for j, distance in distances.items() if distance <= radius}
        graphs.append(graph)
    result = np.full(features.shape[:2], np.nan)
    for t in range(lag, len(features)):
        for i, current in enumerate(graphs[t]):
            previous = graphs[t - lag][i]
            if current and previous:
                result[t, i] = len(current & previous) / len(current | previous)
    return result


@pytest.mark.parametrize("lag", [1, 3, 30])
def test_ring_matches_independent_oracle_with_gaps_and_ties(lag):
    features = np.random.default_rng(42).integers(0, 6, (24, 15, 3)).astype(float)
    features[5:9] = np.nan
    features[13, 2:5, 1] = np.nan
    actual = _module()._retention_series(features, 3, lag)
    np.testing.assert_allclose(actual, _oracle(features, 3, lag), equal_nan=True)


def test_all_boundary_ties_preserved_and_only_lag_graphs_retained():
    module = _module()
    features = np.ones((40, 64, 3))
    peak_graphs = 0
    peak_payload = 0

    def observe(frame, event, arg):
        nonlocal peak_graphs, peak_payload
        if frame.f_code is module._retention_series.__code__:
            graphs = frame.f_locals.get("neighbor_sets", ())
            peak_graphs = max(peak_graphs, len(graphs))
            peak_payload = max(
                peak_payload, sum(array.nbytes for graph in graphs for array in graph)
            )
        return observe

    previous_trace = sys.gettrace()
    try:
        sys.settrace(observe)
        actual = module._retention_series(features, 3, 1)
    finally:
        sys.settrace(previous_trace)
    assert np.isnan(actual[0]).all()
    np.testing.assert_array_equal(actual[1:], 1.0)
    assert peak_graphs <= 2
    assert peak_payload == 2 * 64 * 63 * np.dtype(int).itemsize


def test_public_winner_prefix_column_identity_and_warmup():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    features = np.random.default_rng(71).normal(size=(20, 12, 3))
    frames = [pd.DataFrame(features[:, :, j], columns=list("ABCDEFGHIJKL")) for j in range(3)]
    op = OperatorRegistry.get("cs_knn_neighbor_retention", "pandas_numpy")
    actual = op.calculate(*frames, k=3, lag=3)
    np.testing.assert_allclose(actual, _oracle(features, 3, 3), equal_nan=True)
    prefix = op.calculate(*(frame.iloc[:14] for frame in frames), k=3, lag=3)
    pd.testing.assert_frame_equal(prefix, actual.iloc[:14])
    chunk = op.calculate(*(frame.iloc[7:] for frame in frames), k=3, lag=3)
    pd.testing.assert_frame_equal(chunk.iloc[3:], actual.iloc[10:])
    reverse = op.calculate(*(frame.iloc[:, ::-1] for frame in frames), k=3, lag=3)
    pd.testing.assert_frame_equal(reverse.iloc[:, ::-1], actual)
    with pytest.raises(ValueError):
        op.calculate(frames[0], frames[1].iloc[:, ::-1], frames[2], k=3, lag=3)
