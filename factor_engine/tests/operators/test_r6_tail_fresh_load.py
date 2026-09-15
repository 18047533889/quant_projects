"""Fresh-process load-all proof for the R6 tail pair and static adjacency."""
from __future__ import annotations


def test_fresh_load_all_tail_pair_and_static_adjacency() -> None:
    import numpy as np
    import pandas as pd
    import factor_engine.cleaned_operators as cleaned

    cleaned.load_all()
    from factor_engine.cleaned_operators.common.static_adjacency import StaticAdjacency
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    cols = list("ABCDE")
    target = pd.DataFrame([[10, 20, 30, 40, 50]], columns=cols, dtype=float)
    feature = pd.DataFrame([[1, 2, 3, 4, 5]], columns=cols, dtype=float)
    moran = OperatorRegistry.get("cs_knn_local_moran", "pandas_numpy").calculate(
        target, feature, feature, feature, k=2
    )
    np.testing.assert_allclose(moran, [[0.5, 0.5, 0.0, 0.5, 0.5]])

    idx = pd.date_range("2024-01-01", periods=2)
    factor = pd.DataFrame([[3, 2, 1], [3, 2, 1]], index=idx, columns=list("ABC"), dtype=float)
    ret = pd.DataFrame([[3, 2, 1], [1, 3, 2]], index=idx, columns=list("ABC"), dtype=float)
    pocket = OperatorRegistry.get("panel_factor_pocket_strength", "pandas_numpy").calculate(
        ret, factor, window=2, min_periods=2, threshold=0.5
    )
    np.testing.assert_allclose(pocket.iloc[1], [0.5, 0.5, 0.0])

    graph = StaticAdjacency(("A", "B"), ((0, 1), (1, 0)))
    graph_out = OperatorRegistry.get("panel_peer_graph_aggregate", "pandas_numpy").calculate(
        pd.DataFrame([[1.0, 2.0]], columns=list("AB")), graph
    )
    np.testing.assert_allclose(graph_out, [[2.0, 1.0]])
