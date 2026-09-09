import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_polars_peer_matches_authoritative_cpu_kernel_on_extreme_rows():
    from factor_engine.cleaned_operators.cross_section.peer_ops import (
        _group_peer_beta_deviation as cpu,
    )
    from factor_engine.cleaned_operators.cross_section.polars_peer import (
        _group_peer_beta_deviation as polars,
    )

    rows = np.array(
        [
            [1e308, 1e308, -1e308, -1e308, 1.0],
            [1e16, 1.0, 1.0, 1.0, 1.0],
            [1.0, np.nan, 3.0, 4.0, 5.0],
        ]
    )
    weights = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, np.inf, 2.0, 3.0],
        ]
    )
    groups = np.array([["g"] * 5, ["g"] * 5, ["g", "g", "g", "g", None]], dtype=object)
    columns = list("ABCDE")
    expected = cpu(
        pd.DataFrame(rows, columns=columns),
        pd.DataFrame(groups, columns=columns),
        pd.DataFrame(weights, columns=columns),
    )
    actual = polars(
        pl.DataFrame(dict(zip(columns, rows.T))),
        pl.DataFrame({name: groups[:, i].tolist() for i, name in enumerate(columns)}),
        pl.DataFrame(dict(zip(columns, weights.T))),
    )
    np.testing.assert_allclose(actual.select(columns).to_numpy(), expected.to_numpy(), equal_nan=True)


def test_polars_peer_preserves_metadata_columns():
    from factor_engine.cleaned_operators.cross_section.polars_peer import (
        _group_peer_beta_deviation,
    )

    beta = pl.DataFrame({"date": [1], "A": [1.0], "B": [2.0]})
    group = pl.DataFrame({"date": [1], "A": ["g"], "B": ["g"]})
    weight = pl.DataFrame({"date": [1], "A": [1.0], "B": [1.0]})
    out = _group_peer_beta_deviation(beta, group, weight)
    assert out["date"].to_list() == [1]
    np.testing.assert_allclose(out.select(["A", "B"]).to_numpy(), [[-1.0, 1.0]])
    with pytest.raises(ValueError):
        _group_peer_beta_deviation(beta, group.select(["date", "A"]), weight)
    with pytest.raises(ValueError, match="requires a wide panel"):
        _group_peer_beta_deviation(
            beta.with_columns(pl.lit("panel").alias("stock_code")),
            group.with_columns(pl.lit("panel").alias("stock_code")),
            weight.with_columns(pl.lit("panel").alias("stock_code")),
        )
