from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("ascending", [True, False])
def test_cs_bucket_backends_match_average_percentile_oracle_with_ties_and_nulls(
    ascending: bool,
) -> None:
    import polars as pl
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    source = pd.DataFrame(
        [
            [1.0, 1.0, 2.0, 4.0, np.nan],
            [5.0, 5.0, 5.0, 5.0, np.nan],
            [3.0, np.nan, 1.0, 2.0, 2.0],
        ],
        columns=list("ABCDE"),
    )
    bins = 4
    finite = pd.DataFrame(np.isfinite(source), index=source.index, columns=source.columns)
    clean = source.where(finite)
    rank = clean.rank(axis=1, method="average", ascending=ascending, na_option="keep")
    count = finite.sum(axis=1).astype(float)
    rank01 = rank.sub(1).div((count - 1).replace(0, np.nan), axis=0)
    single = count.eq(1)
    rank01.loc[single] = (finite.loc[single].astype(float) * 0.5).where(finite.loc[single])
    expected = (np.floor(rank01 * bins) + 1.0).clip(1, bins).where(finite)

    pandas_op = OperatorRegistry.get("cs_bucket", backend="pandas_numpy")
    polars_op = OperatorRegistry.get("cs_bucket", backend="polars")
    assert pandas_op is not None and polars_op is not None
    pandas_result = pandas_op.calculate(source, bins, ascending)
    polars_result = polars_op.calculate(pl.from_pandas(source), bins, ascending).to_pandas()
    pd.testing.assert_frame_equal(pandas_result, expected)
    pd.testing.assert_frame_equal(polars_result, expected)


def test_cs_bucket_singleton_uses_canonical_percentile_bucket() -> None:
    import polars as pl
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    source = pd.DataFrame({"A": [5.0], "B": [np.nan]})
    expected = pd.DataFrame({"A": [6.0], "B": [np.nan]})
    for backend, value in (
        ("pandas_numpy", source),
        ("polars", pl.from_pandas(source)),
    ):
        result = OperatorRegistry.get("cs_bucket", backend=backend).calculate(value, 10, True)
        if backend == "polars":
            result = result.to_pandas()
        pd.testing.assert_frame_equal(result, expected)
