from __future__ import annotations

import pandas as pd


def test_logical_source_wrapper_preserves_native_series_index_order() -> None:
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.planner.logical_plan import PlanNode
    from tests.helpers import InMemorySeriesSource

    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(
        data={
            "x": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx),
            "y": pd.Series([2.0, 2.0, 2.0, 2.0], index=idx),
        }
    )
    plan = PlanNode(
        op="divide",
        inputs=(
            PlanNode(op="column", inputs=(), attrs={"name": "x"}),
            PlanNode(op="column", inputs=(), attrs={"name": "y"}),
        ),
        attrs={},
    )
    out = PandasBackend().execute(plan, ExecutionContext(data_source=source))
    expected = source.data["x"] / source.data["y"]
    pd.testing.assert_series_equal(out, expected, check_names=False)
