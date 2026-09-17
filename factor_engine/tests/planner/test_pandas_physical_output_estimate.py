from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.physical_lowerer import contract_for_plan


def test_small_multiindex_series_physical_bytes_fit_cse_reservation() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=4), ["AAA"]],
        names=["timestamp", "instrument"],
    )
    value = pd.Series(np.arange(4.0), index=index)
    contract = contract_for_plan(
        PlanNode(op="ts_mean", inputs=[], attrs={"window": 2}),
        rows=len(value),
        instruments=1,
        backend="pandas_numpy",
    )
    assert contract.output_bytes >= int(value.memory_usage(deep=True))
    assert contract.admissible_peak_bytes >= int(value.memory_usage(deep=True))


def test_wide_dataframe_physical_bytes_fit_cse_reservation() -> None:
    columns = [f"asset_{i:04d}" for i in range(64)]
    value = pd.DataFrame(
        np.arange(12 * len(columns), dtype=float).reshape(12, len(columns)),
        index=pd.date_range("2024-01-02", periods=12),
        columns=columns,
    )
    contract = contract_for_plan(
        PlanNode(op="rank", inputs=[], attrs={}),
        rows=value.size,
        instruments=len(columns),
        backend="pandas_numpy",
    )
    measured = int(value.memory_usage(index=True, deep=True).sum())
    assert contract.output_bytes >= measured
    assert contract.admissible_peak_bytes >= measured


def test_non_pandas_backends_keep_payload_only_output_estimate() -> None:
    rows = 128
    for backend in ("polars", "cupy", "numpy"):
        contract = contract_for_plan(
            PlanNode(op="abs", inputs=[], attrs={}),
            rows=rows,
            instruments=8,
            backend=backend,
        )
        assert contract.output_bytes == rows * 8
