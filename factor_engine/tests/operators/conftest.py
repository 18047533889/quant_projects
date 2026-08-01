# -*- coding: utf-8 -*-
"""Cross-cutting operator certification guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def strict_fiscal_parameter_domain_certification_guard() -> None:
    """Fail every operator certification run if fiscal v2 semantics drift.

    This fixture is intentionally located in ``tests/operators`` so the
    primitive certification stage that runs ``test_production_convergence.py``
    cannot certify only default fiscal parameters.
    """
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from backend.production_signature import signature_for

    load_all()

    expected_revision_positions = {
        "period_change": 5,
        "period_average": 4,
        "period_cagr": 6,
        "quarter_from_cumulative": 3,
        "ttm_from_quarterly": 4,
        "ttm_from_cumulative": 3,
        "yoy_by_period": 5,
    }
    for canonical, input_index in expected_revision_positions.items():
        signature = signature_for(canonical)
        assert signature is not None, canonical
        constraints = {row.name: row for row in signature.params}
        revision = constraints.get("revision_policy")
        assert revision is not None, canonical
        assert revision.input_index == input_index, canonical
        assert set(revision.choices) == {"first_available", "latest_available"}, canonical

    values = pd.DataFrame({"A": [10.0, 20.0, 25.0, 40.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q2", "2023Q2", "2023Q3"]})
    expected = {"first_available": 20.0, "latest_available": 25.0}

    pl = pytest.importorskip("polars")
    pl_values = pl.DataFrame({"A": values["A"].tolist()})
    pl_periods = pl.DataFrame({"A": periods["A"].tolist()})
    for policy, target in expected.items():
        pandas_result = OperatorRegistry.get("period_lag", "pandas_numpy").calculate(
            values, periods, 1, policy
        )
        polars_result = OperatorRegistry.get("period_lag", "polars").calculate(
            pl_values, pl_periods, 1, policy
        )
        assert pandas_result.iloc[-1, 0] == target
        assert polars_result["A"][-1] == target
        np.testing.assert_allclose(
            polars_result.to_numpy(),
            pandas_result.to_numpy(dtype=float),
            equal_nan=True,
        )

    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown import compile_plan_to_sql
    from planner.logical_plan import PlanNode

    panel = pd.DataFrame(
        {
            "ts": [1, 2, 3, 4],
            "inst": ["A"] * 4,
            "x": [10.0, 20.0, 25.0, 40.0],
            "pid": ["2023Q1", "2023Q2", "2023Q2", "2023Q3"],
        }
    )
    connection = duckdb.connect()
    connection.register("panel", panel)
    for policy, target in expected.items():
        plan = PlanNode(
            "period_lag",
            [
                PlanNode("column", attrs={"name": "x"}),
                PlanNode("column", attrs={"name": "pid"}),
            ],
            attrs={"periods": 1, "revision_policy": policy},
        )
        compiled = compile_plan_to_sql(
            plan,
            dataset="panel",
            time_column="ts",
            instrument_column="inst",
        )
        assert compiled is not None
        actual = connection.execute(
            compiled.query.replace("{{panel}}", "panel")
        ).df().sort_values("ts")
        assert actual["value"].iloc[-1] == target
