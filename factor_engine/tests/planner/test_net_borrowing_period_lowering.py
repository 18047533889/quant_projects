import numpy as np
import pandas as pd
import polars as pl

from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.optimizer import Optimizer
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.sources.datasource import DataSource


class _NoReadSource(DataSource):
    def load_column(self, name):
        raise AssertionError("compile must not read source data")


def _column(name):
    return PlanNode(op="column", attrs={"name": name}, inputs=[])


def test_five_input_period_alignment_survives_real_compile():
    formula = (
        'fin_net_borrowing_cashflow('
        'field("cash_from_borrowing", table="StockCashFlow"),'
        'field("cash_from_bonds_issue", table="StockCashFlow"),'
        'field("borrowing_repayment", table="StockCashFlow"),'
        'field("total_assets", table="StockBalance"),'
        'field("report_period_end_date", table="StockBalance"))'
    )
    engine = FactorEngine(PandasBackend(), _NoReadSource(), run_mode="research")
    plan, _analysis = engine.compile(Factor(
        name="net_borrowing_period", expr=parse_expr(formula, surface="compat_research"),
        source_expr=formula, surface="compat_research",
    ))
    assert plan.op == "fin_net_borrowing_cashflow"
    assert len(plan.inputs) == 5


def test_four_input_legacy_shape_still_lowers_to_safe_division():
    plan = PlanNode(
        op="fin_net_borrowing_cashflow",
        inputs=[_column(name) for name in ("borrow", "bonds", "repay", "assets")],
        attrs={},
    )
    optimized = Optimizer().optimize(plan)
    assert optimized.op == "safe_div_null"


def test_canonical_five_input_value_uses_period_only_for_alignment():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    frame = lambda values: pd.DataFrame({"A": values}, index=index)
    pandas_op = OperatorRegistry.get("fin_net_borrowing_cashflow", "pandas_numpy")
    pandas_actual = pandas_op._calculate_series(
        frame([10.0, 20.0, 1.0, np.nan]),
        frame([5.0, 7.0, 1.0, 2.0]),
        frame([3.0, 4.0, 1.0, 1.0]),
        frame([100.0, 200.0, 0.0, 10.0]),
        frame(["2023Q1", "2023Q2", "2023Q3", "2023Q4"]),
    )
    np.testing.assert_allclose(
        pandas_actual["A"], [0.12, 0.115, np.nan, np.nan], equal_nan=True
    )

    polars_frame = lambda values: pl.DataFrame({"A": values})
    polars_op = OperatorRegistry.get("fin_net_borrowing_cashflow", "polars")
    polars_actual = polars_op._calculate_series(
        polars_frame([10.0, 20.0, 1.0, None]),
        polars_frame([5.0, 7.0, 1.0, 2.0]),
        polars_frame([3.0, 4.0, 1.0, 1.0]),
        polars_frame([100.0, 200.0, 0.0, 10.0]),
        polars_frame(["2023Q1", "2023Q2", "2023Q3", "2023Q4"]),
    )
    np.testing.assert_allclose(
        polars_actual["A"].to_numpy(),
        [0.12, 0.115, np.nan, np.nan],
        equal_nan=True,
    )
