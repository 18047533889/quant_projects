"""Native list reductions must also pass DataAccess's real SQL admission gate."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from data_access.read.sql_escape import _check_sql_function_allowlist
from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql
from factor_engine.backend.sql_pushdown.plan_fixtures import column, literal
from factor_engine.planner.logical_plan import PlanNode

CASES = [
    ("ts_max_drawdown", {"window": 12, "min_periods": 4}),
    ("ts_nth_value", {"window": 12, "n": 2, "min_periods": 4}),
    ("ts_dense_rank", {"window": 12, "min_periods": 4}),
    ("ts_percent_rank", {"window": 12, "min_periods": 4}),
    ("ts_mad", {"window": 12, "min_periods": 4}),
    ("ts_kurt", {"window": 12}),
    ("ts_trimmed_mean", {"window": 12, "min_periods": 4, "trim_ratio": 0.2}),
    *[(name, {"window": 12, "k": 3, "min_periods": 4})
      for name in ("ts_topk_mean", "ts_topk_sum", "ts_topk_std",
                   "ts_bottomk_mean", "ts_bottomk_sum", "ts_bottomk_std")],
    # Four related tail templates are not in SQL_IMPLEMENTED_CANONICALS;
    # keep their routing fail-closed (tested separately below).
    ("val1_earnings_yield_slope", {"window": 12, "min_periods": 5}),
    ("vax_ret_per_liquidity_unit", {"window": 12, "min_periods": 5}),
]


@pytest.mark.parametrize("canonical,attrs", CASES, ids=[case[0] for case in CASES])
def test_emitted_list_reductions_are_admitted_and_execute(canonical, attrs):
    inputs = ([column("r"), column("amount")] if canonical == "vax_ret_per_liquidity_unit"
              else [column("x")])
    plan = PlanNode(op=canonical, inputs=inputs, attrs=attrs)
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst",
    )
    assert compiled is not None
    # Do not whitelist macros, table functions or weaken the production gate.
    _check_sql_function_allowlist(compiled.query, strict=True)
    t = np.arange(32, dtype=float)
    frame = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=32),
        "inst": ["A"] * 32,
        "x": 100 + t + 3 * np.sin(t),
        "r": 0.01 * np.sin(t),
        "amount": 1000 + 7 * t,
    })
    frame.loc[10, "x"] = np.nan
    with duckdb.connect(":memory:") as connection:
        connection.register("panel", frame)
        result = connection.execute(compiled.query.replace("{{panel}}", "panel")).df()
    assert len(result) == len(frame)
    assert np.isfinite(result["value"]).any(), canonical
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    reference = OperatorRegistry.get(canonical, backend="pandas_numpy")
    panels = ([pd.DataFrame({"A": frame[name].to_numpy()}) for name in ("r", "amount")]
              if canonical == "vax_ret_per_liquidity_unit"
              else [pd.DataFrame({"A": frame["x"].to_numpy()})])
    if canonical in {"ts_dense_rank", "ts_percent_rank"}:
        # These SQL-only names have no Pandas registry slot. Verify their
        # documented window-local rank definitions against an independent
        # oracle rather than inventing a reference backend certification.
        values = frame["x"].to_numpy()
        expected = np.full(len(values), np.nan)
        for i, current in enumerate(values):
            tail = values[max(0, i + 1 - attrs["window"]):i + 1]
            tail = tail[np.isfinite(tail)]
            if not np.isfinite(current) or len(tail) < attrs["min_periods"]:
                continue
            smaller = tail[tail < current]
            expected[i] = (1 + len(np.unique(smaller)) if canonical == "ts_dense_rank"
                           else len(smaller) / (len(tail) - 1) if len(tail) > 1 else 0.0)
    else:
        assert reference is not None, canonical
        expected = reference.calculate(*panels, **attrs)["A"].to_numpy()
    np.testing.assert_allclose(
        result["value"].to_numpy(), expected, rtol=1e-8, atol=1e-10, equal_nan=True,
    )


def test_list_macro_still_denied_by_dataaccess():
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        _check_sql_function_allowlist("SELECT list_max([1, 2])", strict=True)


@pytest.mark.parametrize("canonical", [
    "ts_expected_shortfall", "ts_quantile_skew", "ts_quantile_kurtosis", "ts_tail_ratio",
])
def test_unregistered_tail_templates_remain_fail_closed(canonical):
    plan = PlanNode(op=canonical, inputs=[column("x")], attrs={"window": 12})
    assert compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst",
    ) is None
