from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _loaded_registry():
    from backend.cleaned_bridge import ensure_cleaned_loaded
    from cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    return OperatorRegistry


def test_lqtp_exact_aliases_share_canonical_expr() -> None:
    from api.dsl_parser import parse_expr

    cases = {
        "decay_linear(close, 5)": "ts_decay_linear",
        "ts_rank_pct(close, 5)": "ts_rank",
        "ts_ewm_mean(close, 12)": "ts_ema",
        "ts_expanding_rank(close)": "expanding_rank",
        "ts_hump_decay(close, 0.01)": "hump_decay",
        "fp_beta(ret, benchmark_ret, 20)": "rolling_beta_to_market",
    }
    for formula, canonical in cases.items():
        expr = parse_expr(formula, surface="lqtp")
        assert getattr(expr, "op", None) == canonical


def test_safe_log_and_momentum_are_macros_not_duplicate_kernels() -> None:
    from api.dsl_parser import parse_expr

    safe_log = parse_expr("safe_log(close)", surface="lqtp")
    momentum = parse_expr("momentum(close, 20)", surface="lqtp")
    assert safe_log.op == "where"
    assert momentum.op == "ts_delta"
    registry = _loaded_registry()
    assert registry.get("safe_log_null") is None
    assert registry.get("momentum") is None


def test_lqtp_sma_keeps_two_distinct_semantics() -> None:
    from api.dsl_parser import parse_expr

    simple = parse_expr("sma(close, 7)", surface="lqtp")
    recursive = parse_expr("sma(close, 7, 2)", surface="lqtp")
    assert simple.op == "ts_mean"
    assert recursive.op == "ts_sma_cn"


def test_lqtp_bare_true_range_expands_before_field_analysis() -> None:
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer

    analysis = Analyzer().lower(parse_expr("true_range + close", surface="lqtp"))
    assert analysis.referenced_columns == {"high", "low", "close"}


def test_real_turnover_rate_expands_to_explicit_source_inputs() -> None:
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer

    expr = parse_expr("real_turnover_rate()", surface="lqtp")
    assert expr.op == "real_turnover_rate"
    analysis = Analyzer().lower(expr)
    assert analysis.referenced_columns == {"volume", "turnover_base"}


def test_minute_bar_reports_missing_frequency_source_instead_of_unknown_function() -> None:
    from api.dsl_parser import DSLParseError, parse_expr

    with pytest.raises(DSLParseError, match="minute-bar source"):
        parse_expr("minute_bar(close, 5, 0)", surface="lqtp")


def test_ambiguous_lqtp_names_fail_instead_of_guessing() -> None:
    from api.dsl_parser import DSLParseError, parse_expr

    with pytest.raises(DSLParseError, match="exact semantic definition"):
        parse_expr("ts_sumac(close, 20)", surface="lqtp")
    with pytest.raises(DSLParseError, match="exact semantic definition"):
        parse_expr("ts_regression_slope_sequence(close, 20)", surface="lqtp")


def test_chinese_sma_is_not_simple_rolling_mean() -> None:
    registry = _loaded_registry()
    op = registry.get("ts_sma_cn", "pandas_numpy")
    panel = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = op.calculate(panel, 3, 1)
    expected = [1.0, 4.0 / 3.0, 17.0 / 9.0, 70.0 / 27.0]
    np.testing.assert_allclose(out["A"].to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_where_scalar_branches_broadcast_to_panel() -> None:
    registry = _loaded_registry()
    op = registry.get("where", "pandas_numpy")
    cond = pd.DataFrame({"A": [1.0, 0.0, np.nan], "B": [0.0, 1.0, 1.0]})
    out = op.calculate(cond, 1, 0)
    expected = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, 1.0]])
    np.testing.assert_allclose(out.to_numpy(dtype=float), expected, equal_nan=True)


def test_fastpath_output_is_canonicalized_before_execution() -> None:
    from planner.logical_plan import PlanNode
    from planner.optimizer import Optimizer

    x = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    mean = PlanNode(op="ts_mean", inputs=[x], attrs={"window": 20})
    std = PlanNode(op="ts_std", inputs=[x], attrs={"window": 20})
    numerator = PlanNode(op="subtract", inputs=[x, mean], attrs={})
    root = PlanNode(op="divide", inputs=[numerator, std], attrs={})
    out = Optimizer().optimize(root)
    assert out.op == "ts_zscore"
    assert out.attrs["window"] == 20
    assert "d" not in out.attrs


def test_data_access_preflight_blocks_operator_and_derived_field_names() -> None:
    from storage.sources.data_access_source import (
        DataAccessColumnPreflightError,
        DataAccessSource,
        MissingDataDependencyError,
    )

    source = DataAccessSource(dataset="ashare_stock_daily")
    with pytest.raises(MissingDataDependencyError, match="derived/source-backed"):
        source._resolve_columns(["ebitda_approx"])
    with pytest.raises(MissingDataDependencyError, match="derived/source-backed"):
        source._resolve_columns(["turnover_base"])
    with pytest.raises(DataAccessColumnPreflightError, match="operator name"):
        source._resolve_columns(["true_range"])


def test_research_tools_have_supported_runtime_entrypoint() -> None:
    _loaded_registry()
    from research_tools.runtime import ResearchToolRuntime

    selection = ResearchToolRuntime.select("bartlett_test")
    assert selection.canonical == "bartlett_test"
    assert selection.backend in selection.available_backends


def test_factor_like_research_operator_remains_composable() -> None:
    from api.dsl_parser import parse_expr

    expr = parse_expr("ts_expanding_rank(close)", surface="lqtp")
    assert expr.op == "expanding_rank"


def test_pandas_first_production_is_independent_from_duckdb() -> None:
    _loaded_registry()
    from backend.backend_certification import backend_certification
    from cleaned_operators.operator_spec import build_operator_spec

    spec = build_operator_spec("ts_quantile")
    assert spec is not None
    assert spec.allow_in_production is True
    cert = backend_certification("ts_quantile")
    assert cert.pandas_numpy == "production"
    assert cert.has_any_production_backend is True


def test_pandas_first_production_parameters_are_bounded() -> None:
    _loaded_registry()
    from backend.pandas_first_signature import check_pandas_first_plan_signatures
    from planner.logical_plan import PlanNode

    x = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    valid = PlanNode(
        op="ts_quantile",
        inputs=[
            x,
            PlanNode(op="literal", inputs=[], attrs={"value": 20}),
            PlanNode(op="literal", inputs=[], attrs={"value": 0.75}),
        ],
        attrs={},
    )
    invalid = PlanNode(
        op="ts_quantile",
        inputs=[
            x,
            PlanNode(op="literal", inputs=[], attrs={"value": 20}),
            PlanNode(op="literal", inputs=[], attrs={"value": 1.5}),
        ],
        attrs={},
    )
    assert check_pandas_first_plan_signatures(valid) == []
    assert any("[0, 1]" in message for message in check_pandas_first_plan_signatures(invalid))


def test_production_source_preflight_accepts_compatible_macros_then_checks_plan() -> None:
    from api.dsl_parser import parse_factor
    from runtime.production_policy import assert_production_factors

    factor = parse_factor("safe_log(close) + momentum(close, 20)", surface="lqtp")
    assert_production_factors([factor], mode="production")


def test_recursive_sma_stays_research_until_checkpointed() -> None:
    _loaded_registry()
    from cleaned_operators.operator_spec import build_operator_spec

    spec = build_operator_spec("ts_sma_cn")
    assert spec is not None
    assert spec.status == "research"
    assert spec.allow_in_production is False
