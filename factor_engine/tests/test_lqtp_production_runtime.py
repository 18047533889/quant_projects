from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _loaded_registry():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    ensure_cleaned_loaded()
    return OperatorRegistry


def _analysis(formula: str):
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer
    return Analyzer().lower(parse_expr(formula, surface="lqtp"))


def test_lqtp_exact_aliases_share_canonical_expr() -> None:
    from factor_engine.api.dsl_parser import parse_expr
    cases = {
        "decay_linear(close, 5)": "ts_decay_linear",
        "ts_rank_pct(close, 5)": "ts_rank",
        "ts_ewm_mean(close, 12)": "ts_ema",
        "ts_expanding_rank(close)": "expanding_rank",
        "ts_hump_decay(close, 0.01)": "hump_decay",
        "fp_beta(ret, benchmark_ret, 20)": "rolling_beta_to_market",
    }
    for formula, canonical in cases.items():
        assert parse_expr(formula, surface="lqtp").op == canonical


def test_surface_and_dialect_are_orthogonal() -> None:
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr
    with pytest.raises(DSLParseError):
        parse_expr("safe_log(close)", surface="daily", dialect="native")
    assert parse_expr("safe_log(close)", surface="daily", dialect="lqtp").op == "where"
    with pytest.raises(ValueError, match="unsupported LQTP dialect_version"):
        parse_expr("safe_log(close)", surface="daily", dialect="lqtp", dialect_version="1900-01-01")


def test_safe_log_and_momentum_are_macros_not_duplicate_kernels() -> None:
    from factor_engine.api.dsl_parser import parse_expr
    assert parse_expr("safe_log(close)", surface="lqtp").op == "where"
    assert parse_expr("momentum(close, 20)", surface="lqtp").op == "ts_delta"
    registry = _loaded_registry()
    assert registry.get("safe_log_null") is None
    assert registry.get("momentum") is None


def test_lqtp_sma_keeps_two_distinct_semantics() -> None:
    from factor_engine.api.dsl_parser import parse_expr
    assert parse_expr("sma(close, 7)", surface="lqtp").op == "ts_mean"
    assert parse_expr("sma(close, 7, 2)", surface="lqtp").op == "ts_sma_cn"


def test_lqtp_bare_true_range_expands_before_field_analysis() -> None:
    assert _analysis("true_range + close").referenced_columns == {"high", "low", "close"}


def test_parameterized_datatable_attribute_becomes_opaque_source_ref() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis('BenchmarkIndexDailyBar(index="000985.SH").Close')
    assert len(analysis.referenced_columns) == 1
    name = next(iter(analysis.referenced_columns))
    spec = decode_source_ref(name)
    assert spec is not None
    assert spec.table == "BenchmarkIndexDailyBar"
    assert spec.field == "Close"
    assert spec.params_dict() == {"index": "000985.SH"}


def test_legacy_benchmark_index_attribute_is_supported() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis('benchmark_index(index="000985.SH").Close')
    spec = decode_source_ref(next(iter(analysis.referenced_columns)))
    assert spec is not None and spec.table == "BenchmarkIndexDailyBar"


def test_real_turnover_rate_uses_turnover_base_source_ref() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("real_turnover_rate()")
    assert "volume" in analysis.referenced_columns
    refs = [decode_source_ref(name) for name in analysis.referenced_columns]
    refs = [x for x in refs if x is not None]
    assert len(refs) == 1
    assert refs[0].table == "TurnoverBaseDaily"
    assert refs[0].field == "TurnoverBase"


def test_minute_helpers_accept_legacy_daily_field_shorthand() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis("minute_bar(close, 5, 0)")
    spec = decode_source_ref(next(iter(analysis.referenced_columns)))
    assert spec is not None
    assert spec.table == "StockMinuteBar"
    assert spec.field == "Close"
    assert spec.transform == "minute_bar"
    assert spec.transform_params_dict() == {"index": 0, "period": 5}


def test_financial_asof_and_lag_are_source_level_transforms() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    for formula, transform in [
        ("asof(StockIncome.TotalOperatingRevenue)", "financial_asof"),
        ("financial_lag(StockIncome.TotalOperatingRevenue, 2)", "financial_lag"),
        ("lag(StockIncome.TotalOperatingRevenue, 2)", "financial_lag"),
    ]:
        analysis = _analysis(formula)
        spec = decode_source_ref(next(iter(analysis.referenced_columns)))
        assert spec is not None and spec.table == "StockIncome"
        assert spec.transform == transform


def test_default_market_risk_helpers_inject_benchmark_source() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    from factor_engine.api.dsl_parser import parse_expr
    for formula in [
        "rolling_beta_to_market(ret, 20)",
        "tail_beta(ret, 20, 0.05)",
        "residual_momentum_capm(ret, 20)",
    ]:
        expr = parse_expr(formula, surface="lqtp")
        analysis = _analysis(formula)
        refs = [decode_source_ref(name) for name in analysis.referenced_columns]
        assert any(ref is not None and ref.table == "BenchmarkIndexDailyBar" and ref.params_dict().get("index") == "000985.SH" for ref in refs)
        assert expr is not None


def test_historical_risk_macros_and_cvar_kernel() -> None:
    from factor_engine.api.dsl_parser import parse_expr
    assert parse_expr("historical_var(ret, 20, 0.05)", surface="lqtp").op == "neg"
    assert parse_expr("historical_cvar(ret, 20, 0.05)", surface="lqtp").op == "lqtp_historical_cvar"
    registry = _loaded_registry()
    op = registry.get("lqtp_historical_cvar", "pandas_numpy")
    panel = pd.DataFrame({"A": [-0.10, -0.05, 0.0, 0.05]})
    out = op.calculate(panel, 4, 0.5)
    assert np.isclose(out.iloc[-1, 0], 0.075)


def test_intermediate_is_versioned_source_ref() -> None:
    from factor_engine.api.source_ref import decode_source_ref
    analysis = _analysis('intermediate("turnover_zscore", 1)')
    spec = decode_source_ref(next(iter(analysis.referenced_columns)))
    assert spec is not None and spec.table == "Intermediate"
    assert spec.params_dict() == {"name": "turnover_zscore", "version": 1}


def test_ambiguous_lqtp_names_fail_instead_of_guessing() -> None:
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr
    for formula in ("ts_sumac(close, 20)", "ts_regression_slope_sequence(close, 20)"):
        with pytest.raises(DSLParseError, match="recognized"):
            parse_expr(formula, surface="lqtp")


def test_chinese_sma_is_not_simple_rolling_mean() -> None:
    op = _loaded_registry().get("ts_sma_cn", "pandas_numpy")
    panel = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = op.calculate(panel, 3, 1)
    np.testing.assert_allclose(out["A"].to_numpy(), [1.0, 4/3, 17/9, 70/27], rtol=1e-12, atol=1e-12)


def test_where_scalar_branches_broadcast_to_panel() -> None:
    op = _loaded_registry().get("where", "pandas_numpy")
    cond = pd.DataFrame({"A": [1.0, 0.0, np.nan], "B": [0.0, 1.0, 1.0]})
    out = op.calculate(cond, 1, 0)
    np.testing.assert_allclose(out.to_numpy(dtype=float), [[1,0],[0,1],[np.nan,1]], equal_nan=True)


def test_fastpath_output_is_canonicalized_before_execution() -> None:
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.optimizer import Optimizer
    x = PlanNode(op="column", inputs=[], attrs={"name":"close"})
    root = PlanNode(op="divide", inputs=[PlanNode(op="subtract", inputs=[x, PlanNode(op="ts_mean",inputs=[x],attrs={"window":20})],attrs={}), PlanNode(op="ts_std",inputs=[x],attrs={"window":20})], attrs={})
    out = Optimizer().optimize(root)
    assert out.op == "ts_zscore" and out.attrs["window"] == 20 and "d" not in out.attrs


def test_research_tools_and_factor_research_both_have_entrypoints() -> None:
    registry = _loaded_registry()
    from research_tools.runtime import ResearchToolRuntime
    selection = ResearchToolRuntime.select("bartlett_test")
    assert selection.backend in selection.available_backends
    for name in ("ts_sma_cn","lqtp_historical_cvar","idio_vol","rank_corr"):
        assert registry.get(name, "pandas_numpy") is not None


def test_pandas_first_production_is_independent_from_duckdb() -> None:
    registry = _loaded_registry()
    from factor_engine.backend.backend_certification import backend_certification
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec
    assert registry.get("ts_quantile", "pandas_numpy") is not None
    spec = build_operator_spec("ts_quantile")
    assert spec is not None and spec.allow_in_production is True
    assert backend_certification("ts_quantile").pandas_numpy == "production"


def test_recursive_sma_is_production_hardened() -> None:
    _loaded_registry()
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec
    spec = build_operator_spec("ts_sma_cn")
    assert spec is not None and spec.status == "production" and spec.allow_in_production is True
