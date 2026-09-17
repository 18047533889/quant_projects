import pytest

from factor_engine.tools.catalog_r13_nested_recipes import migrate_r13_nested_formula


FAILED_TRANSPORT_FORMULAS = (
    "ts_quantile_transport_slope(rank(ts_cross_quantilogram(ret, turnover_ratio, window=120, target_q=0.1, source_q=0.1, lag=1, target_side='upper', source_side='upper', fixed_threshold=False)), window=60)",
    "ts_quantile_transport_slope(rank(cs_rank_copula_mi(subtract(exp(ts_sum(log(add(ret, 1.0)), 20)), 1.0), turnover_ratio, grid=8)), window=60)",
    "ts_quantile_transport_curvature(rank(cs_rank_copula_mi(subtract(exp(ts_sum(log(add(ret, 1.0)), 20)), 1.0), turnover_ratio, grid=8)), window=60)",
    "ts_quantile_transport_slope(rank(ts_extremogram(ret, window=120, quantile=0.1, lag=1, side='upper', fixed_threshold=False)), window=60)",
    "ts_quantile_transport_curvature(rank(ts_extremogram(ret, window=120, quantile=0.1, lag=1, side='upper', fixed_threshold=False)), window=60)",
    "ts_quantile_transport_slope(rank(ts_quantile_crossing_spectral_concentration(ret, window=120, quantile=0.1, side='upper', fixed_threshold=False)), window=60)",
)

EXPLICIT_SPLIT_LOGIC = (
    "分位传输设计明确采用 recent_window=20, old_window=40 的相邻样本窗口"
)


@pytest.mark.parametrize("formula", FAILED_TRANSPORT_FORMULAS)
def test_reviewed_transport_repair_really_parses_and_lowers(formula):
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    result = migrate_r13_nested_formula(
        formula, logic=EXPLICIT_SPLIT_LOGIC, enabled=True,
    )
    assert "window=60" not in result.formula
    assert "recent_window=20, old_window=40" in result.formula
    assert len(result.changes) == 1
    lowered = Analyzer().lower(
        DSLParser(surface="compat_research").parse(result.formula),
    )
    assert lowered.ir.op in {
        "ts_quantile_transport_slope", "ts_quantile_transport_curvature",
    }
    from benchmarks.benchmark_run_many_streaming_20260906 import Source
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine

    engine = FactorEngine(PandasBackend(), Source(__import__("pandas").Series(dtype=float)))
    engine.compile(Factor(name="transport", expr=DSLParser(surface="compat_research").parse(result.formula)))
    assert migrate_r13_nested_formula(
        result.formula, logic=EXPLICIT_SPLIT_LOGIC, enabled=True,
    ).changes == ()


@pytest.mark.parametrize(
    ("formula", "logic"),
    [
        ("ts_quantile_transport_slope(ret, window=61)", "分位传输斜率"),
        ("ts_quantile_transport_slope(ret, window=60)", "generic state"),
        ("ts_quantile_transport_slope(ret, window=60)", "分位传输斜率"),
        ("ts_quantile_transport_slope(ret, window=60)", "recent_window=20 only"),
        ("ts_quantile_transport_slope(ret, window=60)", "old_window=40 only"),
        ("ts_last_pivot_low(low, 3, 3)", "pivot support"),
        ("ts_markov_transition_surprisal(ret, window=60, min_periods=3)", "markov"),
    ],
)
def test_ambiguous_residuals_fail_closed(formula, logic):
    result = migrate_r13_nested_formula(formula, logic=logic, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


def test_canonical_20_40_transport_matches_quantile_ols_golden():
    import numpy as np
    import pandas as pd

    import factor_engine.cleaned_operators.alpha_language_distribution  # noqa: F401
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    values = np.r_[np.linspace(-2.0, 1.0, 40), np.linspace(-1.0, 3.0, 20)]
    panel = pd.DataFrame({"A": values})
    q = np.linspace(0.1, 0.9, 9)
    z = q - 0.5
    design = np.column_stack([np.ones_like(z), z, z * z])
    delta = np.quantile(values[-20:], q) - np.quantile(values[:40], q)
    expected = np.linalg.lstsq(design, delta, rcond=None)[0]
    slope = OperatorRegistry.get(
        "ts_quantile_transport_slope", "pandas_numpy",
    ).calculate(panel, recent_window=20, old_window=40)
    curvature = OperatorRegistry.get(
        "ts_quantile_transport_curvature", "pandas_numpy",
    ).calculate(panel, recent_window=20, old_window=40)
    assert slope.iloc[-1, 0] == pytest.approx(expected[1])
    assert curvature.iloc[-1, 0] == pytest.approx(expected[2])


def test_unicode_before_reviewed_call_preserves_exact_source_text():
    formula = (
        "add(field('中文字段', table='Demo'), "
        "ts_quantile_transport_slope(ret, window=60))"
    )
    result = migrate_r13_nested_formula(formula, logic=EXPLICIT_SPLIT_LOGIC, enabled=True)
    assert result.formula == (
        "add(field('中文字段', table='Demo'), "
        "ts_quantile_transport_slope(ret, recent_window=20, old_window=40))"
    )


def test_nested_matching_transport_calls_fail_closed_as_one_formula():
    formula = (
        "ts_quantile_transport_slope("
        "ts_quantile_transport_curvature(ret, window=60), window=60)"
    )
    result = migrate_r13_nested_formula(formula, logic=EXPLICIT_SPLIT_LOGIC, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


@pytest.mark.parametrize(
    "formula",
    [
        "ts_quantile_transport_slope(ret, window=60, window=60)",
        "ts_quantile_transport_slope(ret, window=60, **opts)",
    ],
)
def test_duplicate_or_expanded_keywords_fail_closed(formula):
    result = migrate_r13_nested_formula(formula, logic=EXPLICIT_SPLIT_LOGIC, enabled=True)
    assert result.formula == formula
    assert result.changes == ()


@pytest.mark.parametrize("formula", [None, b"x", 1])
def test_formula_requires_string(formula):
    with pytest.raises(TypeError, match="formula must be a string"):
        migrate_r13_nested_formula(formula, enabled=False)


def test_utf8_input_budget_counts_bytes_before_parse():
    with pytest.raises(ValueError, match="input budget"):
        migrate_r13_nested_formula("'" + "界" * 22_000 + "'", enabled=False)
