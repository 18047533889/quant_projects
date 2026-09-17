import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r19_type_recipes import migrate_formula


@pytest.mark.parametrize(
    ("source", "expected", "change"),
    [
        ("ts_spectral_entropy(volume, window=60)",
         "ts_activity_spectral_entropy(volume, window=60)", "TYPE_REPAIR"),
        ("ts_spectral_entropy(close, window=60)",
         "ts_detrended_level_spectral_entropy(close, window=60)", "TYPE_REPAIR"),
        ("rank(ts_spectral_entropy(add(foo, bar), window=60))",
         "rank(ts_spectral_flatness(add(foo, bar), window=60))", "NON_EQUIVALENT"),
        ("event_historical_response_mean(ashare_limit_up_touch(high, high_limit, 0.005), ret)",
         "event_historical_response_mean(ret, ashare_limit_up_touch(high, high_limit, 0.005))",
         "TYPE_REPAIR"),
    ],
)
def test_exact_type_repairs(source, expected, change):
    formula, changes = migrate_formula(source, "reviewed")
    assert formula == expected
    assert len(changes) == 1
    assert change in changes[0]
    assert migrate_formula(formula) == (formula, [])


def test_return_entropy_and_correct_event_order_are_unchanged():
    for formula in (
        "ts_spectral_entropy(ret, window=60)",
        "event_historical_response_mean(ret, gt(abs(ret), 0.02))",
        "event_historical_response_mean(capital_change_magnitude(TotalCapital), ret)",
    ):
        assert migrate_formula(formula) == (formula, [])


def test_rewrites_parse_and_lower_with_real_contracts():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.ir.analyzer import Analyzer

    cases = (
        "ts_spectral_entropy(volume, window=60)",
        "ts_spectral_entropy(add(foo, bar), window=60)",
    )
    for source in cases:
        formula, _ = migrate_formula(source)
        Analyzer().lower(DSLParser(surface="compat_research").parse(formula))


def test_spectral_redesign_keeps_noise_direction_and_original_input():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    rng = np.random.default_rng(19)
    n = 300
    idx = pd.date_range("2025-01-01", periods=n)
    structured = pd.DataFrame({"A": np.sin(np.arange(n) * 2 * np.pi / 12)}, index=idx)
    noise = pd.DataFrame({"A": rng.normal(size=n)}, index=idx)
    op = OperatorRegistry.get("ts_spectral_flatness", backend="pandas_numpy")
    structured_value = op.calculate(structured, window=60).iloc[-100:, 0].mean()
    noise_value = op.calculate(noise, window=60).iloc[-100:, 0].mean()
    assert noise_value > structured_value


def test_signed_activity_transform_uses_generic_redesign_and_accepts_negatives():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    formula, changes = migrate_formula(
        "ts_spectral_entropy(ts_zscore(turnover_ratio, 60), window=60)"
    )
    assert formula == "ts_spectral_flatness(ts_zscore(turnover_ratio, 60), window=60)"
    assert "NON_EQUIVALENT" in changes[0]
    load_all()
    idx = pd.date_range("2025-01-01", periods=100)
    signed = pd.DataFrame({"A": np.linspace(-2.0, 2.0, 100)}, index=idx)
    out = OperatorRegistry.get("ts_spectral_flatness", backend="pandas_numpy").calculate(
        signed, window=60
    )
    assert np.isfinite(out.iloc[-1, 0])


def test_event_repair_executes_on_response_and_derived_mask():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    idx = pd.date_range("2025-01-01", periods=100)
    ret = pd.DataFrame({"A": np.tile([-0.01, 0.02], 50)}, index=idx)
    numeric_signal = pd.DataFrame({"A": (np.arange(100) % 9 == 0).astype(float)}, index=idx)
    event = (numeric_signal != 0.0).astype(float)
    result = OperatorRegistry.get(
        "event_historical_response_mean", backend="pandas_numpy"
    ).calculate(ret, event, history_window=60, horizon=5, mode="sum", min_events=2)
    assert np.isfinite(result.iloc[-1, 0])


def test_input_validation_is_bounded():
    malformed = "ts_spectral_entropy("
    assert migrate_formula(malformed) == (malformed, [])
    qualified_wrong_table = "ts_spectral_entropy(field('ret', table='ArbitraryTable'), 60)"
    assert migrate_formula(qualified_wrong_table)[0].startswith("ts_spectral_flatness")
    with pytest.raises(ValueError, match="input budget"):
        migrate_formula("x" * 65_537)
    with pytest.raises(TypeError):
        migrate_formula("ret", logic=None)
