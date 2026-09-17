import numpy as np
import pandas as pd
import pytest

from factor_engine.tools.catalog_r19_parameter_recipes import migrate_formula


@pytest.mark.parametrize("source, expected", [
    ("category_age(PublicStatus)", "category_age(PublicStatus, 252)"),
    ("ts_last_pivot_low(low, 3, 3)", "ts_last_pivot_low(low, 3, 3, 250)"),
    ("ts_support_break(close, low, 3, 3, 3)", "ts_support_break(close, low, 3, 3, 3, 3)"),
    ("cs_rank_gaussian(ret, 3.0)", "cs_rank_gaussian(ret, 'blom')"),
    ("ts_quantile_skew(ret, 60, 0.1, 0.25, 0.75, 0.9)", "ts_quantile_skew(ret, 60, 0.1, 0.5, 0.9, 5)"),
    ("ts_extremogram(ret, window=120, quantile=0.9, side='upper')", "ts_extremogram(ret, window=120, quantile=0.1, side='upper')"),
    ("ts_gap_fill_ratio(close, open, pre_close, 0.01)", "ts_gap_fill_ratio(close, open, pre_close, 60)"),
    ("ts_transition_count(gt(ret, 0.0), 20, 0.0)", "ts_transition_count(gt(ret, 0.0), 20, 'break')"),
])
def test_exact_repairs_are_audited_and_idempotent(source, expected):
    got, changes = migrate_formula(source, "reviewed")
    assert got == expected
    assert changes and all(c.startswith("SEMANTIC_REDESIGN") for c in changes)
    assert migrate_formula(got, "reviewed") == (got, [])


def test_unrecognised_shapes_fail_closed_and_inputs_are_bounded():
    assert migrate_formula("category_age(x, 60)") == ("category_age(x, 60)", [])
    assert migrate_formula("cs_rank_gaussian(x, 'van_der_waerden')")[1] == []
    assert migrate_formula("category_age(") == ("category_age(", [])
    with pytest.raises(ValueError): migrate_formula("x" * 65_537)


def test_nested_and_unicode_source_spans_are_preserved():
    source = "add(field('中文'),\n category_age(PublicStatus))"
    assert migrate_formula(source)[0] == "add(field('中文'),\n category_age(PublicStatus, 252))"


def test_fiscal_x_keyword_is_renamed_without_changing_value_expression():
    source = "fiscal_sign_consistency(x=add(a, b), period_id=FiscalPeriodId)"
    got, changes = migrate_formula(source)
    assert got == "fiscal_sign_consistency(signal=add(a, b), period_id=FiscalPeriodId)"
    assert changes == ["PARAMETER_ALIAS fiscal_sign_consistency: obsolete keyword x -> declared keyword signal; value expression unchanged"]


def test_invalid_ohlcv_technical_signatures_are_explicitly_redefined_on_close():
    kama = "KAMA(open, high, low, close, volume)"
    macd = "MACD_hist(open, high, low, close, volume)"
    assert migrate_formula(kama)[0] == "KAMA(close, 10, 2, 30)"
    assert migrate_formula(macd)[0] == "MACD_hist(close, 12, 26, 9)"
    assert "open/high/low/volume inputs removed" in migrate_formula(kama)[1][0]


def test_kama_missing_fast_slow_uses_documented_canonical_windows():
    assert migrate_formula("KAMA(close, 10)")[0] == "KAMA(close, 10, 2, 30)"


def test_two_sided_mean_shift_is_composed_from_supported_directions():
    source = "ts_rolling_sr_gaussian_mean_shift_score(ret, window=120, shift_sigma=1.0, baseline_window=40, side='two_sided')"
    got, changes = migrate_formula(source)
    assert got.startswith("flex_max(ts_rolling_sr_gaussian_mean_shift_score")
    assert "side='up'" in got and "side='down'" in got
    assert "maximum" in changes[0]


def test_hammer_inactive_knobs_are_removed():
    source = "candlestick_pattern(open, high, low, close, pattern='hammer', body_window=20, shadow_window=20, penetration=0.3)"
    got, changes = migrate_formula(source)
    assert got == "candlestick_pattern(open, high, low, close, pattern='hammer')"
    assert changes[0].startswith("PARAMETER_CANONICALIZATION")


@pytest.mark.parametrize("name", ["group_peer_deviation_index", "group_peer_beta_deviation"])
def test_under_specified_peer_calls_become_equal_weight_ex_self_deviations(name):
    source = f"{name}(safe_div_null(a, b), IndustryCode)"
    got, changes = migrate_formula(source)
    expected_value = "safe_div_null(a, b)"
    assert got == f"subtract({expected_value}, group_ex_self_mean({expected_value}, IndustryCode))"
    assert "equal-weight industry peer mean excluding self" in changes[0]
    assert "fabricated" in changes[0]


def test_multiscale_permutation_template_uses_minimal_feasible_fixed_scale_window():
    source = "ts_multiscale_permutation_entropy_slope(ret, window=120, order=3, min_patterns=20)"
    expected = "ts_multiscale_permutation_entropy_slope(ret, window=256, order=3, min_patterns=20)"
    got, changes = migrate_formula(source)
    assert got == expected
    assert "fixed scales [1,2,4,8]" in changes[0]
    assert "32 coarse points / 30 ordinal patterns" in changes[0]
    assert migrate_formula(got) == (got, [])


@pytest.mark.parametrize("name", ["ts_quantile_transport_slope", "ts_quantile_transport_curvature", "ts_mmd_rbf_shift"])
def test_old_single_distribution_window_is_redefined_as_two_windows(name):
    got, changes = migrate_formula(f"{name}(ret, window=60)")
    assert got == f"{name}(ret, recent_window=20, old_window=40)"
    assert "not equivalent" in changes[0]


def test_known_positional_and_support_parameter_repairs():
    assert migrate_formula("ts_downside_deviation(ret, 0.0, 20, 10)")[0] == "ts_downside_deviation(ret, 20, 0.0, 10)"
    assert migrate_formula("cs_knn_tangent_residual(a, b, c, k=10)")[0] == "cs_knn_tangent_residual(a, b, c, k=20)"
    assert migrate_formula("ts_markov_transition_surprisal(ret, window=60, bins=3, lag=1, min_periods=3)")[0].endswith("min_count=3)")
    assert migrate_formula("ts_autocorr_decay_half_life(ret, 60, 1, 20, 0.05)")[0] == "ts_autocorr_decay_half_life(ret, 60, 10, False, 20)"


def test_state_integral_signed_lower_is_redefined_on_absolute_magnitude():
    source = "ts_state_integral(z, upper=1.0, lower=-1.0, max_run=20, half_life=10.0)"
    got, changes = migrate_formula(source)
    assert got == "ts_state_integral(z, upper=1.0, lower=0.0, max_run=20, half_life=10.0)"
    assert "thresholds apply to |z|" in changes[0]


@pytest.mark.parametrize("name", ["event_frequency", "event_fano_factor", "event_interval_memory", "event_local_variation", "event_fano_excess", "event_hawkes_branching_ratio_proxy"])
def test_response_panel_in_event_window_slot_becomes_default_window(name):
    source = f"{name}(event, field('ret', table='StockDailyBarAdj'))"
    got, changes = migrate_formula(source)
    assert got == f"{name}(event, 60)"
    assert "response panel removed" in changes[0]


def test_small_panel_semantics_for_repaired_state_and_gaussian():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    load_all()
    frame = pd.DataFrame([[1, 2], [1, 3], [2, 3], [2, 3]], dtype=float)
    age = OperatorRegistry.get("category_age", "pandas_numpy").calculate(frame, window=3)
    assert age.iloc[-1, 0] == 1.0
    values = pd.DataFrame(np.arange(32.0).reshape(2, 16))
    gaussian = OperatorRegistry.get("cs_rank_gaussian", "pandas_numpy").calculate(values, method="blom")
    assert np.isfinite(gaussian.to_numpy()).all()
