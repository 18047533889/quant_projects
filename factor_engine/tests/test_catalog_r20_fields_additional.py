from factor_engine.tools.catalog_r20_fields_proposals import redesign


def _rewrite(formula: str) -> str:
    return redesign(formula, {})[0]


def test_financial_period_follows_income_value_source():
    got = _rewrite(
        'ttm_from_cumulative(operating_revenue,'
        'field("report_period_end_date",table="StockIndicator"),4)'
    )
    assert "table='StockIncome'" in got
    assert "table='StockIndicator'" not in got


def test_revision_inputs_are_observable_and_same_statement():
    got = _rewrite(
        'report_revision_magnitude(OperatingProfit,prior_vintage_value,'
        'current_period_id,previous_period_id,revision_event)'
    )
    assert 'prior_vintage_value' not in got
    assert 'current_period_id' not in got
    assert got.count("table='StockIncome'") >= 3
    assert 'and_(eq(' in got


def test_capital_change_age_uses_observed_total_capital_event():
    got = _rewrite('capital_change_age(CapitalChangeDate)')
    assert got.startswith('ts_days_since(ne(')
    assert 'StockCapitalDaily' in got
    assert 'CapitalChangeDate' not in got


def test_state_and_efficiency_placeholders_have_explicit_definitions():
    assert 'ts_sum' in _rewrite('ts_state_residual_life(positive_momentum_state)')
    got = _rewrite('neg(EFF)')
    assert 'safe_div_null' in got
    assert 'delay(close, 20)' in got


def test_crossing_second_series_uses_declared_y_parameter():
    got = _rewrite('ts_crossing_speed(x=ema(close,12),source=ema(close,26))')
    assert 'source=' not in got
    assert 'y=ema(close, 26)' in got
