from __future__ import annotations


def _analyze(formula: str):
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.ir.analyzer import Analyzer

    factor = parse_factor(formula, name=f"warmup::{formula[:20]}", surface="extended")
    return Analyzer().lower(factor.expr)


def test_multi_stage_chart_patterns_add_stage_histories():
    cup = _analyze(
        "pattern_cup_handle(close,high,low,80,15,0.10,0.05,0.02,0.15)"
    )
    pennant = _analyze(
        "pattern_bull_pennant(close,high,low,volume,20,12,0.08,0.12,0.0)"
    )
    retest = _analyze("pattern_breakout_retest(close,60,10,0.02)")
    assert cup.lookback >= 95
    assert pennant.lookback >= 32
    assert retest.lookback >= 70


def test_fundamental_warmup_distinguishes_period_daily_and_elementwise():
    ratio = _analyze("fin_ratio(fundamental_x,fundamental_y)")
    surprise_z = _analyze(
        "fin_surprise_zscore(actual,expected,scale_base,252)"
    )
    ttm = _analyze(
        "fin_ttm_cumulative(fundamental_x,period_id,fiscal_quarter,4)"
    )
    assert ratio.lookback == 0
    assert surprise_z.lookback >= 251
    assert ttm.lookback >= 320


def test_recursive_indicators_declare_full_history_requirement():
    for formula in (
        "KAMA(close,10,2,30)",
        "Supertrend(high,low,close,10,3.0)",
        "PSAR(high,low,0.02,0.2)",
    ):
        assert _analyze(formula).requires_full_history is True
