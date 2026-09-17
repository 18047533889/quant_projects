"""Finite-history contract for the bounded prior-window Huber coefficient."""

from factor_engine.api.dsl_parser import parse_factor
from factor_engine.cleaned_operators import load_all
from factor_engine.ir.analyzer import Analyzer
from factor_engine.runtime.execution_contract import (
    execution_contract,
    own_history_requirement,
)


def _analyze(formula: str):
    factor = parse_factor(formula, name="huber-prior-history", surface="extended")
    return Analyzer().lower(factor.expr)


def test_fresh_load_huber_prior_is_bounded_by_its_real_window_and_fit_lag():
    load_all()
    canonical = "ts_huber_regression_coeff_prior"

    contract = execution_contract(canonical)
    assert contract.state_model == "stateless"
    assert contract.chunking == "independent"

    default = own_history_requirement(canonical)
    explicit = own_history_requirement(canonical, {"window": 80, "min_periods": 20})
    assert (default.kind, default.rows) == ("finite", 60)
    assert (explicit.kind, explicit.rows) == ("finite", 80)


def test_analyzer_composes_nested_predictor_lookback_without_full_history():
    direct = _analyze(
        "ts_huber_regression_coeff_prior(ret,volume,amount,window=80,"
        "coefficient_index=1,min_periods=20,add_intercept=True)"
    )
    nested = _analyze(
        "ts_huber_regression_coeff_prior(ret,ts_pct(add(volume,1),1),"
        "ts_pct(add(amount,1),1),window=80,coefficient_index=1,"
        "min_periods=20,add_intercept=True)"
    )

    assert direct.requires_full_history is False
    assert direct.lookback == 80
    assert nested.requires_full_history is False
    assert nested.lookback == 81
