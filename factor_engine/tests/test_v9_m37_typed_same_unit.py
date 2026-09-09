import numpy as np
import pandas as pd
import pytest

from factor_engine.api.columns import field
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.literal import Literal
from factor_engine.ir.analyzer import Analyzer, SameUnitInputContractError


FIRST_PASSAGE = (
    "ts_first_passage_bias",
    "ts_first_passage_hit_probability",
    "ts_first_passage_conditional_time",
)


@pytest.fixture(autouse=True, scope="module")
def _load_registry():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()


def _call(name, x, scale):
    kwargs = {
        "window": 6,
        "barrier": 1.0,
        "horizon": 2,
        "min_anchors": 2,
        "scale_horizon": 1,
    }
    if name != "ts_first_passage_bias":
        kwargs["side"] = "upper"
    return CleanedCall(name, args=(x, scale), kwargs=tuple(sorted(kwargs.items())))


def _rolling(name, x, window=3):
    return CleanedCall(name, args=(x, Literal(window)))


def test_same_unit_contract_is_shared_and_catalog_visible():
    declared = (("x", "scale"),)
    for name in FIRST_PASSAGE:
        assert OperatorRegistry.get(name).metadata.same_unit_input_groups == declared
        assert OperatorRegistry.catalog()[name]["same_unit_input_groups"] == declared


@pytest.mark.parametrize("name", FIRST_PASSAGE)
def test_known_price_and_price_std_are_accepted(name):
    close = field("close")
    Analyzer(production=True, market="ashare").lower(
        _call(name, close, _rolling("ts_std", close))
    )


@pytest.mark.parametrize("name", FIRST_PASSAGE)
def test_known_price_ratio_mismatch_is_rejected_at_analysis(name):
    with pytest.raises(SameUnitInputContractError, match="same-unit.*violated"):
        Analyzer().lower(_call(name, field("close"), field("return")))


def test_unknown_derived_unit_fails_closed_only_in_production():
    close = field("close")
    unknown_scale = _rolling("ts_mean", close)
    Analyzer().lower(_call("ts_first_passage_bias", close, unknown_scale))
    with pytest.raises(SameUnitInputContractError, match="unit is unknown"):
        Analyzer(production=True, market="ashare").lower(
            _call("ts_first_passage_bias", close, unknown_scale)
        )


def test_log_price_return_volatility_is_explicitly_unsupported_without_proof():
    close = field("close")
    log_price = CleanedCall("log", args=(close,))
    return_vol = _rolling("ts_std", field("return"))
    with pytest.raises(SameUnitInputContractError, match="unit is unknown"):
        Analyzer(production=True, market="ashare").lower(
            _call("ts_first_passage_bias", log_price, return_vol)
        )


@pytest.mark.parametrize("name", FIRST_PASSAGE)
def test_raw_dataframe_public_runtime_does_not_guess_units(name):
    index = pd.date_range("2026-01-01", periods=12)
    columns = ["low_price", "high_price"]
    x = pd.DataFrame(
        np.column_stack((100.0 + np.arange(12), 10_000.0 + np.arange(12))),
        index=index,
        columns=columns,
    )
    scale = pd.DataFrame(1.0, index=index, columns=columns)
    op = OperatorRegistry.get(name)
    kwargs = dict(window=6, barrier=1.0, horizon=2, min_anchors=2, scale_horizon=1)
    if name != "ts_first_passage_bias":
        kwargs["side"] = "upper"
    result = op.calculate(x, scale, **kwargs)
    assert result.shape == x.shape
    for column in columns:
        single = op.calculate(x[[column]], scale[[column]], **kwargs)
        np.testing.assert_allclose(
            result[column].to_numpy(), single[column].to_numpy(), equal_nan=True
        )


def test_known_mismatch_rejection_precedes_runtime_kernel(monkeypatch):
    import factor_engine.cleaned_operators.first_passage as module

    def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("runtime kernel / panel IO was reached")

    monkeypatch.setattr(module, "_first_passage_series", unexpected_runtime)
    with pytest.raises(SameUnitInputContractError):
        Analyzer().lower(
            _call("ts_first_passage_bias", field("close"), field("return"))
        )


@pytest.mark.parametrize("name", FIRST_PASSAGE)
def test_raw_runtime_acceptance_and_prefix_are_future_independent(name):
    op = OperatorRegistry.get(name)
    index = pd.date_range("2026-01-01", periods=90)
    prefix_values = 100.0 + np.arange(30, dtype=float)
    future_values = prefix_values[-1] + 100.0 * np.arange(1, 61, dtype=float)
    full = pd.DataFrame(
        np.concatenate((prefix_values, future_values)), index=index, columns=["A"]
    )
    scale = pd.DataFrame(1.0, index=index, columns=["A"])
    kwargs = dict(window=10, barrier=1.0, horizon=2, min_anchors=2, scale_horizon=1)
    if name != "ts_first_passage_bias":
        kwargs["side"] = "upper"

    short = op.calculate(full.iloc[:30], scale.iloc[:30], **kwargs)
    long = op.calculate(full, scale, **kwargs)
    np.testing.assert_allclose(short, long.iloc[:30], equal_nan=True)


@pytest.mark.parametrize("name", FIRST_PASSAGE)
def test_window_warmup_chunk_matches_full_history(name):
    op = OperatorRegistry.get(name)
    rng = np.random.default_rng(370)
    index = pd.date_range("2026-01-01", periods=40)
    x = pd.DataFrame(rng.normal(size=(40, 2)).cumsum(axis=0), index=index, columns=["A", "B"])
    scale = pd.DataFrame(0.8, index=index, columns=x.columns)
    kwargs = dict(window=10, barrier=1.0, horizon=2, min_anchors=2, scale_horizon=1)
    if name != "ts_first_passage_bias":
        kwargs["side"] = "upper"

    full = op.calculate(x, scale, **kwargs)
    split = 25
    warmup = kwargs["window"]
    chunk = op.calculate(
        x.iloc[split - warmup :], scale.iloc[split - warmup :], **kwargs
    )
    np.testing.assert_allclose(
        chunk.iloc[warmup:].to_numpy(), full.iloc[split:].to_numpy(), equal_nan=True
    )
