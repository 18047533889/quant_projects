import numpy as np
import pytest

from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel
from quant_evaluator.runtime.evaluator import evaluate


def _inputs():
    rng = np.random.default_rng(20260922)
    time = np.arange(100, 145, dtype=np.int64)
    assets = np.arange(60, dtype=np.int64)
    signal = rng.normal(size=(len(time), len(assets)))
    labels = np.tanh(signal) * 0.05
    batch = FactorBatch(
        ("up", "down"), AxisRef("time", "int64", len(time), time),
        AxisRef("asset", "int64", len(assets), assets),
        np.stack((signal, -signal), axis=-1),
    )
    label = LabelBundle(
        "forward", labels, 1, decision_time=tuple(time),
        observation_time=tuple(time), label_start_time=tuple(time + 1),
        label_end_time=tuple(time + 2), asset_axis=batch.asset_axis,
    )
    return batch, label


@pytest.mark.parametrize("values", [
    np.ones((2, 2), dtype=bool),
    np.ones((2, 2), dtype=np.complex128),
    np.full((2, 2), "not-a-return", dtype=object),
    np.ones((2, 2), dtype="timedelta64[D]"),
])
def test_holding_return_panel_rejects_non_real_numeric_values(values):
    time = AxisRef("time", "int64", 2, np.arange(2, dtype=np.int64))
    asset = AxisRef("asset", "int64", 2, np.arange(2, dtype=np.int64))
    with pytest.raises(ValueError, match="real numeric"):
        HoldingReturnPanel(values, time, asset, "fixture:returns", "close_to_close")


@pytest.mark.parametrize("values", [
    np.ones((2, 1), dtype=bool),
    np.ones((2, 1), dtype=np.complex128),
    np.full((2, 1), "not-a-return", dtype=object),
    np.ones((2, 1), dtype="timedelta64[D]"),
])
def test_probe_portfolio_rejects_non_real_numeric_values(values):
    with pytest.raises(ValueError, match="real numeric"):
        ProbePortfolioArtifact(values, time_index=(0, 1), factor_ids=("f",))


def test_public_series_can_be_rebound_as_typed_probe_for_drawdown():
    batch, label = _inputs()
    first = evaluate(batch, label, metrics=("long_short_returns",))
    series = first.artifacts["long_short_returns"]
    assert first.metric_values == {}
    portfolio = ProbePortfolioArtifact(
        series.values, time_index=tuple(series.time_axis.time_index),
        factor_ids=batch.factor_ids,
    )
    second = evaluate(
        batch, label, metrics=("max_drawdown", "sharpe_ratio"),
        portfolio_returns=portfolio,
    )
    assert np.isfinite(second.artifacts["max_drawdown"].values).all()
    assert np.isfinite(second.artifacts["sharpe_ratio"].values).all()


@pytest.mark.parametrize("prices", [
    np.ones((2, 2), dtype=bool),
    np.ones((2, 2), dtype=np.complex128) * (100 + 2j),
    np.full((2, 2), "100", dtype=object),
    np.ones((2, 2), dtype="timedelta64[D]"),
])
def test_price_constructor_cannot_coerce_invalid_dtype_into_valid_returns(prices):
    axis = AxisRef("time", "int64", 2, np.arange(2, dtype=np.int64))
    assets = AxisRef("asset", "int64", 2, np.arange(2, dtype=np.int64))
    with pytest.raises(ValueError, match="real numeric"):
        HoldingReturnPanel.from_prices(prices, time_axis=axis, asset_axis=assets,
            source_ref="fixture:prices", price_basis="close_to_close")
