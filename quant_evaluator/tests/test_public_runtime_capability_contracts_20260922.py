"""Public runtime evidence for the documented QE capability/transport contract."""

import numpy as np
import pytest

from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import SeriesMetricArtifact
from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
from quant_evaluator.runtime.evaluator import evaluate


def _inputs(*, factors=3, periods=80, assets=64):
    rng = np.random.default_rng(20260922)
    labels_array = rng.normal(size=(periods, assets))
    factor_values = np.stack(
        [
            (0.20 - 0.08 * index) * labels_array
            + rng.normal(size=(periods, assets))
            for index in range(factors)
        ],
        axis=-1,
    )
    time_axis = AxisRef("time", "int64", periods, np.arange(periods, dtype=np.int64))
    asset_axis = AxisRef("asset", "int64", assets, np.arange(assets, dtype=np.int64))
    batch = FactorBatch(
        tuple(f"f{index}" for index in range(factors)),
        time_axis,
        asset_axis,
        factor_values,
    )
    labels = LabelBundle(
        "forward",
        labels_array,
        1,
        decision_time=tuple(range(periods)),
        label_start_time=tuple(range(1, periods + 1)),
        label_end_time=tuple(range(2, periods + 2)),
        asset_axis=asset_axis,
    )
    return batch, labels


def test_three_documented_statistical_metrics_execute_via_public_api():
    batch, labels = _inputs()
    metric_ids = (
        "hac_tstat",
        "block_bootstrap_ci",
        "benjamini_hochberg_correction",
    )
    bundle = evaluate(
        EvaluationRequest(batch, labels, metric_ids=metric_ids, tier="research")
    )

    for metric_id in metric_ids:
        assert metric_id in bundle.artifacts
        assert np.isfinite(bundle.artifacts[metric_id].values).all()
        assert all(bundle.get_metric(metric_id, factor_id).valid for factor_id in batch.factor_ids)


def test_series_results_live_in_typed_artifacts_not_metric_values():
    batch, labels = _inputs(factors=2)
    bundle = evaluate(batch, labels, metrics=("rank_ic_series",))

    artifact = bundle.artifacts["rank_ic_series"]
    assert isinstance(artifact, SeriesMetricArtifact)
    assert artifact.values.shape == (batch.num_times, batch.num_factors)
    assert "rank_ic_series" not in bundle.metric_values


def test_int64_and_immutable_object_coordinates_run_but_mutable_objects_fail_closed():
    batch, labels = _inputs(factors=1)
    assert batch.time_axis.values.dtype == np.dtype("int64")
    assert batch.asset_axis.values.dtype == np.dtype("int64")
    assert evaluate(batch, labels, metrics=("rank_ic",)).get_metric("rank_ic", "f0").valid

    object_axis = AxisRef(
        "asset", "object", 2,
        np.asarray(["000001.SZ", "600000.SH"], dtype=object),
    )
    assert object_axis.values.tolist() == ["000001.SZ", "600000.SH"]
    assert not object_axis.values.flags.writeable

    class MutableCoordinate:
        pass

    with pytest.raises(ValueError, match="immutable scalar"):
        AxisRef(
            "asset", "object", 1,
            np.asarray([MutableCoordinate()], dtype=object),
        )


def test_portfolio_construction_and_drawdown_use_explicit_typed_inputs():
    batch, labels = _inputs(factors=2)
    legacy = evaluate(batch, labels, metrics=("long_short_returns",))
    assert isinstance(legacy.artifacts["long_short_returns"], SeriesMetricArtifact)

    rng = np.random.default_rng(9122)
    prices = 100.0 * np.cumprod(
        1.0 + rng.normal(0.0002, 0.01, (batch.num_times, batch.num_assets)), axis=0
    )
    holding_returns = HoldingReturnPanel.from_prices(
        prices,
        time_axis=batch.time_axis,
        asset_axis=batch.asset_axis,
        source_ref="fixture:close-prices",
        price_basis="close_to_close",
    )
    portfolio_spec = PortfolioSpec(holding=2, n_quantiles=5, per_side_cost=0.0005)

    first = evaluate(
        batch,
        labels,
        metrics=("long_short_returns",),
        holding_returns=holding_returns,
        portfolio_spec=portfolio_spec,
    )
    long_short = first.artifacts["long_short_returns"]
    assert isinstance(long_short, SeriesMetricArtifact)
    with pytest.raises(InvalidContractError, match="ProbePortfolioArtifact"):
        evaluate(batch, labels, metrics=("max_drawdown",))

    portfolio = ProbePortfolioArtifact(
        long_short.values,
        time_index=long_short.time_axis.time_index,
        factor_ids=batch.factor_ids,
        provenance={"source_metric": "long_short_returns"},
    )
    second = evaluate(
        batch, labels, metrics=("max_drawdown",), portfolio_returns=portfolio
    )
    assert np.isfinite(second.artifacts["max_drawdown"].values).all()
