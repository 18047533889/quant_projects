import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.metrics.interactions.pairwise import (
    PairwiseMeasurementStatus,
    compute_pairwise_artifacts,
)


def _batch(values, validity=None):
    array = np.asarray(values, dtype=float)
    return FactorBatch(
        factor_ids=("A", "B"),
        time_axis=AxisRef("time", "date", array.shape[0]),
        asset_axis=AxisRef("asset", "str", array.shape[1]),
        values=array,
        validity=validity,
        context_refs={},
    )


def test_pairwise_artifact_preserves_true_zero_and_sparse_counts():
    # Orthogonal centered vectors have exactly zero Pearson correlation.
    values = np.array([
        [[-1.0, -1.0], [-1.0, 1.0]],
        [[1.0, -1.0], [1.0, 1.0]],
    ])
    artifact, = compute_pairwise_artifacts(
        _batch(values), min_obs=4, window_ref="2d", universe_ref="U", sample_ref="S"
    )
    assert artifact.status is PairwiseMeasurementStatus.COMPUTED
    assert artifact.correlation == 0.0
    assert artifact.pair_count == 4
    assert artifact.daily_pair_counts == (2, 2)
    assert artifact.n_days == 2


def test_pairwise_artifact_insufficient_and_constant_are_not_zero():
    sparse = np.array([[[1.0, 2.0], [np.nan, 3.0]]])
    insufficient, = compute_pairwise_artifacts(
        _batch(sparse), min_obs=2, window_ref="1d", universe_ref="U", sample_ref="S"
    )
    assert insufficient.status is PairwiseMeasurementStatus.INSUFFICIENT_DATA
    assert insufficient.correlation is None
    assert insufficient.pair_count == 1

    constant = np.array([[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]])
    artifact, = compute_pairwise_artifacts(
        _batch(constant), min_obs=2, window_ref="1d", universe_ref="U", sample_ref="S"
    )
    assert artifact.status is PairwiseMeasurementStatus.CONSTANT_INPUT
    assert artifact.correlation is None


def test_pairwise_windows_record_signed_drift_and_hac_uncertainty():
    base = np.array([-2.0, -1.0, 1.0, 2.0])
    values = np.empty((40, 4, 2))
    values[:, :, 0] = base
    values[:20, :, 1] = base
    values[20:, :, 1] = -base
    artifact, = compute_pairwise_artifacts(
        _batch(values), min_obs=30, window_ref="40d", universe_ref="U",
        sample_ref="S", windows={"early": (0, 20), "late": (20, 40)},
        uncertainty_min_days=20,
    )
    early, late = artifact.windows
    assert early.correlation == pytest.approx(1.0)
    assert late.correlation == pytest.approx(-1.0)
    assert early.pair_count == late.pair_count == 80
    # Existing HAC authority reports non-finite SE for a zero-variance series;
    # the pairwise layer must leave uncertainty unavailable, not invent certainty.
    assert early.confidence_interval == (None, None)
    assert late.confidence_interval == (None, None)
    assert early.uncertainty_scale == "raw_correlation_hac_daily_corr"
