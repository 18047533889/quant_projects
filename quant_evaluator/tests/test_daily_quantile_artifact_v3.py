import numpy as np
import pytest


def _fixtures():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    t, n = 8, 20
    values = np.empty((t, n, 2))
    base = np.arange(n, dtype=float)
    values[:, :, 0] = base
    values[:, :, 1] = base[::-1]
    labels = np.tile(np.linspace(-0.02, 0.02, n), (t, 1))
    return FactorBatch(("up", "down"), AxisRef("t", "int", t), AxisRef("a", "str", n), values), LabelBundle(
        target_id="h1", values=labels, horizon=1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(t)), label_end_time=tuple(range(1, t + 1)),
    )


def test_builder_preserves_daily_tqf_returns_and_counts():
    from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
    from quant_evaluator.metrics.registry_adapters import build_daily_quantile_return_artifact
    batch, labels = _fixtures()
    expected, counts = compute_quantile_returns_fast(batch, labels, n_quantiles=5, min_assets=2)
    artifact = build_daily_quantile_return_artifact(
        batch, labels, n_quantiles=5, min_assets=2, min_periods=4,
        tie_status_ref="ties:1", tradability_ref="tradable:1", risk_exposure_ref="risk:1",
    )
    assert artifact.values.shape == (8, 5, 2)
    np.testing.assert_allclose(artifact.values, expected, equal_nan=True)
    np.testing.assert_array_equal(artifact.counts, counts)
    assert artifact.artifact_kind == "daily_quantile"
    assert artifact.provenance_refs_present
    artifact.require_certification_ready(lambda ref: ref in {"ties:1", "tradable:1", "risk:1"})


def test_daily_artifact_is_distinct_from_qf_and_roundtrips_losslessly():
    from quant_evaluator.contracts.artifact_types import DailyQuantileReturnArtifact, QuantileReturnArtifact
    from quant_evaluator.metrics.registry_adapters import build_daily_quantile_return_artifact
    batch, labels = _fixtures()
    artifact = build_daily_quantile_return_artifact(batch, labels, n_quantiles=5, min_assets=2)
    assert not isinstance(artifact, QuantileReturnArtifact)
    restored = DailyQuantileReturnArtifact.from_dict(artifact.to_dict())
    assert restored == artifact
    with pytest.raises(ValueError, match="lacks tie/tradability/risk"):
        restored.require_certification_ready(lambda ref: True)
    with pytest.raises(ValueError, match="read-only"):
        artifact.values[0, 0, 0] = 999
