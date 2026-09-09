import numpy as np
import pytest

from quant_evaluator import evaluate_horizons
from quant_evaluator.api.horizons import HorizonEvaluationBundle
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import SeriesMetricArtifact
from quant_evaluator.metrics.predictive import compute_rank_ic_decay
from quant_evaluator.metrics.statistical_evidence import build_horizon_curve_evidence


def inputs(t=36, n=50):
    rng = np.random.default_rng(32011)
    x = rng.normal(size=(t, n))
    other = rng.normal(size=(t, n))
    times = np.arange(t)
    assets = np.asarray([f"S{i:03d}" for i in range(n)])
    time_axis = AxisRef("time", "int64", t, times)
    asset_axis = AxisRef("asset", "str", n, assets)
    batch = FactorBatch(
        ("short_predictor", "other"), time_axis, asset_axis,
        np.stack([x, other], axis=-1), value_hash="factor-content-ref",
    )
    labels = {}
    for horizon in (1, 5, 10, 20):
        values = x.copy() if horizon == 1 else rng.normal(size=(t, n))
        if horizon == 20:
            # Immature tail is adversarially predictive and must not leak.
            values[-10:] = x[-10:]
        labels[horizon] = LabelBundle(
            f"forward_h{horizon}", values, horizon,
            decision_time=tuple(times), observation_time=tuple(times),
            label_start_time=tuple(times), label_end_time=tuple(times + horizon),
            asset_axis=asset_axis, source_ref=f"labels:h{horizon}",
        )
    return batch, labels


def test_public_horizon_curve_keeps_daily_artifacts_and_per_factor_evidence():
    batch, labels = inputs()
    result = evaluate_horizons(
        batch, labels, as_of=35, sample_policy="per_horizon",
        min_assets=20, min_periods=10,
    )
    assert isinstance(result, HorizonEvaluationBundle)
    assert result.horizons == (1, 5, 10, 20)
    assert set(result.curve_evidence) == set(batch.factor_ids)
    assert all(isinstance(result.daily_ic_artifacts[h], SeriesMetricArtifact) for h in result.horizons)
    assert all(result.daily_ic_artifacts[h].values.shape == (36, 2) for h in result.horizons)
    curve = result.curve_evidence["short_predictor"]
    assert curve.predictive_ic[0] > 0.99
    assert max(abs(v) for v in curve.predictive_ic[1:] if np.isfinite(v)) < 0.25
    assert curve.label_refs == tuple(labels[h].content_hash for h in result.horizons)
    assert result.label_content_refs[20] == labels[20].content_hash
    assert result.provenance["samples_comparable_across_horizons"] is False
    assert result.provenance["zero_tolerance"] == 1e-12
    # Serial ACF is one scalar per factor; that is not the predictive
    # cross-horizon curve and cannot substitute for it.
    acf = compute_rank_ic_decay(result.daily_ic_artifacts[1].values, min_periods=10)
    assert acf.shape == (2,)
    assert len(curve.predictive_ic) == 4
    assert not np.allclose(
        np.repeat(acf[0], len(curve.predictive_ic)),
        curve.predictive_ic,
        equal_nan=True,
    )


def test_common_sample_intersects_maturity_and_label_validity_before_every_leaf():
    batch, labels = inputs()
    # One mature H5 cell is invalid, so it must disappear from every horizon.
    validity = np.ones(labels[5].values.shape, dtype=bool)
    validity[3, 7] = False
    from dataclasses import replace
    labels = dict(labels)
    labels[5] = replace(labels[5], validity=validity)
    result = evaluate_horizons(
        batch, labels, as_of=35, sample_policy="common", min_assets=20, min_periods=10,
    )
    assert result.provenance["samples_comparable_across_horizons"] is True
    assert result.maturity_counts == {1: 35, 5: 31, 10: 26, 20: 16}
    assert all(np.array_equal(result.evaluation_masks[h], result.common_mask) for h in result.horizons)
    assert not result.common_mask[3, 7]
    assert not result.common_mask[16:].any()
    assert np.isnan(result.daily_ic_artifacts[20].values[16:]).all()
    assert np.isnan(result.daily_ic_artifacts[1].values[16:]).all()
    assert len(set(result.sample_counts.values())) == 1
    with pytest.raises(ValueError):
        result.common_mask[0, 0] = True


def test_per_horizon_maturity_counts_grow_only_when_that_label_matures():
    batch, labels = inputs()
    before = evaluate_horizons(batch, labels, as_of=35, min_assets=20, min_periods=5,
                               sample_policy="per_horizon")
    after = evaluate_horizons(batch, labels, as_of=36, min_assets=20, min_periods=5,
                              sample_policy="per_horizon")
    assert after.maturity_counts[1] == before.maturity_counts[1] + 1
    assert after.maturity_counts[20] == before.maturity_counts[20] + 1
    # H20 still has a much shorter evaluable history; no tail is zero-filled.
    assert after.maturity_counts[20] < after.maturity_counts[1]
    assert np.isnan(before.daily_ic_artifacts[20].values[16:]).all()


def test_horizon_and_coordinate_contracts_fail_closed():
    batch, labels = inputs()
    with pytest.raises(ValueError, match="at least two"):
        evaluate_horizons(batch, {1: labels[1]}, as_of=35)
    bad_key = dict(labels)
    bad_key[5] = labels[10]
    with pytest.raises(ValueError, match="mapping key"):
        evaluate_horizons(batch, bad_key, as_of=35)
    from dataclasses import replace
    reversed_axis = AxisRef("asset", "str", batch.num_assets, batch.asset_axis.values[::-1])
    swapped = dict(labels)
    swapped[5] = replace(labels[5], asset_axis=reversed_axis)
    with pytest.raises(ValueError, match="asset axis"):
        evaluate_horizons(batch, swapped, as_of=35)
    shifted = dict(labels)
    shifted[5] = replace(labels[5], observation_time=tuple(np.arange(batch.num_times) - 1))
    with pytest.raises(ValueError, match="observation axis"):
        evaluate_horizons(batch, shifted, as_of=35)


def test_cpu_cuda_leaf_parity():
    pytest.importorskip("cupy")
    batch, labels = inputs(t=24, n=30)
    cpu = evaluate_horizons(batch, labels, as_of=23, min_assets=10, min_periods=3)
    gpu = evaluate_horizons(batch, labels, as_of=23, min_assets=10, min_periods=3,
                            backend="cuda")
    for horizon in cpu.horizons:
        np.testing.assert_allclose(
            cpu.daily_ic_artifacts[horizon].values,
            gpu.daily_ic_artifacts[horizon].values,
            equal_nan=True, atol=1e-10,
        )
    for factor_id in batch.factor_ids:
        cpu_curve = cpu.curve_evidence[factor_id]
        gpu_curve = gpu.curve_evidence[factor_id]
        assert cpu_curve.horizons == gpu_curve.horizons
        assert cpu_curve.observation_counts == gpu_curve.observation_counts
        assert cpu_curve.label_refs == gpu_curve.label_refs
        np.testing.assert_allclose(
            cpu_curve.predictive_ic,
            gpu_curve.predictive_ic,
            equal_nan=True,
            atol=1e-10,
        )
        assert cpu_curve.fit_status == gpu_curve.fit_status
        if cpu_curve.half_life is not None:
            assert cpu_curve.half_life == pytest.approx(gpu_curve.half_life, abs=1e-10)
        else:
            assert gpu_curve.half_life is None


def test_horizon_curve_zero_policy_is_explicit_and_preserves_raw_means():
    refs = {1: "h1", 5: "h5", 10: "h10"}
    exact = build_horizon_curve_evidence(
        {1: [0.3] * 4, 5: [0.1] * 4, 10: [0.0] * 4},
        label_refs=refs, min_periods=4,
    )
    residual = build_horizon_curve_evidence(
        {1: [0.3] * 4, 5: [0.1] * 4, 10: [-1e-17] * 4},
        label_refs=refs, min_periods=4,
    )
    assert exact.fit_status == residual.fit_status == "NOT_FIT"
    assert exact.half_life is residual.half_life is None
    assert residual.predictive_ic[-1] == -1e-17
    assert residual.zero_tolerance == 1e-12


def test_horizon_curve_zero_policy_retains_identifiable_sign_changes():
    evidence = build_horizon_curve_evidence(
        {1: [0.3] * 4, 5: [-0.1] * 4, 10: [1e-17] * 4},
        label_refs={1: "h1", 5: "h5", 10: "h10"}, min_periods=4,
    )
    assert evidence.fit_status == "SIGN_CHANGE"
    assert evidence.half_life is None


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan, True, "1e-12"])
def test_horizon_curve_zero_tolerance_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="zero_tolerance"):
        build_horizon_curve_evidence(
            {1: [0.2], 5: [0.1]}, label_refs={1: "h1", 5: "h5"},
            min_periods=1, zero_tolerance=bad,
        )
