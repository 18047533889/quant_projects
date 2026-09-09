import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError, UnsupportedMetricError
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.evaluator import evaluate


METRICS = (
    "industry_exposure", "size_exposure", "beta_exposure",
    "liquidity_exposure", "volatility_exposure", "momentum_exposure",
    "max_absolute_style_exposure", "exposure_drift", "purity_ratio",
)


def _contracts(seed=380, missing=False):
    rng = np.random.default_rng(seed)
    t, n, f, k = 24, 96, 3, 6
    dates = tuple(range(t))
    assets = np.asarray([f"A{i:03d}" for i in range(n)], dtype=object)
    risk = rng.normal(size=(t, n, k))
    values = np.empty((t, n, f))
    values[:, :, 0] = .7 * risk[:, :, 0] - .2 * risk[:, :, 1] + rng.normal(0, .2, (t, n))
    values[:, :, 1] = -.4 * risk[:, :, 3] + rng.normal(0, .3, (t, n))
    values[:, :, 2] = rng.normal(size=(t, n))
    validity = np.ones_like(values, dtype=bool)
    risk_validity = np.ones_like(risk, dtype=bool)
    if missing:
        validity[2:6, :11, 0] = False
        values[8, :, 1] = np.nan
        risk_validity[4:9, :17, 2] = False
        risk[12, :, :] = np.nan
    batch = FactorBatch(
        tuple(f"f{i}" for i in range(f)),
        AxisRef("time", "int", t, np.asarray(dates)),
        AxisRef("asset", "object", n, assets), values, validity=validity,
    )
    labels = LabelBundle(
        "ret", rng.normal(size=(t, n)), 1, decision_time=dates,
        label_start_time=tuple(x + 1 for x in dates),
        label_end_time=tuple(x + 2 for x in dates), asset_axis=batch.asset_axis,
    )
    panel = ExposurePanel(
        risk, style_names=("industry", "size", "beta", "liquidity", "volatility", "momentum"),
        source_ref="risk:test-v3", provider="gpu-oracle-fixture",
        date_index=dates, security_ids=tuple(assets), factor_ids=batch.factor_ids,
        universe_snapshot_ref="universe:test-v3", validity=risk_validity,
    )
    return batch, labels, panel


@pytest.mark.parametrize("missing", [False, True])
def test_public_cuda_exposure_matches_cpu_and_emits_typed_no_fallback_artifacts(missing):
    pytest.importorskip("cupy")
    batch, labels, panel = _contracts(missing=missing)
    options = dict(metrics=METRICS, exposure_panel=panel,
                   metric_parameters={"max_absolute_style_exposure": {"min_finite": 5},
                                      "purity_ratio": {"min_finite": 5}})
    cpu = evaluate(batch, labels, **options)
    gpu = evaluate(batch, labels, backend="cuda_strict", **options)
    for metric_id in METRICS:
        np.testing.assert_allclose(
            gpu.artifacts[metric_id].values, cpu.artifacts[metric_id].values,
            rtol=1e-10, atol=1e-10, equal_nan=True,
        )
        artifact = gpu.artifacts[metric_id]
        assert artifact.provenance["execution_backend"] == "cuda_strict"
        assert artifact.provenance["no_fallback"] is True
        assert artifact.provenance["exposure_source_ref"] == panel.source_ref
        for factor_id in batch.factor_ids:
            assert gpu.get_metric(metric_id, factor_id).observation_count == \
                cpu.get_metric(metric_id, factor_id).observation_count


def test_cuda_exposure_rejects_axis_mismatch_before_device_execution():
    pytest.importorskip("cupy")
    batch, labels, panel = _contracts()
    bad = ExposurePanel(
        panel.values, style_names=panel.style_names, source_ref=panel.source_ref,
        provider=panel.provider, date_index=panel.date_index,
        security_ids=tuple(reversed(panel.security_ids)), factor_ids=panel.factor_ids,
        universe_snapshot_ref=panel.universe_snapshot_ref,
    )
    with pytest.raises(InvalidContractError, match="security axis"):
        evaluate(batch, labels, metrics=("size_exposure",), exposure_panel=bad,
                 backend="cuda_strict")


def test_cuda_rejects_unimplemented_neutralized_exposure_instead_of_fallback():
    pytest.importorskip("cupy")
    batch, labels, panel = _contracts()
    with pytest.raises(UnsupportedMetricError, match="neutralized_rank_ic"):
        evaluate(batch, labels, metrics=("neutralized_rank_ic",),
                 exposure_panel=panel, backend="cuda_strict")


def _known_math_contract(risk, factor, style_names):
    t, n, _ = risk.shape
    dates = tuple(range(t))
    assets = np.asarray([f"K{i:02d}" for i in range(n)], dtype=object)
    batch = FactorBatch(
        ("known",), AxisRef("time", "int", t, np.asarray(dates)),
        AxisRef("asset", "object", n, assets), factor[:, :, None],
    )
    labels = LabelBundle(
        "ret", np.zeros((t, n)), 1, decision_time=dates,
        label_start_time=tuple(x + 1 for x in dates),
        label_end_time=tuple(x + 2 for x in dates), asset_axis=batch.asset_axis,
    )
    panel = ExposurePanel(
        risk, style_names=style_names, source_ref="risk:known",
        provider="independent-math-oracle", date_index=dates,
        security_ids=tuple(assets), factor_ids=batch.factor_ids,
        universe_snapshot_ref="universe:known",
    )
    return batch, labels, panel


def test_cuda_exposure_matches_independent_known_ols_drift_and_purity_math():
    pytest.importorskip("cupy")
    rng = np.random.default_rng(812)
    t, n = 4, 40
    risk = rng.normal(size=(t, n, 2))
    beta = np.asarray(((1, -1), (2, -1), (4, 1), (7, 3)), dtype=float)
    factor = np.einsum("tnk,tk->tn", risk, beta) + np.arange(t)[:, None] * .3
    batch, labels, panel = _known_math_contract(risk, factor, ("industry", "size"))
    result = evaluate(
        batch, labels,
        metrics=("industry_exposure", "size_exposure", "exposure_drift", "purity_ratio"),
        metric_parameters={"purity_ratio": {"min_finite": 1}},
        exposure_panel=panel, backend="cuda_strict",
    )
    # Independent standardized-loading oracle: beta_k * sd(risk_k) / sd(factor).
    expected = np.empty_like(beta)
    for day in range(t):
        design = np.column_stack((np.ones(n), risk[day]))
        coef = np.linalg.solve(design.T @ design, design.T @ factor[day])[1:]
        expected[day] = coef * np.std(risk[day], axis=0) / np.std(factor[day])
    assert result.get_metric("industry_exposure", "known").value == pytest.approx(
        float(expected[:, 0].mean()), abs=1e-11
    )
    assert result.get_metric("size_exposure", "known").value == pytest.approx(
        float(expected[:, 1].mean()), abs=1e-11
    )
    expected_drift = float(np.mean([
        np.mean(np.abs(expected[day + 1] - expected[day]))
        for day in range(t - 1)
    ]))
    assert result.get_metric("exposure_drift", "known").value == pytest.approx(expected_drift, abs=1e-11)
    # Exact linear construction has R²=1 each day, hence time-mean 1-R² is 0.
    assert result.get_metric("purity_ratio", "known").value == pytest.approx(0.0, abs=1e-11)


def test_cuda_rank_deficiency_and_insufficient_observations_stay_invalid():
    pytest.importorskip("cupy")
    rng = np.random.default_rng(813)
    t, n = 5, 30
    one = rng.normal(size=(t, n))
    rank_deficient = np.stack((one, one), axis=-1)
    batch, labels, panel = _known_math_contract(
        rank_deficient, one, ("industry", "industry_copy")
    )
    singular = evaluate(batch, labels, metrics=("industry_exposure",),
                        exposure_panel=panel, backend="cuda_strict")
    assert not singular.get_metric("industry_exposure", "known").valid

    risk = rng.normal(size=(t, n, 2))
    factor = rng.normal(size=(t, n))
    batch, labels, panel = _known_math_contract(risk, factor, ("industry", "size"))
    validity = np.zeros_like(risk, dtype=bool)
    validity[:, :9, :] = True
    sparse = ExposurePanel(
        panel.values, style_names=panel.style_names, source_ref=panel.source_ref,
        provider=panel.provider, date_index=panel.date_index,
        security_ids=panel.security_ids, factor_ids=panel.factor_ids,
        universe_snapshot_ref=panel.universe_snapshot_ref, validity=validity,
    )
    insufficient = evaluate(batch, labels, metrics=("industry_exposure",),
                            exposure_panel=sparse, backend="cuda_strict")
    assert not insufficient.get_metric("industry_exposure", "known").valid


def test_cuda_exposure_preserves_factor_tiling(monkeypatch):
    pytest.importorskip("cupy")
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession

    batch, labels, panel = _contracts()
    monkeypatch.setattr(
        DeviceEvaluationSession, "estimate_tile",
        lambda self, metrics, t, n, itemsize: 1,
    )
    gpu = evaluate(batch, labels, metrics=("industry_exposure", "purity_ratio"),
                   exposure_panel=panel, backend="cuda_strict")
    cpu = evaluate(batch, labels, metrics=("industry_exposure", "purity_ratio"),
                   exposure_panel=panel)
    np.testing.assert_allclose(gpu.artifacts["industry_exposure"].values,
                               cpu.artifacts["industry_exposure"].values, atol=1e-10)
    np.testing.assert_allclose(gpu.artifacts["purity_ratio"].values,
                               cpu.artifacts["purity_ratio"].values, atol=1e-10)
    assert gpu.metadata["factor_tiles_processed"] == len(batch.factor_ids)


def test_public_residual_ic_masks_label_and_exposure_validity_and_matches_oracle():
    from scipy.stats import rankdata

    rng = np.random.default_rng(901)
    t, n, k = 12, 30, 2
    dates = tuple(range(t))
    assets = np.asarray([f"R{i:02d}" for i in range(n)], dtype=object)
    risk = rng.normal(size=(t, n, k))
    alpha = rng.normal(size=(t, n))
    factor = .8 * risk[:, :, 0] - .3 * risk[:, :, 1] + alpha
    labels_array = alpha + rng.normal(0, .1, size=(t, n))
    risk_validity = np.ones_like(risk, dtype=bool)
    label_validity = np.ones_like(labels_array, dtype=bool)
    risk_validity[:, :3, :] = False
    label_validity[:, 3:6] = False

    def run(risk_values, label_values):
        batch = FactorBatch(
            ("resid",), AxisRef("time", "int", t, np.asarray(dates)),
            AxisRef("asset", "object", n, assets), factor[:, :, None],
        )
        labels = LabelBundle(
            "ret", label_values, 1, decision_time=dates,
            label_start_time=tuple(x + 1 for x in dates),
            label_end_time=tuple(x + 2 for x in dates), asset_axis=batch.asset_axis,
            validity=label_validity,
        )
        panel = ExposurePanel(
            risk_values, style_names=("industry", "size"),
            source_ref="risk:masked", provider="masked-oracle", date_index=dates,
            security_ids=tuple(assets), factor_ids=batch.factor_ids,
            universe_snapshot_ref="universe:masked", validity=risk_validity,
        )
        return evaluate(batch, labels,
                        metrics=("neutralized_rank_ic", "residual_rank_ic"),
                        exposure_panel=panel)

    clean = run(risk.copy(), labels_array.copy())
    poisoned_risk = risk.copy()
    poisoned_labels = labels_array.copy()
    poisoned_risk[~risk_validity] = 1e12
    poisoned_labels[~label_validity] = -1e12
    poisoned = run(poisoned_risk, poisoned_labels)

    daily = []
    for day in range(t):
        ols_rows = risk_validity[day].all(axis=1) & np.isfinite(factor[day])
        design = np.column_stack((np.ones(ols_rows.sum()), risk[day, ols_rows]))
        beta = np.linalg.solve(design.T @ design, design.T @ factor[day, ols_rows])
        residual = factor[day, ols_rows] - design @ beta
        asset_rows = np.flatnonzero(ols_rows)
        keep = label_validity[day, asset_rows] & np.isfinite(labels_array[day, asset_rows])
        assert keep.sum() >= 10
        rr = rankdata(residual[keep], method="average")
        yr = rankdata(labels_array[day, asset_rows[keep]], method="average")
        daily.append(np.corrcoef(rr, yr)[0, 1])
    expected = float(np.mean(daily))
    for metric_id in ("neutralized_rank_ic", "residual_rank_ic"):
        assert clean.get_metric(metric_id, "resid").value == pytest.approx(expected, abs=1e-12)
        assert poisoned.get_metric(metric_id, "resid").value == pytest.approx(expected, abs=1e-12)


def test_public_residual_ic_final_joint_min_obs_is_not_half_floor():
    rng = np.random.default_rng(902)
    t, n = 5, 20
    risk = rng.normal(size=(t, n, 2))
    factor = rng.normal(size=(t, n))
    batch, labels, panel = _known_math_contract(risk, factor, ("industry", "size"))
    label_values = rng.normal(size=(t, n))
    validity = np.zeros_like(label_values, dtype=bool)
    validity[:, :9] = True
    sparse_labels = LabelBundle(
        "ret", label_values, 1, decision_time=labels.decision_time,
        label_start_time=labels.label_start_time, label_end_time=labels.label_end_time,
        asset_axis=batch.asset_axis, validity=validity,
    )
    result = evaluate(batch, sparse_labels, metrics=("neutralized_rank_ic",),
                      exposure_panel=panel)
    assert not result.get_metric("neutralized_rank_ic", "known").valid
