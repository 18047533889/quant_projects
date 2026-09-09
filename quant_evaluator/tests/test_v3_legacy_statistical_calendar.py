import numpy as np
import pytest

from quant_evaluator.metrics.robustness import (
    compute_hac_variance, compute_hac_tstat, compute_block_bootstrap_ci,
)


def test_hac_missing_calendar_date_is_insufficient_not_compressed():
    x = np.sin(np.arange(70) / 6) + .05
    gapped = x.copy()
    gapped[25] = np.nan
    assert np.isnan(compute_hac_variance(gapped)[0])
    assert np.isnan(compute_hac_tstat(gapped)[0][0])
    assert np.isfinite(compute_hac_variance(np.delete(x, 25))[0])
    np.testing.assert_allclose(compute_hac_variance(np.r_[np.nan, x, np.nan]),
                               compute_hac_variance(x))


def test_bootstrap_identical_candidates_share_exact_draws_and_permute():
    x = np.random.default_rng(12).normal(size=80)
    matrix = np.column_stack([x, x, x + 1])
    kwargs = dict(block_length=6, num_bootstrap=151, random_seed=3)
    lo, hi = compute_block_bootstrap_ci(matrix, **kwargs)
    assert lo[0] == lo[1] and hi[0] == hi[1]
    np.testing.assert_allclose([lo[2], hi[2]], [lo[0] + 1, hi[0] + 1])
    single = compute_block_bootstrap_ci(x, **kwargs)
    np.testing.assert_array_equal(single[0], lo[:1])
    np.testing.assert_array_equal(single[1], hi[:1])
    order = [2, 0, 1]
    permuted = compute_block_bootstrap_ci(matrix[:, order], **kwargs)
    np.testing.assert_array_equal(permuted[0], lo[order])
    np.testing.assert_array_equal(permuted[1], hi[order])


def test_block_bootstrap_uses_independent_manual_block_oracle():
    x = np.arange(24, dtype=float) ** 2
    rng = np.random.default_rng(13)
    means = []
    for _ in range(40):
        starts = rng.integers(0, 21, size=6)
        draw = [x[start + offset] for start in starts for offset in range(4)]
        means.append(sum(draw) / 24)
    expected = np.quantile(means, [.05, .95])
    lo, hi = compute_block_bootstrap_ci(x, 4, 40, .9, 13)
    np.testing.assert_allclose([lo[0], hi[0]], expected, atol=1e-12)


def test_gapped_or_infinite_column_cannot_affect_valid_candidate_draws():
    x = np.random.default_rng(42).normal(size=80)
    y = x.copy()
    y[30] = np.nan
    z = x.copy()
    z[40] = np.inf
    kwargs = dict(num_bootstrap=80, random_seed=8)
    lower, upper = compute_block_bootstrap_ci(np.column_stack([y, x, z]), **kwargs)
    assert np.isnan(lower[[0, 2]]).all() and np.isnan(upper[[0, 2]]).all()
    alone = compute_block_bootstrap_ci(x, **kwargs)
    assert lower[1] == alone[0][0] and upper[1] == alone[1][0]


@pytest.mark.parametrize("kwargs", [{"max_lag": -1}, {"max_lag": True}, {"kernel": "bad"}])
def test_hac_invalid_policy_rejected_even_for_empty_input(kwargs):
    with pytest.raises(ValueError):
        compute_hac_tstat(np.empty((0, 2)), **kwargs)


@pytest.mark.parametrize("kwargs", [{"confidence_level": np.nan}, {"block_length": True},
                                    {"num_bootstrap": 1.5}, {"random_seed": -1}])
def test_bootstrap_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_block_bootstrap_ci(np.ones((80, 1)), **kwargs)


def test_l20_real_gpu_calendar_guards_and_shared_draws():
    cp = pytest.importorskip("cupy")
    from quant_evaluator.kernels.gpu.robustness import hac_variance, hac_tstat, block_bootstrap_ci
    x = np.random.default_rng(6).normal(size=80)
    gapped = x.copy()
    gapped[32] = np.nan
    values = np.column_stack([x, x, gapped])
    device = cp.asarray(values)
    np.testing.assert_allclose(hac_variance(device), compute_hac_variance(values), equal_nan=True)
    for actual, expected in zip(hac_tstat(device), compute_hac_tstat(values)):
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    kwargs = dict(num_bootstrap=100, random_seed=45)
    gpu = block_bootstrap_ci(device, **kwargs)
    cpu = compute_block_bootstrap_ci(values, **kwargs)
    for actual, expected in zip(gpu, cpu):
        np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-12)
        assert actual[0] == actual[1]


def test_public_hac_gap_invalid_and_semantic_versions_changed():
    from quant_evaluator.api.requests import EvaluationRequest
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    rng = np.random.default_rng(44)
    values = rng.normal(size=(80, 30, 1))
    labels = values[:, :, 0] * .3 + rng.normal(size=(80, 30))
    values[35] = np.nan
    batch = FactorBatch(("f",), AxisRef("time", "int64", 80, np.arange(80)),
                        AxisRef("asset", "int64", 30, np.arange(30)), values)
    label = LabelBundle("forward", labels, 1, decision_time=tuple(range(80)),
                        label_start_time=tuple(range(1, 81)), label_end_time=tuple(range(2, 82)),
                        asset_axis=batch.asset_axis)
    result = evaluate(EvaluationRequest(batch, label,
        metric_ids=("hac_tstat", "hac_pvalue", "block_bootstrap_ci"), tier="research"))
    for name in ("hac_tstat", "hac_pvalue", "block_bootstrap_ci"):
        metric = result.get_metric(name, "f")
        assert not metric.valid and metric.value is None
        assert metric.metric_version == ("2.0.0" if name == "block_bootstrap_ci" else "4.0.0")


def test_hac_dependent_corrections_also_have_new_public_semantic_version():
    from quant_evaluator.registry.metrics import get_metric
    for name in ("hac_tstat", "hac_pvalue", "block_bootstrap_ci",
                 "bonferroni_correction", "benjamini_hochberg_correction",
                 "holm_bonferroni_correction", "sidak_correction"):
        assert get_metric(name).metric_version == ("2.0.0" if name == "block_bootstrap_ci" else "4.0.0")
