import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.interactions.substitution import (
    FactorRelation, compute_marginal_contribution, detect_substitutable_factors,
)
from quant_evaluator.metrics.interactions.conditional import (
    SAME_DATE_DESCRIPTIVE, compute_conditional_ic, compute_incremental_ic,
)
from quant_evaluator.metrics.interactions.pairwise import (
    compute_correlation_matrix, compute_pairwise_artifacts,
    compute_pairwise_correlation, compute_rolling_pairwise_correlation,
)


def bundles(values, labels):
    t, n, f = values.shape
    fb = FactorBatch(tuple(f"f{i}" for i in range(f)), AxisRef("t", "int", t),
                     AxisRef("a", "int", n), values)
    times = tuple(range(t))
    lb = LabelBundle("y", labels, 1, decision_time=times,
                     label_start_time=times, label_end_time=tuple(x + 1 for x in times))
    return fb, lb


def test_x02_complement_is_not_substitute_orientation_and_budget():
    n, t = 64, 20
    a = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    b = np.tile([1.0, -1.0], n // 2)
    fb, lb = bundles(np.tile(np.c_[a, b], (t, 1, 1)), np.tile(a + b, (t, 1)))
    assert detect_substitutable_factors(fb, lb, min_assets=30, min_periods=t) == []
    evidence = detect_substitutable_factors(
        fb, lb, min_assets=30, min_periods=t, return_evidence=True,
    )
    assert evidence[0].relation is FactorRelation.COMPLEMENTARY
    assert evidence[0].orientation == "frozen_as_supplied"
    assert evidence[0].signed_similarity == pytest.approx(0.0, abs=1e-12)
    assert evidence[0].factor_id_a == "f0" and evidence[0].factor_id_b == "f1"
    assert evidence[0].common_support_ref.startswith("common-support:")
    assert evidence[0].method_version == "paired_relation.v2"
    with pytest.raises(ValueError, match="budget"):
        detect_substitutable_factors(fb, lb, candidate_pairs=[(0, 1)], max_pairs=0)
    same, same_lb = bundles(np.tile(np.c_[a, a], (t, 1, 1)), np.tile(a, (t, 1)))
    assert detect_substitutable_factors(same, same_lb, min_periods=t)
    opposite, opposite_lb = bundles(np.tile(np.c_[a, -a], (t, 1, 1)), np.tile(a, (t, 1)))
    assert detect_substitutable_factors(opposite, opposite_lb, min_periods=t) == []
    aa = np.tile([1.0, -1.0], n // 2)
    masked_values = np.tile(np.c_[aa, np.r_[aa[:32], -aa[32:]]], (t, 1, 1))
    validity = np.ones_like(masked_values, dtype=bool); validity[:, 32:, 1] = False
    masked_fb, masked_lb = bundles(masked_values, np.tile(aa, (t, 1)))
    masked_fb = FactorBatch(masked_fb.factor_ids, masked_fb.time_axis, masked_fb.asset_axis,
                            masked_fb.values, validity=validity)
    masked_evidence = detect_substitutable_factors(
        masked_fb, masked_lb, min_periods=t, return_evidence=True,
    )
    assert masked_evidence[0].signed_similarity == pytest.approx(1.0)


def test_x03_common_support_constant_candidate_and_conditional_counts():
    n = 64
    baseline = np.r_[np.arange(32.0), np.arange(32.0)[::-1]]
    labels = np.r_[np.arange(32.0), np.arange(32.0)]
    target = np.r_[np.ones(32), np.full(32, np.nan)]
    fb, lb = bundles(np.c_[baseline, target][None, :, :], labels[None, :])
    lift, base, diagnostic = compute_marginal_contribution(
        fb, lb, 1, (0,), min_assets=30, return_diagnostics=True,
    )
    assert lift[0] == pytest.approx(0.0, abs=1e-12)
    assert base[0] == pytest.approx(1.0)
    assert diagnostic.common_support_counts[0] == 32
    assert diagnostic.native_baseline_counts[0] == 64
    assert diagnostic.native_target_counts[0] == 32
    assert diagnostic.native_baseline_ic[0] == pytest.approx(0.0, abs=1e-12)
    assert diagnostic.method_version == "paired_marginal.v2"

    cond = np.arange(n, dtype=float)
    tested = np.sin(cond)
    y = tested.copy(); y[0] = np.nan
    cfb, clb = bundles(np.c_[cond, tested][None, :, :], y[None, :])
    _, counts = compute_conditional_ic(cfb, clb, 0, quantiles=2, min_assets=5)
    assert counts.shape == (1, 2, 2)
    assert counts[0, 0].sum() == n - 1


def test_x04_pairwise_invariant_to_nan_factor_and_constant_diagonal_undefined():
    n = 64
    a = np.arange(n, dtype=float); b = 3 * a + 2
    fb2, _ = bundles(np.c_[a, b][None, :, :], a[None, :])
    fb3, _ = bundles(np.c_[a, b, np.full(n, np.nan)][None, :, :], a[None, :])
    assert compute_correlation_matrix(fb2, cross_sectional=True)[0, 1] == pytest.approx(
        compute_correlation_matrix(fb3, cross_sectional=True)[0, 1])
    sparse = compute_pairwise_artifacts(
        fb3, candidate_pairs=[(0, 1)], max_pairs=1, min_obs=30,
        window_ref="w", universe_ref="u", sample_ref="s",
    )
    assert len(sparse) == 1 and sparse[0].pair_count == n
    with pytest.raises(ValueError, match="budget"):
        compute_pairwise_artifacts(
            fb3, candidate_pairs=[(0, 1)], max_pairs=0, min_obs=30,
            window_ref="w", universe_ref="u", sample_ref="s",
        )
    constant, _ = bundles(np.ones((1, n, 1)), a[None, :])
    assert np.isnan(compute_correlation_matrix(constant, cross_sectional=True)[0, 0])
    assert np.isnan(compute_pairwise_correlation(constant, min_obs=30)[0, 0])
    assert np.isnan(compute_rolling_pairwise_correlation(
        constant, window=1, min_obs=30,
    )[0, 0, 0])


def test_x04_hundred_thousand_factor_plan_rejects_before_dense_or_pair_allocation():
    factor_count = 100_000
    huge = FactorBatch(
        tuple(f"f{i}" for i in range(factor_count)), AxisRef("t", "int", 1),
        AxisRef("a", "int", 0), np.empty((1, 0, factor_count)),
    )
    with pytest.raises(ValueError, match="candidate pair budget exceeded"):
        compute_pairwise_artifacts(
            huge, window_ref="w", universe_ref="u", sample_ref="s",
        )
    with pytest.raises(ValueError, match="dense pairwise output budget"):
        compute_pairwise_correlation(huge)
    with pytest.raises(ValueError, match="dense correlation output budget"):
        compute_correlation_matrix(huge, cross_sectional=True)
    with pytest.raises(ValueError, match="dense rolling pairwise output budget"):
        compute_rolling_pairwise_correlation(huge, window=1)


def test_x08_x09_descriptive_scope_duplicate_controls_and_zero_residual():
    rng = np.random.default_rng(4); n = 64
    control = rng.normal(size=n); test = rng.normal(size=n); label = test + control
    fb1, lb1 = bundles(np.c_[control, test][None, :, :], label[None, :])
    one = compute_incremental_ic(fb1, lb1, (0,), 1, return_diagnostics=True)
    fb2, lb2 = bundles(np.c_[control, control, test][None, :, :], label[None, :])
    dup = compute_incremental_ic(fb2, lb2, (0, 1), 2, return_diagnostics=True)
    assert one[0][0] == pytest.approx(dup[0][0], abs=1e-12)
    assert dup[3].estimation_scope == SAME_DATE_DESCRIPTIVE
    assert dup[3].test_projection[0]["status"] == "RANK_DEFICIENT"

    exact, exact_lb = bundles(np.c_[control, control][None, :, :], control[None, :])
    result = compute_incremental_ic(exact, exact_lb, (0,), 1, return_diagnostics=True)
    assert np.isnan(result[0][0])
    assert result[3].test_projection[0]["status"] == "NO_RESIDUAL_VARIANCE"

    # Centered authority remains stable when values have a huge location offset.
    huge = 1e12 + np.arange(n, dtype=float) * 1e-2
    residual_signal = np.sin(np.arange(n))
    large, large_lb = bundles(
        np.c_[huge, huge + residual_signal][None, :, :], residual_signal[None, :]
    )
    large_result = compute_incremental_ic(
        large, large_lb, (0,), 1, method="spearman", return_diagnostics=True,
    )
    assert np.isfinite(large_result[0][0])
    assert large_result[3].method_definition == "raw_value_residuals_then_spearman_correlation"


def test_x09_gpu_near_collinear_controls_preserve_real_residual():
    cp = pytest.importorskip("cupy")
    from quant_evaluator.kernels.gpu.interactions import batched_incremental_ic

    n = 40
    basis = np.linalg.qr(np.column_stack((
        np.ones(n), np.linspace(-1, 1, n), np.linspace(-1, 1, n) ** 2,
        np.linspace(-1, 1, n) ** 3,
    )))[0]
    z, almost_duplicate, signal = basis[:, 1], basis[:, 1] + 1e-12 * basis[:, 2], basis[:, 3]
    values = np.c_[z, almost_duplicate, signal][None, :, :]
    fb, lb = bundles(values, signal[None, :])
    cpu = compute_incremental_ic(fb, lb, (0, 1), 2, min_assets=30)[0]
    gpu = batched_incremental_ic(
        cp.asarray(values.transpose(0, 2, 1)), cp.asarray(signal[None, :]),
        (0, 1), 2, min_assets=30,
    )[0]
    assert np.isfinite(cpu[0]) and np.isfinite(float(gpu[0]))
    assert float(gpu[0]) == pytest.approx(cpu[0], abs=1e-10)
