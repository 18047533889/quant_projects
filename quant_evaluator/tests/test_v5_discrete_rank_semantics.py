import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from quant_evaluator.metrics.ic import _spearman_rank_correlation


def test_binary_spearman_is_defined_separately_from_evidence_quality():
    x = np.repeat([0., 1.], 20)
    y = np.arange(40, dtype=float)
    assert _spearman_rank_correlation(x, y, min_obs=20) == pytest.approx(spearmanr(x, y).statistic)
    assert np.isnan(_spearman_rank_correlation(np.zeros(40), y, min_obs=20))


def test_actual_cuda_binary_spearman_and_finite_rank():
    import cupy as cp
    from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic
    from quant_evaluator.kernels.gpu.rank import batched_rank, batched_distinct_level_count
    x = np.repeat([0., 1.], 20)
    y = np.arange(40, dtype=float)
    result, count = batched_spearman_ic(cp.asarray(x[None, None, :]), cp.asarray(y[None, :]), min_obs=20)
    assert cp.asnumpy(result)[0, 0] == pytest.approx(spearmanr(x, y).statistic)
    assert int(cp.asnumpy(count)[0, 0]) == 40
    values = np.array([[3., np.inf, 1., -np.inf, np.nan, 3.]])
    expected = np.full_like(values, np.nan)
    mask = np.isfinite(values[0])
    expected[0, mask] = rankdata(values[0, mask], method="average")
    np.testing.assert_allclose(cp.asnumpy(batched_rank(cp.asarray(values))), expected, equal_nan=True)
    np.testing.assert_allclose(cp.asnumpy(batched_rank(cp.asarray(values), pct=True)), expected / 3., equal_nan=True)
    assert cp.asnumpy(batched_distinct_level_count(cp.asarray(values))).tolist() == [2]


def test_rank_axis_is_honored_and_unsupported_tie_policy_rejected():
    import cupy as cp
    from quant_evaluator.kernels.gpu.rank import batched_rank
    values = np.array([[2., 3., 5.], [1., 4., 2.]])
    np.testing.assert_allclose(cp.asnumpy(batched_rank(cp.asarray(values), axis_n=0)), rankdata(values, axis=0))
    with pytest.raises(ValueError, match="method"):
        batched_rank(cp.asarray(values), method="dense")
    with pytest.raises(ValueError, match="axis"):
        batched_rank(cp.asarray(values), axis_n=3)


def test_empty_rank_axis_has_no_phantom_members():
    import cupy as cp
    from quant_evaluator.kernels.gpu.rank import batched_rank, batched_distinct_level_count
    empty = cp.empty((2, 0))
    assert batched_rank(empty).shape == (2, 0)
    np.testing.assert_array_equal(cp.asnumpy(batched_distinct_level_count(empty)), [0, 0])


def test_quantile_minimum_is_checked_per_actual_bucket():
    from quant_evaluator.metrics.probe_portfolio._core import build_quantile_masks
    values = np.arange(100,dtype=float)[None,:]
    masks, diagnostics = build_quantile_masks(values,20,10,return_diagnostics=True)
    assert not masks.any()
    np.testing.assert_array_equal(diagnostics["bucket_counts"],np.full((20,1),5))
    assert diagnostics["actual_bucket_count"].tolist() == [20]
    assert not diagnostics["bucket_valid"].any()
    uneven = np.array([[0.]+[1.]*99])
    masks, diagnostics = build_quantile_masks(uneven,20,5,return_diagnostics=True)
    singleton=diagnostics["bucket_counts"]==1
    assert singleton.any()
    assert not diagnostics["bucket_valid"][singleton].any()
    assert all(np.count_nonzero(row) == 0 or np.count_nonzero(row) >= 5 for row in masks[:,0])


def test_finite_quantile_assignment_and_unsupported_methods_are_explicit():
    import cupy as cp
    from quant_evaluator.kernels.gpu.rank import batched_quantile_assignment
    values=cp.array([[1.,2.,3.,4.,cp.inf,-cp.inf,cp.nan]])
    result=cp.asnumpy(batched_quantile_assignment(values,2))
    assert result[0,4:].tolist() == [-1,-1,-1]
    assert sorted(result[0,:4].tolist()) == [0,0,1,1]
    with pytest.raises(ValueError,match="method"):
        batched_quantile_assignment(values,2,method="average")


def test_raw_rank_icir_aliases_share_registry_authority():
    from quant_evaluator.contracts.metric_instance import MetricInstance
    from quant_evaluator.registry.metrics import get_metric
    assert MetricInstance("rank_icir_raw").instance_id == MetricInstance("ic_ir").instance_id
    assert get_metric("rank_ic").metric_version == "3.0.0"
    assert get_metric("ic_ir").metric_version == "3.0.0"
