"""Reuse QE-produced candidate IC, never stale scores across effective masks."""
from dataclasses import replace
import numpy as np
import pytest
import runpy
from pathlib import Path


def fixture():
    return runpy.run_path(str(Path(__file__).with_name('test_pair_ic_cache.py')))['fixture']()


def test_coverage_candidate_ic_is_reused_by_joint_scoring(monkeypatch):
    from factor_optimizer.research_batch import _pair_ic, PairICCache, BatchOptimizationConfig
    from factor_optimizer.research_fitness import paired_series, RawSeriesCache
    from quant_evaluator.metrics import ic
    batch, labels = fixture()
    x = batch.values[:, :, 0]
    idx = tuple(range(40, 100))
    cache, raw_cache = PairICCache(), RawSeriesCache()
    expected = paired_series(x, -x, batch, labels, idx)
    _pair_ic(x, -x, batch, labels, idx, BatchOptimizationConfig(), candidate_cache=cache)
    real = ic.compute_daily_ic
    work = []
    def measured(factors, *args, **kwargs):
        work.append(len(factors.factor_ids))
        return real(factors, *args, **kwargs)
    monkeypatch.setattr(ic, 'compute_daily_ic', measured)
    for _ in range(2):
        actual = paired_series(x, -x, batch, labels, idx,
                               raw_cache=raw_cache, candidate_ic_cache=cache)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        actual[1][:] = 999.
    assert sum(work) == 1  # RAW once; candidate never recomputed.


@pytest.mark.parametrize('change', ['candidate', 'raw_mask', 'labels', 'validity',
                                    'minimum_assets', 'indices', 'precision'])
def test_candidate_reuse_invalidates_every_effective_input(change):
    from factor_optimizer.research_batch import _pair_ic, PairICCache, BatchOptimizationConfig
    from factor_optimizer.research_fitness import paired_series
    batch, labels = fixture()
    x, c = batch.values[:, :, 0].copy(), -batch.values[:, :, 0].copy()
    idx, minimum = tuple(range(40, 100)), 20
    if change == 'precision':
        tiny = np.finfo(np.longdouble).eps*4
        if tiny >= np.finfo(float).eps:
            pytest.skip('no extended precision')
        y = np.tile(np.longdouble(1)+np.arange(40, dtype=np.longdouble)*tiny, (300, 1))
        labels = replace(labels, values=y)
    cache = PairICCache()
    _pair_ic(x, c, batch, labels, idx, BatchOptimizationConfig(), candidate_cache=cache)
    if change == 'candidate':
        c[:, :20] *= -1
    elif change == 'raw_mask':
        x[45:55, :25] = np.nan
    elif change in ('labels', 'precision'):
        labels = replace(labels, values=labels.values[:, ::-1].copy())
    elif change == 'validity':
        v = np.ones(labels.values.shape, bool)
        v[45:55, :25] = False
        labels = replace(labels, validity=v)
    elif change == 'minimum_assets':
        minimum = 50
    else:
        idx = tuple(range(45, 105))
    expected = paired_series(x, c, batch, labels, idx, minimum_assets=minimum)
    actual = paired_series(x, c, batch, labels, idx, minimum_assets=minimum,
                           candidate_ic_cache=cache)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
