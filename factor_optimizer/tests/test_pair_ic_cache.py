"""Catch stale RAW IC reuse, changed coverage and ineffective caching."""
from dataclasses import replace
import importlib
import runpy
from pathlib import Path
import numpy as np
import pytest


def fixture():
    f = runpy.run_path(str(Path(__file__).with_name("test_research_diagnostics.py")))
    batch, labels = f["panel"](40)
    rng = np.random.default_rng(665)
    x = rng.normal(size=batch.values.shape[:2])
    y = .01*x + rng.normal(0, .03, x.shape)
    return replace(batch, values=x[:, :, None]), replace(labels, values=y)


def cache():
    module = importlib.import_module("factor_optimizer.research_batch")
    assert hasattr(module, "PairICCache"), "bounded paired IC cache is missing"
    return module.PairICCache()


def equal(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_cached_reference_avoids_recomputing_raw_but_keeps_exact_scores(monkeypatch):
    from factor_optimizer.research_batch import _pair_ic, BatchOptimizationConfig
    from quant_evaluator.runtime import evaluator
    c = cache()
    batch, labels = fixture()
    x = batch.values[:, :, 0]
    indices = tuple(range(40, 100))
    config = BatchOptimizationConfig()
    expected = _pair_ic(x, -x, batch, labels, indices, config)
    actual_evaluate = evaluator.evaluate
    work = []
    def observed(factors, *args, **kwargs):
        work.append(len(factors.factor_ids))
        return actual_evaluate(factors, *args, **kwargs)
    monkeypatch.setattr(evaluator, "evaluate", observed)
    equal(_pair_ic(x, -x, batch, labels, indices, config, reference_cache=c), expected)
    equal(_pair_ic(x, -2*x, batch, labels, indices, config, reference_cache=c), expected)
    assert sum(work) <= 3, "RAW references must not be evaluated again on a cache hit"


@pytest.mark.parametrize("change", ["candidate_missing", "labels", "validity", "raw", "minimum_assets"])
def test_cache_invalidates_effective_reference_inputs(change):
    from factor_optimizer.research_batch import _pair_ic, BatchOptimizationConfig
    c = cache()
    batch, labels = fixture()
    x, candidate = batch.values[:, :, 0].copy(), -batch.values[:, :, 0].copy()
    idx, config = tuple(range(40, 100)), BatchOptimizationConfig()
    _pair_ic(x, candidate, batch, labels, idx, config, reference_cache=c)
    if change == "candidate_missing":
        candidate[40:80, :25] = np.nan
    elif change == "labels":
        labels = replace(labels, values=-labels.values)
    elif change == "validity":
        validity = np.ones(labels.values.shape, bool)
        validity[40:80, :25] = False
        labels = replace(labels, validity=validity)
    elif change == "raw":
        x[:, :20] *= -1
    else:
        config = replace(config, minimum_assets=50)
    expected = _pair_ic(x, candidate, batch, labels, idx, config)
    actual = _pair_ic(x, candidate, batch, labels, idx, config, reference_cache=c)
    equal(actual, expected)
    if change == "candidate_missing":
        assert actual[2] == pytest.approx(1/3), "lost days must remain in RAW coverage denominator"


def test_cache_is_bounded_and_does_not_expose_mutable_evidence():
    c = cache()
    values = np.array([.2, np.nan])
    c.put("a", values)
    values[0] = 999
    read = c.get("a")
    assert read[0] == .2
    read[0] = 777
    assert c.get("a")[0] == .2
    c.put("b", values)
    c.put("c", values)
    assert c.get("a") is None
    assert c.get("b") is not None
    assert c.get("c") is not None


def test_extended_precision_labels_are_not_rounded_in_cache_identity():
    from factor_optimizer.research_batch import _pair_ic, BatchOptimizationConfig
    c = cache()
    batch, labels = fixture()
    x = batch.values[:, :, 0]
    tiny = np.finfo(np.longdouble).eps*4
    if tiny >= np.finfo(float).eps:
        pytest.skip("platform has no extended-precision longdouble")
    y = np.tile(np.longdouble(1)+np.arange(40, dtype=np.longdouble)*tiny, (300, 1))
    labels = replace(labels, values=y)
    idx, config = tuple(range(40, 100)), BatchOptimizationConfig()
    _pair_ic(x, -x, batch, labels, idx, config, reference_cache=c)
    changed = replace(labels, values=y[:, ::-1].copy())
    expected = _pair_ic(x, -x, batch, changed, idx, config)
    equal(_pair_ic(x, -x, batch, changed, idx, config, reference_cache=c), expected)


def test_joint_metric_cache_also_preserves_extended_precision_labels():
    from factor_optimizer.research_fitness import paired_series, RawSeriesCache
    batch, labels = fixture()
    x = batch.values[:, :, 0]
    tiny = np.finfo(np.longdouble).eps*4
    if tiny >= np.finfo(float).eps:
        pytest.skip("platform has no extended-precision longdouble")
    y = np.tile(np.longdouble(1)+np.arange(40, dtype=np.longdouble)*tiny, (300, 1))
    labels = replace(labels, values=y)
    idx, c = tuple(range(40, 100)), RawSeriesCache()
    paired_series(x, -x, batch, labels, idx, raw_cache=c)
    changed = replace(labels, values=y[:, ::-1].copy())
    expected = paired_series(x, -x, batch, changed, idx)
    actual = paired_series(x, -x, batch, changed, idx, raw_cache=c)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("change", [None, "target_id", "decision_time", "asset_axis"])
def test_candidate_cache_context_and_work_reuse(monkeypatch, change):
    from factor_optimizer.research_batch import _pair_ic, BatchOptimizationConfig
    from quant_evaluator.metrics import ic as ic_module
    from quant_evaluator.contracts.factor_batch import AxisRef
    batch, labels = fixture()
    x = batch.values[:, :, 0]
    idx, c, config = tuple(range(40, 100)), cache(), BatchOptimizationConfig()
    calls, compute = [], ic_module.compute_daily_ic
    def counted(factors, *args, **kwargs):
        calls.append(factors.num_factors)
        return compute(factors, *args, **kwargs)
    monkeypatch.setattr(ic_module, "compute_daily_ic", counted)
    first = _pair_ic(x, -x, batch, labels, idx, config, candidate_cache=c)
    other = labels
    if change == "target_id":
        other = replace(labels, target_id="other")
    elif change == "decision_time":
        other = replace(labels,
                        decision_time=tuple(t + 1000 for t in labels.decision_time),
                        label_start_time=tuple(t + 1000 for t in labels.label_start_time),
                        label_end_time=tuple(t + 1000 for t in labels.label_end_time))
    elif change == "asset_axis":
        coords = np.asarray(labels.asset_axis.values).copy(); coords[0] = "other"
        other = replace(labels, asset_axis=AxisRef("asset", labels.asset_axis.dtype, len(coords), coords))
    second = _pair_ic(x, -x, batch, other, idx, config, candidate_cache=c)
    equal(first, second)
    assert calls == ([3, 2] if change is None else [3, 3])


def test_prepared_train_split_reuses_label_slice_without_changing_ic(monkeypatch):
    import factor_optimizer.research_batch as rb

    batch, labels = fixture()
    raw = batch.values[:, :, 0]
    indices = tuple(range(40, 100))
    config = rb.BatchOptimizationConfig()
    expected = rb._pair_ic(raw, -raw, batch, labels, indices, config)
    prepared = rb._prepare_pair_ic_split(batch, labels, indices)

    def unexpected_slice(*_args, **_kwargs):
        pytest.fail("prepared TRAIN split must not rebuild LabelBundle")

    monkeypatch.setattr(rb, "_subset_labels", unexpected_slice)
    actual = rb._pair_ic(
        raw, -raw, batch, labels, indices, config, prepared_split=prepared)
    equal(actual, expected)
    with pytest.raises(ValueError, match="does not match"):
        rb._pair_ic(
            raw, -raw, batch, labels, indices[:-1], config,
            prepared_split=prepared)
    with pytest.raises(ValueError, match="does not match"):
        rb._pair_ic(
            raw, -raw, batch, replace(labels, target_id="different"),
            indices, config, prepared_split=prepared)


def test_batch_scope_pair_ic_caches_reuse_identical_factor_panels(monkeypatch):
    from factor_optimizer.research_batch import _pair_ic, BatchOptimizationConfig
    from quant_evaluator.metrics import ic as ic_module

    batch, labels = fixture()
    raw = batch.values[:, :, 0]
    candidate = -raw
    indices = tuple(range(40, 100))
    config = BatchOptimizationConfig()
    expected = _pair_ic(raw, candidate, batch, labels, indices, config)
    reference_cache, candidate_cache = cache(), cache()
    real_compute = ic_module.compute_daily_ic
    computed_columns = []

    def counted(factors, *args, **kwargs):
        computed_columns.append(factors.num_factors)
        return real_compute(factors, *args, **kwargs)

    monkeypatch.setattr(ic_module, "compute_daily_ic", counted)
    first = _pair_ic(raw, candidate, batch, labels, indices, config,
                     reference_cache=reference_cache, candidate_cache=candidate_cache)
    second = _pair_ic(raw.copy(), candidate.copy(), batch, labels, indices, config,
                      reference_cache=reference_cache, candidate_cache=candidate_cache)
    equal(first, expected)
    equal(second, expected)
    assert computed_columns == [2]


@pytest.mark.parametrize("change", ["candidate", "labels"])
def test_batch_scope_pair_ic_caches_miss_changed_values_or_labels(monkeypatch, change):
    from factor_optimizer.research_batch import _pair_ic, BatchOptimizationConfig
    from quant_evaluator.metrics import ic as ic_module

    batch, labels = fixture()
    raw = batch.values[:, :, 0]
    candidate = -raw.copy()
    other_labels = labels
    indices = tuple(range(40, 100))
    config = BatchOptimizationConfig()
    reference_cache, candidate_cache = cache(), cache()
    _pair_ic(raw, candidate, batch, labels, indices, config,
             reference_cache=reference_cache, candidate_cache=candidate_cache)
    if change == "candidate":
        candidate[40:80, :25] *= -1
    else:
        other_labels = replace(labels, values=-labels.values)
    expected = _pair_ic(raw, candidate, batch, other_labels, indices, config)

    real_compute = ic_module.compute_daily_ic
    computed_columns = []
