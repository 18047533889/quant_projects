"""Exact same-mask sign-opposite RankIC cache reuse."""
from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic
from factor_optimizer.research_batch import (
    BatchOptimizationConfig, PairICCache, _candidate_ic_key, _pair_ic, _subset_labels,
)
from factor_optimizer.research_ic_antithetic import cache_negated_candidate_ic


def _fixture(dtype=np.float64, label_dtype=np.float64):
    rng = np.random.default_rng(20261002)
    t, n = 64, 73
    candidate = rng.normal(size=(t, n)).astype(dtype)
    candidate[:, ::7] = np.round(candidate[:, ::7], 1)
    candidate[::13] = 0.0
    candidate[::19, 1] = np.nan
    labels = rng.normal(size=(t, n)).astype(label_dtype)
    labels[:, ::9] = np.round(labels[:, ::9], 1)
    if label_dtype == np.longdouble:
        labels = labels + np.arange(t, dtype=np.longdouble)[:, None] * np.finfo(np.longdouble).eps
    raw = rng.normal(size=(t, n)).astype(dtype)
    raw[::17, 2] = np.nan
    label_valid = rng.random((t, n)) > .07
    factor_valid = rng.random((t, n)) > .11
    common = (np.isfinite(raw) & np.isfinite(candidate) & np.isfinite(labels)
              & factor_valid & label_valid)
    times = np.arange(t)
    asset_axis = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    time_axis = AxisRef("time", "int", t, times)
    target = LabelBundle(
        "antithetic-synthetic", labels, 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2), validity=label_valid, asset_axis=asset_axis,
    )
    return candidate, common, target, time_axis, asset_axis


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("label_dtype", [np.float64, np.longdouble])
def test_antithetic_cache_key_and_ic_match_direct_negative_compute(dtype, label_dtype):
    candidate, common, target, time_axis, asset_axis = _fixture(dtype, label_dtype)
    ids = ("candidate",)
    positive = FactorBatch(ids, time_axis, asset_axis, candidate[:, :, None],
                           validity=common[:, :, None])
    negative = FactorBatch(ids, time_axis, asset_axis, (-candidate)[:, :, None],
                           validity=common[:, :, None])
    positive_ic, positive_counts = compute_daily_ic(
        positive, target, method="spearman", min_assets=20)
    negative_ic, negative_counts = compute_daily_ic(
        negative, target, method="spearman", min_assets=20)

    assert np.array_equal(negative_counts, positive_counts)
    assert np.array_equal(np.isnan(negative_ic), np.isnan(positive_ic))
    finite = np.isfinite(positive_ic)
    assert np.array_equal(negative_ic[finite], -positive_ic[finite])
    assert np.array_equal(negative_ic[finite].view(np.uint64),
                          (-positive_ic[finite]).view(np.uint64))

    cache = PairICCache()
    key = cache_negated_candidate_ic(cache, candidate, common, target, 20,
                                     positive_ic[:, 0])
    expected_key = _candidate_ic_key(np.where(common, -candidate, np.nan), target, 20)
    assert key == expected_key
    cached = cache.get(expected_key)
    assert cached is not None
    assert np.array_equal(cached, negative_ic[:, 0], equal_nan=True)
    assert np.array_equal(cached.view(np.uint64), negative_ic[:, 0].view(np.uint64))
    # Returned evidence cannot mutate the cache entry.
    cached[:] = 99
    assert np.array_equal(cache.get(expected_key), negative_ic[:, 0], equal_nan=True)


def test_key_preserves_signed_zero_and_effective_common_mask_identity():
    candidate, common, target, _, _ = _fixture()
    row, column = np.argwhere(target.validity & np.isfinite(target.values))[0]
    candidate[row, column] = 0.0
    common[row, column] = True
    cache = PairICCache()
    key = cache_negated_candidate_ic(cache, candidate, common, target, 20,
                                     np.ones(len(candidate)))
    expected = _candidate_ic_key(np.where(common, -candidate, np.nan), target, 20)
    assert key == expected
    assert np.signbit(np.where(common, -candidate, np.nan)[row, column])
    changed_mask = common.copy()
    changed_mask[row, column] = False
    changed = _candidate_ic_key(np.where(changed_mask, -candidate, np.nan), target, 20)
    assert changed != key


@pytest.mark.parametrize("candidate, mask, minimum_assets, ic", [
    (np.ones((4, 3), dtype=bool), np.ones((4, 3), dtype=bool), 2, np.ones(4)),
    (np.ones((4, 3)), np.ones((4, 3), dtype=np.uint8), 2, np.ones(4)),
    (np.ones((4, 3)), np.ones((4, 3), dtype=bool), True, np.ones(4)),
    (np.ones((4, 3)), np.ones((4, 3), dtype=bool), 2, np.ones((4, 1))),
    (np.ones((4, 3)), np.ones((4, 3), dtype=bool), 2, np.array([1., 1., np.inf, 1.])),
    (np.ones((0, 3)), np.ones((0, 3), dtype=bool), 2, np.array([])),
])
def test_invalid_antithetic_inputs_are_rejected(candidate, mask, minimum_assets, ic):
    _, _, target, _, _ = _fixture()
    target = target.__class__(
        "invalid-input", np.ones((4, 3)), 1,
        decision_time=(0, 1, 2, 3), label_start_time=(1, 2, 3, 4),
        label_end_time=(2, 3, 4, 5),
        asset_axis=AxisRef("asset", "str", 3, np.array(["a", "b", "c"])),
    )
    with pytest.raises((TypeError, ValueError)):
        cache_negated_candidate_ic(PairICCache(), candidate, mask, target,
                                  minimum_assets, ic)


def test_pair_ic_opt_in_populates_negative_cache_without_second_ic_call(monkeypatch):
    rng = np.random.default_rng(3002)
    t, n = 64, 73
    raw = rng.normal(size=(t, n))
    candidate = raw + rng.normal(size=(t, n))
    raw[::13, 2] = np.nan
    candidate[::17, 3] = np.nan
    labels_values = rng.normal(size=(t, n))
    label_validity = rng.random((t, n)) > .05
    times = np.arange(t)
    time_axis = AxisRef("time", "int", t, times)
    asset_axis = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("raw",), time_axis, asset_axis, raw[:, :, None])
    labels = LabelBundle(
        "pair-ic-antithetic", labels_values, 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2), validity=label_validity, asset_axis=asset_axis,
    )
    indices = tuple(range(t))
    config = BatchOptimizationConfig(minimum_assets=20)
    reference_cache, candidate_cache = PairICCache(), PairICCache()

    import quant_evaluator.metrics.ic as ic_module
    original = ic_module.compute_daily_ic
    calls = []

    def observe(*args, **kwargs):
        calls.append(args[0].num_factors)
        return original(*args, **kwargs)

    monkeypatch.setattr(ic_module, "compute_daily_ic", observe)
    positive = _pair_ic(raw, candidate, batch, labels, indices, config,
                        reference_cache=reference_cache, candidate_cache=candidate_cache,
                        cache_opposite_candidate=True)
    calls_after_positive = len(calls)
    assert calls_after_positive == 1

    target = _subset_labels(labels, indices)
    available = np.isfinite(raw) & np.isfinite(target.values)
    available &= target.validity
    common = available & np.isfinite(-candidate)
    expected_key = _candidate_ic_key(np.where(common, -candidate, np.nan), target,
                                     config.minimum_assets)
    assert candidate_cache.get(expected_key) is not None
    label_values_changed = labels.values.copy()
    label_values_changed[0, 0] += .25
    changed_labels = _subset_labels(replace(labels, values=label_values_changed), indices)
    assert candidate_cache.get(_candidate_ic_key(
        np.where(common, -candidate, np.nan), changed_labels, config.minimum_assets)) is None
    changed_mask = common.copy()
    row, column = np.argwhere(common)[0]
    changed_mask[row, column] = False
    assert candidate_cache.get(_candidate_ic_key(
        np.where(changed_mask, -candidate, np.nan), target, config.minimum_assets)) is None
    assert candidate_cache.get(_candidate_ic_key(
        np.where(common, -candidate, np.nan), target, config.minimum_assets + 1)) is None

    negative = _pair_ic(raw, -candidate, batch, labels, indices, config,
                        reference_cache=reference_cache, candidate_cache=candidate_cache)
    assert len(calls) == calls_after_positive
    assert np.array_equal(positive[1], negative[1])
    assert positive[2] == negative[2]


def test_exact_zero_ic_falls_back_to_bit_exact_negative_computation(monkeypatch):
    t, n = 32, 4
    x = np.tile(np.array([0., 1., 2., 3.]), (t, 1))
    y = np.tile(np.array([1., 3., 0., 2.]), (t, 1))
    common = np.ones((t, n), dtype=bool)
    times = np.arange(t)
    time_axis = AxisRef("time", "int", t, times)
    asset_axis = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("raw",), time_axis, asset_axis, x[:, :, None])
    labels = LabelBundle(
        "orthogonal-ranks", y, 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2), asset_axis=asset_axis,
    )
    positive_batch = FactorBatch(("candidate",), time_axis, asset_axis, x[:, :, None],
                                 validity=common[:, :, None])
    negative_batch = FactorBatch(("candidate",), time_axis, asset_axis, (-x)[:, :, None],
                                 validity=common[:, :, None])
    positive_ic, _ = compute_daily_ic(positive_batch, labels, method="spearman", min_assets=3)
    negative_ic, _ = compute_daily_ic(negative_batch, labels, method="spearman", min_assets=3)
    assert np.isfinite(positive_ic).all() and np.all(positive_ic == 0.0)
    assert np.isfinite(negative_ic).all() and np.all(negative_ic == 0.0)

    cache = PairICCache()
    assert cache_negated_candidate_ic(cache, x, common, labels, 3, positive_ic[:, 0]) is None
    negative_key = _candidate_ic_key(np.where(common, -x, np.nan), labels, 3)
    assert cache.get(negative_key) is None

    config = BatchOptimizationConfig(minimum_assets=3)
    indices = tuple(range(t))
    refs, candidates = PairICCache(), PairICCache()
    import quant_evaluator.metrics.ic as ic_module
    original = ic_module.compute_daily_ic
    calls = []

    def observe(*args, **kwargs):
        calls.append(args[0].num_factors)
        return original(*args, **kwargs)

    monkeypatch.setattr(ic_module, "compute_daily_ic", observe)
    _pair_ic(x, x, batch, labels, indices, config, reference_cache=refs,
             candidate_cache=candidates, cache_opposite_candidate=True)
    after_positive = len(calls)
    _pair_ic(x, -x, batch, labels, indices, config, reference_cache=refs,
             candidate_cache=candidates)
    assert len(calls) == after_positive + 1

    uncached = _pair_ic(x, -x, batch, labels, indices, config)
    reused_fallback = _pair_ic(x, -x, batch, labels, indices, config,
                               reference_cache=PairICCache(), candidate_cache=PairICCache())
    for left, right in zip(uncached, reused_fallback):
        assert np.array_equal(left, right, equal_nan=True)


def test_zero_ic_fallback_still_validates_mask_and_identity_inputs():
    candidate, common, target, _, _ = _fixture()
    zero_ic = np.zeros(len(candidate))
    with pytest.raises(ValueError, match="common_mask"):
        cache_negated_candidate_ic(PairICCache(), candidate,
            common.astype(np.uint8), target, 20, zero_ic)
    with pytest.raises(ValueError, match="minimum_assets"):
        cache_negated_candidate_ic(PairICCache(), candidate, common,
            target, True, zero_ic)
    invalid_common = common.copy()
    row, column = np.argwhere(~target.validity)[0]
    candidate[row, column] = 0.0
    invalid_common[row, column] = True
    with pytest.raises(ValueError, match="common_mask"):
        cache_negated_candidate_ic(PairICCache(), candidate, invalid_common,
            target, 20, zero_ic)


def test_full_optimizer_train_scores_and_validation_match_without_helper(monkeypatch):
    import json
    from factor_optimizer.research_batch import optimize_factor_batch
    import factor_optimizer.adapters.preprocessing as preprocessing
    import factor_optimizer.candidate_catalog as candidate_catalog
    import factor_optimizer.research_ic_antithetic as antithetic
    import quant_evaluator.metrics.ic as ic_module

    rng = np.random.default_rng(9041)
    t, n = 240, 40
    times = np.arange(t)
    signal = np.tile(np.linspace(-1., 1., n), (t, 1))
    labels_values = signal.copy()
    values = signal + .6 * rng.normal(size=(t, n))
    time_axis = AxisRef("time", "int", t, times)
    asset_axis = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("signal",), time_axis, asset_axis, values[:, :, None])
    labels = LabelBundle(
        "optimizer-antithetic", labels_values, 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2), asset_axis=asset_axis,
    )
    config = BatchOptimizationConfig(
        families=("CAUSAL_SMOOTHING", "SIGN_ORIENTATION"), selection_objective="rank_ic",
        bootstrap_draws=99,
    )
    def one_plan(*, natural_time_scale, training_context_ref, **_kwargs):
        return (preprocessing.compile_smoothing_repair(
            "CAUSAL_SMOOTHING",
            {"method": "EWMA", "natural_time_scale_relative": .5},
            natural_time_scale=natural_time_scale,
            training_context_ref=training_context_ref,
        ),)

    monkeypatch.setattr(preprocessing, "compile_admissible_smoothing_grid", one_plan)
    monkeypatch.setattr(candidate_catalog, "optimizer_candidate_specs", lambda _config: [])
    original_ic = ic_module.compute_daily_ic
    calls = []

    def observe(*args, **kwargs):
        calls.append(args[0].num_factors)
        return original_ic(*args, **kwargs)

    monkeypatch.setattr(ic_module, "compute_daily_ic", observe)
    with_helper = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    helper_calls = len(calls)
    calls.clear()
    monkeypatch.setattr(antithetic, "cache_negated_candidate_ic", lambda *a, **k: None)
    without_helper = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    no_helper_calls = len(calls)

    left, right = with_helper.factors["signal"], without_helper.factors["signal"]
    assert (left.status, left.selected_family, left.plan_identity, left.train_gain,
            left.validation_lower_bound, left.validation_candidate_identity,
            left.validation_coverage, left.reason) == (
            right.status, right.selected_family, right.plan_identity, right.train_gain,
            right.validation_lower_bound, right.validation_candidate_identity,
            right.validation_coverage, right.reason)
    assert json.dumps([dict(x) for x in left.candidates], sort_keys=True, allow_nan=True) == \
           json.dumps([dict(x) for x in right.candidates], sort_keys=True, allow_nan=True)
    assert np.array_equal(with_helper.optimized.values, without_helper.optimized.values,
                          equal_nan=True)
    assert helper_calls == no_helper_calls - 1
    assert [record["orientation"] for record in left.candidates] == [1, -1]
