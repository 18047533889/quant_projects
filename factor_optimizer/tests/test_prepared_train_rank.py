import sys

import numpy as np
import pandas as pd

from factor_optimizer.adapters.repair_execution import ValueRepairPlan, compile_value_repair
from factor_optimizer.shape_rank_reuse import (
    MAX_RANK_FREEZE_COPY_BYTES,
    RANK_CONTRACT,
    RankFeatureCache,
    _READ_ONLY_VALUE_REPAIR_TRANSFORMS,
    _candidate_frame_for_plan,
    apply_u_shape_from_prepared_rank,
)


def _frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
        "asset_id": ["a", "b", "c", "d"] * 2,
        "value": [1.0, 1.0, np.nan, 4.0, 3.0, 2.0, np.inf, 8.0],
    })


def _plan(family="U_SHAPE_REPAIR", center=.4, context="train:test"):
    return compile_value_repair(
        family, {"center": center, "power": 2.0, "asymmetry": True},
        natural_time_scale=10.0, training_context_ref=context,
    )


def test_prepared_rank_is_computed_once_and_matches_plan(monkeypatch):
    from factor_optimizer.adapters import repair_execution

    frame = _frame()
    plans = [_plan(family, center)
             for family in ("U_SHAPE_REPAIR", "INVERTED_U_REPAIR")
             for center in (.2, .4, .7)]
    expected = [plan.execute(frame, allow_research=True) for plan in plans]
    cache = RankFeatureCache(max_bytes=1_000_000)
    calls = []
    original = repair_execution._execute_fe_cs_rank

    def tracked(values):
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(repair_execution, "_execute_fe_cs_rank", tracked)
    prepared = cache.prepare_train_rank(frame, training_context_ref="train:test")
    assert prepared is not None
    for plan, expected_values in zip(plans, expected):
        actual = apply_u_shape_from_prepared_rank(
            plan, prepared, expected_index=frame.index, allow_research=True)
        pd.testing.assert_series_equal(actual, expected_values, check_exact=True)
    assert calls == [len(frame)]
    assert not prepared._rank.flags.writeable
    assert prepared.retained_bytes <= cache.max_bytes


def test_public_cache_still_fingerprints_mutable_dataframes():
    frame = _frame()
    cache = RankFeatureCache()
    cache.average_rank(frame, training_context_ref="train:test")
    changed = frame.copy(deep=True)
    changed.loc[0, "value"] = -123.0
    cache.average_rank(changed, training_context_ref="train:test")
    assert cache.rank_calls == 2
    assert cache.hits == 0


def test_rank_freeze_budget_is_distinct_and_covers_copy_peak(monkeypatch):
    import factor_optimizer.shape_rank_reuse as rank_reuse

    assert MAX_RANK_FREEZE_COPY_BYTES == 256 * 1024 * 1024
    rows = 8_500_000
    metadata = (sys.getsizeof(pd.RangeIndex(rows)) + sys.getsizeof("train:test")
                + sys.getsizeof(RANK_CONTRACT) + sys.getsizeof(None) + 512)
    assert rows * 8 + metadata <= 128 * 1024 * 1024
    assert 2 * rows * 8 <= MAX_RANK_FREEZE_COPY_BYTES
    frame = _frame()
    monkeypatch.setattr(rank_reuse, "MAX_RANK_FREEZE_COPY_BYTES", 2 * len(frame) * 8 - 1)
    cache = RankFeatureCache(max_bytes=1_000_000)
    assert cache.prepare_train_rank(frame, training_context_ref="train:test") is None
    assert cache.rank_calls == 0


def test_metadata_counts_toward_retained_budget_and_evicts_rank():
    frame = _frame()
    index = pd.RangeIndex(len(frame))
    metadata = (sys.getsizeof(index) + sys.getsizeof("train:test")
                + sys.getsizeof(RANK_CONTRACT) + sys.getsizeof(index.name) + 512)
    cache = RankFeatureCache(max_bytes=len(frame) * 8 + metadata - 1)
    assert cache.prepare_train_rank(frame, training_context_ref="train:test") is None
    assert cache.rank_calls == 1
    assert cache.retained_bytes == 0


def test_non_range_index_skips_precomputation(monkeypatch):
    from factor_optimizer.adapters import repair_execution

    frame = _frame()
    frame.index = pd.Index([f"row-{i}" for i in range(len(frame))])
    cache = RankFeatureCache(max_bytes=1_000_000)
    def unexpected(_frame):
        raise AssertionError("FE rank must not run during unsupported preparation")
    monkeypatch.setattr(repair_execution, "_execute_fe_cs_rank", unexpected)
    assert cache.prepare_train_rank(frame, training_context_ref="train:test") is None
    assert cache.rank_calls == 0


def test_prepared_rank_survives_source_mutation_and_rejects_context_mismatch():
    frame = _frame()
    plan = _plan()
    expected = plan.execute(frame, allow_research=True)
    cache = RankFeatureCache(max_bytes=1_000_000)
    prepared = cache.prepare_train_rank(frame, training_context_ref="train:test")
    assert prepared is not None
    frame.loc[:, "value"] = 500.0
    frame.loc[:, "asset_id"] = "changed"
    actual = apply_u_shape_from_prepared_rank(
        plan, prepared, expected_index=prepared.index, allow_research=True)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert apply_u_shape_from_prepared_rank(
        _plan(context="train:other"), prepared,
        expected_index=prepared.index, allow_research=True,
    ) is None
    assert apply_u_shape_from_prepared_rank(
        plan, prepared, expected_index=pd.RangeIndex(1, len(frame) + 1),
        allow_research=True,
    ) is None


def test_each_shared_transform_executes_without_mutating_source():
    params_by_transform = {
        "raw": {},
        "sign": {"multiplier": 1.0},
        "rank_shape": {"center": .4, "power": 2.0,
                       "asymmetric": True, "inverted": False},
        "cs_rank": {"method": "average"},
        "fp_cs_rank_min": {},
        "ts_rank_history": {"window": 2, "method": "average"},
        "ts_zscore_history": {"window": 2, "cap": 3.0},
        "trailing_sma": {"window": 2, "min_periods": 2},
        "capped_zscore": {"cap": 3.0},
        "tail_hinge": {"hinge": "top", "hinge_value": 1.5},
        "tail_saturation": {"quantile": .75, "side": "both"},
        "robust_scale": {"scale": "std", "center": "mean"},
    }
    assert set(params_by_transform) == _READ_ONLY_VALUE_REPAIR_TRANSFORMS
    frame = _frame()
    before = frame.copy(deep=True)
    for transform, params in params_by_transform.items():
        plan = ValueRepairPlan("TEST", transform, tuple(sorted(params.items())),
                               "train:test", 10.0)
        candidate = _candidate_frame_for_plan(plan, frame)
        assert candidate is frame
        plan.execute(candidate, allow_research=True)
        pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_unknown_plan_gets_isolated_candidate_frame():
    class UnknownPlan:
        transform = "unreviewed"

    frame = _frame()
    candidate = _candidate_frame_for_plan(UnknownPlan(), frame)
    assert candidate is not frame
    candidate.loc[:, "value"] = -100.0
    pd.testing.assert_series_equal(frame["value"], _frame()["value"])

def test_layered_decay_plan_shares_frame_and_successful_execution_is_read_only():
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan

    frame = _frame()
    before = frame.copy(deep=True)
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    candidate = _candidate_frame_for_plan(plan, frame)

    assert candidate is frame
    plan.execute(candidate, allow_research=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_layered_decay_failed_execution_does_not_mutate_shared_frame(monkeypatch):
    import pytest
    import factor_optimizer.adapters.layered_decay as layered_decay
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan

    frame = _frame()
    before = frame.copy(deep=True)
    plan = LayeredDecayPlan((5.0,) * 20, "train:test")
    candidate = _candidate_frame_for_plan(plan, frame)

    def fail_quantile_assignment(_values):
        raise RuntimeError("synthetic candidate failure")

    monkeypatch.setattr(layered_decay, "_assign_daily_quantiles", fail_quantile_assignment)
    assert candidate is frame
    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        plan.execute(candidate, allow_research=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_layered_decay_subclass_keeps_candidate_isolation():
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan

    class CustomLayeredDecayPlan(LayeredDecayPlan):
        pass

    frame = _frame()
    before = frame.copy(deep=True)
    plan = CustomLayeredDecayPlan((5.0,) * 20, "train:test")
    candidate = _candidate_frame_for_plan(plan, frame)

    assert candidate is not frame
    candidate.loc[:, "value"] = -100.0
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
