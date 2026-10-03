"""Characterize the TRAIN search frame transport before copy removal."""

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _fixture():
    rng = np.random.default_rng(82)
    times, assets = 240, 40
    y = np.stack([
        rng.permutation(np.linspace(-1, 1, assets)) for _ in range(times)
    ]) ** 2
    values = np.stack((y, -y), axis=-1)
    validity = np.ones(values.shape, dtype=bool)
    validity[::17, :2, 0] = False
    values[~validity] = np.nan
    time_axis = AxisRef("time", "int", times, np.arange(times))
    asset_axis = AxisRef(
        "asset", "str", assets, np.array([f"a{i}" for i in range(assets)])
    )
    batch = FactorBatch(
        ("baseline", "reverse"), time_axis, asset_axis, values,
        validity=validity,
    )
    labels = LabelBundle(
        "synthetic", y, 1, decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )
    return batch, labels


def _run(batch, labels):
    from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch

    config = BatchOptimizationConfig(
        families=("SIGN_ORIENTATION",), selection_objective="rank_ic",
        bootstrap_draws=99,
    )
    return optimize_factor_batch(batch, labels, config=config, allow_research=True)


def _ledger(result):
    return {
        factor_id: (
            item.status, item.selected_family, item.plan_identity,
            item.train_gain, item.validation_lower_bound,
            item.validation_candidate_identity,
            tuple((candidate.get("family"), candidate.get("parameters"),
                   candidate.get("orientation"), candidate.get("status"),
                   candidate.get("plan_identity"), candidate.get("train_gain"),
                   candidate.get("reason")) for candidate in item.candidates),
        )
        for factor_id, item in result.factors.items()
    }


def _ledger_after_first_failure(result):
    """Candidate ledger after the injected first representation failure."""
    item = result.factors["ranked"]
    failed_index = next(
        index for index, candidate in enumerate(item.candidates)
        if "injected executor failure" in str(candidate.get("reason"))
    )
    return tuple(
        (candidate.get("family"), candidate.get("parameters"),
         candidate.get("orientation"), candidate.get("status"),
         candidate.get("plan_identity"), candidate.get("train_gain"),
         candidate.get("reason"))
        for candidate in item.candidates[failed_index + 1:]
    )


def _rank_fixture():
    rng = np.random.default_rng(821)
    times, assets = 240, 40
    values = np.stack([
        rng.permutation(np.linspace(-1, 1, assets)) for _ in range(times)
    ])
    time_axis = AxisRef("time", "int", times, np.arange(times))
    asset_axis = AxisRef(
        "asset", "str", assets, np.array([f"a{i}" for i in range(assets)])
    )
    batch = FactorBatch(("ranked",), time_axis, asset_axis, values[:, :, None])
    labels = LabelBundle(
        "ranked-synthetic", values ** 2, 1, decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )
    return batch, labels


def _run_rank(batch, labels):
    from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch

    config = BatchOptimizationConfig(
        families=("REPRESENTATION_RANK",), selection_objective="rank_ic",
        bootstrap_draws=99,
    )
    return optimize_factor_batch(batch, labels, config=config, allow_research=True)


def test_identity_transport_preserves_admitted_sign_baseline_and_ownership(monkeypatch):
    batch, labels = _fixture()
    values_before = batch.values.copy()
    validity_before = batch.validity.copy()
    copied = _run(batch, labels)

    # Exercise the exact transport substitution at the reachable search-frame
    # seam while retaining pandas' normal copy behavior everywhere else.
    import pandas as pd
    original_copy = pd.DataFrame.copy
    transported = []

    import inspect
    import linecache

    def copy_or_transport(frame, *args, **kwargs):
        caller = inspect.currentframe().f_back
        source_call = (
            caller.f_code.co_name == "optimize_factor_batch"
            and caller.f_code.co_filename.endswith("factor_optimizer/research_batch.py")
            and "search_frame = frame.copy()" in linecache.getline(
                caller.f_code.co_filename, caller.f_lineno
            )
        )
        if source_call and list(frame.columns) == ["date", "asset_id", "value"]:
            transported.append(frame)
            return frame
        return original_copy(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", copy_or_transport)
    identity = _run(batch, labels)

    assert len(transported) == len(batch.factor_ids)
    assert _ledger(identity) == _ledger(copied)
    assert identity.factors["reverse"].status == "improved"
    assert identity.factors["reverse"].selected_family == "SIGN_ORIENTATION"
    assert identity.factors["reverse"].validation_candidate_identity is not None
    assert identity.factors["baseline"].selected_family == "NO_OP_RAW"
    np.testing.assert_array_equal(identity.optimized.values, copied.optimized.values)
    assert np.array_equal(batch.values, values_before, equal_nan=True)
    np.testing.assert_array_equal(batch.validity, validity_before)
    assert not np.shares_memory(identity.optimized.values, batch.values)
    assert not np.shares_memory(identity.optimized.validity, batch.validity)


def test_executor_mutation_then_exception_does_not_change_later_candidates(monkeypatch):
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    batch, labels = _rank_fixture()
    source_before = batch.values.copy()
    original_execute = ValueRepairPlan.execute

    def run_with_failure(mutate):
        failed = False

        def execute(self, frame, *args, **kwargs):
            nonlocal failed
            if self.family == "REPRESENTATION_RANK" and not failed:
                failed = True
                if mutate:
                    frame.loc[:, "date"] = -1
                    frame.loc[:, "asset_id"] = "corrupted"
                    frame.loc[:, "value"] = -777.0
                raise RuntimeError("injected executor failure")
            return original_execute(self, frame, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(ValueRepairPlan, "execute", execute)
            result = _run_rank(batch, labels)
        assert failed
        return result

    control = run_with_failure(mutate=False)
    mutated = run_with_failure(mutate=True)

    for result in (control, mutated):
        item = result.factors["ranked"]
        assert any(
            candidate.get("status") == "ineligible"
            and "injected executor failure" in str(candidate.get("reason"))
            for candidate in item.candidates
        )
        assert any(candidate.get("status") == "train_evaluated"
                   for candidate in item.candidates)
    assert _ledger_after_first_failure(mutated) == _ledger_after_first_failure(control)
    np.testing.assert_array_equal(mutated.optimized.values, control.optimized.values)
    np.testing.assert_array_equal(batch.values, source_before)


def test_rebuilt_train_frame_restores_exact_axes_order_and_ownership():
    import pandas as pd
    from factor_optimizer.candidate_recovery import rebuild_train_frame

    times = AxisRef("time", "int", 3, np.array([11, 17, 29]))
    assets = AxisRef("asset", "str", 2, np.array(["b", "a"]))
    baseline = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    frame = rebuild_train_frame(times, assets, baseline)

    assert list(frame.columns) == ["date", "asset_id", "value"]
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["date"].tolist() == [11, 11, 17, 17, 29, 29]
    assert frame["asset_id"].tolist() == ["b", "a"] * 3
    assert np.array_equal(
        frame["value"].to_numpy().reshape(baseline.shape), baseline, equal_nan=True
    )
    frame.loc[:, "value"] = -777.0
    assert np.array_equal(
        baseline, [[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]], equal_nan=True
    )


def test_mutating_isolated_candidate_frame_skips_shared_frame_recovery(monkeypatch):
    import factor_optimizer.candidate_recovery as candidate_recovery
    import factor_optimizer.shape_rank_reuse as rank_reuse
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    batch, labels = _rank_fixture()
    source_before = batch.values.copy()
    original_execute = ValueRepairPlan.execute
    original_candidate_frame = rank_reuse._candidate_frame_for_plan
    original_recovery = candidate_recovery.rebuild_train_frame
    recovery_calls = []
    failed = False

    def copied_frame(plan, frame):
        return original_candidate_frame(plan, frame).copy(deep=True)

    def record_recovery(*args, **kwargs):
        recovery_calls.append(args)
        return original_recovery(*args, **kwargs)

    def mutate_then_fail(self, frame, *args, **kwargs):
        nonlocal failed
        if self.family == "REPRESENTATION_RANK" and not failed:
            failed = True
            frame.loc[:, "value"] = -777.0
            raise RuntimeError("injected executor failure")
        return original_execute(self, frame, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(rank_reuse, "_candidate_frame_for_plan", copied_frame)
        scoped.setattr(candidate_recovery, "rebuild_train_frame", record_recovery)
        scoped.setattr(ValueRepairPlan, "execute", mutate_then_fail)
        result = _run_rank(batch, labels)

    assert failed
    assert recovery_calls == []
    assert any(candidate.get("status") == "train_evaluated"
               for candidate in result.factors["ranked"].candidates)
    np.testing.assert_array_equal(batch.values, source_before)


def test_recovery_failure_retains_current_factor_raw_and_continues(monkeypatch):
    import factor_optimizer.candidate_recovery as candidate_recovery
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    single, labels = _rank_fixture()
    batch = FactorBatch(
        ("ranked", "other"), single.time_axis, single.asset_axis,
        np.concatenate((single.values, single.values), axis=2),
    )
    source_before = batch.values.copy()
    original_execute = ValueRepairPlan.execute
    failed = False
    recovery_attempts = []

    def fail_recovery(*args, **kwargs):
        recovery_attempts.append(args)
        raise OSError("injected recovery failure")

    def mutate_then_fail(self, frame, *args, **kwargs):
        nonlocal failed
        if self.family == "REPRESENTATION_RANK" and not failed:
            failed = True
            frame.loc[:, "date"] = -1
            frame.loc[:, "asset_id"] = "corrupted"
            frame.loc[:, "value"] = -777.0
            raise RuntimeError("primary executor failure")
        return original_execute(self, frame, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(candidate_recovery, "rebuild_train_frame", fail_recovery)
        scoped.setattr(ValueRepairPlan, "execute", mutate_then_fail)
        result = _run_rank(batch, labels)

    assert failed
    failed_factor = result.factors["ranked"]
    assert failed_factor.status == "error_raw_retained"
    assert failed_factor.selected_family == "NO_OP_RAW"
    assert "primary executor failure" in failed_factor.reason
    assert any(row.get("recovery_error") == "OSError: injected recovery failure"
               and "primary executor failure" in row.get("reason", "")
               for row in failed_factor.candidates)
    assert len(recovery_attempts) == 1
    assert result.factors["other"].status != "error_raw_retained"
    assert np.array_equal(result.optimized.values, source_before, equal_nan=True)
    assert np.array_equal(batch.values, source_before, equal_nan=True)


def _run_cached_rank_and_shape(batch, labels):
    from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch

    config = BatchOptimizationConfig(
        families=("U_SHAPE_REPAIR", "REPRESENTATION_RANK"),
        maximum_candidates=64, selection_objective="rank_ic", bootstrap_draws=99,
    )
    return optimize_factor_batch(batch, labels, config=config, allow_research=True)

def test_mutation_recovery_invalidates_populated_train_rank_cache(monkeypatch):
    import factor_optimizer.shape_rank_reuse as rank_reuse
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    batch, labels = _rank_fixture()
    original_execute = ValueRepairPlan.execute
    original_prepare = rank_reuse.RankFeatureCache.prepare_train_rank
    original_bypass = rank_reuse.RankFeatureCache._bypass
    original_apply_prepared = rank_reuse.apply_u_shape_from_prepared_rank
    original_apply_rank = rank_reuse.apply_u_shape_from_rank

    def run_case(mutate):
        state = {
            "failed": False,
            "cache": None,
            "prepared": None,
            "before_fault": None,
            "bypass_state": None,
            "prepared_calls_after_failure": 0,
            "ordinary_calls_after_failure": 0,
            "ordinary_cache_clean": True,
        }

        def force_admission(*args, **kwargs):
            return True

        def prepare(cache, *args, **kwargs):
            result = original_prepare(cache, *args, **kwargs)
            state["cache"] = cache
            state["prepared"] = result
            return result

        def bypass(cache):
            before = (cache.hits, cache.misses, cache.rank_calls,
                      cache.bypasses, cache.evictions, cache._rank is not None)
            result = original_bypass(cache)
            after = (cache.hits, cache.misses, cache.rank_calls,
                     cache.bypasses, cache.evictions, cache._rank is not None)
            if before[5] and state["bypass_state"] is None:
                state["bypass_state"] = (before, after, cache._key, cache._rank)
            return result

        def apply_prepared(plan, prepared, *args, **kwargs):
            if state["failed"]:
                state["prepared_calls_after_failure"] += 1
            return original_apply_prepared(plan, prepared, *args, **kwargs)

        def apply_rank(plan, frame, cache, *args, **kwargs):
            if state["failed"]:
                if state["ordinary_calls_after_failure"] == 0:
                    state["ordinary_cache_clean"] = cache._key is None and cache._rank is None
                state["ordinary_calls_after_failure"] += 1
            return original_apply_rank(plan, frame, cache, *args, **kwargs)

        def execute(plan, frame, *args, **kwargs):
            if plan.family == "REPRESENTATION_RANK" and not state["failed"]:
                state["failed"] = True
                cache = state["cache"]
                assert state["prepared"] is not None
                assert cache is not None and cache._rank is not None and cache._key is not None
                state["before_fault"] = (
                    cache.hits, cache.misses, cache.rank_calls,
                    cache.bypasses, cache.evictions,
                )
                if mutate:
                    frame.loc[:, "value"] = -777.0
                raise RuntimeError("injected rank-cache executor failure")
            return original_execute(plan, frame, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(rank_reuse, "should_admit_u_shape_rank_reuse", force_admission)
            scoped.setattr(rank_reuse.RankFeatureCache, "prepare_train_rank", prepare)
            scoped.setattr(rank_reuse.RankFeatureCache, "_bypass", bypass)
            scoped.setattr(rank_reuse, "apply_u_shape_from_prepared_rank", apply_prepared)
            scoped.setattr(rank_reuse, "apply_u_shape_from_rank", apply_rank)
            scoped.setattr(ValueRepairPlan, "execute", execute)
            result = _run_cached_rank_and_shape(batch, labels)
        return result, state

    control, control_state = run_case(mutate=False)
    recovered, recovered_state = run_case(mutate=True)

    for result, state in ((control, control_state), (recovered, recovered_state)):
        assert state["failed"]
        assert state["prepared"] is not None
        assert state["before_fault"] is not None
        assert any(candidate.get("status") == "ineligible"
                   and "injected rank-cache executor failure" in str(candidate.get("reason"))
                   for candidate in result.factors["ranked"].candidates)
    assert recovered_state["bypass_state"] is not None
    before, after, cache_key, cached_rank = recovered_state["bypass_state"]
    assert after[0:3] == before[0:3]
    assert after[3] == before[3] + 1
    assert after[4] == before[4] + 1
    assert before[5] is True
    assert cache_key is None and cached_rank is None
    assert recovered_state["prepared_calls_after_failure"] == 0
    assert recovered_state["ordinary_calls_after_failure"] > 0
    assert recovered_state["ordinary_cache_clean"]
    assert _ledger(recovered) == _ledger(control)
    assert np.array_equal(recovered.optimized.values, control.optimized.values, equal_nan=True)
