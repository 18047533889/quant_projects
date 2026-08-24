"""Tests for P0-10 data capability boundary (scope isolation).

These verify the authorization-only capability model: the runner builds one
per-scope capability from the protocol's split plan, and the train /
validation / sealed-test execution contexts enforce their own scope's non-empty
mask without ever receiving another scope's capability or data.
"""

import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.data_capabilities import (
    DataCapability,
    DataScope,
    TestDataCapability,
    TrainDataCapability,
    ValidationDataCapability,
    build_data_capabilities,
)
from factor_optimizer.search.runner import (
    SearchConfig,
    SearchRunner,
    SealedTestExecutor,
    TrainEvaluationContext,
    ValidationEvaluationContext,
)


def _plan(
    train=None,
    validation=None,
    test=None,
    split_id="p1",
):
    return SplitPlan(
        split_id,
        train if train is not None else [True, True, False],
        validation if validation is not None else [False, False, True],
        test if test is not None else [False, False, False],
        {},
    )


def _protocol(plan):
    return EvaluationProtocol(
        plan,
        lambda trial, fid: {
            "score": 0.5,
            "cost": 1.0,
            "evidence_ref": "ev-1",
        },
    )


def _budget():
    return SearchBudget(max_trials=3, max_evaluations=3)


def _trial():
    return Trial(
        trial_id="t1",
        mutation_id="m1",
        status=TrialStatus.VALIDATING,
    )


def test_build_data_capabilities_scopes():
    plan = _plan()
    caps = build_data_capabilities(plan)
    assert set(caps) == {DataScope.TRAIN, DataScope.VALIDATION, DataScope.TEST}
    assert isinstance(caps[DataScope.TRAIN], TrainDataCapability)
    assert isinstance(caps[DataScope.VALIDATION], ValidationDataCapability)
    assert isinstance(caps[DataScope.TEST], TestDataCapability)
    assert all(isinstance(cap, DataCapability) for cap in caps.values())


def test_can_evaluate_reflects_scope_mask():
    # test scope is entirely masked off -> not evaluable.
    caps = build_data_capabilities(
        _plan(train=[True, True, False], validation=[False, False, True], test=[False, False, False])
    )
    assert caps[DataScope.TRAIN].can_evaluate(
        _plan()
    )  # default plan has train True
    assert not caps[DataScope.TEST].can_evaluate(
        _plan(test=[False, False, False])
    )
    assert caps[DataScope.VALIDATION].can_evaluate(
        _plan(validation=[False, False, True])
    )


def test_runner_builds_capabilities_and_records_split_plan():
    plan = _plan()
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=_protocol(plan),
    )
    # runner._data_capabilities built from the protocol split plan
    assert runner._data_capabilities is not None
    # R46 P0-R: the search runner's object graph contains ONLY train and
    # validation capabilities — never a TEST capability.
    assert set(runner._data_capabilities) == {DataScope.TRAIN, DataScope.VALIDATION}
    assert isinstance(runner._data_capabilities[DataScope.TRAIN], TrainDataCapability)
    assert isinstance(runner._data_capabilities[DataScope.VALIDATION], ValidationDataCapability)
    # the search split plan is recorded on the config
    assert runner.config._search_split_plan is plan
    # the runner's train capability is NOT a test capability
    assert not isinstance(runner._data_capabilities[DataScope.TRAIN], TestDataCapability)


def test_runner_without_protocol_has_no_capabilities():
    runner = SearchRunner(
        config=SearchConfig(
            budget=_budget(), require_evaluation_protocol=False
        ),
        proposal_fn=lambda: _trial(),
        evaluation_fn=lambda trial, fid: {"score": 0.5, "cost": 1.0, "evidence_ref": "e"},
    )
    assert runner._data_capabilities is None
    assert not hasattr(runner.config, "_search_split_plan")


def test_train_context_requires_non_empty_train_mask():
    plan = _plan(train=[False, False, False], validation=[False, False, True])
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=_protocol(plan),
    )
    ctx = runner.create_train_evaluation_context()
    assert isinstance(ctx, TrainEvaluationContext)
    with pytest.raises(ValueError, match="no training data"):
        ctx.evaluate(_trial(), 0)


def test_validation_context_requires_non_empty_validation_mask():
    plan = _plan(validation=[False, False, False], train=[True, True, False])
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=_protocol(plan),
    )
    ctx = runner.create_validation_evaluation_context()
    assert isinstance(ctx, ValidationEvaluationContext)
    with pytest.raises(ValueError, match="no validation data"):
        ctx.evaluate(_trial(), 0)


def test_sealed_test_executor_requires_non_empty_test_mask():
    plan = _plan(test=[False, False, False])
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=_protocol(plan),
    )
    executor = runner.create_sealed_test_executor()
    assert isinstance(executor, SealedTestExecutor)
    with pytest.raises(ValueError, match="no test data"):
        executor.evaluate_sealed_test(
            None, None, _plan(test=[False, False, False])
        )


def test_train_and_validation_never_construct_test_capability():
    plan = _plan()
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=_protocol(plan),
    )
    # R46 P0-R: the search runner's object graph contains ONLY train and
    # validation capabilities.  No TEST capability is reachable from the
    # search runner.
    assert set(runner._data_capabilities) == {DataScope.TRAIN, DataScope.VALIDATION}
    assert not isinstance(
        runner._data_capabilities[DataScope.TRAIN], TestDataCapability
    )
    assert not isinstance(
        runner._data_capabilities[DataScope.VALIDATION], TestDataCapability
    )
    # No method on the runner returns a TEST capability.
    assert runner._test_authority_broker is None
    executor = runner.create_sealed_test_executor()
    assert executor.has_test_authority is False
