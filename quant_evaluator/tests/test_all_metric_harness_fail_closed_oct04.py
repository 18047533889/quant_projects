"""Behavioral checks for failure handling in the all-metric harness itself."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_evaluator.contracts.errors import InvalidContractError


def _harness_module():
    path = Path(__file__).parent / "metrics" / "test_all_metrics_ab_contract.py"
    spec = importlib.util.spec_from_file_location("all_metric_ab_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_does_not_hide_internal_error_after_known_contract_failure(monkeypatch):
    """Catch _run accepting one known input error while hiding a later bug."""
    harness = _harness_module()
    outcomes = iter((
        InvalidContractError("Metric sample requires missing probe_pnl input"),
        RuntimeError("internal recurrence failure"),
    ))

    def evaluate(*args, **kwargs):
        raise next(outcomes)

    monkeypatch.setattr(harness, "evaluate", evaluate)
    qe_inputs = (None, None, {}, {}, {})
    with pytest.raises(RuntimeError, match="internal recurrence failure"):
        harness._run(None, None, "sample", qe_inputs)


def test_ic_decay_is_declared_unavailable_only_for_horizon_mean_ic(monkeypatch):
    """Catch the declared gap allowing an unrelated ic_decay failure."""
    harness = _harness_module()
    qe_inputs = (None, None, {}, {}, {})

    def unavailable(*args, **kwargs):
        raise InvalidContractError(
            "Metric ic_decay requires unavailable artifact builder HorizonMeanIC"
        )

    monkeypatch.setattr(harness, "evaluate", unavailable)
    harness.test_every_metric_is_executable_or_declares_its_requirement(
        "ic_decay", qe_inputs
    )

    def unrelated_contract(*args, **kwargs):
        raise InvalidContractError("ic_decay: unexpected factor-axis contract failure")

    monkeypatch.setattr(harness, "evaluate", unrelated_contract)
    with pytest.raises(AssertionError):
        harness.test_every_metric_is_executable_or_declares_its_requirement(
            "ic_decay", qe_inputs
        )


def test_batch_single_probe_does_not_drop_candidate_on_internal_error(monkeypatch):
    """Catch batch coverage passing after silently removing a broken metric."""
    harness = _harness_module()
    candidates = ("broken", "m1", "m2", "m3", "m4", "m5")
    monkeypatch.setattr(harness, "_BATCH", candidates)

    def evaluate(fb, lb, *, metrics, **kwargs):
        if tuple(metrics) == ("broken",):
            raise RuntimeError("batch candidate failed internally")
        return SimpleNamespace(metric_values={
            metric_id: SimpleNamespace(value=1.0) for metric_id in metrics
        })

    monkeypatch.setattr(harness, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="batch candidate failed internally"):
        harness.test_batch_evaluation_equals_single_metric_evaluation(
            (None, None, {}, {}, {})
        )
