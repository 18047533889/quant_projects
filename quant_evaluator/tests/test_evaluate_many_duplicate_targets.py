"""A batch evaluation must never silently discard a label result."""

import numpy as np
import pytest

from quant_evaluator.contracts.label_bundle import LabelBundle

from quant_evaluator.api import evaluate_many as api_module


@pytest.mark.parametrize("backend", ["cpu", "auto", "cuda_strict"])
def test_duplicate_target_ids_fail_before_any_evaluation(monkeypatch, backend):
    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("duplicate labels must be rejected before evaluation")

    monkeypatch.setattr(api_module, "evaluate", unexpected_evaluation)
    labels = tuple(LabelBundle(
        "same", np.array([float(seed), float(seed + 1)]), 1,
        decision_time=(0, 1), label_start_time=(0, 1), label_end_time=(1, 2),
    ) for seed in (0, 10))

    with pytest.raises(ValueError, match="unique target_id"):
        api_module.evaluate_many(object(), labels, backend=backend)
